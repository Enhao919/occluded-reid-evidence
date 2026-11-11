"""Train only the allocation branch, preserving the frozen native model."""

import os
import random


def fit(
    phase,
    arm,
    model,
    cfg,
    train,
    cases,
    references,
    evidence,
    plan,
    binding,
    log,
    resume,
):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from allocation import balanced_point_loss
    from roles import support_coefficient
    from checkpoints import state_hash
    from occreid.bpbreid_losses import reference_triplet
    from occreid.losses import weighted_ce
    from reference_state import digest, publish_resume, require
    import evaluate_checkpoint as e

    policy = plan["training"]
    by_path = {r["path"]: i for i, r in enumerate(train)}
    dataset = e.ImageDataset(cfg["data_root"], train, cfg["image_size"])
    cache = {}

    def image(index):
        if index not in cache:
            _, value, sha = dataset[index]
            require(sha == train[index]["sha256"], "Changed training image")
            cache[index] = value.pin_memory()
        return cache[index]

    def batch(ids, epoch):
        selected, targets = [], []
        for case_index in ids.tolist():
            case, point = cases[case_index], evidence[case_index]
            for side in ("a", "b"):
                index = by_path[case[side]]
                if index not in selected:
                    selected.append(index)
                joint = torch.tensor(point["joint"])
                conflict = torch.tensor(point["conflict"])
                valid = (joint != 0) | conflict
                target_direction = joint if side == "a" else -joint
                state = torch.where(
                    target_direction > 0, 0, torch.where(target_direction < 0, 1, 2)
                )
                if valid.any():
                    targets.append(
                        dict(
                            image=selected.index(index),
                            grid=torch.tensor(point[side + "_grid"])[valid],
                            part=torch.tensor(point["part"])[valid],
                            state=state[valid],
                        )
                    )
            cams = case["reference_cameras"]
            for identity in case["ids"]:
                for j in range(2):
                    cam = cams[(epoch + j) % len(cams)]
                    candidates = references[identity, cam]
                    index = candidates[(epoch + case_index + j) % len(candidates)]
                    if index not in selected:
                        selected.append(index)
        x = torch.stack([image(i) for i in selected]).cuda(non_blocking=True)
        pid = torch.tensor([train[i]["pid"] for i in selected], device="cuda")
        source = torch.tensor(selected, device="cuda")
        return x, pid, source, targets

    head = model.allocation
    initial = state_hash(head.state_dict())
    start = head.state_score.weight.detach().clone()
    optimizer = torch.optim.AdamW(head.parameters(), lr=policy["lr"])
    rng = torch.Generator().manual_seed(policy["seed"])
    out = phase / "runs" / arm
    out.mkdir(parents=True, exist_ok=True)
    last = out / "last.pt"
    first, history, audits = 0, [], []
    if last.exists():
        require(resume, "Existing allocation run requires resume")
        packet = torch.load(last, map_location="cpu", weights_only=False)
        require(
            packet["format"] == "target_allocation_resume_v1"
            and packet["binding"] == binding
            and packet["arm"] == arm
            and packet["initial"] == initial,
            "Wrong allocation resume",
        )
        head.load_state_dict(packet["model"], strict=True)
        optimizer.load_state_dict(packet["optimizer"])
        first, history, audits = packet["epoch"], packet["history"], packet["audits"]
        rng.set_state(packet["rng"]["shuffle"])
        random.setstate(packet["rng"]["python"])
        np.random.set_state(packet["rng"]["numpy"])
        torch.set_rng_state(packet["rng"]["torch"])
        torch.cuda.set_rng_state_all(packet["rng"]["cuda"])

    for epoch in range(first + 1, policy["epochs"] + 1):
        head.train()
        order = torch.randperm(len(cases), generator=rng)
        sums = dict(gilt=0.0, support=0.0, protection=0.0, total=0.0)
        gradients, counts, image_count = [], [0, 0, 0], 0
        steps = list(order.split(policy["pairs_per_batch"]))
        for step, ids in enumerate(steps, 1):
            images, pid, source, targets = batch(ids, epoch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(images)
            triplet, triplet_audit = reference_triplet(
                output["parts"], pid, source, cfg["triplet_margin"]
            )
            ce = weighted_ce(
                output["concat_logits"], pid, smoothing=cfg["label_smoothing"]
            )
            gilt = (
                cfg["loss_weights"]["concat_ce"] * ce
                + cfg["loss_weights"]["local_triplet"] * triplet
            )
            support, nc = balanced_point_loss(output["source_states"], targets)
            protection = (
                1
                - F.cosine_similarity(
                    output["parts"], output["native_parts"].float(), dim=-1
                )
            ).mean()
            coefficient = support_coefficient(arm, policy)
            loss = (
                gilt + coefficient * support + policy["protection_weight"] * protection
            )
            if step == 1 and epoch in (1, 3, 6):
                components = dict(gilt=gilt, support=support, protection=protection)
                vectors = {}
                for name, value in components.items():
                    gs = torch.autograd.grad(
                        value,
                        tuple(head.parameters()),
                        retain_graph=True,
                        allow_unused=True,
                    )
                    vectors[name] = torch.cat(
                        [
                            (torch.zeros_like(p) if g is None else g).flatten()
                            for p, g in zip(head.parameters(), gs)
                        ]
                    )
                norms = {name: float(g.norm()) for name, g in vectors.items()}
                audit = dict(
                    epoch=epoch,
                    step=step,
                    gradient_norm=norms,
                    state_counts=nc,
                    identity_support_gradient_cosine=float(
                        F.cosine_similarity(vectors["gilt"], vectors["support"], dim=0)
                    ),
                    triplet=triplet_audit,
                    all_finite=all(
                        torch.isfinite(g).all().item() for g in vectors.values()
                    ),
                )
                require(audit["all_finite"], "Nonfinite component gradients")
                audits.append(audit)
                log("gradient_audit", arm=arm, **audit)
            loss.backward()
            require(
                torch.isfinite(loss).item()
                and all(
                    p.grad is not None and torch.isfinite(p.grad).all().item()
                    for p in head.parameters()
                ),
                "Disconnected or nonfinite allocation gradients",
            )
            gradients.append(float(head.state_score.weight.grad.norm()))
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            for name, value in dict(
                gilt=gilt, support=support, protection=protection, total=loss
            ).items():
                sums[name] += float(value.detach())
            counts = [a + b for a, b in zip(counts, nc)]
            image_count += len(images)
        row = dict(
            epoch=epoch,
            updates=len(steps),
            losses={k: v / len(steps) for k, v in sums.items()},
            state_counts=counts,
            images=image_count,
            nonzero_gradient_steps=sum(g > 0 for g in gradients),
            max_gradient=max(gradients),
            parameter_delta=float((head.state_score.weight.detach() - start).norm()),
            order_sha256=e.json_digest(order.tolist()),
        )
        require(
            row["nonzero_gradient_steps"] > 0 and row["parameter_delta"] > 0,
            "Allocation loss stopped updating",
        )
        history.append(row)
        packet = dict(
            format="target_allocation_resume_v1",
            arm=arm,
            epoch=epoch,
            binding=binding,
            initial=initial,
            model=head.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=dict(
                shuffle=rng.get_state(),
                python=random.getstate(),
                numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(),
            ),
            history=history,
            audits=audits,
            config=cfg,
            policy=policy,
            frozen_parent=plan["native"],
            model_state_sha256=state_hash(head.state_dict()),
        )
        publish_resume(out, packet, torch.save)
        if epoch in (3, 5, 6):
            endpoint = out / f"epoch{epoch}_weights.pt"
            if not endpoint.exists():
                os.link(last, endpoint)
            require(digest(endpoint) == digest(last), "Conflicting immutable endpoint")
        log("training", arm=arm, **row)
    saved = torch.load(last, map_location="cpu", weights_only=False)
    clone = torch.optim.AdamW(head.parameters(), lr=policy["lr"])
    clone.load_state_dict(saved["optimizer"])
    require(
        saved["epoch"] == policy["epochs"]
        and saved["history"] == history
        and saved["model_state_sha256"] == state_hash(head.state_dict())
        and torch.equal(saved["rng"]["shuffle"], rng.get_state()),
        "Invalid resume roundtrip",
    )
    for key, saved_state in saved["optimizer"]["state"].items():
        restored = clone.state_dict()["state"][key]
        require(
            all(
                (
                    torch.equal(value.cpu(), restored[name].cpu())
                    if torch.is_tensor(value)
                    else value == restored[name]
                )
                for name, value in saved_state.items()
            ),
            "Optimizer roundtrip mismatch",
        )
    model.eval()
    return dict(
        initial=initial,
        history=history,
        audits=audits,
        endpoint_sha256=digest(last),
        state_sha256=state_hash(head.state_dict()),
        resume_roundtrip=True,
    )
