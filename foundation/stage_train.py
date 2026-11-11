"""Run one foundation training stage in its original source environment."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

from .retrain import ROOT, file_sha, read, require, specification, write


ENDPOINTS = {"p120": 120, "r180": 180}


def checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def two_versions(save):
    def retained(path, *args, **kwargs):
        path = Path(path)
        if path.name != "last.pt" or not path.is_file():
            return save(path, *args, **kwargs)
        pinned = path.with_name(f".previous-{uuid.uuid4().hex}.pt")
        try:
            try:
                os.link(path, pinned)
            except OSError:
                shutil.copyfile(path, pinned)
            result = save(path, *args, **kwargs)
            pinned.replace(path.with_name("last.previous.pt"))
            return result
        finally:
            if pinned.exists():
                pinned.unlink()
    return retained


def configure_device(smoke):
    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 2))
    torch.set_num_interop_threads(1)
    if smoke:
        require(not torch.cuda.is_available(), "CPU smoke must hide CUDA")
        return torch.device("cpu")
    require(torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), "Use one BF16-capable GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch.device("cuda:0")


def phase_position(work, stage, target, smoke):
    path = work / stage / "last.pt"
    if not path.exists():
        directory = path.parent
        if directory.exists() and any(directory.iterdir()):
            # Preserve logs from an interruption before the first committed epoch.
            directory.rename(work / f"{stage}.uncommitted.{time.time_ns()}")
        return path, 0
    state = checkpoint(path)
    expected_format = 3 if stage == "r180" else 2
    require(state.get("format_version") == expected_format and state.get("training_state")
            and set(("model", "optimizer", "scheduler", "scaler", "rng")) <= set(state)
            and not state["extra"].get("epoch_truncated")
            and bool(state["config"].get("portable_smoke", False)) == smoke, f"Invalid full checkpoint {path}")
    if stage == "r180":
        require("donor_rng" in state, "Refinement checkpoint lacks donor RNG")
    epoch = state["epoch"]
    require(0 < epoch <= target and state["scheduler"]["last_epoch"] == epoch, "Checkpoint schedule differs")
    return path, epoch


def parent_packet(work, stage, smoke):
    path, epoch = phase_position(work, stage, 2 if smoke else ENDPOINTS[stage], smoke)
    require(epoch == (2 if smoke else ENDPOINTS[stage]), f"Finish {stage} before dependent training")
    return path, checkpoint(path)


def calibrate(model, cfg, samples, device, *, r180):
    """Fit only the two foreground BN moments using clean training images."""
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from occreid.bpbreid_model import reference_pool
    from occreid.data import ReIDDataset

    model.eval()
    dtype = torch.bfloat16 if device.type == "cuda" else None
    amp = lambda: torch.autocast(device.type, dtype=dtype, enabled=dtype is not None)
    linear, bn1, relu = model.head.foreground_reduce.layers[:3]
    batch_size = 64 if r180 else 128
    if device.type == "cpu":
        batch_size = 8
    loader = DataLoader(ReIDDataset(samples, cfg["image_size"], False, parts=8), batch_size=batch_size,
                        shuffle=False, num_workers=2 if device.type == "cuda" else 0,
                        pin_memory=device.type == "cuda")
    projected, raw, masses = [], [], []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            with amp():
                _, maps = model.backbone((image - .5) / .5)
                spatial = model.head.msf(maps)
                probability = model.head.pixel_classifier(spatial).float().softmax(1)[:, 1:]
                foreground, _ = reference_pool(spatial, probability)
                mass = probability.amax(1).mean((-1, -2)).clamp_min(1e-6)
                if r180:
                    raw.append(foreground.float().cpu())
                    masses.append(mass.cpu())
                else:
                    projected.append(linear(foreground / mass[:, None]).float().cpu())
        if r180:
            for values, mass in zip(torch.cat(raw).split(512), torch.cat(masses).split(512)):
                with amp():
                    projected.append(linear(values.to(device) / mass.to(device)[:, None]).cpu())
        z = torch.cat(projected)
        require(len(z) == len(samples) and len(z) > 1, "Incomplete training-only calibration")
        stats = dict(mean1=z.double().mean(0).float(), var1=z.double().var(0, unbiased=True).float())
        activation = []
        if r180:
            for values in z.split(512):
                y = F.batch_norm(values.to(device), stats["mean1"].to(device), stats["var1"].to(device),
                                 bn1.weight, bn1.bias, False, 0., bn1.eps)
                activation.append(F.relu(y).cpu())
        else:
            bn1.running_mean.copy_(stats["mean1"])
            bn1.running_var.copy_(stats["var1"])
            with amp():
                activation.append(relu(bn1(z.to(device=device, dtype=dtype or torch.float32))).float().cpu())
        a = torch.cat(activation).double()
        stats.update(mean2=a.mean(0).float(), var2=a.var(0, unbiased=True).float())
        require(all(bool(torch.isfinite(value).all()) for value in stats.values()), "Nonfinite BN moments")
    return stats


def calibration_stage(work, data, device, smoke):
    import torch
    from occreid.model import build_model

    source, parent = parent_packet(work, "r180", smoke)
    binding = file_sha(source)
    output = work / "r180_train_only_bn.pt"
    if output.exists():
        require(checkpoint(output)["parent_sha256"] == binding, "Calibration parent differs")
        print(f"Calibration already committed at {output}", flush=True)
        return
    cfg = parent["config"]
    model = build_model(cfg, None)
    model.load_state_dict(parent["model"], strict=True)
    model.head.context_residual = None
    model.to(device).requires_grad_(False)
    stats = calibrate(model, cfg, data["train"], device, r180=True)
    temp = output.with_suffix(".tmp")
    torch.save(dict(GWAP_recal=stats, parent_sha256=binding, training_images=len(data["train"])), temp)
    temp.replace(output)
    print(f"Fitted foreground BN from {len(data['train'])} training images", flush=True)


def read_training_split(data_root):
    from occreid.data import read_dataset
    from occreid.engine import dataset_signature

    spec = specification()
    data = read_dataset(data_root)
    require(
        all(len(data[split]) == count for split, count in spec["split_counts"].items())
        and data["num_classes"] == spec["train_identities"],
        "Dataset does not match the Occluded-Duke split",
    )
    require(
        dataset_signature(data["train"]) == spec["dataset_signature"],
        "Training identity mapping differs from the Occluded-Duke protocol",
    )
    data.update(query=[], gallery=[])
    return data


def smoke_subset(data):
    groups = {}
    for sample in data["train"]:
        groups.setdefault(sample.pid, {}).setdefault(sample.cam, []).append(sample)
    for pid_a, cams_a in groups.items():
        for pid_b, cams_b in groups.items():
            if pid_b <= pid_a:
                continue
            cameras = sorted(
                cam for cam in cams_a.keys() & cams_b.keys()
                if len(cams_a[cam]) >= 2 and len(cams_b[cam]) >= 2
            )
            if len(cameras) >= 2:
                data["train"] = [
                    sample
                    for group in (cams_a, cams_b)
                    for camera in cameras[:2]
                    for sample in group[camera][:2]
                ]
                return
    raise ValueError("Could not find a two-identity, two-camera smoke batch")


def build_config(args, label_signature):
    cfg = read(ROOT / "configs" / f"{args.stage}.json")
    cfg.update(
        data_root=str(args.data_root),
        label_root=str(args.part_labels),
        label_signature=label_signature,
        output_dir=str(args.work / args.stage),
        pretrained=str(args.pretrained),
        eval_period=0,
        checkpoint_period=0,
    )
    if args.stage == "r180":
        cfg["repair"]["parent_checkpoint"] = str(args.work / "p120/last.pt")
    if args.smoke:
        cfg.update(
            epochs=2, image_size=[128, 64], batch_size=8, workers=0, test_batch_size=8,
            precision="fp32", warmup_epochs=0, log_interval=1, diagnostic_interval=1,
            portable_smoke=True,
        )
        if args.stage == "r180":
            cfg["repair"].update(warmup_epochs=0, ramp_epochs=1)
    return cfg


def train_stage(args, cfg, data, device):
    from occreid import engine

    stage, work = args.stage, args.work
    target = 2 if args.smoke else ENDPOINTS[stage]
    last, current = phase_position(work, stage, target, args.smoke)
    if current == target:
        print(f"{stage} already committed through epoch {current}", flush=True)
        return

    resume = last if current else None
    if stage == "p120":
        engine.save_checkpoint = two_versions(engine.save_checkpoint)
        engine.train(cfg, data, device, resume=resume, stop_after=target)
    else:
        from occreid import refine

        parent_packet(work, "p120", args.smoke)
        refine.save_refinement = two_versions(refine.save_refinement)
        refine.refine(cfg, data, device, resume=resume, stop_after=target)

    _, epoch = phase_position(work, stage, target, args.smoke)
    require(epoch == target, f"Training did not reach {stage} epoch {target}")
    state = checkpoint(last)
    write(
        last.parent / "RETRAIN_ENDPOINT.json",
        {
            "epoch": epoch, "sha256": file_sha(last), "stage": stage,
            "updates": sum(row["successful_updates"] for row in state["extra"]["history"]),
            "smoke": args.smoke, "training_state": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("p120", "r180", "calibrate", "import-labels", "check-labels"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--part-labels", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--heatmaps", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / ("baseline_src" if args.stage == "p120" else "src")))

    from occreid.vei_labels import audit_vei_labels, import_official_labels, validate_manifest

    data = read_training_split(args.data_root)
    if args.stage == "import-labels":
        if args.heatmaps is None:
            parser.error("import-labels requires --heatmaps")
        summary = import_official_labels(
            data["train"], args.heatmaps, args.part_labels, layout="auto",
            mode="official_compatible",
        )
        print(json.dumps(summary, indent=2))
        return

    label_manifest = validate_manifest(args.part_labels)
    require(label_manifest["mode"] == "official_compatible", "Use the BPBreID-compatible part cache")
    if args.stage == "check-labels":
        summary = audit_vei_labels(data["train"], args.part_labels)
        require(summary["missing"] == 0, f"Missing part labels: {summary['missing']}")
        require(summary["usable_fraction"] >= 0.5, "Insufficient usable part labels")
        print(json.dumps(summary, indent=2))
        return

    if args.work is None or args.pretrained is None:
        parser.error("Foundation stages require --work and --pretrained")
    device = configure_device(args.smoke)
    if args.smoke:
        smoke_subset(data)
    if args.stage == "calibrate":
        calibration_stage(args.work, data, device, args.smoke)
    else:
        train_stage(args, build_config(args, label_manifest["signature"]), data, device)


if __name__ == "__main__":
    main()
