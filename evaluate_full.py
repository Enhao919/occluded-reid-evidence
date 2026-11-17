"""Evaluate the final TIERNet ranking on Occluded-Duke."""

import json

import numpy as np

from portable_model import load_complete
from checkpoints import state_hash
from evaluate_checkpoint import ImageDataset, aggregate, atomic_json, json_digest, query_records, require
from foundation.retrain import dataset_manifest, specification


FEATURE_KEYS = ("global", "foreground", "parts", "visibility")


def extract(model, data, rows, config, args, *, keep_features):
    import torch
    from torch.utils.data import DataLoader

    from bounded_graph import evidence_quality

    bank = {key: [] for key in FEATURE_KEYS}
    quality = []
    receipts = {}
    loader = DataLoader(
        ImageDataset(data, rows, config["image_size"]),
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=args.device == "cuda",
    )
    with torch.inference_mode():
        for indexes, images, hashes in loader:
            if keep_features:
                receipts.update((rows[i]["path"], sha) for i, sha in zip(indexes.tolist(), hashes))
            with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
                output = model(images.to(args.device, non_blocking=True))
            values = evidence_quality(output).float().cpu()
            require(torch.isfinite(values).all().item(), "Nonfinite evidence quality")
            quality.append(values)
            if keep_features:
                for key in bank:
                    value = output.get(key + "_eval", output[key]).float().cpu()
                    require(torch.isfinite(value).all().item(), f"Nonfinite feature: {key}")
                    bank[key].append(value)
    features = {key: torch.cat(values) for key, values in bank.items()} if keep_features else None
    return features, torch.cat(quality).numpy(), receipts


def main(args):
    import torch

    from occreid.evaluation import retrieval_distance
    from reciprocal_graph import assemble_owned, rerank_owned
    from runtime import seed_everything

    destination = args.output / "RESULT.json"
    require(not destination.exists(), "Choose a new evaluation output directory")
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, config, _ = load_complete(args.work, args.device, args.reference, args.checkpoint)
    seed_everything(101)
    frozen_state = state_hash(model.state_dict())
    data, manifest = dataset_manifest(args.data_root)
    train, query, gallery = (manifest[key] for key in ("train", "query", "gallery"))

    print("Calibrating evidence quality on the training split", flush=True)
    _, train_quality, _ = extract(model, data, train, config, args, keep_features=False)
    positive = train_quality[train_quality > 0].astype(np.float64)
    require(len(positive) > 0, "No positive evidence scores on the training split")
    scale = float(np.median(positive))

    print("Extracting query and gallery features", flush=True)
    q, q_quality, q_receipts = extract(model, data, query, config, args, keep_features=True)
    g, g_quality, g_receipts = extract(model, data, gallery, config, args, keep_features=True)
    require(
        json_digest({**q_receipts, **g_receipts}) == specification()["evaluation_image_set_sha256"],
        "Query or gallery images differ from the evaluation protocol",
    )
    require(state_hash(model.state_dict()) == frozen_state, "Model state changed during evaluation")
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    def distance(a, b):
        return retrieval_distance(a, b, mode="parts", rule="bpbreid", chunk_size=128, device=args.device)

    print("Re-ranking the full gallery with ECGR", flush=True)
    full = assemble_owned(dict(qg=distance(q, g), qq=distance(q, q).T.copy(), gg=distance(g, g)))
    del q, g
    scores = rerank_owned(full, len(query), quality=np.concatenate([q_quality, g_quality]), quality_scale=scale)
    metrics = aggregate(query_records(scores, query, gallery, "TIERNet"))
    result = {
        "dataset": "Occluded-Duke",
        "rank1_percent": 100 * metrics["rank1"],
        "map_percent": 100 * metrics["mAP"],
        "query_images": len(query),
        "gallery_images": len(gallery),
    }
    atomic_json(destination, result)
    print(json.dumps(result, indent=2), flush=True)
