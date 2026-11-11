"""Train and evaluate the evidence-constrained Occluded-Duke model."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from foundation.retrain import (
    check_pretrained,
    dataset_manifest,
    file_sha,
    load_supervision,
    read,
    require,
    run_lock,
    specification,
    write,
)


ROOT = Path(__file__).resolve().parent
WORK_DIR = Path("runs/occluded_duke")


def command(*arguments):
    subprocess.run(
        [sys.executable, "-B", "-u", *map(str, arguments)],
        cwd=ROOT,
        check=True,
    )


def local_path(value):
    return Path(value).expanduser().resolve()


def train(args):
    pretrained = check_pretrained(args.pretrained)
    labels = args.part_labels / "manifest.json"
    require(labels.is_file(), f"Missing label manifest: {labels}. Run import-labels first.")
    _, _, supervision = load_supervision(args.source_supervision)
    data_root, _ = dataset_manifest(args.data_root)
    binding = {
        "dataset_signature": specification()["dataset_signature"],
        "pretrained_sha256": pretrained,
        "label_manifest_sha256": file_sha(labels),
        "shared_source": supervision,
    }
    work = args.work_dir
    resume = args.action == "resume"
    with run_lock(work):
        record = work / "training_inputs.json"
        if record.exists():
            require(resume, f"Use resume for the existing run: {work}")
            require(read(record) == binding, "Training inputs differ from the saved run")
        else:
            require(not resume, f"No training run to resume: {work}")
            require(
                all(p.name == ".train.lock" for p in work.iterdir()),
                f"Choose an empty work directory: {work}",
            )
            command(
                "-m", "foundation.stage_train", "check-labels",
                "--data-root", data_root, "--part-labels", args.part_labels,
            )
            write(record, binding)
        for stage in ("p120", "r180", "calibrate"):
            command(
                "-m", "foundation.stage_train", stage, "--work", work / "foundation",
                "--data-root", data_root, "--part-labels", args.part_labels,
                "--pretrained", args.pretrained,
            )
        command("portable_model.py", "--work", work)
        extra = ["--resume"] if resume else []
        command(
            "train_evidence.py", "--work", work, "--data-root", data_root,
            "--source-supervision", args.source_supervision, *extra,
        )
    print(f"Training complete. Evaluate the checkpoints in {work}.", flush=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)

    def add_command(name, help_text):
        subparser = commands.add_parser(name, help=help_text)
        subparser.add_argument("--data-root", type=local_path, required=True)
        return subparser

    check = add_command("check-data", "Verify the external Occluded-Duke inputs")
    check.add_argument("--part-labels", type=local_path)
    check.add_argument("--pretrained", type=local_path)
    check.add_argument("--source-supervision", type=local_path)

    labels = add_command("import-labels", "Convert BPBreID heatmaps to the eight-part cache")
    labels.add_argument("--heatmaps", type=local_path, required=True)
    labels.add_argument("--part-labels", type=local_path, required=True)

    for action, help_text in (
        ("train", "Train P120, R180 and the frozen evidence branch"),
        ("resume", "Continue a previously committed training run"),
    ):
        training = add_command(action, help_text)
        training.add_argument("--part-labels", type=local_path, required=True)
        training.add_argument("--pretrained", type=local_path, required=True)
        training.add_argument("--source-supervision", type=local_path, required=True)
        training.add_argument("--work-dir", type=local_path, default=str(WORK_DIR))
        training.add_argument("--gpu", type=int, default=0)

    test = add_command("test", "Evaluate the three scores on the complete gallery")
    test.add_argument("--work-dir", dest="work", type=local_path, default=str(WORK_DIR))
    test.add_argument("--output", type=local_path, required=True)
    test.add_argument("--native-checkpoint", type=local_path)
    test.add_argument("--bn-statistics", type=local_path)
    test.add_argument("--evidence-checkpoint", type=local_path)
    test.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    test.add_argument("--gpu", type=int, default=0)
    test.add_argument("--batch-size", type=int, default=256)
    test.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if getattr(args, "gpu", 0) < 0:
        parser.error("--gpu must be nonnegative")
    if args.action == "test":
        reference = (args.native_checkpoint, args.bn_statistics, args.evidence_checkpoint)
        if any(reference) and not all(reference):
            parser.error("Supply all three reference checkpoints together")
        if args.batch_size <= 0 or args.workers < 0:
            parser.error("--batch-size must be positive and --workers nonnegative")
        args.reference = reference if all(reference) else None
    return args


def main():
    args = parse_arguments()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        str(args.gpu) if getattr(args, "device", "cuda") == "cuda" and hasattr(args, "gpu") else "-1"
    )
    if args.action == "check-data":
        data_root, rows = dataset_manifest(args.data_root, check_evaluation_images=True)
        summary = {"splits": {split: len(items) for split, items in rows.items()}}
        if args.pretrained:
            summary["pretrained_sha256"] = check_pretrained(args.pretrained)
        if args.part_labels:
            command(
                "-m", "foundation.stage_train", "check-labels",
                "--data-root", data_root, "--part-labels", args.part_labels,
            )
        if args.source_supervision:
            _, _, summary["shared_source"] = load_supervision(args.source_supervision)
        print(json.dumps(summary, indent=2))
    elif args.action == "import-labels":
        command(
            "-m", "foundation.stage_train", "import-labels",
            "--data-root", args.data_root, "--part-labels", args.part_labels,
            "--heatmaps", args.heatmaps,
        )
    elif args.action in ("train", "resume"):
        train(args)
    else:
        if args.reference:
            for path in args.reference:
                require(path.is_file(), f"Missing reference checkpoint: {path}")
        from evaluate_full import main as evaluate

        evaluate(args)


if __name__ == "__main__":
    main()
