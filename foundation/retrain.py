"""Input checks and run-directory utilities for the training pipeline."""

from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import uuid


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def specification():
    return read(ROOT / "configs/occluded_duke.json")


def json_sha(value):
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def dataset_manifest(data_root, *, check_evaluation_images=False):
    """Derive the locked sample order from an external Occluded-Duke installation."""
    from foundation.src.occreid.data import read_dataset

    data = read_dataset(data_root)
    root = data["root"]
    spec = specification()
    require(data["num_classes"] == spec["train_identities"], "Wrong training identity count")
    original_ids = {label: pid for pid, label in data["id_mapping"].items()}
    receipts = {}
    manifest = {}
    for split, count in spec["split_counts"].items():
        samples = data[split]
        require(len(samples) == count, f"Wrong {split} split size: expected {count}, got {len(samples)}")
        rows = []
        for sample in samples:
            path = Path(sample.path)
            relative = path.relative_to(root).as_posix()
            row = {"path": relative, "pid": sample.pid, "cam": sample.cam}
            if split == "train":
                row.update(original_pid=original_ids[sample.pid], source=sample.source, sha256=file_sha(path))
            else:
                row["original_pid"] = sample.pid
                if check_evaluation_images:
                    receipts[relative] = file_sha(path)
            rows.append(row)
        require(
            json_sha(rows) == spec["split_manifest_sha256"][split],
            f"{split} split differs from the reference protocol",
        )
        manifest[split] = rows
    if check_evaluation_images:
        require(
            json_sha(receipts) == spec["evaluation_image_set_sha256"],
            "Query/gallery image bytes differ from the reference protocol",
        )
    return root, manifest


def load_supervision(directory):
    spec = specification()["shared_source"]
    cohort = read(directory / "cohort.json")
    cohort = {name: cohort[name] for name in ("cases", "references")}
    with gzip.open(directory / "weak_points.json.gz", "rb") as stream:
        payload = stream.read()
    fingerprints = {
        "cohort_sha256": json_sha(cohort),
        "weak_points_sha256": hashlib.sha256(payload).hexdigest(),
    }
    require(fingerprints == spec, "Shared-source annotations differ from the reference recipe")
    points = json.loads(payload)
    require(len(cohort["cases"]) == len(points), "Shared-source pairs and point annotations differ")
    return cohort, points, fingerprints


def check_pretrained(path):
    require(path.is_file(), f"Missing SOLIDER Swin-B checkpoint: {path}")
    sha = file_sha(path)
    require(sha in specification()["pretrained_sha256"], "SOLIDER Swin-B checkpoint SHA256 differs")
    return sha


@contextmanager
def run_lock(work):
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".train.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
