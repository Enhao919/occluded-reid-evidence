from dataclasses import dataclass
from pathlib import Path
import math
import re
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as TF

PART_NAMES = ('background', 'head', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg')
FLIP = [0, 1, 2, 4, 3, 6, 5]
SPLITS = ('bounding_box_train', 'query', 'bounding_box_test')


@dataclass(frozen=True)
class Sample:
    path: str
    pid: int
    cam: int
    source: int


def resolve_root(root):
    root = Path(root).expanduser().resolve()
    for p in (root, root/'Occluded-Duke', root/'Occluded-Duke'/'Occluded-Duke'):
        if all((p/s).is_dir() for s in SPLITS):
            return p
    raise FileNotFoundError(f'Missing {SPLITS} under {root}')


def read_dataset(root):
    root = resolve_root(root)
    result = {}
    for split in SPLITS:
        rows = []
        for p in sorted((root/split).iterdir()):
            if p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            m = re.fullmatch(r'(-?\d+)_c([1-8])_f\d+\.(jpg|jpeg|png)', p.name, re.I)
            if not m:
                raise ValueError(f'Malformed Occluded-Duke filename: {p}')
            pid, cam = int(m[1]), int(m[2])-1
            if pid >= 0:
                rows.append(Sample(str(p), pid, cam, len(rows)))
        if not rows:
            raise ValueError(f'No images: {root/split}')
        result[split] = rows
    train_ids = {x.pid for x in result[SPLITS[0]]}
    test_ids = {x.pid for s in SPLITS[1:] for x in result[s]}
    if train_ids & test_ids:
        raise ValueError(f'Train/test identity overlap: {sorted(train_ids & test_ids)[:8]}')
    mapping = {p: i for i, p in enumerate(sorted(train_ids))}
    result['train'] = [Sample(x.path, mapping[x.pid], x.cam, x.source) for x in result[SPLITS[0]]]
    result['gallery'] = result[SPLITS[2]]
    result['num_classes'] = len(mapping)
    result['id_mapping'] = mapping
    result['root'] = root
    return result


def flip_labels(label):
    flipped = label.flip(-1)
    table = torch.tensor(FLIP + list(range(7, 256)), device=label.device)
    return table[flipped.long()]


class IdentityBatchSampler(Sampler):
    """Fixed-length epoch, deterministic PK batches with camera diversity."""
    def __init__(self, samples, batch_size=32, instances=4, seed=42):
        if instances < 2 or batch_size % instances or batch_size//instances < 2:
            raise ValueError('Require batch=P*K with P>=2 and K>=2')
        self.groups = {}
        for i, s in enumerate(samples):
            self.groups.setdefault(s.pid, {}).setdefault(s.cam, []).append(i)
        self.samples, self.batch_size, self.instances = samples, batch_size, instances
        self.pids = sorted(self.groups)
        self.p = batch_size//instances
        if len(self.pids) < self.p:
            raise ValueError('Fewer identities than P in PK batch')
        self.seed, self.epoch = seed, 0
        self.batches = max(1, math.ceil(len(samples)/batch_size))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        for _ in range(self.batches):
            batch = []
            for pid in rng.choice(self.pids, self.p, replace=False):
                groups = self.groups[pid]
                cameras = list(groups)
                rng.shuffle(cameras)
                picked = [int(rng.choice(groups[c])) for c in cameras[:self.instances]]
                remaining = [i for v in groups.values() for i in v if i not in picked]
                need = self.instances-len(picked)
                if need and remaining:
                    take = min(need, len(remaining))
                    picked.extend(rng.choice(remaining, take, replace=False).tolist())
                if len(picked) < self.instances:
                    pool = [i for v in groups.values() for i in v]
                    picked.extend(rng.choice(pool, self.instances-len(picked), replace=True).tolist())
                batch.extend(picked)
            yield batch


def cache_path(cache, image):
    return Path(cache)/Path(image).parent.name/(Path(image).name+'.npz')


def read_label(path, image_size, parts=6):
    with np.load(path, allow_pickle=False) as z:
        label = z['label'].astype(np.uint8)
        confidence = z['confidence'].astype(np.float32)
    if label.shape != (image_size[1], image_size[0]) or confidence.shape != label.shape:
        raise ValueError(f'Cache/image dimensions mismatch: {path}')
    if not np.isin(label, list(range(parts+1))+[255]).all() or not np.isfinite(confidence).all():
        raise ValueError(f'Invalid label/confidence: {path}')
    if (confidence < 0).any() or (confidence > 1).any():
        raise ValueError(f'Confidence outside [0,1]: {path}')
    return label, confidence


class ReIDDataset(Dataset):
    def __init__(self, samples, size=(384,128), training=False, label_root=None, parts=6):
        self.samples, self.size, self.training = samples, tuple(size), training
        self.label_root = label_root
        self.parts=parts
        if training and label_root is not None and parts==8:
            from .vei_labels import validate_manifest
            validate_manifest(label_root)
        if training and label_root is not None:
            missing = [s.path for s in samples if not cache_path(label_root,s.path).is_file()]
            if missing:
                raise FileNotFoundError(f'{len(missing)} missing label caches; run prepare-labels. First: {missing[0]}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        with Image.open(s.path) as source:
            image = source.convert('RGB')
        h,w = self.size
        if self.training and self.label_root is not None:
            y,c = read_label(cache_path(self.label_root,s.path),image.size,self.parts)
        else:
            y = np.full((image.height,image.width),255,np.uint8)
            c = np.zeros(y.shape,np.float32)
        label = Image.fromarray(y)
        conf = Image.fromarray(c)
        image = image.resize((w,h),Image.Resampling.BICUBIC)
        label = label.resize((w,h),Image.Resampling.NEAREST)
        conf = conf.resize((w,h),Image.Resampling.BILINEAR)
        if self.training:
            image = ImageOps.expand(image,10,fill=(128,128,128))
            label = ImageOps.expand(label,10,fill=255)
            conf = ImageOps.expand(conf,10,fill=0)
            top,left = torch.randint(21,(2,)).tolist()
            box = (left,top,left+w,top+h)
            image,label,conf = [x.crop(box) for x in (image,label,conf)]
        image = TF.to_tensor(image)
        label = torch.from_numpy(np.array(label,dtype=np.int64))
        conf = torch.from_numpy(np.array(conf,dtype=np.float32))
        if self.training and torch.rand(()) < .5:
            if self.parts==8:
                from .vei_labels import flip_eight
                label=flip_eight(label)
            else:label=flip_labels(label)
            image,conf = image.flip(-1),conf.flip(-1)
        return {'image':image,'label':label,'confidence':conf,'pid':s.pid,'cam':s.cam,'source':s.source}


def seed_worker(_):
    import random
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
