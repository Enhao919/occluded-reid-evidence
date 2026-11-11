"""Portable pixel loading and JSON receipts, identical to the measured pipeline."""

import hashlib
import io
import json
import os
import statistics
from pathlib import Path


def require(condition,message):
    if not condition:
        raise ValueError(message)


def json_digest(value):
    payload = json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp,path)


def image_path(root,relative):
    root = Path(root).resolve()
    path = (root/relative).resolve()
    require(path.is_relative_to(root),'Image path escapes the supplied dataset')
    return path


class ImageDataset:
    def __init__(self,root,rows,size):
        self.root,self.rows,self.size = root,rows,tuple(size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        import numpy as np
        import torch
        from PIL import Image

        payload = image_path(self.root,self.rows[index]['path']).read_bytes()
        receipt = hashlib.sha256(payload).hexdigest()
        claim = self.rows[index].get('sha256')
        require(claim is None or receipt==claim,'Image content differs from the locked manifest')
        with Image.open(io.BytesIO(payload)) as image:
            h,w = self.size
            array = np.asarray(image.convert('RGB').resize((w,h),Image.Resampling.BICUBIC)).copy()
        return index,torch.from_numpy(array).permute(2,0,1).float()/255,receipt


def extract_once(model,root,rows,config,device,amp_dtype,batch_size,workers,receipts):
    import torch
    from torch.utils.data import DataLoader

    result = {key:[] for key in ('global','foreground','parts','visibility')}
    loader = DataLoader(ImageDataset(root,rows,config['image_size']),batch_size=batch_size,
                        num_workers=workers,shuffle=False,pin_memory=device.type=='cuda')
    with torch.inference_mode():
        for indexes,images,hashes in loader:
            for index,receipt in zip(indexes.tolist(),hashes):
                row = rows[index]
                old = receipts.setdefault(row['path'],receipt)
                require(old==receipt,'Image changed during inference')
            with torch.autocast(device.type,dtype=amp_dtype,enabled=amp_dtype is not None):
                output = model(images.to(device,non_blocking=True))
            for key in result:
                feature = output[key+'_eval'] if key+'_eval' in output else output[key]
                value = feature.detach().float().cpu()
                require(torch.isfinite(value).all().item(),'Nonfinite feature '+key)
                result[key].append(value)
    return {key:torch.cat(values) for key,values in result.items()}


def aggregate(records):
    require(bool(records), 'No scored queries')
    by_id = {}
    for row in records:
        by_id.setdefault(row['pid'], []).append(row)
    return dict(rank1=statistics.mean(row['rank1'] for row in records),
                rank5=statistics.mean(row['rank5'] for row in records),
                rank10=statistics.mean(row['rank10'] for row in records),
                mAP=statistics.mean(row['AP'] for row in records),
                identity_macro_rank1=statistics.mean(statistics.mean(r['rank1'] for r in rs) for rs in by_id.values()),
                valid_queries=len(records), skipped_queries=0)


def query_records(distance, query, gallery, condition):
    import numpy as np
    require(distance.shape == (len(query), len(gallery)), 'Distance shape mismatch')
    require(np.isfinite(distance).all(), 'Nonfinite retrieval distance')
    gids = np.asarray([r['pid'] for r in gallery])
    gcams = np.asarray([r['cam'] for r in gallery])
    records = []
    for i, row in enumerate(query):
        keep = ~((gids == row['pid']) & (gcams == row['cam'])) & (gids >= 0)
        indexes = np.flatnonzero(keep)
        order = indexes[np.argsort(distance[i, keep], kind='stable')]
        match = gids[order] == row['pid']
        require(match.any(), 'Query without a positive: ' + row['path'])
        rank = int(np.flatnonzero(match)[0] + 1)
        ap = float(((match.cumsum() / np.arange(1, len(match) + 1)) * match).sum() / match.sum())
        records.append(dict(condition=condition, path=row['path'], pid=row['pid'], cam=row['cam'],
                            rank1=int(rank <= 1), rank5=int(rank <= 5), rank10=int(rank <= 10), AP=ap,
                            first_positive_rank=rank, top1_path=gallery[int(order[0])]['path'],
                            top1_pid=int(gids[order[0]]), top1_cam=int(gcams[order[0]])))
    return records
