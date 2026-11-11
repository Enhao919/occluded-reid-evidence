"""Independent importer for the authors' public PifPaf/MaskRCNN heatmaps.

Grouping semantics reference BPBreID (Somers et al., WACV 2023).
No Torchreid/OpenPifPaf dependency or external inference labels are required.
"""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .data import cache_path

PART_NAMES=('background','head','left_arm','right_arm','torso','left_leg','right_leg','left_foot','right_foot')
FLIP=(0,1,3,2,4,6,5,8,7)
SCHEMA='vei-eight-pifpaf-v1'
# 17 COCO keypoint heatmaps followed by the 19 PifPaf limb heatmaps.
GROUPS=((0,1,2,3,4,29,30,31,32,33,34,35),(5,7,9,25,27),(6,8,10,26,28),
        (11,12,21,22,23,24),(13,17,18,21),(14,19,20),(15,),(16,))


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def flip_eight(labels):
    table=torch.arange(256,device=labels.device);table[:9]=torch.tensor(FLIP,device=labels.device)
    return table[labels.flip(-1).long()]


def convert_heatmaps(array,image_size,layout='auto',mode='official_compatible'):
    a=np.asarray(array)
    if a.ndim!=3 or not np.isfinite(a).all() or (a<0).any():raise ValueError('Invalid PifPaf heatmap array')
    if layout=='auto':
        possibilities=[name for name,dim in [('chw',a.shape[0]),('hwc',a.shape[-1])] if dim==36]
        if len(possibilities)!=1:raise ValueError(f'Ambiguous/unsupported heatmap layout {a.shape}; specify chw/hwc for 36 channels')
        layout=possibilities[0]
    if layout not in ('chw','hwc'):raise ValueError('layout must be auto/chw/hwc')
    if layout=='hwc':a=a.transpose(2,0,1)
    if a.shape[0]!=36:raise ValueError(f'Expected 36 PifPaf channels, got {a.shape}')
    if mode not in ('official_compatible','quality_filtered'):raise ValueError('Unknown label conversion mode')
    x=torch.from_numpy(a.astype(np.float32,copy=True))
    grouped=torch.stack([x[list(indices)].amax(0).clamp(0,1) for indices in GROUPS])
    background=(grouped.amax(0)<.5).float()
    scores=torch.cat([background[None],grouped])
    scores=F.interpolate(scores[None],size=tuple(image_size),mode='nearest')[0]
    probs=(scores*15).softmax(0)
    values,indices=probs.topk(2,dim=0)
    label=indices[0].to(torch.uint8)
    confidence=torch.ones_like(values[0])
    if mode=='quality_filtered':
        confidence=values[0]
        ignore=(values[0]-values[1])<.1
        label[ignore]=255;confidence[ignore]=0
    return label.numpy(),confidence.numpy()


def validate_manifest(root):
    p=Path(root)/'manifest.json'
    if not p.is_file():raise ValueError(f'VEI label manifest missing: {p}; run import-labels')
    m=json.loads(p.read_text(encoding='utf-8'))
    if m.get('schema')!=SCHEMA or m.get('part_names')!=list(PART_NAMES):
        raise ValueError('VEI requires the official eight-part schema; six-part caches are incompatible')
    signed={k:v for k,v in m.items() if k!='signature'}
    if hashlib.sha256(json.dumps(signed,sort_keys=True).encode()).hexdigest()!=m.get('signature'):
        raise ValueError('VEI label manifest signature mismatch')
    return m


def import_official_labels(samples,source,destination,layout='auto',mode='official_compatible',overwrite=False):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if not source.is_dir():raise FileNotFoundError(f'Unpacked official annotations missing: {source}')
    if source.name!='pifpaf_maskrcnn_filtering':
        filtered=[p for p in source.rglob('pifpaf_maskrcnn_filtering') if p.is_dir()]
        if len(filtered)>1:raise ValueError('Multiple filtered label roots; specify the intended one explicitly')
        if filtered:source=filtered[0]
    if 'pifpaf_maskrcnn_filtering' not in source.parts:
        raise ValueError('Expected the mask-filtered official source tree pifpaf_maskrcnn_filtering, not unfiltered/raw heatmaps')
    existing=validate_manifest(destination) if (destination/'manifest.json').exists() and not overwrite else None
    if existing is not None and (existing['mode']!=mode or existing['layout']!=layout):
        raise ValueError('Existing conversion mode/layout settings differ; choose a new label directory or --overwrite-labels')
    if destination.exists() and any(destination.iterdir()) and not overwrite and existing is None:
        raise FileExistsError('Nonempty label destination without a valid VEI manifest; choose a new directory')
    files={}
    for p in source.rglob('*.npy'):
        # Training labels only; do not resolve a query/gallery duplicate by chance.
        if any(x in ('query','bounding_box_test') for x in p.parts):continue
        stem=p.name.removesuffix('.confidence_fields.npy') if p.name.endswith('.confidence_fields.npy') else p.stem
        for key in (stem,Path(stem).stem):files.setdefault(key,set()).add(p)
    if existing is not None:
        prior={e['image']:e for e in existing['entries']}
        for sample in samples:
            name=Path(sample.path).name;candidates=files.get(name,files.get(Path(name).stem,set()))
            if len(candidates)!=1 or name not in prior or sha256(next(iter(candidates)))!=prior[name]['source_sha256']:
                raise ValueError(f'Existing conversion source differs for {name}; choose a new directory or --overwrite-labels')
        report=audit_vei_labels(samples,destination);report['reused_cache']=True
        return report
    destination.mkdir(parents=True,exist_ok=True)
    entries=[]
    for i,sample in enumerate(samples,1):
        name=Path(sample.path).name
        candidates=files.get(name,files.get(Path(name).stem,set()))
        if len(candidates)!=1:raise ValueError(f'Expected one official train heatmap for {name}, found {len(candidates)}')
        p=next(iter(candidates))
        with Image.open(sample.path) as im:w,h=im.size
        y,c=convert_heatmaps(np.load(p,allow_pickle=False),(h,w),layout,mode)
        output=cache_path(destination,sample.path);output.parent.mkdir(parents=True,exist_ok=True)
        tmp=output.with_name(output.name+'.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,label=y,confidence=c,reason='official_bpbreid',schema=SCHEMA)
        os.replace(tmp,output)
        entries.append({'image':name,'source_sha256':sha256(p),'cache_sha256':sha256(output)})
        if i==1 or i%500==0 or i==len(samples):print(f'Import official labels {i}/{len(samples)}',flush=True)
    manifest={'schema':SCHEMA,'part_names':list(PART_NAMES),'mode':mode,'layout':layout,'source':str(source),
              'reference':'https://github.com/VlSomers/bpbreid','grouping':[list(g) for g in GROUPS],'entries':entries}
    manifest['signature']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    tmp=destination/'manifest.json.tmp';tmp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    os.replace(tmp,destination/'manifest.json')
    return audit_vei_labels(samples,destination)


def audit_vei_labels(samples,root):
    m=validate_manifest(root);entries={x['image']:x for x in m.get('entries',[])}
    result={'images':len(samples),'missing':0,'usable':0,'failed':0,'part_pixels':[0]*9,'schema':SCHEMA,
            'label_signature':m['signature']}
    for s in samples:
        p=cache_path(root,s.path)
        if not p.is_file():result['missing']+=1;continue
        e=entries.get(Path(s.path).name)
        if e is None or sha256(p)!=e['cache_sha256']:raise ValueError(f'Label cache content/manifest mismatch: {p}')
        with np.load(p,allow_pickle=False) as z:y,c=z['label'],z['confidence']
        if y.shape!=c.shape or not np.isin(y,list(range(9))+[255]).all() or not np.isfinite(c).all() or (c<0).any() or (c>1).any():
            raise ValueError(f'Corrupt eight-part cache: {p}')
        usable=bool(((y>0)&(y<9)&(c>.2)).any())
        result['usable' if usable else 'failed']+=1
        for k in range(9):result['part_pixels'][k]+=int((y==k).sum())
    result['usable_fraction']=result['usable']/max(1,len(samples))
    # Portable archive: audit is read-only; report is returned below.
    return result
