"""Read-only fixed internal image protocol, with per-query paired comparisons."""
import json
import hashlib
from pathlib import Path
import numpy as np
from .data import Sample
from .engine import make_loader,save_json
from .evaluation import extract_features,retrieval_distance,ranking_metrics

CONDITIONS=('BA','CA','EA_a','EA_b')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_image(root,row,condition):
    item=row['files'][condition];p=(root/item['relative_path']).resolve()
    if not p.is_relative_to(root):raise ValueError('Protocol image path escape')
    if not p.is_file() or sha(p)!=item['sha256']:raise ValueError('Protocol image hash mismatch: '+str(p))
    return p


def validate_protocol(path,data,cfg):
    path=Path(path).resolve();root=path.parent;obj=json.loads(path.read_text())
    expected=cfg.get('repair',{}).get('expected_protocol_sha256')
    if expected and sha(path)!=expected:raise ValueError('Fixed protocol differs from pinned approved SHA256')
    if obj.get('kind')!='fixed_training_identity_holdout_robustness_v1':raise ValueError('Unknown fixed protocol kind')
    if obj['split_signature']!=cfg.get('validation_signature') or list(obj['image_size'])!=list(cfg['image_size']):
        raise ValueError('Fixed protocol split/image-size mismatch')
    if len(obj['queries'])!=len(data['query']):raise ValueError('Protocol query count mismatch')
    train_ids={s.pid for s in data['train']}
    for i,(row,sample) in enumerate(zip(obj['queries'],data['query'])):
        if (row['index'],row['pid'],row['camera'],Path(row['original']).name)!=(i,sample.pid,sample.cam,Path(sample.path).name):
            raise ValueError('Protocol query source/order/identity mismatch')
        if row['pid'] in train_ids or not set(row.get('donor_training_pids',[])).issubset(train_ids):
            raise ValueError('Protocol train/held identity or donor mismatch')
        if not any(g.pid==sample.pid and g.cam!=sample.cam for g in data['gallery']):
            raise ValueError('Fixed query has no cross-camera gallery positive')
        for condition in CONDITIONS:protocol_image(root,row,condition)
    return sha(path)


def evaluate_protocol(model,data,cfg,device,dtype,output,metadata=None):
    path=Path(cfg['repair']['validation_protocol']).resolve()
    digest=validate_protocol(path,data,cfg);obj=json.loads(path.read_text());out=Path(output)
    out.mkdir(parents=True,exist_ok=True)
    gf,gpid,gcam=extract_features(model,make_loader(data['gallery'],cfg),device,dtype,False)
    report={'kind':'context_repair_fixed_internal_evaluation','protocol_sha256':digest,
            'image_content_sha256':cfg.get('repair_image_content_sha256'),
            'approved_protocol_pinned':bool(cfg['repair'].get('expected_protocol_sha256')),
            'split_signature':cfg['validation_signature'],'conditions':{},'flip':False,'rerank':'none',
            'query_sources':[str(s.path) for s in data['query']],**(metadata or {})}
    arrays={'pid':np.array([s.pid for s in data['query']]),'camera':np.array([s.cam for s in data['query']])}
    for condition in CONDITIONS:
        samples=[Sample(str(protocol_image(path.parent,row,condition)),row['pid'],row['camera'],row['index']) for row in obj['queries']]
        qf,qpid,qcam=extract_features(model,make_loader(samples,cfg),device,dtype,False)
        d=retrieval_distance(qf,gf,'parts',cfg['distance_chunk'],device,cfg['global_weight'],cfg['foreground_weight'],'bpbreid')
        report['conditions'][condition]=ranking_metrics(d,qpid,gpid,qcam,gcam)
        ap=[];rank1=[]
        for i in range(len(samples)):
            r=ranking_metrics(d[i:i+1],qpid[i:i+1],gpid,qcam[i:i+1],gcam)
            ap.append(r['mAP']);rank1.append(r['rank1'])
        arrays[condition+'_ap']=np.array(ap);arrays[condition+'_rank1']=np.array(rank1)
    p=out/'per_query.npz';np.savez_compressed(p,**arrays)
    report.update(per_query_file=p.name,per_query_sha256=sha(p))
    save_json(out/'report.json',report)
    print(json.dumps({'fixed_protocol':report['conditions']},ensure_ascii=False),flush=True)
    return report


def compare_reports(reference,candidate,bootstrap=5000,seed=20261031):
    paths=[Path(reference),Path(candidate)];reports=[json.loads(p.read_text()) for p in paths]
    a,b=reports
    if (a['protocol_sha256'],a['split_signature'])!=(b['protocol_sha256'],b['split_signature']):
        raise ValueError('Cannot compare different fixed protocols or splits')
    if a.get('image_content_sha256')!=b.get('image_content_sha256'):
        raise ValueError('Cannot compare changed underlying image content')
    arrays=[]
    for p,r in zip(paths,reports):
        f=p.parent/r['per_query_file']
        if sha(f)!=r['per_query_sha256']:raise ValueError('Per-query data hash mismatch')
        with np.load(f,allow_pickle=False) as z:arrays.append({k:z[k].copy() for k in z.files})
    x,y=arrays
    if not np.array_equal(x['pid'],y['pid']) or not np.array_equal(x['camera'],y['camera']):
        raise ValueError('Per-query identity/camera alignment changed')
    keys=('clean_mAP','clean_rank1','robust_mAP','robust_rank1')
    def values(z):
        return np.stack((z['BA_ap'],z['BA_rank1'],np.mean([z[k+'_ap'] for k in CONDITIONS[1:]],axis=0),
                         np.mean([z[k+'_rank1'] for k in CONDITIONS[1:]],axis=0)),1)
    delta=values(y)-values(x);ids=np.unique(x['pid'])
    counts=np.array([(x['pid']==pid).sum() for pid in ids])
    sums=np.stack([delta[x['pid']==pid].sum(0) for pid in ids])
    rng=np.random.default_rng(seed);draw=rng.integers(len(ids),size=(bootstrap,len(ids)))
    resampled=sums[draw].sum(1)/counts[draw].sum(1)[:,None]
    mean=delta.mean(0);ci=np.quantile(resampled,[.025,.975],axis=0)
    cases={}
    for k in CONDITIONS:
        base,new=x[k+'_rank1']>.5,y[k+'_rank1']>.5;d=y[k+'_ap']-x[k+'_ap']
        cases[k]={'rescued':int((~base&new).sum()),'broken':int((base&~new).sum()),
                  'worst_AP_changes':[{'index':int(i),'pid':int(x['pid'][i]),'delta_AP':float(d[i]),'source':b['query_sources'][i]} for i in np.argsort(d)[:12]]}
    return {'kind':'paired_internal_refinement_comparison','protocol_sha256':a['protocol_sha256'],
            'deltas':dict(zip(keys,mean.tolist())),
            'identity_bootstrap_CI95':{k:[float(ci[0,i]),float(ci[1,i])] for i,k in enumerate(keys)},
            'passes_pairwise_point_gate':bool((mean>=-1e-8).all()),'strict_robust_mAP_gain':bool(mean[2]>1e-8),
            'cases':cases,'limitations':['Internal screening only; no automatic model promotion.','No 120-epoch or additional-seed claim follows from this comparison.']}
