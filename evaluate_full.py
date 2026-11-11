"""Complete 2210/17661 evaluation, including TRAIN-only evidence calibration."""
import csv,gzip,json,time
import numpy as np
from portable_model import ROOT,load_complete
from checkpoints import state_hash
from evaluate_checkpoint import atomic_json,aggregate,ImageDataset,json_digest,query_records
from foundation.retrain import dataset_manifest, specification


def table(path,rows):
    with gzip.open(path,'wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main(args):
    import torch
    from torch.utils.data import DataLoader
    from bounded_graph import evidence_quality,graph_variant
    from fixed_graph_pinned import assemble_owned,rerank_owned
    from occreid.evaluation import retrieval_distance
    started=time.monotonic();torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
    model,cfg,proof=load_complete(args.work,args.device,args.reference)
    from runtime import seed_everything
    seed_everything(101);before=state_hash(model.state_dict())
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'RESULT.json').exists():raise ValueError('Use a new output directory; never overwrite a completed evaluation')
    data,manifest=dataset_manifest(args.data_root)
    train,query,gallery=(manifest[n] for n in ('train','query','gallery'))
    assert list(map(len,[train,query,gallery]))==[15618,2210,17661]
    all_receipts={}
    def status(stage,**kwargs):
        value=dict(stage=stage,**kwargs);atomic_json(args.output/'STATUS.json',value);print(json.dumps(value),flush=True)
    def extract(rows,role,features):
        result={k:[] for k in ['global','foreground','parts','visibility']};qualities=[]
        loader=DataLoader(ImageDataset(data,rows,cfg['image_size']),batch_size=args.batch_size,
            num_workers=args.workers,shuffle=False,pin_memory=args.device=='cuda')
        with torch.inference_mode():
            for step,(indexes,images,hashes) in enumerate(loader,1):
                for index,sha in zip(indexes.tolist(),hashes):
                    row=rows[index];assert row.get('sha256',sha)==sha
                    if role!='TRAIN':all_receipts[row['path']]=sha
                with torch.autocast(args.device,dtype=torch.bfloat16,enabled=args.device=='cuda'):
                    output=model(images.to(args.device,non_blocking=True))
                quality=evidence_quality(output).float().cpu();assert torch.isfinite(quality).all();qualities.append(quality)
                if features:
                    for key in result:
                        value=output.get(key+'_eval',output[key]).float().cpu()
                        assert torch.isfinite(value).all();result[key].append(value)
                del images,output,quality
                if step==1 or step%40==0 or step==len(loader):status('extract',role=role,batch=step,total_batches=len(loader))
        return ({k:torch.cat(v) for k,v in result.items()} if features else None),torch.cat(qualities).numpy()
    _,tq=extract(train,'TRAIN',False)
    positive=tq[tq>0].astype(np.float64);assert len(positive)>0
    scale=float(np.median(positive))
    calibration=dict(scale=scale,positive_images=len(positive),train_images=15618,
        train_manifest_sha256=specification()['split_manifest_sha256']['train'],official_images_opened=0)
    atomic_json(args.output/'TRAIN_CALIBRATION.json',calibration)
    table(args.output/'train_quality.csv.gz',[dict(path=r['path'],quality=float(v)) for r,v in zip(train,tq)])
    q,qq=extract(query,'query',True);g,gq=extract(gallery,'gallery',True)
    assert len(all_receipts)==19871 and json_digest(all_receipts)==specification()['evaluation_image_set_sha256']
    assert state_hash(model.state_dict())==before
    del model;torch.cuda.empty_cache() if args.device=='cuda' else None
    distance=lambda a,b:retrieval_distance(a,b,mode='parts',rule='bpbreid',chunk_size=128,device=args.device)
    raw=distance(q,g);records={'raw':query_records(raw,query,gallery,'original')}
    full=assemble_owned(dict(qg=raw,qq=distance(q,q).T.copy(),gg=distance(g,g)))
    del q,g
    source=ROOT/'evidence/fixed_graph_pinned.py'
    sha='c7a1c01ee12f7792f94e63129e134bd70a20ab805014ae5a07284bdc5d2d1535'
    status('fixed_graph')
    fixed,_,context=rerank_owned(full.copy(),2210,k1=20,k2=6,blend=.3)
    records['fixed']=query_records(fixed,query,gallery,'original')
    del fixed
    status('evidence_bounded_graph',scale=scale)
    modified,_,neighbors,correction=graph_variant(source,sha)(full,2210,k1=20,k2=6,blend=.3,max_kl=.001,
        quality=np.concatenate([qq,gq]),quality_scale=scale)
    assert context==neighbors and correction['summary']['max_budget_violation']<=1e-8
    assert correction['summary']['changed_rows']>0
    records['complete_evidence_graph']=query_records(modified[:,2210:],query,gallery,'original')
    for name,rows in records.items():table(args.output/(name+'_per_query.csv.gz'),rows)
    scores={k:dict(aggregate(v),correct_queries=sum(r['rank1'] for r in v)) for k,v in records.items()}
    result=dict(state='complete',model='shared-source + three-state part allocation + evidence-bounded graph',
        archived_original_weights=args.reference is not None,proof=proof,scores=scores,primary_score='complete_evidence_graph',
        query_images=2210,gallery_images=17661,full_gallery=True,flip=False,
        same_identity_same_camera_removed=True,model_and_BN_unchanged=True,
        optimizer_updates=0,calibration=calibration,graph=dict(k1=20,k2=6,blend=.3,max_kl=.001),
        modules=dict(shared_source_supervision=True,three_state_allocation=True,evidence_bounded_graph=True),
        graph_correction=correction['summary'],batch_size=args.batch_size,device=args.device,
        seconds=time.monotonic()-started,parameters_searched_on_official=False)
    atomic_json(args.output/'GRAPH_CORRECTION.json',correction)
    atomic_json(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)
