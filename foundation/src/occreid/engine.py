from pathlib import Path
import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import time
import uuid
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from .data import ReIDDataset,IdentityBatchSampler,seed_worker
from .model import OcclusionReID,build_model
from .augment import paired_augment
from .losses import reid_objective
from .diagnostics import GradientHealth,loss_gradient_report,parameter_gradient_norm,shared_loss_gradient_report,diagnostic_due
from .config import lr_factor,validate_resume
from .evaluation import extract_features,retrieval_distance,ranking_metrics,bridge_rerank


def save_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def json_ready(value):
    """Overflow is explicit null+finite/update flags, never invalid JSON tokens."""
    if isinstance(value,float):return value if math.isfinite(value) else None
    if isinstance(value,dict):return {k:json_ready(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_ready(v) for v in value]
    return value


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def precision_settings(precision,device):
    if device.type=='cpu':
        if precision!='fp32':raise ValueError('CPU requires --precision fp32')
        return None
    if precision=='bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU lacks bf16 support; pass --precision fp16 (RTX2080) or fp32')
    return {'fp32':None,'fp16':torch.float16,'bf16':torch.bfloat16}[precision]


def make_optimizer(model,cfg):
    groups=[]
    for module,lr in ((model.backbone,cfg['backbone_lr']),(model.head,cfg['head_lr'])):
        if cfg.get('head_architecture')=='bpbreid':
            for name,p in module.named_parameters():
                if p.requires_grad:
                    groups.append({'params':[p],'lr':lr*(2 if name.endswith('bias') else 1),'weight_decay':cfg['weight_decay']})
            continue
        decay,no_decay=[],[]
        for name,p in module.named_parameters():
            if not p.requires_grad:continue
            (no_decay if p.ndim==1 or name.endswith('bias') or 'relative_position_bias' in name else decay).append(p)
        groups.extend([{'params':decay,'lr':lr,'weight_decay':cfg['weight_decay']},
                       {'params':no_decay,'lr':lr,'weight_decay':0.}])
    if cfg['optimizer']=='sgd':return torch.optim.SGD(groups,momentum=.9,nesterov=cfg.get('sgd_nesterov',True))
    return torch.optim.AdamW(groups)


def clip_gradients(parameters,max_norm):
    if max_norm<0:raise ValueError('gradient_clip must be nonnegative')
    return torch.nn.utils.clip_grad_norm_(parameters,float('inf') if max_norm==0 else max_norm)


def dataset_signature(samples):
    content='\n'.join(f'{Path(s.path).parent.name}/{Path(s.path).name}|{s.pid}|{s.cam}' for s in samples)
    return hashlib.sha256(content.encode()).hexdigest()


def validate_output_directory(output,resume_run_id=None):
    output=Path(output)
    if not output.exists() or not any(output.iterdir()):return
    if resume_run_id is None:
        raise FileExistsError(f'Output is not empty: {output}; choose a new directory or --resume')
    identity=output/'run_identity.json'
    if not identity.is_file():
        raise ValueError('Non-empty output has no run identity; resume into a new empty directory')
    if json.loads(identity.read_text(encoding='utf-8')).get('run_id')!=resume_run_id:
        raise ValueError('Output belongs to a different run; resume into its own or a new empty directory')


def save_checkpoint(path,model,optimizer,scheduler,scaler,epoch,config,signature,extra):
    if config.get('method')=='vei':ensure_checkpoint_space(Path(path).parent,model,optimizer)
    state={'format_version':2 if config.get('method')=='vei' else 1,'training_state':True,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
           'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,
           'config':config,'dataset_signature':signature,'extra':extra,'run_id':extra.get('run_id'),
           'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                  'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    torch.save(state,tmp);os.replace(tmp,path)


def restore_checkpoint(path,model,optimizer,scheduler,scaler,config,signature,state=None):
    if state is None:state=torch.load(path,map_location='cpu',weights_only=False)
    expected=2 if config.get('method')=='vei' else 1
    if state.get('format_version')!=expected:raise ValueError('Checkpoint format/method mismatch')
    if not state.get('training_state',True) or any(k not in state for k in ('optimizer','scheduler','scaler','rng')):
        raise ValueError('Inference-only checkpoint cannot resume training; use last.pt')
    if config.get('method')=='vei' and state.get('extra',{}).get('epoch_truncated') and not config.get('smoke_test'):
        raise ValueError('Truncated diagnostic training checkpoint cannot resume a formal experiment')
    validate_resume(state['config'],config)
    if state['dataset_signature']!=signature:raise ValueError('Resume training data signature mismatch')
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'])
    if torch.cuda.is_available() and state['rng']['cuda'] is not None:torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state


def ensure_checkpoint_space(directory,model,optimizer=None,reserve_bytes=512*1024**2):
    weights=sum(t.numel()*t.element_size() for t in model.state_dict().values())
    moments=0
    if optimizer is not None:
        moments=sum(v.numel()*v.element_size() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))
        if not moments:moments=weights*(2 if isinstance(optimizer,torch.optim.AdamW) else 1)
    needed=weights+moments+reserve_bytes
    free=shutil.disk_usage(directory).free
    if free<needed:raise OSError(f'Insufficient checkpoint space: free {free/2**30:.2f}GiB, need at least {needed/2**30:.2f}GiB including atomic temporary file')
    return needed


def save_inference_checkpoint(path,model,epoch,config,signature,extra):
    ensure_checkpoint_space(Path(path).parent,model)
    state={'format_version':2,'training_state':False,'model':model.state_dict(),'epoch':epoch,'config':config,
           'dataset_signature':signature,'run_id':extra.get('run_id'),'extra':extra}
    path=Path(path);tmp=path.with_name(path.name+'.tmp');torch.save(state,tmp);os.replace(tmp,path)


def initialize_baseline(model,path):
    """Transfer the existing official downstream model; reject partial transfers."""
    weights=torch.load(path,map_location='cpu',weights_only=True)
    weights={k.removeprefix('module.'):v for k,v in weights.items()}
    backbone={k.removeprefix('base.'):v for k,v in weights.items() if k.startswith('base.')}
    model.backbone.load_state_dict(backbone,strict=True)
    model.head.global_neck.load_state_dict({k.removeprefix('bottleneck.'):v for k,v in weights.items() if k.startswith('bottleneck.')},strict=True)
    model.head.global_classifier.load_state_dict({k.removeprefix('classifier.'):v for k,v in weights.items() if k.startswith('classifier.')},strict=True)
    return {'path':str(path),'backbone_tensors':len(backbone),'strict':True}


def initialize_model(model,path,cfg,signature):
    """Explicit compatible downstream weights only; never restore training state."""
    state=torch.load(path,map_location='cpu',weights_only=False)
    if cfg.get('method')!='vei' or state.get('format_version')!=2 or 'model' not in state:raise ValueError('Expected a trusted VEI downstream checkpoint')
    saved=state['config']
    for key in ('method','num_classes','image_size','feature_dim','semantic_weight','label_signature','validation_signature'):
        if saved.get(key)!=cfg.get(key):raise ValueError(f'Warmstart incompatible {key}')
    if state.get('dataset_signature')!=signature:raise ValueError('Warmstart training data signature changed')
    model.load_state_dict(state['model'],strict=True)
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return {'path':str(Path(path).resolve()),'sha256':digest.hexdigest(),'parent_epoch':state['epoch'],
            'parent_epoch_truncated':state.get('extra',{}).get('epoch_truncated',False),
            'mode':'model_only_optimizer_scheduler_rng_reset','strict':True}


def make_loader(samples,cfg,training=False,label_root=None,generator=None):
    dataset=ReIDDataset(samples,cfg['image_size'],training,label_root,parts=8 if cfg.get('method')=='vei' else 6)
    if not training and generator is None:generator=torch.Generator().manual_seed(cfg['seed'])
    options={'num_workers':cfg['workers'],'pin_memory':torch.cuda.is_available(),
             'worker_init_fn':seed_worker,'generator':generator}
    if training:
        sampler=IdentityBatchSampler(samples,cfg['batch_size'],cfg['instances'],cfg['seed'])
        return DataLoader(dataset,batch_sampler=sampler,**options)
    return DataLoader(dataset,batch_size=cfg['test_batch_size'],shuffle=False,**options)


def experiment_metadata(cfg,query,gallery,epoch=None,run_id=None):
    excluded={'data_root','pretrained','label_root','output_dir','workers','seed','test_batch_size',
              'log_interval','diagnostic_interval','gradient_patience','fail_fast','eval_period',
              'checkpoint_period','distance_chunk','flip_test','rerank','global_weight','foreground_weight',
              'num_classes','smoke_test'}
    recipe={k:v for k,v in cfg.items() if k not in excluded}
    signature=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
    datahash=hashlib.sha256((dataset_signature(query)+'|'+dataset_signature(gallery)).encode()).hexdigest()
    return {'experiment':re.sub(r'_seed\d+$','',Path(cfg['output_dir']).name),
            'seed':cfg['seed'],'epoch':epoch,'run_id':run_id,'recipe_signature':signature,
            'evaluation_signature':datahash,'rerank':cfg['rerank'],
            'retrieval_weights':[cfg['global_weight'],cfg['foreground_weight']],
            'smoke_test':cfg.get('smoke_test',False),'method':cfg.get('method','pso'),
            'part_policy':cfg.get('part_policy','anatomical8'),'parsing_supervision':cfg.get('parsing_supervision','flat'),
            'curriculum_epoch_offset':cfg.get('curriculum_epoch_offset',0),
            'retrieval_rule':cfg.get('retrieval_rule','legacy'),'validation_signature':cfg.get('validation_signature'),
            'flip_visibility_policy':('binary_union' if cfg.get('method')=='vei' else 'legacy_average') if cfg['flip_test'] else 'none'}


def evaluate(model,query,gallery,cfg,device,dtype,output=None,save_features=False,metadata=None):
    start=time.monotonic()
    loader=make_loader(query+gallery,cfg)
    features,pids,cams=extract_features(model,loader,device,dtype,cfg['flip_test'])
    n=len(query);q={k:v[:n] for k,v in features.items()};g={k:v[n:] for k,v in features.items()}
    report={'flip_test':cfg['flip_test'],'results':{},'protocol':'standard image-only retrieval; same-ID same-camera removed',
            **(metadata or experiment_metadata(cfg,query,gallery))}
    modes=['global'] if cfg.get('global_only') else (['global','parts','local'] if cfg.get('method')=='vei' else ['global','parts'])
    for mode in modes:
        distance=retrieval_distance(q,g,mode,cfg['distance_chunk'],device,
                                    cfg['global_weight'],cfg['foreground_weight'],cfg.get('retrieval_rule','legacy'))
        report['results'][mode]=ranking_metrics(distance,pids[:n],pids[n:],cams[:n],cams[n:])
        if cfg['rerank']=='bridge':
            refined=bridge_rerank(distance,g)
            report['results'][mode+'_bridge']=ranking_metrics(refined,pids[:n],pids[n:],cams[:n],cams[n:])
    report['seconds']=round(time.monotonic()-start,3)
    report['mean_predicted_visibility']=float(features['visibility'].mean())
    report['zero_visible_image_fraction']=float((features['visibility'].sum(1)==0).float().mean())
    if output:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        save_json(output/'metrics.json',report)
        if save_features:torch.save({'features':features,'pids':torch.from_numpy(pids),'cameras':torch.from_numpy(cams),'num_query':n},output/'features.pt')
    print(json.dumps(report,indent=2),flush=True)
    return report


def train(cfg,data,device,resume=None,init_baseline=None,max_steps=0,stop_after=0,smoke=False,init_model=None):
    cfg=copy.deepcopy(cfg);cfg['num_classes']=data['num_classes']
    output=Path(cfg['output_dir'])
    resumed_state=torch.load(resume,map_location='cpu',weights_only=False) if resume else None
    if resumed_state is not None:validate_resume(resumed_state['config'],cfg)
    run_id=(resumed_state.get('run_id') if resumed_state is not None else None) or uuid.uuid4().hex
    validate_output_directory(output,run_id if resume else None)
    output.mkdir(parents=True,exist_ok=True)
    seed_everything(cfg['seed']);dtype=precision_settings(cfg['precision'],device)
    model=build_model(cfg,None if resume or init_baseline or init_model else cfg['pretrained']).to(device)
    init_report=initialize_baseline(model,init_baseline) if init_baseline else model.pretrained_report
    if init_model:init_report=initialize_model(model,init_model,cfg,dataset_signature(data['train']))
    optimizer=make_optimizer(model,cfg)
    if cfg.get('method')=='vei':ensure_checkpoint_space(output,model,optimizer)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda ep:lr_factor(ep,cfg))
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and dtype==torch.float16,init_scale=1024)
    signature=dataset_signature(data['train']);start_epoch=0;history=[];best=-1.
    health=GradientHealth(cfg['gradient_patience'],cfg['fail_fast'])
    if resume:
        state=restore_checkpoint(resume,model,optimizer,scheduler,scaler,cfg,signature,resumed_state)
        start_epoch=state['epoch'];history=state['extra']['history'];best=state['extra'].get('best',-1.)
        health.streaks=state['extra'].get('gradient_streaks',{})
        init_report=state['extra'].get('initialization',init_report)
        del state
        del resumed_state
    # Only write run metadata AFTER resume validation succeeds.
    save_json(output/'run_identity.json',{'run_id':run_id})
    if '_validation_manifest' in data:save_json(output/'internal_validation_split.json',data['_validation_manifest'])
    (output/'config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    save_json(output/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
        'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        'initialization':init_report,'dataset_signature':signature,'num_classes':cfg['num_classes'],
        'train_images':len(data['train']),'smoke_test':smoke,'benchmark_monitoring':cfg['eval_period']>0})
    generator=torch.Generator()
    loader=make_loader(data['train'],cfg,True,None if cfg.get('global_only') else cfg['label_root'],generator)
    if cfg.get('head_architecture')=='bpbreid' and not resume:
        # Extra optional modules must not change augmentation/drop-path randomness.
        seed_everything(cfg['seed'])
    log=(output/'steps.jsonl').open('a',encoding='utf-8')
    end_epoch=min(cfg['epochs'],stop_after) if stop_after else cfg['epochs']
    if start_epoch>=end_epoch:raise ValueError('Checkpoint already reached requested end epoch')
    try:
        for epoch in range(start_epoch,end_epoch):
            curriculum_epoch=epoch+cfg.get('curriculum_epoch_offset',0)
            model.train();loader.batch_sampler.set_epoch(curriculum_epoch);generator.manual_seed(cfg['seed']+curriculum_epoch)
            start=time.monotonic();sums={};steps=success=0;last_diagnostic=None;cf_checked=False;aux_checked=set()
            for step,batch in enumerate(loader,1):
                batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
                image=batch['image'];b=len(image)
                if cfg.get('method')=='vei':
                    from .vei_augment import make_vei_views
                    view_images,paired,augstats=make_vei_views(batch,cfg,curriculum_epoch,cfg.get('force_view'))
                else:
                    strong,label,conf,augstats=paired_augment(image,batch['label'],batch['confidence'],batch['pid'],
                        epoch/max(1,cfg['occlusion_ramp_epochs']),cfg['occlusion_probability'])
                    paired=dict(batch,label=torch.cat([batch['label'],label]),confidence=torch.cat([batch['confidence'],conf]))
                    view_images=torch.cat([image,strong])
                if cfg.get('research',{}).get('enabled'):
                    from .evidence import set_pair_context
                    set_pair_context(model,b if paired.get('views')==2 else 0)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=dtype,enabled=dtype is not None):
                    predictions=model(view_images)
                # Loss math is deliberately FP32, outside autocast.
                total,raw,weighted,stats=reid_objective(predictions,paired,cfg,curriculum_epoch)
                if not torch.isfinite(total):raise FloatingPointError(f'Nonfinite loss at epoch{epoch+1},step{step}')
                diagnose=diagnostic_due(step,cfg['diagnostic_interval'],stats,cf_checked)
                if cfg.get('research',{}).get('enabled'):
                    from .evidence import active_aux
                    diagnose=diagnose or bool(active_aux(stats)-aux_checked)
                if diagnose:
                    if cfg.get('method')=='vei':
                        live={k:v for k,v in weighted.items() if cfg['loss_weights'].get(k,0)>0 and (k!='cfcon' or stats['contrast_weight']>0)}
                        if cfg.get('research',{}).get('enabled'):
                            from .evidence import diagnostic_live,AUX_NAMES
                            live=diagnostic_live(weighted,cfg,stats)
                            if any(k in live for k in AUX_NAMES):
                                live['_identity_reference']=sum(weighted[k] for k in
                                    ('global_ce','foreground_ce','concat_ce','local_triplet'))
                        report=shared_loss_gradient_report(live,{'input':model.backbone.patch_embed.projection.weight,
                                                                 'last':model.backbone.norm3.weight})
                        identity=report.pop('_identity_reference',None)
                        if identity:
                            for k in AUX_NAMES:
                                if k in report:
                                    report[k]['ratio_to_identity']=(report[k]['gradient_norm']/identity['gradient_norm']
                                        if identity['gradient_norm']>1e-12 else None)
                        if cfg.get('research',{}).get('enabled'):
                            aux_checked.update(active_aux(stats).intersection(report))
                    else:
                        report=loss_gradient_report({k:v for k,v in raw.items() if k in ('global_ce','global_triplet')},predictions['global'])
                        report.update(loss_gradient_report({k:v for k,v in raw.items() if k not in ('global_ce','global_triplet')},predictions['spatial']))
                    failures=health.check(report,stats,raw)
                    cf_checked=cf_checked or ('cfcon' in report and stats.get('contrast_anchors',0)>0 and stats.get('contrast_weight',0)>0)
                    watched={'backbone':model.backbone.patch_embed.projection.weight,
                             'segmentation':model.head.segmentation.weight,
                             'global_classifier':model.head.global_classifier.weight}
                    if cfg.get('method')=='vei':
                        watched['concat_classifier']=model.head.concat_classifier.weight
                        if hasattr(model.head,'scale_logits'):watched['scale_logits']=model.head.scale_logits
                    else:watched['part_classifier']=model.head.part_classifier.weight
                    before={k:p.detach().flatten()[:128].clone() for k,p in watched.items()}
                scaler.scale(total).backward()
                if cfg.get('research',{}).get('enabled'):set_pair_context(model,0)
                scaler.unscale_(optimizer)
                norm=clip_gradients(model.parameters(),cfg['gradient_clip'])
                if not torch.isfinite(norm) and not scaler.is_enabled():
                    raise FloatingPointError(f'Nonfinite gradients at epoch{epoch+1},step{step}')
                gradstats={}
                if diagnose:
                    gradstats={'backbone_gradient':parameter_gradient_norm(model.backbone),
                               'head_gradient':parameter_gradient_norm(model.head)}
                previous_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                updated=not scaler.is_enabled() or scaler.get_scale()>=previous_scale
                success+=int(updated);steps=step
                record={'epoch':epoch+1,'curriculum_epoch':curriculum_epoch+1,'step':step,'total':float(total.detach()),
                        'raw':{k:float(v.detach()) for k,v in raw.items()},
                        'weighted':{k:float(v.detach()) for k,v in weighted.items()},
                        'stats':dict(stats,**augstats),'lr':[g['lr'] for g in optimizer.param_groups],
                        'gradient_norm':float(norm) if torch.isfinite(norm) else None,
                        'optimizer_updated':updated,'amp_scale':scaler.get_scale()}
                predicted_visibility=predictions['visibility'].detach()
                record['predicted_visibility']={
                    'mean':float(predicted_visibility.mean()),
                    'zero_image_fraction':float((predicted_visibility.sum(1)==0).float().mean()),
                    'all_visible_fraction':float((predicted_visibility>0).all(1).float().mean()),
                    'part_support_fraction':(predicted_visibility>0).float().mean(0).tolist()}
                if diagnose:
                    record['loss_gradients']=report;record.update(gradstats)
                    record['parameter_updates']={k:float((p.detach().flatten()[:128]-before[k]).abs().max()) for k,p in watched.items()}
                    record['feature_std']=float(predictions['global'].detach().float().std(0).mean())
                    record['feature_norm']=float(predictions['global'].detach().float().norm(dim=1).mean())
                    record['gradient_failures']=failures
                    if 'scale_weights' in predictions:record['scale_weights']=predictions['scale_weights'].detach().tolist()
                if cfg.get('research',{}).get('enabled'):
                    from .evidence import input_fingerprint
                    record.update(input_fingerprint(view_images,paired,batch,
                        step==1 or step%cfg['diagnostic_interval']==0))
                record=json_ready(record)
                if diagnose:last_diagnostic=record
                log.write(json.dumps(record,allow_nan=False)+'\n')
                for key,value in dict(total=record['total'],**record['raw'],**stats).items():
                    sums[key]=sums.get(key,0)+value
                if step==1 or step%cfg['log_interval']==0:
                    log.flush();print(f"Epoch{epoch+1}/{cfg['epochs']} step{step}/{len(loader)} loss={record['total']:.4f} updated={success}/{steps} lr={record['lr'][0]:.2e}",flush=True)
                if max_steps and step>=max_steps:break
            if not success:raise RuntimeError('No successful optimizer update this epoch; inspect AMP/gradients')
            scheduler.step()
            summary={'epoch':epoch+1,'mean':{k:v/steps for k,v in sums.items()},'steps':steps,
                     'successful_updates':success,'skipped_updates':steps-success,
                     'seconds':round(time.monotonic()-start,3),'last_diagnostic':last_diagnostic,
                     'peak_cuda_GB':torch.cuda.max_memory_allocated(device)/1e9 if device.type=='cuda' else 0.}
            summary['curriculum_epoch']=curriculum_epoch+1
            if cfg['eval_period'] and ((epoch+1)%cfg['eval_period']==0 or epoch+1==cfg['epochs']):
                monitor_key='internal_validation' if data.get('split_name')=='internal_validation' else 'benchmark_monitor'
                summary[monitor_key]=evaluate(model,data['query'],data['gallery'],cfg,device,dtype,output/f'monitor_epoch{epoch+1}',
                    metadata=dict(experiment_metadata(cfg,data['query'],data['gallery'],epoch+1,run_id),initialization=init_report))
                primary='global' if cfg.get('global_only') else 'parts'
                score=summary[monitor_key]['results'][primary]['mAP']
                is_best=score>best;best=max(best,score)
            else:is_best=False
            history.append(summary)
            extra={'history':history,'best':best,'gradient_streaks':health.streaks,
                   'smoke_test':smoke,'epoch_truncated':bool(max_steps and max_steps<len(loader)),
                   'run_id':run_id,'initialization':init_report}
            save_checkpoint(output/'last.pt',model,optimizer,scheduler,scaler,epoch+1,cfg,signature,extra)
            if is_best and not (cfg.get('research',{}).get('enabled') and not cfg['research'].get('retain_best',False)):
                best_name='best_validation.pt' if data.get('split_name')=='internal_validation' else 'best_benchmark.pt'
                if cfg.get('method')=='vei' and cfg['thin_best']:save_inference_checkpoint(output/best_name,model,epoch+1,cfg,signature,extra)
                else:save_checkpoint(output/best_name,model,optimizer,scheduler,scaler,epoch+1,cfg,signature,extra)
            if cfg['checkpoint_period']>0 and (epoch+1)%cfg['checkpoint_period']==0:
                save_checkpoint(output/f'epoch_{epoch+1}.pt',model,optimizer,scheduler,scaler,epoch+1,cfg,signature,extra)
            save_json(output/'history.json',history)
            print(f'Epoch{epoch+1} complete: mean_loss={summary["mean"]["total"]:.4f}, successful_updates={success}',flush=True)
    finally:log.close()
    return model,history
