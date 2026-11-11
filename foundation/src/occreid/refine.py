"""Experimental refinement of a matched SOLIDER/BPBreID baseline.

The original training epoch/recipe remains available through `train`.
This separate recipe never resumes a parent optimizer as its own optimizer.
"""
import copy
import hashlib
import json
import os
import random
import time
import uuid
from pathlib import Path
import numpy as np
import torch
import yaml
from .config import validate_resume,lr_factor
from .engine import (dataset_signature,ensure_checkpoint_space,make_loader,make_optimizer,
                     precision_settings,seed_everything,save_json,validate_output_directory,
                     json_ready,evaluate,experiment_metadata,clip_gradients)
from .model import build_model
from .diagnostics import GradientHealth,shared_loss_gradient_report
from .repair_runtime import differentiable_eval
from .repair_augment import background_intervention
from .repair_objectives import select_relations,repair_objectives
from .bpbreid_losses import reference_objective
from .vei_augment import make_vei_views

RECIPE='context_repair_v1'
PARENT_KEYS=('method','head_architecture','bpbreid_msf','part_policy','identity_masks_detached',
             'parsing_supervision','num_classes','image_size','feature_dim','semantic_weight',
             'label_signature','validation_signature','drop_path','retrieval_rule')


def recipe_source_sha256():
    root=Path(__file__).resolve().parent;h=hashlib.sha256()
    for p in sorted(root.rglob('*.py')):
        h.update(p.relative_to(root).as_posix().encode());h.update(p.read_bytes())
    return h.hexdigest()


def image_content_sha256(samples):
    """Relocation-stable receipt of current bytes, not retroactive parent proof."""
    h=hashlib.sha256()
    for sample in samples:
        p=Path(sample.path)
        h.update(f'{p.parent.name}/{p.name}|{sample.pid}|{sample.cam}|'.encode())
        h.update(bytes.fromhex(sha256_file(p)))
    return h.hexdigest()


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def ensure_training_command(command,cfg):
    if command in ('train','smoke','diagnose') and (cfg.get('repair',{}).get('enabled') or cfg.get('context_repair',{}).get('enabled')):
        raise ValueError('Context repair training requires the separate refine command')
    if command=='refine':
        if not cfg.get('repair',{}).get('enabled'):raise ValueError('refine requires repair.enabled=true')
        if cfg['view_layout']!='single' or cfg.get('force_view') or cfg.get('research',{}).get('enabled'):
            raise ValueError('refine preserves the original single-view recipe')


def validate_parent_state(state,cfg,signature):
    if state.get('format_version')!=2 or state.get('extra',{}).get('epoch_truncated') or state.get('extra',{}).get('smoke_test'):
        raise ValueError('Parent must be a complete reference baseline, not a diagnostic/refinement checkpoint')
    saved=state['config']
    if saved.get('view_layout')!='single':raise ValueError('Parent incompatible view_layout')
    if saved.get('force_view') is not None:raise ValueError('Parent incompatible force_view')
    for key in PARENT_KEYS:
        if saved.get(key)!=cfg.get(key):raise ValueError('Parent incompatible '+key)
    for key in ('label_smoothing','triplet_margin','single_erase_probability'):
        if saved.get(key)!=cfg.get(key):raise ValueError('Parent incompatible '+key)
    for key in ('global_ce','foreground_ce','concat_ce','local_triplet','segmentation'):
        if saved['loss_weights'].get(key)!=cfg['loss_weights'].get(key):raise ValueError('Parent incompatible loss '+key)
    if saved.get('head_architecture')!='bpbreid' or saved.get('context_repair',{}).get('enabled') or saved.get('research',{}).get('enabled'):
        raise ValueError('Parent must be the unmodified BPBreID reference baseline')
    if state.get('dataset_signature')!=signature:raise ValueError('Parent training data/identity mapping mismatch')


def save_refinement(path,model,optimizer,scheduler,scaler,epoch,cfg,signature,extra,donor_rng):
    path=Path(path);ensure_checkpoint_space(path.parent,model,optimizer)
    state={'format_version':3,'training_recipe':RECIPE,'training_state':True,
           'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
           'scaler':scaler.state_dict(),'epoch':epoch,'config':cfg,'dataset_signature':signature,
           'run_id':extra['run_id'],'extra':extra,'donor_rng':donor_rng.get_state(),
           'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                  'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    tmp=path.with_name(path.name+'.tmp');torch.save(state,tmp);os.replace(tmp,path)


def restore_refinement(state,model,optimizer,scheduler,scaler,cfg,signature,teacher_sha,donor_rng):
    if state.get('format_version')!=3 or state.get('training_recipe')!=RECIPE or not state.get('training_state'):
        raise ValueError('Expected complete format3 refinement state')
    if state['extra'].get('teacher_sha256')!=teacher_sha:raise ValueError('Frozen teacher content changed')
    if state.get('dataset_signature')!=signature:raise ValueError('Resume data/identity mapping changed')
    if state['extra'].get('epoch_truncated'):raise ValueError('Truncated self-check cannot resume a formal refinement')
    saved,current=copy.deepcopy(state['config']),copy.deepcopy(cfg)
    for conf in (saved,current):
        for key in ('parent_checkpoint','validation_protocol'):conf.get('repair',{}).pop(key,None)
    validate_resume(saved,current)
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
    donor_rng.set_state(state['donor_rng'])
    random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'])
    if torch.cuda.is_available() and state['rng']['cuda'] is not None:torch.cuda.set_rng_state_all(state['rng']['cuda'])


def initialize_pair(student,teacher,state):
    teacher.load_state_dict(state['model'],strict=True)
    missing,unexpected=student.load_state_dict(state['model'],strict=False)
    expected={k for k in student.state_dict() if k.startswith('head.context_residual.')}
    if unexpected or set(missing)!=expected:raise ValueError('Only new context residual parameters may be absent from parent')
    teacher.eval().requires_grad_(False)


def _gradient_vector(params,scale=1.):
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad.detach()/scale).float().flatten() for p in params])


def accumulate_refinement(model,teacher,batch,cfg,scaler,dtype,donor_rng,epoch,health,checked_aux=None,diagnose=False):
    """Two backward stages, no zero_grad/step/unscale; caller updates once."""
    checked_aux=checked_aux if checked_aux is not None else set()
    images,paired,aug=make_vei_views(batch,cfg,epoch)
    device=images.device
    amp=lambda:torch.autocast(device_type=device.type,dtype=dtype,enabled=dtype is not None)
    targets={'input':model.backbone.patch_embed.projection.weight,'last':model.backbone.norm3.weight}
    model.train()
    with amp():output=model(images)
    total,raw,weighted,stats=reference_objective(output,paired,cfg,epoch)
    if not torch.isfinite(total):raise FloatingPointError('Nonfinite original GiLt')
    report={};failures=[]
    if diagnose:
        live={k:v for k,v in weighted.items() if cfg['loss_weights'].get(k,0)>0}
        report=shared_loss_gradient_report(live,targets);failures=health.check(report,stats,raw)
    record={'raw':{k:float(v.detach()) for k,v in raw.items()},'weighted':{k:float(v.detach()) for k,v in weighted.items()},
            'stats':dict(stats,**aug),'total':float(total.detach()),'loss_gradients':report,
            'feature_std':float(output['parts'].detach().float().std(0,unbiased=False).mean())}
    scaler.scale(total).backward()
    base_vector=_gradient_vector(targets.values(),scaler.get_scale())
    del output,total,raw,weighted
    r=cfg['repair'];ramp=max(0.,min(1.,(epoch-r['warmup_epochs']+1)/r['ramp_epochs'])) if epoch>=r['warmup_epochs'] else 0.
    weights={'identity_guard':r['guard_weight']*ramp,'relation_rescue':r['relation_weight']*ramp}
    record['stats'].update(repair_ramp=ramp,teacher_images=0,aux_student_images=0,
                           repair_accepted=0,repair_damaged=0,repair_violations=0,guard_active=0)
    for name in weights:record['raw'][name]=0.;record['weighted'][name]=0.
    if any(weights.values()):
        with torch.no_grad(),amp():t=teacher(images)
        with torch.no_grad():
            relations=select_relations(t,paired)
            probs=t['segmentation'].float().softmax(1)
            env,mask,donors=background_intervention(images,paired['label'],paired['confidence'],probs,batch['pid'],donor_rng)
        with torch.no_grad(),amp():te=teacher(env)
        # Keep this context alive THROUGH backward: checkpoint recomputation
        # must observe the same BN/DropPath mode as its own forward.
        with differentiable_eval(model):
            with amp():s=model(torch.cat((images,env)))
            b=len(images);clean={k:v[:b] for k,v in s.items()};changed={k:v[b:] for k,v in s.items()}
            aux,auxstats=repair_objectives(clean,changed,t,te,paired,relations,mask.flatten(1).any(1),r['damage_weighted'])
            positive={k for k,v in aux.items() if weights[k]>0 and float(v.detach())>1e-5}
            check=diagnose or bool(positive-checked_aux)
            live={k:aux[k]*weights[k] for k in aux if weights[k]>0}
            if check:
                probe=shared_loss_gradient_report(live,targets)
                den=float(base_vector.norm())
                for k in probe:
                    gradients=torch.autograd.grad(live[k],tuple(targets.values()),retain_graph=True,allow_unused=True)
                    vector=torch.cat([(torch.zeros_like(p) if g is None else g.detach()).float().flatten() for p,g in zip(targets.values(),gradients)])
                    norm=float(vector.norm())
                    probe[k]['ratio_to_reference_GiLt']=norm/den if den>1e-12 else None
                    probe[k]['cosine_with_reference_GiLt']=float(vector@base_vector)/(norm*den) if norm*den>1e-12 else None
                failures+=health.check(probe,auxstats,aux);record['loss_gradients'].update(probe)
                if model.head.context_residual is not None:
                    record['adapter_loss_gradients']=shared_loss_gradient_report(live,{'output':model.head.context_residual.output.weight})
                checked_aux.update(positive)
            extra=sum(live.values())
            if not torch.isfinite(extra):raise FloatingPointError('Nonfinite refinement objective')
            scaler.scale(extra).backward()
            record['total']+=float(extra.detach())
            record['raw'].update({k:float(v.detach()) for k,v in aux.items()})
            record['weighted'].update({k:float(v.detach())*weights[k] for k,v in aux.items()})
            record['stats'].update(auxstats)
        record['stats'].update(teacher_images=2*b,aux_student_images=2*b,
            background_changed_pixels=int(mask.sum()),background_active_sources=int(mask.flatten(1).any(1).sum()),
            donor_indices=donors.cpu().tolist())
    record['stats'].update({k+'_weight':v for k,v in weights.items()})
    record['gradient_failures']=failures
    return record


def refine(cfg,data,device,resume=None,max_steps=0,stop_after=0):
    cfg=copy.deepcopy(cfg);cfg['num_classes']=data['num_classes'];cfg['smoke_test']=bool(max_steps)
    cfg['repair_source_sha256']=recipe_source_sha256();ensure_training_command('refine',cfg)
    signature=dataset_signature(data['train']);parent=Path(cfg['repair']['parent_checkpoint'])
    if not parent.is_file():raise FileNotFoundError('Set --parent-checkpoint to the matched baseline checkpoint')
    if cfg['gradient_clip']!=0 or cfg['checkpoint_period']!=0:raise ValueError('Refinement requires gradient_clip=0, checkpoint_period=0')
    parent_sha=sha256_file(parent);parent_state=torch.load(parent,map_location='cpu',weights_only=False)
    validate_parent_state(parent_state,cfg,signature)
    cfg['repair_image_content_sha256']=image_content_sha256(data['train']+data['query']+data['gallery'])
    parent_image_hash=parent_state['config'].get('repair_image_content_sha256')
    if parent_image_hash and parent_image_hash!=cfg['repair_image_content_sha256']:
        raise ValueError('Parent image content changed')
    # Split/teacher/path guards precede model construction and file writes.
    protocol=cfg['repair'].get('validation_protocol')
    if protocol:
        from .repair_protocol import validate_protocol
        cfg['repair_protocol_sha256']=validate_protocol(protocol,data,cfg)
    state=torch.load(resume,map_location='cpu',weights_only=False) if resume else None
    output=Path(cfg['output_dir']);run_id=(state.get('run_id') if state else None) or uuid.uuid4().hex
    validate_output_directory(output,run_id if resume else None)
    seed_everything(cfg['seed']);dtype=precision_settings(cfg['precision'],device)
    teacher_cfg=copy.deepcopy(cfg);teacher_cfg['context_repair']['enabled']=False
    teacher=build_model(teacher_cfg).to(device);student=build_model(cfg).to(device)
    initialize_pair(student,teacher,parent_state);parent_epoch=int(parent_state['epoch']);del parent_state
    opt=make_optimizer(student,cfg);scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda ep:lr_factor(ep,cfg))
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and dtype==torch.float16,init_scale=1024)
    donor_rng=torch.Generator().manual_seed(cfg['seed']+271828);health=GradientHealth(cfg['gradient_patience'],cfg['fail_fast'])
    start=0;history=[];updates=0
    if state:
        restore_refinement(state,student,opt,scheduler,scaler,cfg,signature,parent_sha,donor_rng)
        start=int(state['epoch']);history=state['extra']['history'];updates=state['extra']['successful_updates']
        health.streaks=state['extra']['gradient_streaks'];del state
    else:seed_everything(cfg['seed'])
    end=min(cfg['epochs'],stop_after) if stop_after else cfg['epochs']
    if start>=end:raise ValueError('Already at requested refinement endpoint')
    output.mkdir(parents=True,exist_ok=True);ensure_checkpoint_space(output,student,opt)
    generator=torch.Generator();loader=make_loader(data['train'],cfg,True,cfg['label_root'],generator)
    save_json(output/'run_identity.json',{'run_id':run_id})
    (output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    save_json(output/'refinement_provenance.json',{'training_recipe':RECIPE,'teacher_sha256':parent_sha,
        'parent_epoch':parent_epoch,'dataset_signature':signature,'training_config':cfg,
        'current_image_content_sha256':cfg['repair_image_content_sha256'],
        'parent_image_bytes_authenticated':bool(parent_image_hash),
        'parent_content_limitation':None if parent_image_hash else 'Old parent has no image-byte receipt; original training bytes cannot be retroactively authenticated.',
        'torch':torch.__version__,'parent_checkpoint':str(parent),'not_a_validated_upgrade':True})
    if '_validation_manifest' in data:save_json(output/'internal_validation_split.json',data['_validation_manifest'])
    with (output/'steps.jsonl').open('a',encoding='utf-8') as log:
        for epoch in range(start,end):
            student.train();loader.batch_sampler.set_epoch(parent_epoch+epoch);generator.manual_seed(cfg['seed']+parent_epoch+epoch)
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            begun=time.monotonic();sums={};success=0;checked=set()
            for step,batch in enumerate(loader,1):
                batch={k:v.to(device,non_blocking=True) for k,v in batch.items()};opt.zero_grad(set_to_none=True)
                diagnose=step==1 or step%cfg['diagnostic_interval']==0
                watched={'backbone':student.backbone.patch_embed.projection.weight,'parser':student.head.segmentation.weight}
                if student.head.context_residual is not None:
                    watched.update(adapter_output=student.head.context_residual.output.weight,adapter_project=student.head.context_residual.project.weight)
                before={k:p.detach().clone() for k,p in watched.items()} if diagnose else {}
                record=accumulate_refinement(student,teacher,batch,cfg,scaler,dtype,donor_rng,epoch,health,checked,diagnose)
                scaler.unscale_(opt);norm=clip_gradients(student.parameters(),0)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite accumulated gradients')
                old_scale=scaler.get_scale();scaler.step(opt);scaler.update()
                updated=not scaler.is_enabled() or scaler.get_scale()>=old_scale
                if not updated:raise RuntimeError('Skipped optimizer update; refinement aborted')
                success+=1;updates+=1
                record.update(epoch=epoch+1,parent_epoch=parent_epoch,step=step,optimizer_updated=True,
                    gradient_norm=float(norm),lr=[g['lr'] for g in opt.param_groups])
                if diagnose:record['parameter_updates']={k:float((p.detach()-before[k]).abs().max()) for k,p in watched.items()}
                record=json_ready(record);log.write(json.dumps(record,allow_nan=False)+'\n')
                for k,v in {'total':record['total'],**record['raw']}.items():sums[k]=sums.get(k,0)+v
                if step==1 or step%cfg['log_interval']==0:
                    log.flush();print(f"Refine{epoch+1}/{cfg['epochs']} step{step}/{len(loader)} loss={record['total']:.4f} accepted={record['stats']['repair_accepted']} updated={success}",flush=True)
                if max_steps and step>=max_steps:break
            if not success:raise RuntimeError('No successful refinement updates')
            scheduler.step()
            row={'epoch':epoch+1,'parent_epoch':parent_epoch,'mean':{k:v/success for k,v in sums.items()},
                 'successful_updates':success,'seconds':time.monotonic()-begun,
                 'peak_cuda_GiB':torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None}
            if cfg['eval_period'] and ((epoch+1)%cfg['eval_period']==0 or epoch+1==cfg['epochs']):
                row['evaluation']=evaluate(student,data['query'],data['gallery'],cfg,device,dtype,output/f'monitor_epoch{epoch+1}',
                    metadata=dict(experiment_metadata(cfg,data['query'],data['gallery'],epoch+1,run_id),parent_epoch=parent_epoch,training_recipe=RECIPE))
                if protocol:
                    from .repair_protocol import evaluate_protocol
                    row['fixed_robustness']=evaluate_protocol(student,data,cfg,device,dtype,output/f'robust_epoch{epoch+1}',
                        {'epoch':epoch+1,'parent_epoch':parent_epoch,'run_id':run_id,'training_recipe':RECIPE})
            history.append(row)
            extra={'history':history,'run_id':run_id,'teacher_sha256':parent_sha,'parent_epoch':parent_epoch,
                   'successful_updates':updates,'gradient_streaks':health.streaks,
                   'epoch_truncated':bool(max_steps and max_steps<len(loader)),'smoke_test':bool(max_steps)}
            save_refinement(output/'last.pt',student,opt,scheduler,scaler,epoch+1,cfg,signature,extra,donor_rng)
            save_json(output/'history.json',history);log.flush()
            print(f"Refine epoch{epoch+1} complete: loss={row['mean']['total']:.4f}, updates={success}",flush=True)
    return student,history
