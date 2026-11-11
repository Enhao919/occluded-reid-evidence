from pathlib import Path
import copy
import yaml

ROOT=Path(__file__).resolve().parents[1]


def load_config(path=None,overrides=None):
    default=ROOT/'configs/a100_40gb.yaml'
    cfg=yaml.safe_load(default.read_text(encoding='utf-8'))
    if path and Path(path).resolve()!=default.resolve():
        custom=yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
        for k,v in custom.items():
            if k not in cfg and k not in ('num_classes','smoke_test'):
                raise ValueError(f'Unknown config key: {k}')
            if isinstance(v,dict):cfg[k].update(v)
            else:cfg[k]=v
    for k,v in (overrides or {}).items():
        if v is not None:cfg[k]=v
    # Resolve YAML and CLI through exactly the same baseline contract.
    if cfg.get('global_only'):
        cfg['occlusion_probability']=0.
        if cfg.get('method')=='vei':cfg['view_layout']='ba2'
        cfg['loss_weights']={k:(v if k in ('global_ce','global_triplet') else 0.)
                             for k,v in cfg['loss_weights'].items()}
    for k in ('data_root','pretrained','label_root','output_dir'):
        p=Path(cfg[k]).expanduser()
        cfg[k]=str((ROOT/p).resolve() if not p.is_absolute() else p.resolve())
    if len(cfg['image_size'])!=2 or any(int(n)<32 or int(n)%32 for n in cfg['image_size']):
        raise ValueError('image_size must contain H,W divisible by32')
    if cfg['batch_size']%cfg['instances'] or cfg['instances']<2 or cfg['batch_size']//cfg['instances']<2:
        raise ValueError('batch_size=P*K with P>=2,K>=2')
    if cfg['feature_dim']%8:raise ValueError('feature_dim must be divisible by8')
    if cfg['optimizer'] not in ('adamw','sgd'):raise ValueError('optimizer: adamw or sgd')
    if not isinstance(cfg['sgd_nesterov'],bool):raise ValueError('sgd_nesterov must be boolean')
    if not 0<=cfg['single_erase_probability']<=1:raise ValueError('single_erase_probability must be in [0,1]')
    if cfg['gradient_clip']<0:raise ValueError('gradient_clip must be nonnegative')
    if cfg['precision'] not in ('fp32','fp16','bf16'):raise ValueError('precision: fp32/fp16/bf16')
    if cfg['epochs']<1 or cfg['warmup_epochs']<0:raise ValueError('Invalid epochs/warmup')
    if not 0<=cfg['global_weight']<=1 or not 0<=cfg['foreground_weight']<=1-cfg['global_weight']:
        raise ValueError('Invalid retrieval mixture weights')
    if any(v<0 for v in cfg['loss_weights'].values()):raise ValueError('Loss weights cannot be negative')
    if cfg['method'] not in ('pso','vei'):raise ValueError('method must be pso or vei')
    if cfg['part_policy'] not in ('anatomical8','bilateral5'):raise ValueError('Invalid part_policy')
    if cfg['head_architecture'] not in ('vei','bpbreid'):raise ValueError('Invalid head_architecture')
    if not isinstance(cfg['bpbreid_msf'],bool):raise ValueError('bpbreid_msf must be boolean')
    if cfg['head_architecture']=='bpbreid' and (cfg['method']!='vei' or cfg['part_policy']!='anatomical8' or cfg['identity_masks_detached']):
        raise ValueError('BPBreID head requires method vei, anatomical8 and reference attention gradients')
    if not isinstance(cfg['identity_masks_detached'],bool):raise ValueError('identity_masks_detached must be boolean')
    if cfg['method']!='vei' and cfg['identity_masks_detached']:raise ValueError('identity_masks_detached requires method vei')
    if cfg['parsing_supervision'] not in ('flat','hierarchical_balanced'):raise ValueError('Invalid parsing_supervision')
    if not isinstance(cfg['curriculum_epoch_offset'],int) or cfg['curriculum_epoch_offset']<0:raise ValueError('Invalid curriculum_epoch_offset')
    if cfg['method']!='vei' and (cfg['part_policy']!='anatomical8' or cfg['parsing_supervision']!='flat' or cfg['curriculum_epoch_offset']!=0):
        raise ValueError('Bilateral policies and curriculum offsets require method vei')
    if cfg['contrast_temperature']<=0 or cfg['contrast_ramp']<1:raise ValueError('Invalid contrast temperature/ramp')
    if cfg['view_layout'] not in ('ba2','pam2','pam3_reference','budget2_intervention','single'):raise ValueError('Invalid view layout')
    if cfg['retrieval_rule'] not in ('legacy','visible_mean','bpbreid'):raise ValueError('Invalid retrieval rule')
    if not 0<=cfg['validation_fraction']<.5:raise ValueError('validation_fraction must be in [0,.5)')
    return cfg


def lr_factor(epoch,cfg):
    import math
    warm=cfg['warmup_epochs']
    if epoch<warm:return .01+.99*epoch/max(1,warm)
    progress=min(1.,max(0.,(epoch-warm)/max(1,cfg['epochs']-warm)))
    floor=cfg['min_lr_ratio']
    return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))


def validate_resume(saved,current):
    for key,default in (('part_policy','anatomical8'),('parsing_supervision','flat'),('curriculum_epoch_offset',0),('identity_masks_detached',False),('head_architecture','vei'),('bpbreid_msf',True)):
        if saved.get(key,default)!=current.get(key,default):raise ValueError(f'Resume config mismatch: {key}')
    # Paths/output placement may change on upload, but the experiment may not.
    mutable={'data_root','pretrained','label_root','output_dir','workers','test_batch_size',
             'log_interval','checkpoint_period','eval_period','distance_chunk','flip_test','rerank'}
    for key,value in saved.items():
        if key in mutable or key=='num_classes':continue
        if key not in current or current[key]!=value:
            raise ValueError(f'Resume config mismatch: {key} ({value!r} -> {current.get(key)!r})')
    if saved['num_classes']!=current['num_classes']:raise ValueError('Resume class count mismatch')
