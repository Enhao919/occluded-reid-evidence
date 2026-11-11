"""Visible-evidence metrics. No predicted gate can disable supervised losses."""
import torch
from torch.nn import functional as F


def _math(x):
    return x if x.dtype == torch.float64 else x.float()


def binary_visibility(labels, confidence, retained=None, min_pixels=32, min_retained=.25, parts=8):
    support=torch.stack([((labels==k)&(confidence>.2)).sum((-1,-2)) for k in range(1,parts+1)],-1)
    v=support>=min_pixels
    if retained is not None:v=v&(retained.detach()>=min_retained)
    return v.to(dtype=torch.float32).detach()


def part_pair_distance(features, visibility):
    if features.ndim!=3 or visibility.shape!=features.shape[:2]:
        raise ValueError('Expected features [N,K,D] and visibility [N,K]')
    f=F.normalize(_math(features),dim=-1)
    v=visibility.detach().to(f)
    weights=torch.minimum(v.T[:,:,None],v.T[:,None,:])
    local=(1-torch.einsum('ikd,jkd->kij',f,f)).clamp(0,2)
    den=weights.sum(0)
    return (weights*local).sum(0)/den.clamp_min(1), den>0


def _pairs(pids, source, valid):
    n=len(pids)
    usable=valid&~torch.eye(n,dtype=torch.bool,device=pids.device)
    if source is not None:usable=usable&(source[:,None]!=source[None,:])
    same=pids[:,None]==pids[None,:]
    return usable&same,usable&~same


def part_averaged_triplet(features,pids,visibility,source=None,margin=.3,temperature=.1):
    d,valid=part_pair_distance(features,visibility)
    pos,neg=_pairs(pids,source,valid)
    active=pos.any(1)&neg.any(1)
    empty={'valid_anchors':0,'violation_rate':0.,'positive_distance':0.,'negative_distance':0.}
    if not active.any():return features.sum()*0,empty
    ap=d[active].masked_fill(~pos[active],-torch.inf).max(1).values
    an=d[active].masked_fill(~neg[active],torch.inf).min(1).values
    violation=ap-an+margin
    z=F.softplus(violation/temperature)*temperature
    return z.mean(),{'valid_anchors':int(active.sum()),'violation_rate':float((violation.detach()>0).float().mean()),
                     'positive_distance':float(ap.detach().mean()),'negative_distance':float(an.detach().mean())}


def positive_negative_npair(positive,negative,temperature=.1):
    if temperature<=0:raise ValueError('temperature must be positive')
    if not positive.numel() or not negative.numel():return (positive.sum()+negative.sum())*0
    p,n=_math(positive),_math(negative)
    return F.softplus(p/temperature+torch.logsumexp(-n/temperature,dim=-1)).mean()


def group_contrast(features,pids,visibility,source=None,temperature=.1,robust=True):
    if features.ndim!=4 or features.shape[0]<2:raise ValueError('Need [V>=2,B,K,D] intervention features')
    if temperature<=0:raise ValueError('temperature must be positive')
    views,b,k,dim=features.shape
    f=F.normalize(_math(features),dim=-1)
    v=visibility.detach().to(f)
    weights=torch.minimum(v.T[:,:,None],v.T[:,None,:]);den=weights.sum(0)
    distances=[]
    for r in range(views):
        for s in range(views):
            local=(1-torch.einsum('ikd,jkd->kij',f[r],f[s])).clamp(0,2)
            distances.append((local*weights).sum(0)/den.clamp_min(1))
    ds=torch.stack(distances)
    upper=ds.max(0).values if robust else ds.mean(0)
    lower=ds.min(0).values if robust else ds.mean(0)
    pos,neg=_pairs(pids,source,den>0)
    active=pos.any(1)&neg.any(1)
    stats={'contrast_anchors':int(active.sum()),'contrast_positive_pairs':int(pos[active].sum()),
           'contrast_negative_pairs':int(neg[active].sum()),'contrast_positive_distance':0.,'contrast_negative_distance':0.}
    if not active.any():return features.sum()*0,stats
    p,n=pos[active],neg[active]
    up,lo=upper[active],lower[active]
    # Select valid anchors before logsumexp: empty rows must never create inf*0.
    log_negative=torch.logsumexp((-lo/temperature).masked_fill(~n,-torch.inf),dim=1)
    terms=F.softplus(up/temperature+log_negative[:,None])
    loss=((terms*p).sum(1)/p.sum(1)).mean()
    stats.update(contrast_positive_distance=float(up.detach()[p].mean()),
                 contrast_negative_distance=float(lo.detach()[n].mean()))
    return loss,stats


def vei_objective(output,batch,config,epoch):
    from .losses import weighted_ce,segmentation_loss,batch_hard_triplet
    b=len(batch['pid']);views=batch['views'];pids=batch['pid'].repeat(views)
    labels,confidence=batch['label'],batch['confidence']
    if len(output['parts'])!=b*views:raise ValueError('View/feature batch mismatch')
    vis=binary_visibility(labels,confidence,batch['retained'],config['supervision_min_pixels'],config['supervision_min_retained'])
    from .vei_parts import merge_visibility,hierarchical_segmentation_loss
    vis=merge_visibility(vis,config.get('part_policy','anatomical8'))
    valid=(vis.sum(1)>0).float();smoothing=config['label_smoothing']
    raw={'global_ce':weighted_ce(output['global_logits'],pids,smoothing=smoothing),
         'foreground_ce':weighted_ce(output['foreground_logits'],pids,valid,smoothing),
         'concat_ce':weighted_ce(output['concat_logits'],pids,valid,smoothing)}
    local=[];global_losses=[];stats={'local_valid_anchors':0,'local_violation_rate':0.,'foreground_valid':int(valid.sum()),
                                 'visible_parts':float(vis.mean())}
    for r in range(views):
        sl=slice(r*b,(r+1)*b)
        z,s=part_averaged_triplet(output['parts'][sl],batch['pid'],vis[sl],batch['source'],
                                  config['triplet_margin'],config['triplet_temperature'])
        local.append(z)
        stats['local_valid_anchors']+=s['valid_anchors'];stats['local_violation_rate']+=s['violation_rate']/views
        if config['loss_weights'].get('global_triplet',0)>0:
            gz,_=batch_hard_triplet(output['global'][sl],batch['pid'],source=batch['source'],
                                    margin=config['triplet_margin'],temperature=config['triplet_temperature'])
            global_losses.append(gz)
    raw['local_triplet']=torch.stack(local).mean()
    raw['global_triplet']=torch.stack(global_losses).mean() if global_losses else output['global'].sum()*0
    parsing_loss=hierarchical_segmentation_loss if config.get('parsing_supervision','flat')=='hierarchical_balanced' else segmentation_loss
    raw['segmentation'],segstats=parsing_loss(output['segmentation'],labels,confidence*valid[:,None,None])
    stats.update(segstats)
    if batch['intervention']:
        if views!=2 or not torch.equal(vis[:b],vis[b:]):raise ValueError('Intervention views must share visible evidence')
        raw['cfcon'],cs=group_contrast(output['parts'].reshape(views,b,*output['parts'].shape[1:]),batch['pid'],
                                      vis[:b],batch['source'],config['contrast_temperature'],config['contrast_robust'])
    else:
        raw['cfcon']=output['parts'].sum()*0
        cs={'contrast_anchors':0,'contrast_positive_pairs':0,'contrast_negative_pairs':0,
            'contrast_positive_distance':0.,'contrast_negative_distance':0.}
    stats.update(cs)
    weights=dict(config['loss_weights'])
    ramp=max(0.,min(1.,(epoch+1-config['contrast_start'])/config['contrast_ramp']))
    weights['cfcon']*=ramp;stats['contrast_weight']=weights['cfcon']
    weighted={k:v*weights[k] for k,v in raw.items()}
    return sum(weighted.values()),raw,weighted,stats
