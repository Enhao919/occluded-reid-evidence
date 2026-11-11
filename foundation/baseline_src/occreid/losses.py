"""All objectives differentiate CURRENT outputs; targets/validity never optimize themselves."""
import torch
from torch.nn import functional as F


def weighted_mean(values,weights):
    weights=weights.to(values).detach()
    return (values*weights).sum()/weights.sum().clamp_min(1e-6)


def weighted_ce(logits,labels,weights=None,smoothing=.1):
    loss=F.cross_entropy(logits.float(),labels,reduction='none',label_smoothing=smoothing)
    return loss.mean() if weights is None else weighted_mean(loss,weights)


def batch_hard_triplet(features,labels,valid=None,source=None,margin=.3,temperature=.1):
    features=F.normalize(features.float(),dim=-1)
    # Explicit squared-distance formula avoids cdist's fp16/zero-distance backward pitfalls.
    square=(2-2*(features@features.T)).clamp_min(1e-8)
    distance=square.sqrt()
    n=len(labels);eye=torch.eye(n,device=labels.device,dtype=torch.bool)
    valid=torch.ones(n,device=labels.device,dtype=torch.bool) if valid is None else valid.bool()
    pair_valid=valid[:,None]&valid[None,:]&~eye
    if source is not None:pair_valid=pair_valid&(source[:,None]!=source[None,:])
    positive=(labels[:,None]==labels[None,:])&pair_valid
    negative=(labels[:,None]!=labels[None,:])&pair_valid
    anchors=positive.any(1)&negative.any(1)
    if not anchors.any():
        return features.sum()*0,{'valid_anchors':0,'violation_rate':0.,'positive_distance':0.,'negative_distance':0.}
    ap=distance.masked_fill(~positive,-1).max(1).values[anchors]
    an=distance.masked_fill(~negative,3).min(1).values[anchors]
    violation=ap-an+margin
    loss=(F.softplus(violation/temperature)*temperature).mean()
    return loss,{'valid_anchors':int(anchors.sum()),'violation_rate':float((violation.detach()>0).float().mean()),
                 'positive_distance':float(ap.detach().mean()),'negative_distance':float(an.detach().mean())}


def supervision_visibility(labels,confidence,min_pixels=3):
    """Detached teacher support; inferred visibility cannot turn off the training loss."""
    weights=[]
    for k in range(1,7):
        mask=(labels==k)&(confidence>.2)
        count=mask.sum((-1,-2))
        value=(confidence*mask).sum((-1,-2))/count.clamp_min(1)
        weights.append(value*(count>=min_pixels))
    return torch.stack(weights,1).detach()


def segmentation_loss(logits,labels,confidence,background_weight=.25):
    size=logits.shape[-2:]
    labels=F.interpolate(labels[:,None].float(),size=size,mode='nearest')[:,0].long()
    confidence=F.interpolate(confidence[:,None].float(),size=size,mode='bilinear',align_corners=False)[:,0]
    valid=labels!=255
    loss=F.cross_entropy(logits.float(),labels,ignore_index=255,reduction='none')
    weight=confidence*valid*torch.where(labels==0,background_weight,1.)
    return weighted_mean(loss,weight),{'valid_pixels':int((weight>0).sum()),
        'foreground_pixels':int(((labels>0)&(labels<logits.shape[1])&(weight>0)).sum())}


def part_consistency(student,target,weights):
    error=1-F.cosine_similarity(student.float(),target.detach().float(),dim=-1)
    return weighted_mean(error,weights)


def reid_objective(output,batch,config,epoch):
    if config.get('method')=='vei':
        if config.get('head_architecture','vei')=='bpbreid':
            from .bpbreid_losses import bpbreid_objective
            return bpbreid_objective(output,batch,config,epoch)
        from .vei_losses import vei_objective
        return vei_objective(output,batch,config,epoch)
    b=len(batch['pid']);pid=batch['pid'];source=batch['source']
    labels,confidence=batch['label'],batch['confidence']
    pids=pid.repeat(2)
    vis=supervision_visibility(labels,confidence)
    smoothing=config['label_smoothing']
    fg_valid=(vis.sum(1)>0).float()
    raw={'global_ce':weighted_ce(output['global_logits'],pids,smoothing=smoothing),
         'foreground_ce':weighted_ce(output['foreground_logits'],pids,fg_valid,smoothing),
         'local_ce':weighted_ce(output['part_logits'].reshape(-1,config['num_classes']),
                                pids[:,None].expand(-1,6).reshape(-1),vis.reshape(-1),smoothing)}
    global_triplets=[];local_triplets=[];stats={}
    for view in range(2):
        selection=slice(view*b,(view+1)*b)
        loss,s=batch_hard_triplet(output['global'][selection],pid,source=source,
                                 margin=config['triplet_margin'],temperature=config['triplet_temperature'])
        global_triplets.append(loss)
        for key,value in s.items():stats[f'global_{key}']=stats.get(f'global_{key}',0)+value/2
        for k in range(6):
            loss,s=batch_hard_triplet(output['parts'][selection,k],pid,vis[selection,k]>.2,source,
                                     config['triplet_margin'],config['triplet_temperature'])
            if s['valid_anchors']:
                local_triplets.append(loss)
            stats['local_valid_anchors']=stats.get('local_valid_anchors',0)+s['valid_anchors']
    raw['global_triplet']=torch.stack(global_triplets).mean()
    raw['local_triplet']=torch.stack(local_triplets).mean() if local_triplets else output['parts'].sum()*0
    raw['segmentation'],segstats=segmentation_loss(output['segmentation'],labels,confidence)
    stats.update(segstats)
    both=torch.minimum(vis[:b],vis[b:])
    raw['consistency']=part_consistency(output['parts'][b:],output['parts'][:b],both)
    stats.update(visible_parts=float((vis>.2).float().mean()),consistency_pairs=int((both>.2).sum()),
                 foreground_valid=int(fg_valid.sum()))
    ramp=max(0.,min(1.,(epoch-config['consistency_start'])/max(1,config['consistency_ramp'])))
    weights=dict(config['loss_weights']);weights['consistency']*=ramp
    weighted={k:v*weights[k] for k,v in raw.items()}
    total=sum(weighted.values())
    return total,raw,weighted,stats
