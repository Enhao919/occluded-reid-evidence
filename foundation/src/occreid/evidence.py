"""Opt-in evidence-preserving training; reference architecture/test rule unchanged."""
import torch
from torch.nn import functional as F

AUX_NAMES=('cf_invariance','group_contrast')

def working(x):
    return x if x.dtype==torch.float64 else x.float()

def eligible_parts(batch):
    y,c=batch['label'],batch['confidence']
    counts=torch.stack([((y==k)&(c>.2)).sum((-1,-2)) for k in range(1,9)],-1)
    retained=batch.get('retained',torch.ones_like(counts))
    return ((counts>=16)&(retained>=.5)).detach()

def cf_invariance(parts,foreground,eligible,batch_size):
    if len(parts)!=2*batch_size:raise ValueError('CF requires two view-major batches')
    p,f=working(parts),working(foreground)
    g=(eligible[:batch_size]&eligible[batch_size:]).detach()
    active=g.sum(-1)>=2
    stats={'cf_valid_sources':int(active.sum()),'cf_valid_parts':int(g[active].sum())}
    if not active.any():return (p.sum()+f.sum())*0,stats
    pn=F.normalize(p,dim=-1);fn=F.normalize(f,dim=-1)
    pd=(pn[:batch_size]-pn[batch_size:]).square().sum(-1)
    fd=(fn[:batch_size]-fn[batch_size:]).square().sum(-1)
    grouped=(pd*g).sum(-1)/g.sum(-1).clamp_min(1)
    return (.5*(grouped+fd))[active].mean(),stats

def group_contrast(parts,eligible,pid,cam,source,temperature=.1):
    if temperature<=0:raise ValueError('positive contrast temperature required')
    p=F.normalize(working(parts),dim=-1);n=len(p)
    g=(eligible[:,None,:]&eligible[None,:,:]).detach()
    count=g.sum(-1)
    different_source=source[:,None]!=source[None,:]
    same=pid[:,None]==pid[None,:]
    positive=same&(cam[:,None]!=cam[None,:])&different_source&(count>=2)
    negative=(~same)&different_source&(count>=2)
    active=positive.any(1)&negative.any(1)
    stats={'group_anchors':int(active.sum()),'group_positive_pairs':int(positive[active].sum()),
           'group_negative_pairs':int(negative[active].sum()),'group_positive_entropy_floor':0.}
    if not active.any():return parts.sum()*0,stats
    sq=p.square().sum(-1).T
    d2=(sq[:,:,None]+sq[:,None,:]-2*torch.einsum('ikd,jkd->kij',p,p)).clamp_min(0)
    zero=d2==0
    d=(d2+zero.to(d2)*1e-16).sqrt().masked_fill(zero,0).permute(1,2,0)
    distance=(d*g).sum(-1)/count.clamp_min(1)
    logits=-distance[active]/temperature
    pos=positive[active];valid=pos|negative[active]
    np_=pos.sum(1).to(logits)
    floor=np_.log()
    loss=torch.logsumexp(logits.masked_fill(~valid,-torch.inf),1)-(logits*pos).sum(1)/np_-floor
    stats['group_positive_entropy_floor']=float(floor.detach().mean())
    return loss.clamp_min(0).mean(),stats

def build_views(batch,cfg,epoch,force_mode=None):
    from .vei_augment import _erase_pair,_crop,_counts
    x,y,c=batch['image'],batch['label'],batch['confidence'];b=len(x)
    if force_mode is None:
        draw=float(torch.rand(()))
        mode='intervention' if draw<.25 else ('crop' if draw<.625 else 'erase')
    else:mode=force_mode
    base=(_counts(y,c)>0).float()
    mask=torch.zeros_like(y,dtype=torch.bool)
    donors=torch.arange(b,device=x.device).repeat(2,1)
    fallback=0
    if mode=='crop':
        z,yz,cz,r,fallback=_crop(batch)
        images,labels,confs,retained=[x,z],[y,yz],[c,cz],[base,r]
    elif mode in ('erase','intervention'):
        a,z,ya,yz,ca,cz,r,mask,donors=_erase_pair(batch,epoch,cfg,mode=='intervention')
        if mode=='intervention':
            images,labels,confs,retained=[a,z],[ya,yz],[ca,cz],[r,r]
        else:images,labels,confs,retained=[x,a],[y,ya],[c,ca],[base,r]
    else:raise ValueError('Unknown evidence view mode '+str(mode))
    pair=dict(batch,label=torch.cat(labels),confidence=torch.cat(confs),
              retained=torch.cat(retained),intervention=mode=='intervention',cf_pair=mode=='intervention',
              intervention_mask=mask,donor_indices=donors,views=2)
    stats={'view_count':2,'view_mode':{'erase':1,'crop':2,'intervention':3}[mode],
           'occluded_fraction':float(mask.float().mean()),'crop_fallback_fraction':fallback/max(1,b)}
    return torch.cat(images),pair,stats

def enabled(cfg,epoch):
    r=cfg.get('research',{})
    return r.get('enabled',False) and epoch>=r.get('warmup_epochs',5)

def set_pair_context(model,batch_size):
    from .backbone import DropPath
    for module in model.modules():
        if isinstance(module,DropPath):module.paired_batch_size=int(batch_size)

def extend_objective(reference,output,batch,cfg,epoch):
    if not enabled(cfg,epoch):return reference
    total,raw,weighted,stats=reference
    raw=dict(raw);weighted=dict(weighted);stats=dict(stats)
    r=cfg['research'];ramp=max(0.,min(1.,(epoch-r.get('warmup_epochs',5))/max(1,r.get('ramp_epochs',5))))
    eligible=eligible_parts(batch);b=len(batch['pid']);v=batch['views']
    zero=(output['parts'].sum()+output['foreground_eval'].sum())*0
    cfstats={'cf_valid_sources':0,'cf_valid_parts':0}
    gcstats={'group_anchors':0,'group_positive_pairs':0,'group_negative_pairs':0,'group_positive_entropy_floor':0.}
    cf,con=zero,zero
    if cfg['loss_weights'].get('cf_invariance',0)>0 and batch.get('cf_pair',False):
        cf,cfstats=cf_invariance(output['parts'],output['foreground_eval'],eligible,b)
    if cfg['loss_weights'].get('group_contrast',0)>0:
        con,gcstats=group_contrast(output['parts'],eligible,batch['pid'].repeat(v),
                                 batch['cam'].repeat(v),batch['source'].repeat(v),r.get('temperature',.1))
    stats.update(cfstats);stats.update(gcstats)
    for name,value in (('cf_invariance',cf),('group_contrast',con)):
        weight=cfg['loss_weights'].get(name,0)*ramp
        raw[name]=value;weighted[name]=value*weight;stats[name+'_weight']=weight
        total=total+weighted[name]
    return total,raw,weighted,stats

def diagnostic_live(weighted,cfg,stats):
    return {k:v for k,v in weighted.items()
            if cfg['loss_weights'].get(k,0)>0
            and (k!='cfcon' or stats.get('contrast_weight',0)>0)
            and (k not in AUX_NAMES or stats.get(k+'_weight',0)>0)}

def active_aux(stats):
    return {name for name,count in (('cf_invariance','cf_valid_sources'),('group_contrast','group_anchors'))
            if stats.get(name+'_weight',0)>0 and stats.get(count,0)>0}


def input_fingerprint(images,paired,batch,include_pixels=False):
    """Exact source/RNG receipts each step; full tensor receipt at fixed probes."""
    import hashlib
    def digest(tensors):
        h=hashlib.sha256()
        for tensor in tensors:
            value=tensor.detach().cpu().contiguous()
            h.update(str((tuple(value.shape),str(value.dtype))).encode())
            h.update(value.numpy().tobytes())
        return h.hexdigest()
    rng=[torch.get_rng_state()]
    if images.is_cuda:rng.append(torch.cuda.get_rng_state(images.device))
    result={'evidence_source_sha256':digest([batch[k] for k in ('pid','cam','source')]),
            'evidence_rng_sha256':digest(rng)}
    if include_pixels:
        result['evidence_pixels_sha256']=digest([images,paired['label'].to(torch.uint8),paired['confidence']])
    return result

