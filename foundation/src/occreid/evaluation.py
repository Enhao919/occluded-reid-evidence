import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import FLIP

FEATURE_KEYS=('global','foreground','parts','visibility')


def merge_flip(original,flipped):
    result={}
    for k in FEATURE_KEYS:
        other=flipped[k]
        if k in ('parts','visibility'):
            indices={8:[0,2,1,3,5,4,7,6],5:list(range(5)),6:[v-1 for v in FLIP[1:]]}[other.shape[1]]
            other=other[:,indices]
        if k=='visibility' and other.shape[1] in (5,8):
            result[k]=((original[k]>0)|(other>0)).float()
        elif k=='parts' and other.shape[1] in (5,8):
            a=(original['visibility']>0).float()[...,None]
            order=[0,2,1,3,5,4,7,6] if other.shape[1]==8 else list(range(5))
            b=(flipped['visibility'][:,order]>0).float()[...,None]
            result[k]=(original[k]*a+other*b)/(a+b).clamp_min(1)
        else:result[k]=(original[k]+other)*.5
    return result


@torch.inference_mode()
def extract_features(model,loader,device,amp_dtype=None,flip=False):
    model.eval();features={k:[] for k in FEATURE_KEYS};pids=[];cameras=[]
    for step,batch in enumerate(loader,1):
        images=batch['image'].to(device,non_blocking=True)
        with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp_dtype is not None):
            output=model(images)
            output=dict(output,**{k:output[k+'_eval'] for k in ('global','foreground') if k+'_eval' in output})
            if flip:
                flipped=model(images.flip(-1))
                flipped=dict(flipped,**{k:flipped[k+'_eval'] for k in ('global','foreground') if k+'_eval' in flipped})
                output=merge_flip(output,flipped)
        for k in FEATURE_KEYS:
            value=output[k].float().cpu()
            if not torch.isfinite(value).all():raise FloatingPointError(f'Nonfinite test feature: {k}')
            features[k].append(value)
        pids.extend(batch['pid'].tolist());cameras.extend(batch['cam'].tolist())
        if step==1 or step%50==0 or step==len(loader):print(f'Extract {step}/{len(loader)}',flush=True)
    return {k:torch.cat(v) for k,v in features.items()},np.array(pids),np.array(cameras)


def retrieval_distance(query,gallery,mode='parts',chunk_size=128,device='cpu',
                       global_weight=.4,foreground_weight=.2,rule='legacy'):
    """No identity/camera inputs. Distances depend ONLY on image representations."""
    if mode not in ('global','parts','local'):raise ValueError(mode)
    if rule not in ('legacy','visible_mean','bpbreid'):raise ValueError(rule)
    if chunk_size<1:raise ValueError('chunk_size must be positive')
    gallery={k:v.to(device=device,dtype=torch.float32) for k,v in gallery.items()}
    g_global=F.normalize(gallery['global'],dim=-1)
    g_fg=F.normalize(gallery['foreground'],dim=-1)
    g_part=F.normalize(gallery['parts'],dim=-1)
    def distance(a,b):
        if rule=='bpbreid':
            return (a.square().sum(-1)[:,None]+b.square().sum(-1)[None,:]-2*a@b.T).clamp_min(0).sqrt()
        return (1-a@b.T).clamp(0,2)
    blocks=[]
    for start in range(0,len(query['global']),chunk_size):
        q={k:v[start:start+chunk_size].to(device=device,dtype=torch.float32) for k,v in query.items()}
        global_distance=distance(F.normalize(q['global'],dim=-1),g_global)
        if mode=='global':blocks.append(global_distance.cpu().numpy());continue
        fg_distance=distance(F.normalize(q['foreground'],dim=-1),g_fg)
        local_sum=torch.zeros_like(global_distance);denominator=torch.zeros_like(global_distance)
        q_part=F.normalize(q['parts'],dim=-1)
        for part in range(q_part.shape[1]):
            weight=q['visibility'][:,part,None]*gallery['visibility'][None,:,part]
            part_distance=distance(q_part[:,part],g_part[:,part])
            local_sum+=weight*part_distance;denominator+=weight
        local_distance=local_sum/denominator.clamp_min(1e-6)
        if mode=='local':blended=local_distance
        elif rule in ('visible_mean','bpbreid'):blended=(local_sum+fg_distance)/(denominator+1)
        else:blended=global_weight*global_distance+foreground_weight*fg_distance+(1-global_weight-foreground_weight)*local_distance
        blended=torch.where(denominator>1e-6,blended,fg_distance if rule=='bpbreid' else global_distance)
        blocks.append(blended.cpu().numpy())
    result=np.concatenate(blocks)
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite retrieval distances')
    return result


def ranking_metrics(distances,qids,gids,qcams,gcams,max_rank=50):
    qids,gids,qcams,gcams=map(np.asarray,(qids,gids,qcams,gcams))
    if distances.shape!=(len(qids),len(gids)) or len(qcams)!=len(qids) or len(gcams)!=len(gids):
        raise ValueError('Distance/metadata shape mismatch')
    if not np.isfinite(distances).all():raise ValueError('Nonfinite distances')
    if len(gids)==0:raise ValueError('No valid gallery')
    rank=min(max_rank,len(gids));curves=[];aps=[]
    for i,row in enumerate(distances):
        order=np.argsort(row,kind='stable')
        keep=~(((gids[order]==qids[i])&(gcams[order]==qcams[i]))|(gids[order]==-1))
        match=(gids[order][keep]==qids[i]).astype(np.int32)
        if not match.any():continue
        cumulative=match.cumsum()
        curve=np.ones(rank,np.float64)
        curve[:min(rank,len(match))]=(cumulative>0)[:rank]
        curves.append(curve)
        aps.append(float(((cumulative/np.arange(1,len(match)+1))*match).sum()/match.sum()))
    if not aps:raise ValueError('No valid cross-camera positive query')
    cmc=np.stack(curves).mean(0)
    return {'mAP':float(np.mean(aps)),'rank1':float(cmc[0]),'rank5':float(cmc[min(4,rank-1)]),
            'rank10':float(cmc[min(9,rank-1)]),'valid_queries':len(aps),
            'skipped_queries':len(qids)-len(aps),'query_images':len(qids),'gallery_images':len(gids)}


def bridge_rerank(distances,gallery,topk=200,references=5,strength=.2):
    """Optional unlabeled reliable-gallery bridge; NOT a Cheb-GR/PSS reproduction.

    Candidate/reference selection is image-only. No gallery identity centroids.
    Complexity avoids a dense gallery-by-gallery NxN matrix.
    """
    features=F.normalize(gallery['global'].float().cpu(),dim=-1)
    quality=gallery['visibility'].mean(1).float().cpu().numpy()
    result=distances.copy();n=len(features);topk=min(topk,n);references=min(references,topk)
    for i,row in enumerate(distances):
        candidates=np.argsort(row)[:topk]
        reliability=np.exp(-row[candidates])*np.maximum(.05,quality[candidates])
        ref_order=np.argsort(reliability)[-references:]
        anchors=candidates[ref_order]
        neighbor=(1-features[candidates]@features[anchors].T).clamp(0,2).numpy()
        weights=reliability[ref_order];weights=weights/(weights.sum()+1e-12)
        # Nonnegative bounded blend; outside candidates stay unchanged.
        result[i,candidates]=(1-strength)*row[candidates]+strength*(neighbor@weights)
    return result
