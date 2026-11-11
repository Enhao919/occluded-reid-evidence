"""Fixed-support, one-sided identity guards and actual intervention rank repair.

No pixel ownership or causal identifiability is certified by these self-checks.
All teacher selection/targets are detached. Original GiLt is left untouched.
"""
import torch
from torch.nn import functional as F

HEADS=('global_logits','foreground_logits','concat_logits')


def working(x):return x if x.dtype==torch.float64 else x.float()


def feature_pack(output):
    parts=F.normalize(working(output['parts']),dim=-1)
    fg=F.normalize(working(output['foreground_eval']),dim=-1)
    return torch.cat((fg[:,None],parts),1)


def class_margin(logits,pid):
    z=working(logits);onehot=F.one_hot(pid,z.shape[1]).bool()
    return z.gather(1,pid[:,None])[:,0]-z.masked_fill(onehot,-torch.inf).amax(1)


@torch.no_grad()
def select_relations(teacher,batch,min_margin=.05):
    f=feature_pack(teacher);pid,cam,source=(batch[k] for k in ('pid','cam','source'))
    b=len(pid);device=f.device
    correct=torch.stack([teacher[k].argmax(1)==pid for k in HEADS]).all(0)
    y,c=batch['label'],batch['confidence']
    counts=torch.stack([((y==k)&(c>.2)).sum((-1,-2)) for k in range(1,9)],1)
    known=(counts>=16)&(teacher['visibility']>0)
    distances=torch.cdist(f.transpose(0,1),f.transpose(0,1),compute_mode='donot_use_mm_for_euclid_dist')
    indices=torch.arange(b,device=device)[:,None].repeat(1,3)
    support=torch.zeros(b,8,dtype=torch.bool,device=device)
    valid=torch.zeros(b,dtype=torch.bool,device=device);margin=f.new_zeros(b)
    reasons={'teacher_wrong':0,'no_positive':0,'no_camera_negative':0,'insufficient_common_parts':0,'small_margin':0}
    for i in range(b):
        if not bool(correct[i]):reasons['teacher_wrong']+=1;continue
        positives=torch.where((pid==pid[i])&(cam!=cam[i])&(source!=source[i])&correct)[0]
        if not len(positives):reasons['no_positive']+=1;continue
        best=None;has_camera_negative=False
        for p in positives.tolist():
            common=known & known[i] & known[p]
            camera_negative=(pid!=pid[i])&(cam==cam[p])
            has_camera_negative=has_camera_negative or bool(camera_negative.any())
            legal=camera_negative&(common.sum(1)>=2)
            if not legal.any():continue
            weights=torch.cat((torch.ones(b,1,dtype=torch.bool,device=device),common),1)
            gaps=((distances[:,i,:].T-distances[:,i,p][None])*weights).sum(1)/weights.sum(1)
            value,n=gaps.masked_fill(~legal,torch.inf).min(0)
            if best is None or float(value)<best[0]:best=(float(value),p,int(n),common[n])
        if best is None:
            reasons['insufficient_common_parts' if has_camera_negative else 'no_camera_negative']+=1
            continue
        value,p,n,s=best
        if value<=min_margin:reasons['small_margin']+=1;continue
        indices[i]=torch.tensor([i,p,n],device=device);support[i]=s;valid[i]=True;margin[i]=value
    return {'indices':indices.detach(),'support':support.detach(),'valid':valid.detach(),
            'teacher_margin':margin.detach(),'target':margin.clamp(max=.3).detach(),'reasons':reasons}


def relation_margins(features,relations,detach_refs=True,reference_features=None):
    a,p,n=relations['indices'].unbind(1)
    references=features if reference_features is None else reference_features
    positive,negative=references[p],references[n]
    if detach_refs:positive,negative=positive.detach(),negative.detach()
    dp=torch.linalg.vector_norm(features[a]-positive,dim=-1)
    dn=torch.linalg.vector_norm(features[a]-negative,dim=-1)
    support=torch.cat((torch.ones(len(a),1,dtype=torch.bool,device=features.device),relations['support']),1)
    gap=((dn-dp)*support).sum(1)/support.sum(1).clamp_min(1)
    return gap*relations['valid']


def masked_mean(value,mask):return (value*mask).sum()/mask.sum().clamp_min(1)


def weighted_rescue(margin,target,accepted,risk):
    return masked_mean(F.relu(target.detach()-margin)*risk.detach(),accepted.detach())


def repair_objectives(clean,environment,teacher,teacher_environment,batch,relations,actual_mask_active,
                      damage_weighted=True):
    cf,ef=feature_pack(clean),feature_pack(environment)
    cm=relation_margins(cf,relations)
    em=relation_margins(ef,relations,reference_features=cf)
    with torch.no_grad():
        tm=relation_margins(feature_pack(teacher_environment),relations,reference_features=feature_pack(teacher))
        correct=torch.stack([teacher_environment[k].argmax(1)==batch['pid'] for k in HEADS])
        votes=correct.sum(0)
        accepted=relations['valid']&actual_mask_active.bool()&(votes>=2)
        risk=((relations['teacher_margin']-tm)/.1).clamp(0,1)
        weight=risk if damage_weighted else torch.ones_like(risk)
    relation_guard=masked_mean(F.relu(relations['target']-cm),relations['valid'])
    guards=[relation_guard];head_counts={}
    for key in HEADS:
        target=class_margin(teacher[key].detach(),batch['pid']).detach()
        active=target>0;head_counts[key]=int(active.sum())
        guards.append(masked_mean(F.relu(target.clamp(max=2)-class_margin(clean[key],batch['pid'])),active))
    guard=torch.stack(guards).mean()
    rescue=weighted_rescue(em,relations['target'],accepted,weight)
    stats={'guard_active':int(relations['valid'].sum())+sum(head_counts.values()),
           'guard_relations':int(relations['valid'].sum()),'guard_head_counts':head_counts,
           'repair_accepted':int(accepted.sum()),'repair_damaged':int((accepted&(risk>0)).sum()),
           'repair_violations':int((accepted&(weight>0)&(em.detach()<relations['target'])).sum()),
           'repair_risk_mean':float(masked_mean(risk,accepted)),
           'teacher_clean_margin':float(masked_mean(relations['teacher_margin'],relations['valid'])),
           'teacher_environment_margin':float(masked_mean(tm,accepted)),
           'student_clean_margin':float(masked_mean(cm.detach(),relations['valid'])),
           'student_environment_margin':float(masked_mean(em.detach(),accepted)),
           'repair_rejections':{'no_relation':int((~relations['valid']).sum()),
               'no_changed_pixels':int((relations['valid']&~actual_mask_active.bool()).sum()),
               'teacher_identity_disagreement':int((relations['valid']&actual_mask_active.bool()&(votes<2)).sum()),
               'no_actual_damage':int((accepted&(risk<=0)).sum()),
               'satisfied_margin':int((accepted&(em.detach()>=relations['target'])).sum())},
           'teacher_environment_wrong':{key:int((~correct[i]&relations['valid']&actual_mask_active.bool()).sum()) for i,key in enumerate(HEADS)},
           'teacher_environment_vote_histogram':{str(i):int(((votes==i)&relations['valid']&actual_mask_active.bool()).sum()) for i in range(4)},
           'relation_rejections':relations['reasons']}
    return {'identity_guard':guard,'relation_rescue':rescue},stats
