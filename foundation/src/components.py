"""Isolated layer-wise candidates. None is claimed novel or effective a priori."""
import copy,types
import torch
from torch import nn
from torch.nn import functional as F

class PartConditionalBN(nn.Module):
    """Separate part identity from visible-evidence selection, with explicit controls."""
    def __init__(self, old, groups=8, visible_only=False):
        super().__init__();assert groups in [1,8]
        self.groups=groups;self.visible_only=visible_only;self.eps=old.eps;self.momentum=old.momentum
        self.weight=nn.Parameter(old.weight.detach().repeat(groups,1));self.bias=nn.Parameter(old.bias.detach().repeat(groups,1))
        self.register_buffer('running_mean',old.running_mean.detach().repeat(groups,1))
        self.register_buffer('running_var',old.running_var.detach().repeat(groups,1))
        self.register_buffer('num_batches_tracked',torch.zeros((),dtype=torch.long))
        self.register_buffer('visible_counts',torch.zeros(groups,dtype=torch.float64))
        self.valid_hint=None
    def forward(self,x):
        assert x.ndim==2 and len(x)%8==0
        if not self.training and self.num_batches_tracked.item()==0:
            # Exact native operator at initialization, rather than algebraically equivalent hand arithmetic.
            return F.batch_norm(x,self.running_mean[0],self.running_var[0],self.weight[0],self.bias[0],False,self.momentum,self.eps)
        n=len(x)//8;z=x.float().reshape(n,8,-1)
        if self.groups==1:z=z.reshape(n*8,1,-1)
        if self.training:
            if self.visible_only:
                assert self.valid_hint is not None and self.valid_hint.shape==(n,8),'training visibility labels missing'
                valid=self.valid_hint.detach().float()
            else:valid=torch.ones((n,8),device=x.device)
            if self.groups==1:valid=valid.reshape(n*8,1)
            count=valid.sum(0);den=count.clamp_min(1)[:,None]
            mean=(z*valid[...,None]).sum(0)/den
            var=((z-mean[None]).square()*valid[...,None]).sum(0)/den
            active=count>1
            mean=torch.where(active[:,None],mean,self.running_mean)
            var=torch.where(active[:,None],var,self.running_var)
            with torch.no_grad():
                momentum=.1 if self.momentum is None else self.momentum
                unbiased=var.detach()*count[:,None]/(count-1).clamp_min(1)[:,None]
                new_mean=torch.lerp(self.running_mean,mean.detach(),momentum)
                new_var=torch.lerp(self.running_var,unbiased,momentum)
                self.running_mean.copy_(torch.where(active[:,None],new_mean,self.running_mean))
                self.running_var.copy_(torch.where(active[:,None],new_var,self.running_var))
                self.visible_counts.add_(count.to(self.visible_counts));self.num_batches_tracked.add_(1)
        else:mean,var=self.running_mean,self.running_var
        result=(z-mean[None])*torch.rsqrt(var[None]+self.eps)
        result=result*self.weight[None]+self.bias[None]
        return result.reshape_as(x).to(x.dtype)

def calibrated_forward(self,global_feature,maps):
    from occreid.bpbreid_model import reference_pool
    spatial=self.msf(maps) if self.use_msf else maps[-1]
    logits=self.pixel_classifier(spatial);probabilities=logits.float().softmax(1);masks=probabilities[:,1:]
    foreground,parts=reference_pool(spatial,masks)
    mass=masks.amax(1).mean((-1,-2));foreground=foreground/mass[:,None].clamp_min(1e-6)
    bn=self.part_reduce.layers[1]
    if isinstance(bn,PartConditionalBN):bn.valid_hint=getattr(self,'_valid_hint',None)
    global_embedding=self.global_reduce(spatial.mean((-1,-2)))
    foreground=self.foreground_reduce(foreground);parts=self.part_reduce(parts)
    global_bn,global_logits=self.global_identity_classifier(global_embedding)
    foreground_bn,foreground_logits=self.foreground_identity_classifier(foreground)
    _,concat_logits=self.concat_parts_identity_classifier(parts.flatten(1))
    winner=probabilities.argmax(1);visibility=torch.stack([(winner==k).any(-1).any(-1) for k in range(1,9)],1).float()
    return dict(global_eval=global_bn,foreground=foreground,foreground_eval=foreground_bn,
                parts=parts,visibility=visibility,anatomical_parts=parts,anatomical_visibility=visibility,
                global_logits=global_logits,foreground_logits=foreground_logits,concat_logits=concat_logits,
                segmentation=logits,spatial=spatial,**{'global':global_embedding})

def install_head(model,stats,feature='none'):
    head=model.head;head.context_residual=None
    with torch.no_grad():
        for bn,mean,var in [(head.foreground_reduce.layers[1],stats['mean1'],stats['var1']),
                            (head.foreground_identity_classifier.bn,stats['mean2'],stats['var2'])]:
            bn.running_mean.copy_(mean);bn.running_var.copy_(var)
    if feature!='none':
        options={'part_bn':(8,False),'visible_shared_bn':(1,True),'visible_part_bn':(8,True)}
        groups,visible=options[feature];old=head.part_reduce.layers[1]
        with torch.random.fork_rng(devices=[]):head.part_reduce.layers[1]=PartConditionalBN(old,groups,visible)
    head.forward=types.MethodType(calibrated_forward,head)
    return model

def validity(labels,confidence,min_pixels=32):
    return torch.stack([((labels==k)&(confidence>.2)).sum((-1,-2))>=min_pixels for k in range(1,9)],1)

def part_targeted_views(batch,cfg,epoch,step):
    """Match erase probability/height/width and global RNG draws; change only rectangle location."""
    from occreid.vei_augment import _counts
    x,y,c=batch['image'],batch['label'],batch['confidence'];images=x.clone();labels=y.clone();conf=c.clone();b,_,h,w=x.shape
    local=torch.Generator().manual_seed(cfg['seed']*10000019+epoch*100003+step)
    # One host copy avoids hundreds of scalar GPU synchronizations while selecting training labels.
    host_y=y.detach().cpu();host_c=c.detach().cpu()
    erased=0;targets=[0]*8;fallback=0
    for i in range(b):
        if torch.rand(())>=cfg['single_erase_probability']:continue
        rh=max(1,int(h*float(torch.empty(()).uniform_(.15,.65))));rw=max(1,int(w*float(torch.empty(()).uniform_(.15,.65))))
        top,left=int(torch.randint(h-rh+1,())),int(torch.randint(w-rw+1,()))
        present=[k for k in range(1,9) if int(((host_y[i]==k)&(host_c[i]>.2)).sum())>=cfg['supervision_min_pixels']]
        if present:
            k=present[int(torch.randint(len(present),(),generator=local))]
            points=torch.nonzero((host_y[i]==k)&(host_c[i]>.2));point=points[int(torch.randint(len(points),(),generator=local))]
            top=max(0,min(h-rh,int(point[0])-int(torch.randint(rh,(),generator=local))))
            left=max(0,min(w-rw,int(point[1])-int(torch.randint(rw,(),generator=local))));targets[k-1]+=1
        else:fallback+=1
        images[i,:,top:top+rh,left:left+rw]=.75;labels[i,top:top+rh,left:left+rw]=0;conf[i,top:top+rh,left:left+rw]=1
        labels[i][y[i]==255]=255;conf[i][y[i]==255]=0;erased+=rh*rw
    pair=dict(batch,label=labels,confidence=conf,retained=(_counts(labels,conf)>0).float(),intervention=False,views=1)
    return images,pair,dict(view_count=1,view_mode=5,occluded_fraction=erased/y.numel(),crop_fallback_fraction=0.,
                           targeted_part_counts=targets,targeting_fallback=fallback)

def deployed_pair_loss(output,batch,temperature=.1):
    # All current in-batch valid positives/negatives; not a claim of full-gallery optimization.
    from core import joint_distance
    features={k:output[k] for k in ['parts','foreground_eval','visibility']}
    d=joint_distance(features,features);pids=batch['pid'].repeat(batch['views']);source=batch['source'].repeat(batch['views'])
    pos=(pids[:,None]==pids[None,:])&(source[:,None]!=source[None,:]);neg=pids[:,None]!=pids[None,:]
    valid=pos[:,:,None]&neg[:,None,:]
    differences=(d[:,:,None]-d[:,None,:])/temperature
    loss=F.softplus(differences[valid]).mean() if valid.any() else output['parts'].sum()*0
    return loss,dict(deployed_valid_triplets=int(valid.sum()),deployed_mean_pair_violation=float((differences[valid]>0).float().mean()) if valid.any() else 0.)
