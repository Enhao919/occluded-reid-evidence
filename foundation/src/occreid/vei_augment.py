"""Synchronized PADE-style views and fixed-evidence occluder interventions."""
import torch
from torch.nn import functional as F


def _counts(y,c):
    return torch.stack([((y==k)&(c>.2)).sum((-1,-2)) for k in range(1,9)],-1)


def _rect(h,w,epoch,ramp=30):
    area=float(torch.empty(()).uniform_(.06,.12+.28*min(1.,max(0.,epoch/max(1,ramp)))))
    mode=int(torch.randint(4,()))
    if mode<2:
        rh,rw=h,max(1,int(w*area));top,left=0,0 if mode==0 else w-rw
    elif mode==2:rh,rw=max(1,int(h*area)),w;top,left=h-rh,0
    else:
        rh,rw=max(1,int(h*area**.5)),max(1,int(w*area**.5))
        top,left=int(torch.randint(h-rh+1,())),int(torch.randint(w-rw+1,()))
    return top,left,rh,rw


def _erase_pair(batch,epoch,cfg,intervention):
    x,y,c,pid=batch['image'],batch['label'],batch['confidence'],batch['pid']
    b,_,h,w=x.shape;a,z=x.clone(),x.clone();ya,yz=y.clone(),y.clone();ca,cz=c.clone(),c.clone()
    masks=torch.zeros((b,h,w),device=x.device,dtype=torch.bool)
    donors=torch.arange(b,device=x.device).repeat(2,1)
    base=_counts(y,c);remaining=base.clone()
    for i in range(b):
        foreground=(y[i]>0)&(y[i]<9)&(c[i]>.2)
        if not foreground.any():continue
        candidates=torch.where(pid!=pid[i])[0]
        if intervention and not len(candidates):continue
        chosen=None
        for _ in range(12):
            top,left,rh,rw=_rect(h,w,epoch,cfg.get('occlusion_ramp_epochs',30));region=(slice(top,top+rh),slice(left,left+rw))
            if foreground[region].sum()<=.5*foreground.sum():chosen=(top,left,rh,rw);break
        if chosen is None:continue
        top,left,rh,rw=chosen;region=(slice(top,top+rh),slice(left,left+rw));masks[i,region[0],region[1]]=True
        if intervention:
            d1=candidates[int(torch.randint(len(candidates),()))]
            second=candidates[pid[candidates]!=pid[d1]]
            d2=second[int(torch.randint(len(second),()))] if len(second) else d1
            donors[0,i],donors[1,i]=d1,d2
            a[i,:,region[0],region[1]]=x[d1,:,region[0],region[1]]
            z[i,:,region[0],region[1]]=x[d2,:,region[0],region[1]]
        else:a[i,:,region[0],region[1]]=torch.rand((3,rh,rw),device=x.device,dtype=x.dtype)
        for label,conf in ((ya,ca),(yz,cz)):
            label[i,region[0],region[1]]=0;conf[i,region[0],region[1]]=1
            # Ignore the inner 1px contour; no target/donor identity at blended edges.
            for edge in ((slice(top,top+1),region[1]),(slice(top+rh-1,top+rh),region[1]),
                         (region[0],slice(left,left+1)),(region[0],slice(left+rw-1,left+rw))):
                label[i,edge[0],edge[1]]=255;conf[i,edge[0],edge[1]]=0
            unknown=y[i]==255
            label[i,unknown]=255;conf[i,unknown]=0
        remaining[i]=_counts(ya[i:i+1],ca[i:i+1])[0]
    retention=remaining/base.clamp_min(1)
    return a,z,ya,yz,ca,cz,retention,masks,donors


def _crop(batch):
    x,y,c=batch['image'],batch['label'],batch['confidence'];b,_,h,w=x.shape
    result=x.clone();labels=y.clone();confidence=c.clone();base=_counts(y,c)
    ratios=(base>0).float();fallback=0
    for i in range(b):
        fg=(y[i]>0)&(y[i]<9)&(c[i]>.2)
        chosen=None
        if fg.any():
            for _ in range(12):
                rh=max(1,int(h*float(torch.empty(()).uniform_(.55,1.))))
                rw=max(1,int(w*float(torch.empty(()).uniform_(.7,1.))))
                top,left=int(torch.randint(h-rh+1,())),int(torch.randint(w-rw+1,()))
                if fg[top:top+rh,left:left+rw].sum()>=.5*fg.sum():chosen=(top,left,rh,rw);break
        if chosen is None:fallback+=1;continue
        top,left,rh,rw=chosen;rs=(slice(top,top+rh),slice(left,left+rw))
        crop_y=y[i:i+1,rs[0],rs[1]];crop_c=c[i:i+1,rs[0],rs[1]]
        ratios[i]=_counts(crop_y,crop_c)[0]/base[i].clamp_min(1)
        result[i]=F.interpolate(x[i:i+1,:,rs[0],rs[1]],size=(h,w),mode='bicubic',align_corners=False).clamp(0,1)[0]
        labels[i]=F.interpolate(crop_y[:,None].float(),size=(h,w),mode='nearest')[0,0].long()
        confidence[i]=F.interpolate(crop_c[:,None],size=(h,w),mode='bilinear',align_corners=False)[0,0]
    return result,labels,confidence,ratios,fallback


def make_vei_views(batch,cfg,epoch,force_mode=None):
    from .evidence import enabled,build_views
    if enabled(cfg,epoch):
        return build_views(batch,cfg,epoch,force_mode)
    x,y,c=batch['image'],batch['label'],batch['confidence'];b=len(x)
    layout=cfg.get('view_layout','budget2_intervention')
    if layout=='single' and force_mode is None:
        images,labels,confidence=x.clone(),y.clone(),c.clone()
        erased=0
        for i in range(b):
            if torch.rand(())>=cfg.get('single_erase_probability',0.):continue
            h,w=y.shape[-2:]
            rh=max(1,int(h*float(torch.empty(()).uniform_(.15,.65))))
            rw=max(1,int(w*float(torch.empty(()).uniform_(.15,.65))))
            top,left=int(torch.randint(h-rh+1,())),int(torch.randint(w-rw+1,()))
            # Author CoarseDropout follows .5/.5 normalization and fills .5,
            # equivalent to .75 in this loader's unnormalized [0,1] image.
            images[i,:,top:top+rh,left:left+rw]=.75
            labels[i,top:top+rh,left:left+rw]=0
            confidence[i,top:top+rh,left:left+rw]=1
            confidence[i][y[i]==255]=0;labels[i][y[i]==255]=255
            erased+=rh*rw
        pair=dict(batch,label=labels,confidence=confidence,retained=(_counts(labels,confidence)>0).float(),intervention=False,views=1)
        return images,pair,{'view_count':1,'view_mode':5,'occluded_fraction':erased/y.numel(),'crop_fallback_fraction':0.}
    if force_mode is not None:mode=force_mode
    elif layout=='ba2':mode='ba'
    elif layout=='pam3_reference':mode='pam3'
    elif layout=='pam2':mode='erase' if torch.rand(())<.5 else 'crop'
    elif layout=='budget2_intervention':
        draw=float(torch.rand(()));mode='erase' if draw<.25 else ('crop' if draw<.75 else 'intervention')
    else:raise ValueError(f'Unknown view_layout: {layout}')
    base=(_counts(y,c)>0).float();images=[x,x];labels=[y,y];confs=[c,c];retained=[base,base]
    mask=torch.zeros_like(y,dtype=torch.bool);donors=torch.arange(b,device=x.device).repeat(2,1);fallback=0
    if mode in ('erase','intervention','pam3'):
        a,z,ya,yz,ca,cz,r,mask,donors=_erase_pair(batch,epoch,cfg,mode=='intervention')
        if mode=='intervention':images=[a,z];labels=[ya,yz];confs=[ca,cz];retained=[r,r]
        else:images=[x,a];labels=[y,ya];confs=[c,ca];retained=[base,r]
    if mode in ('crop','pam3'):
        crop,cy,cc,cr,fallback=_crop(batch)
        if mode=='crop':images=[x,crop];labels=[y,cy];confs=[c,cc];retained=[base,cr]
        else:images.append(crop);labels.append(cy);confs.append(cc);retained.append(cr)
    if mode not in ('ba','erase','crop','intervention','pam3'):raise ValueError(f'Unknown view mode {mode}')
    pair=dict(batch,label=torch.cat(labels),confidence=torch.cat(confs),retained=torch.cat(retained),
              intervention=mode=='intervention',intervention_mask=mask,donor_indices=donors,views=len(images))
    stats={'view_count':len(images),'view_mode':('ba','erase','crop','intervention','pam3').index(mode),
           'occluded_fraction':float(mask.float().mean()),'crop_fallback_fraction':fallback/max(1,b)}
    return torch.cat(images),pair,stats
