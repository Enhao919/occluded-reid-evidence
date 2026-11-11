"""Training-only background interventions with an independent CPU generator."""
import math
import torch
from torch.nn import functional as F


def background_intervention(images,labels,confidence,teacher_probabilities,pids,generator,
                            border=8,min_pixels=64):
    b,_,h,w=images.shape
    if generator.device.type!='cpu':raise ValueError('Donor generator must be CPU')
    probs=F.interpolate(teacher_probabilities.detach().float(),size=(h,w),mode='bilinear',align_corners=False)
    protected=(labels!=0)|(probs.argmax(1)>0)
    protected=F.max_pool2d(protected[:,None].float(),2*border+1,1,border)[:,0]>0
    allowed=(~protected)&(labels==0)&(confidence>.2)&(probs[:,0]>=.5)
    changed=images.clone();mask=torch.zeros_like(labels,dtype=torch.bool)
    donors=torch.full((b,),-1,dtype=torch.long,device=images.device)
    ids=pids.detach().cpu()
    def draw():return float(torch.rand((),generator=generator))
    def randint(n):return int(torch.randint(n,(),generator=generator))
    for i in range(b):
        candidates=torch.where(ids!=ids[i])[0]
        if not len(candidates) or int(allowed[i].sum())<min_pixels:continue
        for _ in range(12):
            area=h*w*(.06+.14*draw());ratio=.5+1.5*draw()
            rh=min(h,max(1,round(math.sqrt(area/ratio))))
            rw=min(w,max(1,round(math.sqrt(area*ratio))))
            top,left=randint(h-rh+1),randint(w-rw+1)
            region=torch.zeros((h,w),dtype=torch.bool,device=images.device)
            region[top:top+rh,left:left+rw]=True;region &= allowed[i]
            if int(region.sum())<min_pixels:continue
            donor=int(candidates[randint(len(candidates))]);donors[i]=donor;mask[i]=region
            changed[i]=torch.where(region[None],images[donor],images[i])
            break
    return changed,mask,donors
