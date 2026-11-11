import torch


def paired_augment(images, labels, confidence, pids, progress, probability=.6):
    """Training donors only; same occluder shared across groups of different IDs.

    Inputs are [0,1] RGB. Original tensors remain intact. Labels of pasted pixels
    describe absence of the TARGET, regardless of the donor's semantic class.
    """
    strong, y, c = images.clone(), labels.clone(), confidence.clone()
    b,_,h,w = images.shape
    max_area = .12 + .28*max(0.,min(1.,progress))
    pasted = 0
    pattern = None
    for i in range(b):
        if torch.rand((),device=images.device).item() >= probability:
            continue
        candidates = torch.where(pids != pids[i])[0]
        if not len(candidates):
            continue
        if pattern is None or i%2 == 0:
            mode = int(torch.randint(4,()).item())
            area = float(torch.empty(()).uniform_(.06,max_area).item())
            if mode < 2:
                rh,rw = h,max(1,int(w*area));top=0;left=0 if mode==0 else w-rw
            elif mode == 2:
                rh,rw = max(1,int(h*area)),w;top=h-rh;left=0
            else:
                rh=max(1,int(h*area**.5));rw=max(1,int(w*area**.5))
                top=int(torch.randint(h-rh+1,()).item());left=int(torch.randint(w-rw+1,()).item())
            donor = int(candidates[torch.randint(len(candidates),()).item()].item())
            pattern = (top,left,rh,rw,donor)
        top,left,rh,rw,donor = pattern
        if pids[donor] == pids[i]:
            donor = int(candidates[0].item())
        region = (slice(top,top+rh),slice(left,left+rw))
        fg = (labels[i]>0)&(labels[i]<7)&(confidence[i]>.2)
        if fg.any() and fg[region].sum() > .6*fg.sum():
            continue
        strong[i,:,region[0],region[1]] = images[donor,:,region[0],region[1]]
        y[i,region[0],region[1]] = 0
        c[i,region[0],region[1]] = 1.
        pasted += rh*rw
    return strong,y,c,{'occluded_fraction':pasted/(b*h*w)}
