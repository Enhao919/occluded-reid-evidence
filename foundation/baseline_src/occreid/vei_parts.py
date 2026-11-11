"""Bilateral identity groups; original anatomical labels remain intact."""
import torch
from torch.nn import functional as F

BILATERAL=((0,),(1,2),(3,),(4,5),(6,7))
COARSE=((0,),(1,),(2,3),(4,),(5,6),(7,8))


def metric_groups(policy):
    if policy=='anatomical8':return tuple((i,) for i in range(8))
    if policy=='bilateral5':return BILATERAL
    raise ValueError(f'Unknown part policy {policy}')


def merge_visibility(visibility,policy):
    if visibility.shape[-1]!=8:raise ValueError('Teacher visibility must have eight anatomical parts')
    if policy=='anatomical8':return visibility
    return torch.stack([visibility[...,list(g)].amax(-1) for g in metric_groups(policy)],-1)


def coarse_logits(logits):
    if logits.shape[1]!=9:raise ValueError('Expected nine anatomical logits')
    x=logits if logits.dtype==torch.float64 else logits.float()
    return torch.stack([torch.logsumexp(x[:,list(g)],dim=1) for g in COARSE],1)


def coarse_labels(labels):
    if ((labels>8)&(labels!=255)).any() or (labels<0).any():raise ValueError('Invalid anatomical label')
    table=torch.full((256,),255,dtype=torch.long,device=labels.device)
    table[:9]=torch.tensor([0,1,2,2,3,4,4,5,5],device=labels.device)
    return table[labels.long()]


def balanced_class_weights(labels,confidence,classes):
    valid=(labels!=255)&(confidence>0)
    counts=torch.bincount(labels[valid].flatten(),weights=confidence[valid].float().flatten(),minlength=classes)
    present=counts[1:]>0
    average=counts[1:][present].mean() if present.any() else counts.new_tensor(1.)
    weights=(average/counts.clamp_min(1e-12)).sqrt().clamp(.5,2.5).detach()
    weights[0]=.25
    return weights


def hierarchical_segmentation_loss(logits,labels,confidence):
    from .losses import weighted_mean
    labels=F.interpolate(labels[:,None].float(),size=logits.shape[-2:],mode='nearest')[:,0].long()
    confidence=F.interpolate(confidence[:,None].float(),size=logits.shape[-2:],mode='bilinear',align_corners=False)[:,0]
    valid=(labels!=255)&(confidence>0)
    targets=(labels,coarse_labels(labels));predictions=(logits.float(),coarse_logits(logits))
    terms=[];max_weight=0.
    for z,y in zip(predictions,targets):
        weights=balanced_class_weights(y,confidence*valid,z.shape[1])
        pixel_weight=confidence*valid*weights[y.clamp(0,z.shape[1]-1)]
        ce=F.cross_entropy(z,y,ignore_index=255,reduction='none')
        terms.append(weighted_mean(ce,pixel_weight));max_weight=max(max_weight,float(weights.max()))
    return (terms[0]+terms[1])*.5,{'valid_pixels':int(valid.sum()),
        'foreground_pixels':int(((labels>0)&(labels<9)&valid).sum()),
        'segmentation_fine':float(terms[0].detach()),'segmentation_coarse':float(terms[1].detach()),
        'segmentation_max_class_weight':max_weight}
