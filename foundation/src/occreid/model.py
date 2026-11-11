import torch
from torch import nn
from torch.nn import functional as F
from .backbone import swin_base_patch4_window7_224


class MultiScaleHead(nn.Module):
    def __init__(self,num_classes,dim=256,visibility_threshold=.45):
        super().__init__()
        self.threshold=visibility_threshold
        self.projections=nn.ModuleList([
            nn.Sequential(nn.Conv2d(c,dim,1,bias=False),nn.GroupNorm(8,dim),nn.GELU())
            for c in (256,512,1024)])
        self.fusion=nn.Sequential(nn.Conv2d(dim,dim,3,padding=1,bias=False),
                                  nn.GroupNorm(8,dim),nn.GELU())
        self.segmentation=nn.Conv2d(dim,7,1)
        self.global_neck=nn.BatchNorm1d(1024)
        self.global_neck.bias.requires_grad_(False)
        self.global_classifier=nn.Linear(1024,num_classes,bias=False)
        self.local_neck=nn.LayerNorm(dim)
        self.foreground_classifier=nn.Linear(dim,num_classes,bias=False)
        self.part_classifier=nn.Linear(dim,num_classes,bias=False)
        for layer in (self.global_classifier,self.foreground_classifier,self.part_classifier):
            nn.init.normal_(layer.weight,std=.001)

    def forward(self,global_feature,maps):
        size=maps[1].shape[-2:]
        spatial=sum(F.interpolate(p(m),size=size,mode='bilinear',align_corners=False)
                    for p,m in zip(self.projections,maps[1:]))/3
        spatial=self.fusion(spatial)
        logits=self.segmentation(spatial)
        # FP32 masks prevent tiny pooling denominators under mixed precision.
        probs=logits.float().softmax(1)
        part_masks=probs[:,1:].square()
        parts=torch.einsum('bkhw,bdhw->bkd',part_masks,spatial.float())
        parts=parts/part_masks.sum((-1,-2)).clamp_min(1e-6)[...,None]
        fg_mask=1-probs[:,0:1]
        foreground=(spatial.float()*fg_mask).sum((-1,-2))/fg_mask.sum((-1,-2)).clamp_min(1e-6)
        winner=probs.argmax(1)
        support=torch.stack([(winner==k)&(probs[:,k]>self.threshold) for k in range(1,7)],1)
        # Channel softmax, not spatial softmax: a missing class can have no support.
        visibility=(probs[:,1:]*support).flatten(2).sum(-1)
        visibility=(visibility/(max(1,spatial.shape[-2]*spatial.shape[-1]*.02))).clamp(0,1)
        return {'global':global_feature,'foreground':foreground,'parts':parts,
                'global_logits':self.global_classifier(self.global_neck(global_feature)),
                'foreground_logits':self.foreground_classifier(self.local_neck(foreground)),
                'part_logits':self.part_classifier(self.local_neck(parts)),
                'segmentation':logits,'visibility':visibility,'spatial':spatial}


class OcclusionReID(nn.Module):
    def __init__(self,num_classes,config,pretrained=None):
        super().__init__()
        self.backbone=swin_base_patch4_window7_224(
            img_size=config['image_size'],semantic_weight=config['semantic_weight'],
            convert_weights=False,drop_path_rate=config['drop_path'],
            with_cp=config['checkpointing'])
        self.head=MultiScaleHead(num_classes,config['feature_dim'],config['visibility_threshold'])
        self.pretrained_report=None
        if pretrained:
            self.backbone.init_weights(str(pretrained))
            self.pretrained_report={'path':str(pretrained),'strict':True,
                                    'tensors':len(self.backbone.state_dict())}

    def forward(self,images):
        # Match the original SOLIDER downstream normalization exactly.
        global_feature,maps=self.backbone((images-.5)/.5)
        return self.head(global_feature,maps)


def build_model(config,pretrained=None):
    if config.get('method','pso')=='vei':
        from .vei_model import VEIReID
        return VEIReID(config['num_classes'],config,pretrained)
    return OcclusionReID(config['num_classes'],config,pretrained)
