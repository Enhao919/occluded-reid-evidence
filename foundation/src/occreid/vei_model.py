"""Single SOLIDER trunk with global/foreground/multi-level body-part branches."""
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import swin_base_patch4_window7_224


def _neck(dim):
    layer=nn.BatchNorm1d(dim);layer.bias.requires_grad_(False)
    return layer


class VEIHead(nn.Module):
    def __init__(self,num_classes,dim=256,visibility_threshold=.45,learnable_scales=True,min_visibility_pixels=2,part_policy='anatomical8',identity_masks_detached=False):
        super().__init__()
        self.threshold=visibility_threshold;self.min_visibility_pixels=min_visibility_pixels;self.part_policy=part_policy
        self.identity_masks_detached=identity_masks_detached
        from .vei_parts import metric_groups
        metric_groups(part_policy)
        self.projections=nn.ModuleList([nn.Sequential(nn.Conv2d(c,dim,1,bias=False),nn.GroupNorm(8,dim),nn.GELU())
                                       for c in (256,512,1024)])
        self.fusion=nn.Sequential(nn.Conv2d(dim,dim,3,padding=1,bias=False),nn.GroupNorm(8,dim),nn.GELU())
        self.segmentation=nn.Conv2d(dim,9,1)
        self.scale_norms=nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.scale_logits=nn.Parameter(torch.zeros(8,3),requires_grad=learnable_scales)
        self.global_neck=_neck(1024);self.foreground_neck=_neck(dim);self.concat_neck=_neck(8*dim)
        self.global_classifier=nn.Linear(1024,num_classes,bias=False)
        self.foreground_classifier=nn.Linear(dim,num_classes,bias=False)
        self.concat_classifier=nn.Linear(8*dim,num_classes,bias=False)
        for layer in (self.global_classifier,self.foreground_classifier,self.concat_classifier):nn.init.normal_(layer.weight,std=.001)

    def forward(self,global_feature,maps):
        size=maps[1].shape[-2:]
        levels=[F.interpolate(p(m),size=size,mode='bilinear',align_corners=False) for p,m in zip(self.projections,maps[1:])]
        spatial=self.fusion(sum(levels)/3)
        logits=self.segmentation(spatial)
        probabilities=logits.float().softmax(1);masks=probabilities[:,1:]
        # Parsing keeps explicit supervision. Identity/metric objectives may learn
        # feature values without steering the semantic pooling masks themselves.
        masks=masks.detach() if self.identity_masks_detached else masks
        den=masks.sum((-1,-2)).clamp_min(1e-6)[...,None]
        descriptors=[norm(torch.einsum('bkhw,bdhw->bkd',masks,x.float())/den) for x,norm in zip(levels,self.scale_norms)]
        scale_weights=self.scale_logits.float().softmax(-1)
        parts=(torch.stack(descriptors,2)*scale_weights[None,:,:,None]).sum(2)
        foreground_mask=masks.amax(1,keepdim=True)
        foreground=(spatial.float()*foreground_mask).sum((-1,-2))/foreground_mask.sum((-1,-2)).clamp_min(1e-6)
        global_bn=self.global_neck(global_feature)
        foreground_bn=self.foreground_neck(foreground)
        concat_bn=self.concat_neck(parts.flatten(1))
        winner=probabilities.argmax(1)
        support=torch.stack([(winner==k)&(probabilities[:,k]>self.threshold) for k in range(1,9)],1)
        visibility=(support.sum((-1,-2))>=self.min_visibility_pixels).float()
        anatomical_parts,anatomical_visibility=parts,visibility
        if self.part_policy=='bilateral5':
            from .vei_parts import BILATERAL
            group_masks=torch.stack([masks[:,list(g)].sum(1) for g in BILATERAL],1)
            group_den=group_masks.sum((-1,-2)).clamp_min(1e-6)[...,None]
            descriptors=[norm(torch.einsum('bkhw,bdhw->bkd',group_masks,x.float())/group_den)
                         for x,norm in zip(levels,self.scale_norms)]
            group_weights=torch.stack([self.scale_logits[list(g)].mean(0) for g in BILATERAL]).softmax(-1)
            parts=(torch.stack(descriptors,2)*group_weights[None,:,:,None]).sum(2)
            group_probs=torch.cat([probabilities[:,:1],group_masks],1)
            winner=group_probs.argmax(1)
            support=torch.stack([(winner==k)&(group_probs[:,k]>self.threshold) for k in range(1,6)],1)
            visibility=(support.sum((-1,-2))>=self.min_visibility_pixels).float()
        return {'global':global_feature,'foreground':foreground,'parts':parts,'visibility':visibility,
                'anatomical_parts':anatomical_parts,'anatomical_visibility':anatomical_visibility,
                'global_eval':global_bn,'foreground_eval':foreground_bn,
                'global_logits':self.global_classifier(global_bn),'foreground_logits':self.foreground_classifier(foreground_bn),
                'concat_logits':self.concat_classifier(concat_bn),'segmentation':logits,'spatial':spatial,'scale_weights':scale_weights}


class VEIReID(nn.Module):
    def __init__(self,num_classes,config,pretrained=None):
        super().__init__()
        self.backbone=swin_base_patch4_window7_224(img_size=config['image_size'],semantic_weight=config['semantic_weight'],
            convert_weights=False,drop_path_rate=config['drop_path'],with_cp=config['checkpointing'])
        if config.get('head_architecture','vei')=='bpbreid':
            from .bpbreid_model import BPBreIDHead
            self.head=BPBreIDHead(num_classes,config['feature_dim'],config.get('bpbreid_msf',True),config.get('context_repair'))
        else:
            self.head=VEIHead(num_classes,config['feature_dim'],config['visibility_threshold'],
                              config.get('learnable_scales',True),config.get('visibility_min_tokens',2),config.get('part_policy','anatomical8'),
                              config.get('identity_masks_detached',False))
        self.pretrained_report=None
        if pretrained:
            self.backbone.init_weights(str(pretrained))
            self.pretrained_report={'path':str(pretrained),'strict':True,'tensors':len(self.backbone.state_dict())}

    def forward(self,images):
        feature,maps=self.backbone((images-.5)/.5)
        return self.head(feature,maps)
