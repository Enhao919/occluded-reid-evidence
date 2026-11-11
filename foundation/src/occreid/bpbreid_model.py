"""SOLIDER appearance backbone with the author's BPBreID/KPR part-head layers."""
import torch
from torch import nn
from torch.nn import functional as F

from .third_party.bpbreid_layers import (MultiStageFusion, AfterPoolingDimReduceLayer,
                                         PixelToPartClassifier, BNClassifier)


def reference_pool(spatial, masks):
    """Same GAP foreground / GWAP parts as BPBreID, without a [B,K,C,H,W] allocation."""
    with torch.autocast(device_type=spatial.device.type, enabled=False):
        x = spatial if spatial.dtype == torch.float64 else spatial.float()
        m = masks.to(x)
        parts = torch.einsum('bkhw,bchw->bkc', m, x) / m.sum((-1, -2)).clamp_min(1e-6)[..., None]
        foreground = torch.einsum('bhw,bchw->bc', m.amax(1), x) / (x.shape[-1] * x.shape[-2])
    return foreground, parts


class BPBreIDHead(nn.Module):
    def __init__(self, num_classes, dim=512, use_msf=True, context_repair=None):
        super().__init__()
        self.use_msf = use_msf
        self.pixel_classifier = PixelToPartClassifier(1024, 8)
        self.global_reduce = AfterPoolingDimReduceLayer(1024, dim)
        self.foreground_reduce = AfterPoolingDimReduceLayer(1024, dim)
        self.part_reduce = AfterPoolingDimReduceLayer(1024, dim)
        self.global_identity_classifier = BNClassifier(dim, num_classes)
        self.foreground_identity_classifier = BNClassifier(dim, num_classes)
        self.concat_parts_identity_classifier = BNClassifier(8 * dim, num_classes)
        # Optional MSF is initialized after shared heads: matched-seed ablations
        # retain identical classifier/parser/reduction initialization.
        self.msf = MultiStageFusion(1920, 1024) if use_msf else nn.Identity()
        self.context_residual = None
        if (context_repair or {}).get('enabled', False):
            from .context_residual import ContextResidual
            self.context_residual = ContextResidual(1024, context_repair.get('hidden',64),
                context_repair.get('max_relative_scale',.25))

    @property
    def segmentation(self):
        return self.pixel_classifier.classifier

    @property
    def global_classifier(self):
        return self.global_identity_classifier.classifier

    @property
    def concat_classifier(self):
        return self.concat_parts_identity_classifier.classifier

    def forward(self, global_feature, maps):
        spatial = self.msf(maps) if self.use_msf else maps[-1]
        logits = self.pixel_classifier(spatial)
        probabilities = logits.float().softmax(1)
        masks = probabilities[:, 1:]
        if self.context_residual is not None:
            spatial = self.context_residual(spatial, probabilities)
        foreground, parts = reference_pool(spatial, masks)
        global_embedding = self.global_reduce(spatial.mean((-1, -2)))
        foreground = self.foreground_reduce(foreground)
        parts = self.part_reduce(parts)
        global_bn, global_logits = self.global_identity_classifier(global_embedding)
        foreground_bn, foreground_logits = self.foreground_identity_classifier(foreground)
        _, concat_logits = self.concat_parts_identity_classifier(parts.flatten(1))
        winner = probabilities.argmax(1)
        visibility = torch.stack([(winner == k).any(-1).any(-1) for k in range(1, 9)], 1).float()
        return {'global': global_embedding, 'global_eval': global_bn,
                'foreground': foreground, 'foreground_eval': foreground_bn,
                'parts': parts, 'visibility': visibility,
                'anatomical_parts': parts, 'anatomical_visibility': visibility,
                'global_logits': global_logits, 'foreground_logits': foreground_logits,
                'concat_logits': concat_logits, 'segmentation': logits, 'spatial': spatial}
