"""Opt-in, identity-initialized correction of the reference ReID spatial values.

This is an experimental module, not a certified target-person segmenter.
The original parsing probabilities continue to serve the reference head.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ContextResidual(nn.Module):
    def __init__(self, channels=1024, hidden=64, max_relative_scale=.25):
        super().__init__()
        if hidden < 8 or hidden % 8 or not 0 < max_relative_scale <= 1:
            raise ValueError('hidden must be divisible by8; relative scale in (0,1]')
        self.max_relative_scale = float(max_relative_scale)
        # Adding capacity must not move the original augmentation/DropPath RNG.
        with torch.random.fork_rng(devices=[]):
            self.project = nn.Conv2d(2*channels, hidden, 1)
            self.norm = nn.GroupNorm(8, hidden)
            self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
            self.output = nn.Conv2d(hidden, channels, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, spatial, probabilities):
        if probabilities.shape[1] != 9 or probabilities.shape[-2:] != spatial.shape[-2:]:
            raise ValueError('Expected matching background + eight-part probabilities')
        with torch.autocast(device_type=spatial.device.type, enabled=False):
            x = spatial if spatial.dtype == torch.float64 else spatial.float()
            p = probabilities.detach().to(x)
            masks = p[:, 1:]
            prototypes = torch.einsum('bkhw,bchw->bkc', masks, x)
            prototypes = prototypes / masks.sum((-1,-2)).clamp_min(1e-6)[...,None]
            conditional = masks / masks.sum(1,keepdim=True).clamp_min(1e-6)
            context = torch.einsum('bkhw,bkc->bchw', conditional, prototypes)
            hidden = F.gelu(self.norm(self.project(torch.cat((x,context-x),1))))
            hidden = F.gelu(self.local(hidden))
            rms = x.square().mean(1,keepdim=True).clamp_min(1e-6).sqrt().detach()
            delta = (1-p[:,:1])*self.max_relative_scale*rms*self.output(hidden).tanh()
            return (x+delta).to(spatial.dtype)
