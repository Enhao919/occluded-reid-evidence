"""Target-conditioned pre-pooling allocation with an explicit uncertain state.

The three states describe cross-camera support, not ground-truth pixel ownership.
The original foreground/global paths and the native retrieval function are retained.
"""

import torch
from torch import nn
from torch.nn import functional as F


def pool_parts(spatial, masks):
    return (
        torch.einsum("bkhw,bchw->bkc", masks.float(), spatial.float())
        / masks.float().sum((-1, -2)).clamp_min(1e-6)[..., None]
    )


class TargetAllocation(nn.Module):
    def __init__(
        self, channels=1024, hidden=32, max_log_gain=0.4, max_visibility_drop=0.25
    ):
        super().__init__()
        self.max_log_gain = max_log_gain
        self.max_visibility_drop = max_visibility_drop
        self.stem = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GroupNorm(8, hidden), nn.GELU()
        )
        self.state_score = nn.Conv2d(hidden * 3 + 3, 3, 1)
        nn.init.zeros_(self.state_score.weight)
        nn.init.zeros_(self.state_score.bias)

    def forward(self, spatial, masks):
        z = self.stem(spatial.float())
        b, c, h, w = z.shape
        k = masks.shape[1]
        parts = pool_parts(z, masks)
        whole = pool_parts(z, masks.amax(1, keepdim=True))[:, 0]
        local = z[:, None].expand(-1, k, -1, -1, -1)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=z.device),
            torch.linspace(-1, 1, w, device=z.device),
            indexing="ij",
        )
        coordinates = torch.stack((xx, yy))[None, None].expand(b, k, -1, -1, -1)
        features = torch.cat(
            (
                local,
                local * whole[:, None, :, None, None],
                local * parts[..., None, None],
                coordinates,
                masks[:, :, None],
            ),
            2,
        )
        return self.state_score(features.flatten(0, 1)).reshape(b, k, 3, h, w)

    def allocate(self, masks, visibility, logits):
        probability = logits.float().softmax(2)
        target, other = probability[:, :, 0], probability[:, :, 1]
        gain = self.max_log_gain * (target - other)
        adjusted = masks.float() * gain.exp()
        other_support = ((other - target) * masks).sum((-1, -2)) / masks.sum(
            (-1, -2)
        ).clamp_min(1e-6)
        visible = visibility.float() * (
            1 - self.max_visibility_drop * other_support.relu()
        )
        return adjusted, visible, gain


def balanced_point_loss(logits, targets):
    sampled, labels = [], []
    for row in targets:
        grid = row["grid"].to(logits.device).float()[None, None]
        value = F.grid_sample(
            logits[row["image"]].float(),
            grid.expand(logits.shape[1], -1, -1, -1),
            align_corners=False,
        )[:, :, 0].permute(2, 0, 1)
        part = row["part"].to(logits.device).long() - 1
        state = row["state"].to(logits.device).long()
        sampled.append(value[torch.arange(len(part), device=logits.device), part])
        labels.append(state)
    if not labels:
        return logits.sum() * 0, [0, 0, 0]
    labels, sampled = torch.cat(labels), torch.cat(sampled)
    loss = F.cross_entropy(sampled, labels, reduction="none")
    counts = [(labels == i).sum() for i in range(3)]
    selected = [loss[labels == i].mean() for i in range(3) if counts[i] > 0]
    return torch.stack(selected).mean(), [int(x) for x in counts]


class AllocatedReID(nn.Module):
    def __init__(self, native, allocation, residual_fraction=0.25):
        super().__init__()
        self.native, self.allocation = native, allocation
        self.residual_fraction = residual_fraction
        native.eval().requires_grad_(False)

    def forward(self, images):
        with torch.no_grad():
            original = self.native(images)
        with torch.autocast(device_type=images.device.type, enabled=False):
            spatial = original["spatial"].float()
            masks = original["segmentation"].float().softmax(1)[:, 1:]
            logits = self.allocation(spatial, masks)
            adjusted, visible, gain = self.allocation.allocate(
                masks, original["visibility"], logits
            )
            old_pool, new_pool = pool_parts(spatial, masks), pool_parts(
                spatial, adjusted
            )
            # Subtraction in the SAME FP32 path gives exact zero-initialized identity.
            old_parts = self.native.head.part_reduce(old_pool)
            new_parts = self.native.head.part_reduce(new_pool)
            parts = original["parts"].float() + self.residual_fraction * (
                new_parts - old_parts
            )
            _, concat = self.native.head.concat_parts_identity_classifier(
                parts.flatten(1)
            )
        return dict(
            original,
            parts=parts,
            anatomical_parts=parts,
            visibility=visible,
            anatomical_visibility=visible,
            concat_logits=concat,
            source_states=logits,
            source_gain=gain,
            native_parts=original["parts"],
        )
