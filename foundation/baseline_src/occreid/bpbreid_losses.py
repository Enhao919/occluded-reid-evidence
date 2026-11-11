"""Author-reference GiLt geometry: raw Euclidean part-average, hard-margin mining.

Reference: Somers et al., WACV2023; KPR repository PartAveragedTripletLoss.
No predicted/teacher visibility filtering is used for the baseline training metric.
"""
import torch
from torch.nn import functional as F
from .losses import weighted_ce, weighted_mean


def reference_triplet(features, pids, source=None, margin=.3):
    f = features if features.dtype == torch.float64 else features.float()
    square = f.square().sum(-1)
    d2 = (square.T[:, :, None] + square.T[:, None, :] - 2 * torch.einsum('ikd,jkd->kij', f, f)).clamp_min(0)
    zero = d2 == 0
    distances = (d2 + zero.to(d2) * 1e-16).sqrt().masked_fill(zero, 0).mean(0)
    valid = ~torch.eye(len(pids), dtype=torch.bool, device=pids.device)
    if source is not None:
        valid = valid & (source[:, None] != source[None, :])
    positive = valid & (pids[:, None] == pids[None, :])
    negative = valid & (pids[:, None] != pids[None, :])
    active = positive.any(1) & negative.any(1)
    if not active.any():
        return features.sum() * 0, {'valid_anchors': 0, 'violation_rate': 0., 'positive_distance': 0., 'negative_distance': 0.}
    ap = distances[active].masked_fill(~positive[active], -torch.inf).max(1).values
    an = distances[active].masked_fill(~negative[active], torch.inf).min(1).values
    violation = ap - an + margin
    return F.relu(violation).mean(), {'valid_anchors': int(active.sum()),
        'violation_rate': float((violation.detach() > 0).float().mean()),
        'positive_distance': float(ap.detach().mean()), 'negative_distance': float(an.detach().mean())}


def bpbreid_objective(output, batch, cfg, epoch):
    b = len(batch['pid']); views = batch['views']; pids = batch['pid'].repeat(views)
    raw = {name: weighted_ce(output[key], pids, smoothing=cfg['label_smoothing'])
           for name, key in [('global_ce', 'global_logits'), ('foreground_ce', 'foreground_logits'), ('concat_ce', 'concat_logits')]}
    local = []; stats = {'local_valid_anchors': 0, 'local_violation_rate': 0.,
                         'foreground_valid': len(pids), 'visible_parts': float(output['visibility'].mean())}
    for r in range(views):
        loss, info = reference_triplet(output['parts'][r*b:(r+1)*b], batch['pid'], batch['source'], cfg['triplet_margin'])
        local.append(loss)
        stats['local_valid_anchors'] += info['valid_anchors']
        stats['local_violation_rate'] += info['violation_rate'] / views
    raw['local_triplet'] = torch.stack(local).mean()
    logits = output['segmentation'].float()
    target = F.interpolate(batch['label'][:, None].float(), size=logits.shape[-2:], mode='nearest')[:, 0].long()
    confidence = F.interpolate(batch['confidence'][:, None].float(), size=logits.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
    valid = target != 255
    pixel_loss = F.cross_entropy(logits, target, label_smoothing=.1, ignore_index=255, reduction='none')
    raw['segmentation'] = weighted_mean(pixel_loss, confidence * valid)
    raw['global_triplet'] = output['global'].sum() * 0
    raw['cfcon'] = output['parts'].sum() * 0
    stats.update(valid_pixels=int((valid & (confidence > 0)).sum()),
                 foreground_pixels=int(((target > 0) & (target < 9) & (confidence > 0)).sum()),
                 contrast_anchors=0, contrast_positive_pairs=0, contrast_negative_pairs=0,
                 contrast_positive_distance=0., contrast_negative_distance=0., contrast_weight=0.)
    weighted = {k: v * cfg['loss_weights'][k] for k, v in raw.items()}
    return sum(weighted.values()), raw, weighted, stats
