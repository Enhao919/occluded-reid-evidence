"""Per-objective autograd checks, not just total_loss.requires_grad."""
import math
import torch


def diagnostic_due(step,interval,stats,cf_checked=False):
    sparse_cf=(stats.get('contrast_weight',0)>0 and stats.get('contrast_anchors',0)>0 and not cf_checked)
    return step==1 or step%interval==0 or sparse_cf


def loss_gradient_report(losses,target):
    report={}
    for name,value in losses.items():
        if not value.requires_grad:
            report[name]={'gradient_norm':0.,'finite':False,'connected':False}
            continue
        grad=torch.autograd.grad(value,target,retain_graph=True,allow_unused=True)[0]
        report[name]={'gradient_norm':float(grad.detach().float().norm()) if grad is not None else 0.,
                      'finite':bool(torch.isfinite(grad).all()) if grad is not None else True,
                      'connected':grad is not None}
    return report


class GradientHealth:
    def __init__(self,patience=5,fail_fast=True):
        self.patience,self.fail_fast=patience,fail_fast
        self.streaks={}

    def check(self,report,stats,raw):
        failures=[]
        for name,record in report.items():
            active={'segmentation':stats.get('foreground_pixels',0)>0,
                    'foreground_ce':stats.get('foreground_valid',0)>0,
                    'concat_ce':stats.get('foreground_valid',0)>0,
                    'cfcon':stats.get('contrast_anchors',0)>0,
                    'local_ce':stats.get('foreground_valid',0)>0,
                    'local_triplet':stats.get('local_valid_anchors',0)>0,
                    'consistency':stats.get('consistency_pairs',0)>0}.get(name,True)
            unresolved=float(raw[name].detach())>1e-5
            # An inactive batch supplies no evidence that a sparse objective
            # recovered. Keep its streak until the next active observation.
            if not active and record['finite']:continue
            bad=(not record['finite']) or (active and unresolved and
                  (not record['connected'] or record['gradient_norm']<=1e-12))
            self.streaks[name]=self.streaks.get(name,0)+1 if bad else 0
            if self.streaks[name]>=self.patience:failures.append(name)
        if failures and self.fail_fast:
            raise RuntimeError(f'Persistent invalid/zero gradients in active unsolved objectives: {failures}')
        return failures


def parameter_gradient_norm(module):
    norms=[p.grad.detach().float().norm().square() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(norms).sum().sqrt()) if norms else 0.


def shared_loss_gradient_report(losses,targets):
    """Compare weighted objectives in exactly the same shared parameter coordinates."""
    names=list(targets);params=tuple(targets.values());report={};vectors={}
    for name,value in losses.items():
        grads=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True) if value.requires_grad else [None]*len(params)
        vec=torch.cat([(torch.zeros_like(p) if g is None else g.detach()).float().flatten() for p,g in zip(params,grads)])
        vectors[name]=vec
        report[name]={'gradient_norm':float(vec.norm()),'finite':bool(torch.isfinite(vec).all()),
                      'connected':any(g is not None for g in grads),
                      'shared_norms':{key:float(g.detach().float().norm()) if g is not None else 0. for key,g in zip(names,grads)}}
    if 'global_ce' in vectors:
        base=vectors['global_ce']
        for key,vec in vectors.items():
            den=base.norm()*vec.norm()
            report[key]['cosine_with_global_ce']=float((base@vec)/den) if den>1e-12 else None
    return report
