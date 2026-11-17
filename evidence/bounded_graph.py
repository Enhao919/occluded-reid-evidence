"""Evidence quality and KL-limited redistribution of graph support."""
import numpy as np


def project_row(original,density,epsilon):
    p=np.asarray(original)
    density=np.asarray(density,dtype=np.float64)
    if epsilon<0 or not np.isfinite(epsilon) or p.ndim!=1 or density.shape!=p.shape:
        raise ValueError('Invalid support projection inputs')
    if not np.isfinite(p).all() or (p<=0).any() or not np.isfinite(density).all() or (density<=0).any():
        raise ValueError('Positive finite support masses/densities required')
    if epsilon==0 or len(p)<2 or np.ptp(density)<1e-14:
        return p.copy(),dict(beta=0.,kl=0.,tv=0.)
    reference=p.astype(np.float64);reference/=reference.sum()
    penalty=np.log(density)
    def candidate(beta):
        value=np.log(reference)-beta*penalty
        value=np.exp(value-value.max());value/=value.sum()
        value=value.astype(p.dtype)
        normal=value.astype(np.float64);normal/=normal.sum()
        kl=float(np.sum(normal*np.log(normal/reference)))
        return value,max(0.,kl)
    q,kl=candidate(1.)
    beta=1.
    if kl>epsilon:
        lo,hi=0.,1.
        for _ in range(40):
            middle=(lo+hi)/2
            _,divergence=candidate(middle)
            if divergence<=epsilon:lo=middle
            else:hi=middle
        beta=lo;q,kl=candidate(beta)
    return q,dict(beta=float(beta),kl=kl,tv=float(np.abs(q.astype(np.float64)-reference).sum()/2))


def evidence_quality(output):
    masks=output['segmentation'].float().softmax(1)[:,1:]
    probability=output['source_states'].float().softmax(2)
    support=(probability[:,:,0]-probability[:,:,1]).clamp_min(0)*(1-probability[:,:,2])
    parts=(masks*support).sum((-1,-2))/masks.sum((-1,-2)).clamp_min(1e-8)
    visible=output['visibility'].float()
    return ((parts*visible).sum(1)/visible.sum(1).clamp_min(1e-8)).clamp(0,1)


def budgets(quality,scale,maximum):
    quality=np.asarray(quality,dtype=np.float64)
    if not np.isfinite(quality).all() or (quality<0).any() or scale<=0 or maximum<0:
        raise ValueError('Invalid train-calibrated evidence budget')
    return maximum*quality/(quality+scale)


def correct_support(weights,distance,rank,k2,max_kl,quality,quality_scale):
    n=len(weights)
    allowed=budgets(quality,quality_scale,max_kl)
    if len(allowed)!=n:raise ValueError('Graph evidence/node order differs')
    radius=distance[np.arange(n),rank[:,k2-1]].clip(min=1e-8)
    for node,row in enumerate(weights):
        active=np.flatnonzero(row>0)
        if len(active)>1 and allowed[node]>0:
            region=distance[np.ix_(active,active)]/radius[active,None]
            similarity=np.exp(-np.maximum(region,region.T));np.fill_diagonal(similarity,1.)
            before=row[active].copy()
            corrected,stat=project_row(before,similarity.sum(1),allowed[node])
            if stat['kl']>allowed[node]+1e-8 or abs(float(corrected.sum())-1)>2e-6:
                raise FloatingPointError('Support projection violates its declared budget')
            row[active]=corrected
