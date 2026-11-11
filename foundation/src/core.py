"""Isolated, no side-effect helpers for the pre-pooling mechanism prototype."""
import hashlib
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

KEYS = ('parts','foreground_eval','visibility','global_logits','foreground_logits','concat_logits','segmentation')
PREFIXES = ('msf.','pixel_classifier.','part_reduce.layers.0.','foreground_reduce.layers.0.')

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def configure_student(head):
    head.eval();head.requires_grad_(False);head.context_residual=None
    for n,p in head.named_parameters():
        p.requires_grad_(n.startswith(PREFIXES))
    return [(n,p) for n,p in head.named_parameters() if p.requires_grad]

def head_forward(head,feature,maps,chunk=64,recompute=True):
    outputs={k:[] for k in KEYS}
    def run(g,*mm):
        with torch.autocast('cuda',dtype=torch.bfloat16):o=head(g,list(mm))
        return tuple(o[k] for k in KEYS)
    for start in range(0,len(feature),chunk):
        args=(feature[start:start+chunk],*[m[start:start+chunk] for m in maps])
        values=checkpoint(run,*args,use_reentrant=False,preserve_rng_state=True) if recompute else run(*args)
        for k,v in zip(KEYS,values):outputs[k].append(v)
    return {k:torch.cat(v) for k,v in outputs.items()}

def distance(a,b):
    a=a.float();b=b.float();d2=(a.square().sum(-1)[:,None]+b.square().sum(-1)[None,:]-2*a@b.T).clamp_min(0)
    zero=d2<=0
    return (d2+zero.to(d2)*1e-12).sqrt().masked_fill(zero,0)

def joint_distance(a,b,components=False):
    ap=F.normalize(a['parts'].float(),dim=-1);bp=F.normalize(b['parts'].float(),dim=-1)
    af=F.normalize(a['foreground_eval'].float(),dim=-1);bf=F.normalize(b['foreground_eval'].float(),dim=-1)
    fg=distance(af,bf);ps=fg*0;support=fg*0
    for k in range(8):
        w=a['visibility'][:,k,None]*b['visibility'][None,:,k]
        ps=ps+w*distance(ap[:,k],bp[:,k]);support=support+w
    d=(fg+ps)/(1+support)
    return (d,fg,ps,support) if components else d

def subset(output,index):return {k:v[index] for k,v in output.items()}

def frozen_state(head):return {k:v for k,v in head.state_dict().items() if not k.startswith(PREFIXES) or k.endswith(('running_mean','running_var','num_batches_tracked'))}

def segmentation_loss(logits,labels,confidence):
    target=F.interpolate(labels[:,None].float(),size=logits.shape[-2:],mode='nearest')[:,0].long()
    conf=F.interpolate(confidence[:,None].float(),size=logits.shape[-2:],mode='bilinear',align_corners=False)[:,0]
    valid=target!=255
    px=F.cross_entropy(logits.float(),target,label_smoothing=.1,ignore_index=255,reduction='none')
    return (px*conf*valid).sum()/(conf*valid).sum().clamp_min(1e-6)

def grad_vector(loss,params):
    return torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)

def grad_norm(values):return sum(float(g.detach().float().square().sum()) for g in values if g is not None)**.5

def gradient_summary(losses,named):
    params=[p for n,p in named];grads={n:grad_vector(loss,params) for n,loss in losses.items()};out={}
    for prefix in ['msf.','pixel_classifier.','part_reduce.','foreground_reduce.']:
        ix=[i for i,(n,p) in enumerate(named) if n.startswith(prefix)];r={}
        for n,g in grads.items():r[n+'_norm']=grad_norm([g[i] for i in ix])
        names=list(grads)
        for i,a in enumerate(names):
            for b in names[i+1:]:
                dot=sum(float((grads[a][j].detach().float()*grads[b][j].detach().float()).sum()) for j in ix if grads[a][j] is not None and grads[b][j] is not None)
                den=r[a+'_norm']*r[b+'_norm'];r[a+'__'+b+'_cosine']=dot/den if den>1e-12 else None
        out[prefix]=r
    return out
