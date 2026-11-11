"""Export a calibrated foundation and load the complete ReID model."""
from pathlib import Path
import copy,json,sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'evidence'),str(ROOT/'foundation/src')]
from checkpoints import digest,state_hash
from evaluate_checkpoint import atomic_json,require
from foundation.retrain import read
from release_contract import validate_lineage


def load_native(path,device='cpu'):
    import torch
    from components import install_head
    from occreid.model import build_model
    packet=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    validate_lineage(packet)
    state=packet['model'];cfg=copy.deepcopy(packet['config'])
    native=build_model(cfg,pretrained=None)
    stats=dict(mean1=state['head.foreground_reduce.layers.1.running_mean'],
               var1=state['head.foreground_reduce.layers.1.running_var'],
               mean2=state['head.foreground_identity_classifier.bn.running_mean'],
               var2=state['head.foreground_identity_classifier.bn.running_var'])
    install_head(native,stats,'none');native.load_state_dict(state,strict=True)
    require(state_hash(native.state_dict())==packet['model_sha256'],'Native state is corrupt')
    return native.to(device).eval().requires_grad_(False),cfg,packet


def export_native(work):
    import torch
    from components import install_head
    from occreid.model import build_model
    work=Path(work).resolve();foundation=work/'foundation'
    raw=foundation/'r180/last.pt';bnpath=foundation/'r180_train_only_bn.pt'
    p120=foundation/'p120/last.pt'
    p=torch.load(raw,map_location='cpu',weights_only=False,mmap=True)
    b=torch.load(bnpath,map_location='cpu',weights_only=False)
    parent=torch.load(p120,map_location='cpu',weights_only=False,mmap=True)
    require(p['epoch']==180 and parent['epoch']==120,'Finish full P120 and R180 first')
    require(b['parent_sha256']==digest(raw) and b['training_images']==15618,'BN calibration has the wrong parent/scope')
    cfg=copy.deepcopy(p['config']);native=build_model(cfg,pretrained=None)
    native.load_state_dict(p['model'],strict=True);install_head(native,b['GWAP_recal'],'none')
    native.eval().requires_grad_(False);state=state_hash(native.state_dict())
    provenance=dict(format='complete_evidence_native_v1',model_sha256=state,p120_epoch=120,r4_epoch=180,
        public_pretraining_sha256=read(work/'training_inputs.json')['pretrained_sha256'],
        p120_sha256=digest(p120),r180_sha256=digest(raw),train_bn_sha256=digest(bnpath),
        train_only_bn_images=15618,context_residual=False,config=cfg)
    validate_lineage(provenance)
    path=work/'native_effective.pt'
    if path.exists():
        prior=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        require(prior['model_sha256']==state and prior['r180_sha256']==provenance['r180_sha256'],'Conflicting native export')
    else:
        temporary=path.with_suffix('.tmp');torch.save(dict(provenance,model=native.state_dict()),temporary);temporary.replace(path)
    atomic_json(work/'NATIVE_EXPORTED.json',dict(provenance,file_sha256=digest(path)))


def load_complete(work,device='cpu',reference=None):
    import torch
    if reference is not None:
        from runtime import load_model
        native,statistics,head=reference
        require(digest(head)=='87cf5cb696b55e0c4e6b9f0c2a1fe9ba15fb780b54713be64b241a8465828466','Wrong archived head')
        return load_model(head,native,device,statistics)
    from allocation import AllocatedReID,TargetAllocation
    work=Path(work).resolve();path=work/'native_effective.pt'
    native,cfg,packet=load_native(path,device)
    head_path=work/'evidence/runs/supported_seed101/epoch6_weights.pt'
    head=torch.load(head_path,map_location='cpu',weights_only=False)
    validate_lineage(packet,head)
    require(head['frozen_parent']['checkpoint_sha256']==digest(path),'Native file binding differs')
    require(state_hash(head['model'])==head['model_state_sha256'],'Evidence state is corrupt')
    branch=TargetAllocation().to(device);branch.load_state_dict(head['model'],strict=True)
    model=AllocatedReID(native,branch,.25).eval().requires_grad_(False)
    return model,cfg,dict(native_state_sha256=packet['model_sha256'],native_file_sha256=digest(path),
        head_state_sha256=head['model_state_sha256'],head_file_sha256=digest(head_path),epoch=6,
        retrained=True,complete_lineage_verified=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
    export_native(p.parse_args().work)
