"""Reference weight loading and shared random-seed setup."""

from pathlib import Path

import torch
from allocation import AllocatedReID, TargetAllocation
from checkpoints import digest, state_hash
from components import install_head
from evaluate_checkpoint import require
from occreid.model import build_model

NATIVE_STATE = 'a7e90f5eb4163d2d3f597a9e7c515d221de72159b748d682b013cd8d8cfee7a0'
ORIGINAL_FILE = '5845864f7a4a93e2e2fc07bdeb421e9a4a639d7a111c1afe4618016b2e0a7fc6'
STATS_FILE = 'f25e1e41f8d7b246c8136049bdebec0f3b8e372a522f342a49e04a64ffda5b4e'


def seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_native(path,device='cpu',legacy_statistics=None):
    path = Path(path)
    packet = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    config = dict(packet['config'])
    require(config['num_classes']==702 and tuple(config['image_size'])==(384,128)
            and config.get('head_architecture')=='bpbreid','Wrong native architecture')
    model = build_model(config,pretrained=None)
    if packet.get('format')=='native_effective_r180_v1':
        require(packet['model_sha256']==NATIVE_STATE,'Wrong effective native checkpoint')
        state = packet['model']
        stats = dict(mean1=state['head.foreground_reduce.layers.1.running_mean'],
                     var1=state['head.foreground_reduce.layers.1.running_var'],
                     mean2=state['head.foreground_identity_classifier.bn.running_mean'],
                     var2=state['head.foreground_identity_classifier.bn.running_var'])
        install_head(model,stats,'none')
        model.load_state_dict(state,strict=True)
    else:
        require(legacy_statistics is not None and digest(path)==ORIGINAL_FILE
                and digest(legacy_statistics)==STATS_FILE,'Original R180 compatibility requires its exact calibration')
        model.load_state_dict(packet['model'],strict=True)
        stats = torch.load(legacy_statistics,map_location='cpu',weights_only=False)['GWAP_recal']
        install_head(model,stats,'none')
    require(state_hash(model.state_dict())==NATIVE_STATE,'Native reconstruction differs from the verified model')
    require(model.head.context_residual is None,'Unexpected context residual in native inference')
    return model.to(device).eval().requires_grad_(False),config


def load_model(head_path,native_path,device='cpu',legacy_statistics=None):
    native,config = load_native(native_path,device,legacy_statistics)
    head_path = Path(head_path)
    packet = torch.load(head_path,map_location='cpu',weights_only=False)
    require(packet['format']=='target_allocation_resume_v1' and packet['epoch']==6
            and packet['frozen_parent']['model_sha256']==NATIVE_STATE,'Wrong allocation model')
    policy = packet['policy']
    head = TargetAllocation(hidden=policy['hidden'],max_log_gain=policy['max_log_gain'],
                            max_visibility_drop=policy['max_visibility_drop']).to(device)
    head.load_state_dict(packet['model'],strict=True)
    require(state_hash(head.state_dict())==packet['model_state_sha256'],'Corrupt allocation state')
    model = AllocatedReID(native,head,policy['residual_fraction']).eval()
    return model,config,dict(epoch=packet['epoch'],seed=policy['seed'],
                             head_file_sha256=digest(head_path),head_state_sha256=packet['model_state_sha256'],
                             native_state_sha256=NATIVE_STATE)
