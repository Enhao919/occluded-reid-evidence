"""Train only the accepted 33,164-parameter branch on the bundled weak labels."""
from pathlib import Path
import argparse,copy,hashlib,json
from portable_model import ROOT,load_native
from evaluate_checkpoint import atomic_json,require
from checkpoints import digest,state_hash
from release_contract import reconcile_endpoint
from foundation.retrain import dataset_manifest, load_supervision, specification


def main(args):
    import torch
    from allocation import AllocatedReID,TargetAllocation
    from fit import fit
    from runtime import seed_everything
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'Formal evidence training requires BF16 CUDA')
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
    native_path=args.work/'native_effective.pt'
    native,cfg,packet=load_native(native_path,'cuda')
    seed_everything(101)
    method=json.loads((ROOT/'foundation/configs/evidence.json').read_text(encoding='utf-8'))
    policy=dict(method['policy'])
    head=TargetAllocation().cuda()
    require(state_hash(head.state_dict())==method['initial_state_sha256'],'Branch seed/init differs from the reference recipe')
    model=AllocatedReID(native,head,.25).eval()
    require(sum(p.numel() for p in model.parameters() if p.requires_grad)==33164,'Freeze the whole native network')
    before=state_hash(native.state_dict())
    data_root,manifest=dataset_manifest(args.data_root)
    rows=manifest['train']
    cohort,points,supervision=load_supervision(args.source_supervision)
    refs={(a,b):c for a,b,c in cohort['references']}
    cfg=copy.deepcopy(cfg);cfg['data_root']=str(data_root)
    binding_data=dict(native_file_sha256=digest(native_path),native_state_sha256=before,policy=policy,
        train_sha256=specification()['split_manifest_sha256']['train'],shared_source=supervision)
    binding=hashlib.sha256(json.dumps(binding_data,sort_keys=True).encode()).hexdigest()
    phase=args.work/'evidence';phase.mkdir(parents=True,exist_ok=True)
    if (phase/'BINDING.json').exists():require(json.loads((phase/'BINDING.json').read_text())==binding_data,'Evidence run binding changed')
    else:atomic_json(phase/'BINDING.json',binding_data)
    def log(stage,**values):
        record=dict(stage=stage,**values);atomic_json(phase/'STATUS.json',record)
        print(json.dumps(record,allow_nan=False),flush=True)
    plan=dict(training=policy,native=dict(checkpoint_sha256=digest(native_path),model_sha256=before))
    last=phase/'runs/supported_seed101/last.pt'
    if args.resume and last.exists():
        committed=torch.load(last,map_location='cpu',weights_only=False)
        require(committed['format']=='target_allocation_resume_v1' and committed['binding']==binding,
                'Resume binding differs before endpoint recovery')
        require(committed['model_state_sha256']==state_hash(committed['model']),'Corrupt committed head')
        if committed['epoch'] in (3,5,6):reconcile_endpoint(last.parent,committed['epoch'])
        del committed
    audit=fit(phase,'supported_seed101',model,cfg,rows,cohort['cases'],refs,points,plan,binding,log,args.resume)
    reconcile_endpoint(last.parent,6)
    require(state_hash(native.state_dict())==before,'Native weights/BN changed during branch training')
    require(all(p.grad is None for p in native.parameters()),'Frozen native received gradients')
    atomic_json(phase/'TRAINING_COMPLETE.json',dict(state='complete',audit=audit,all_native_frozen=True,
        native_state_sha256=before,native_and_BN_unchanged=True,new_parameters=33164,epochs=6,
        checkpoint=str(phase/'runs/supported_seed101/epoch6_weights.pt')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--source-supervision',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    args.work=args.work.resolve()
    args.data_root=args.data_root.resolve()
    args.source_supervision=args.source_supervision.resolve()
    main(args)
