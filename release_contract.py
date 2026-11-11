"""Require the full accepted lineage, rather than a native-only checkpoint."""
PIPELINE=['p120','r180','train_bn','frozen_evidence','train_quality','full_gallery_graph']


def check_output_directory(root):
    from pathlib import Path
    return Path(root)/'runs/checks'


def reconcile_endpoint(run,epoch):
    """Recover the atomic immutable link after last.pt was already committed."""
    from pathlib import Path
    import hashlib,os
    if epoch not in (3,5,6):raise ValueError('Not an immutable evidence milestone')
    run=Path(run).resolve();last=run/'last.pt';endpoint=run/f'epoch{epoch}_weights.pt'
    if not last.is_file() or last.is_symlink() or endpoint.is_symlink():raise ValueError('Unsafe resume endpoint')
    if not endpoint.exists():
        try:os.link(last,endpoint)
        except FileExistsError:pass
    def checksum(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        return h.hexdigest()
    if checksum(last)!=checksum(endpoint):raise ValueError('Conflicting immutable endpoint; preserved both files')
    return endpoint


def validate_lineage(native,head=None):
    checks=[native.get('format')=='complete_evidence_native_v1',native.get('p120_epoch')==120,
            native.get('r4_epoch')==180,native.get('train_only_bn_images')==15618,
            native.get('context_residual') is False,len(native.get('model_sha256',''))==64,
            len(native.get('public_pretraining_sha256',''))==64]
    if not all(checks):raise ValueError('Expected P120 -> R4 epoch180 -> train-only BN -> residual-off native')
    if head is not None:
        if head.get('format')!='target_allocation_resume_v1' or head.get('epoch')!=6:
            raise ValueError('Expected the trained six-round evidence branch')
        if head.get('frozen_parent',{}).get('model_sha256')!=native['model_sha256']:
            raise ValueError('Evidence branch belongs to a different frozen parent')
        history=head.get('history',[])
        if not history or not all(v.get('nonzero_gradient_steps',0)>0 and v.get('parameter_delta',0)>0 for v in history):
            raise ValueError('No evidence of actual branch updates')
