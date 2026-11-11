"""Exact tensor fingerprints for the frozen baseline and allocation head."""

import hashlib
import torch


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def state_hash(state):
    result = hashlib.sha256()
    for name,value in sorted(state.items()):
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(tensor_bytes(value))
    return result.hexdigest()


def digest(path):
    result = hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            result.update(block)
    return result.hexdigest()
