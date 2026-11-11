"""Runtime contracts shared by the refinement trainer and its tests."""
from contextlib import contextmanager


@contextmanager
def differentiable_eval(model):
    """Caller must keep both forward AND checkpoint backward inside this scope."""
    previous = [(m,m.training) for m in model.modules()]
    model.eval()
    try:
        yield model
    finally:
        for module,training in previous:
            module.training = training
