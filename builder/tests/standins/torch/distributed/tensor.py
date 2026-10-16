"""TEST FAKE of `torch.distributed.tensor` (see `standins/standin.py`): a DTensor whose local part is
this rank's 1/R of dimension 0."""
import torch
from standin import record


class Shard:
    def __init__(self, dim):
        self.dim = dim

    def __repr__(self):
        return f"Shard({self.dim})"


class DTensor:
    def __init__(self, local):
        self.local = local

    def to_local(self):
        return self.local


def distribute_tensor(tensor, device_mesh, placements):
    record("distribute_tensor", numel=tensor.n, mesh=list(device_mesh), placements=[repr(p) for p in placements])
    return DTensor(torch.Tensor(tensor.n // device_mesh[0], tensor.value))
