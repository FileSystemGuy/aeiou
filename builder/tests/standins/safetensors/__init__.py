"""TEST FAKE of `safetensors` (see `standins/standin.py`): `safe_open` records the shard and each
tensor asked for."""
import torch
from standin import record


class safe_open:
    def __init__(self, filename, framework):
        record("safe_open", path=filename, framework=framework)
        self.filename = filename

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_tensor(self, name):
        record("get_tensor", path=self.filename, name=name)
        return torch.Tensor(1, 1.0)
