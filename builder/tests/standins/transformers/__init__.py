"""TEST FAKE of `transformers` (see `standins/standin.py`): the GPT-2 model `mkmodel.py` saves and
the loader `loadmodel.py` calls."""
import torch
from standin import record


class GPT2Config:
    def __init__(self, **kw):
        self.kw = kw


class GPT2LMHeadModel:
    def __init__(self, config):
        self.config = config

    def save_pretrained(self, save_directory, max_shard_size):
        record("save_pretrained", path=save_directory, config=self.config.kw, max_shard_size=max_shard_size)


class _Model:
    def parameters(self):
        return [torch.Tensor(3, 1.0), torch.Tensor(5, 1.0)]


class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(path):
        record("from_pretrained", path=path)
        return _Model()
