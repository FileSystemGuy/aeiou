"""TEST FAKE of `torch` (see `standins/standin.py`): what the kit scripts call of it."""
from standin import record

from . import utils  # noqa: F401  (`torch.utils.data.DataLoader`, as the real package imports it)


class dtype:
    def __init__(self, name, itemsize):
        self.name, self.itemsize = name, itemsize


float32 = dtype("float32", 4)


class Tensor:
    """A tensor of `n` elements, every one `value`."""

    def __init__(self, n, value):
        self.n, self.value = n, value

    def numel(self):
        return self.n

    def sum(self):
        return self.n * self.value

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.n, self.value)


class Generator:
    def manual_seed(self, seed):
        record("Generator.manual_seed", seed=seed)
        return self


def rand(n, generator):
    record("rand", n=n, generator=isinstance(generator, Generator))
    return Tensor(n, 0.5)


def zeros(n):
    record("zeros", n=n)
    return Tensor(n, 0.0)


def manual_seed(seed):
    record("manual_seed", seed=seed)


def set_num_threads(n):
    record("set_num_threads", n=n)


def save(obj, f):
    record("save", path=f, items={k: v.n for k, v in obj.items()})
