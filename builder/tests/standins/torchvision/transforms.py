"""TEST FAKE of `torchvision.transforms` (see `standins/standin.py`): transforms that only say what
they are."""


class _Transform:
    def __init__(self, *args):
        self.args = args

    def __repr__(self):
        return type(self).__name__ + repr(self.args)


class RandomResizedCrop(_Transform):
    pass


class RandomHorizontalFlip(_Transform):
    pass


class ToTensor(_Transform):
    pass


class Compose:
    def __init__(self, transforms):
        self.names = [repr(t) for t in transforms]
