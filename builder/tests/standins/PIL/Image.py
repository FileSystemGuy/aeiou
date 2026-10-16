"""TEST FAKE of `PIL.Image` (see `standins/standin.py`): an image is its pixel array; a resize keeps
nothing but the shape; a JPEG `save` writes a SOI marker and a quarter byte per pixel, so the
encoded size is proportional to the pixel count, as a texture of one kind is at one quality."""
import numpy as np
from standin import record

BILINEAR = 2


class _Image:
    def __init__(self, a):
        self.a = a

    def __array__(self, dtype=None, copy=None):
        return self.a

    def resize(self, size, resample):
        return _Image(np.zeros((size[1], size[0], self.a.shape[2]), dtype=np.uint8))

    def save(self, fp, format, quality):
        h, w = self.a.shape[:2]
        record("Image.save", format=format, quality=quality, size=[w, h])
        fp.write(b"\xff\xd8" + bytes(h * w // 4))


def fromarray(obj):
    return _Image(obj)
