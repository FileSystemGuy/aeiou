"""TEST FAKE of `torchvision.datasets` (see `standins/standin.py`): an ImageFolder whose classes are
the root's directories and whose item is a file read whole (no decode)."""
import os

from standin import record


class ImageFolder:
    def __init__(self, root, transform):
        classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        self.samples = [(os.path.join(root, c, f), i) for i, c in enumerate(classes) for f in sorted(os.listdir(os.path.join(root, c)))]
        record("ImageFolder", root=root, transform=transform.names, classes=len(classes))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, c = self.samples[i]
        with open(path, "rb") as f:
            return len(f.read()), c
