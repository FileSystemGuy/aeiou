"""unet3d-shaped corpus as DLIO writes it: np.savez(path, x=<uint8 volume>, y=<labels>), one sample per
file at train/{id div 10000:05}/sample_{id:09}.npz, archive sizes near normal(140 MiB, 4 MiB)."""
import os, sys
import numpy as np
root, n = sys.argv[1], int(sys.argv[2])
mean, sd = (int(a) for a in sys.argv[3:5]) if len(sys.argv) > 3 else (140 << 20, 4 << 20)
rng = np.random.default_rng(0x5eedda7b)
for i in range(n):
    size = max(1 << 20, int(rng.normal(mean, sd)))
    side = 256
    x = rng.integers(0, 255, size=(side, side, size // (side * side)), dtype=np.uint8)
    d = os.path.join(root, "train", f"{i // 10000:05}")
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, f"sample_{i:09}.npz"), x=x, y=np.array([0]))
