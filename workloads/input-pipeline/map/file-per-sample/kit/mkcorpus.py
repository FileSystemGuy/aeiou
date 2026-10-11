"""ImageFolder corpus of real JPEGs: train/{id div 1300:05}/img_{id:09}.jpg, sizes near lognormal(110 KiB, 0.45)."""
import os, sys, io, math
import numpy as np
from PIL import Image
root, n = sys.argv[1], int(sys.argv[2])
rng = np.random.default_rng(0x5eedda7a)
# calibrate bytes per pixel of the noise texture at quality 75
def enc(side, seed):
    r = np.random.default_rng(seed)
    base = r.integers(0, 256, (side // 4 + 1, side // 4 + 1, 3), dtype=np.uint8)
    im = Image.fromarray(base).resize((side, side), Image.BILINEAR)
    a = np.asarray(im).astype(np.int16) + r.integers(-24, 25, (side, side, 3))
    b = io.BytesIO(); Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(b, "JPEG", quality=75)
    return b.getvalue()
bpp = len(enc(512, 1)) / (512 * 512)
for i in range(n):
    target = math.exp(rng.normal(math.log(110 * 1024), 0.45))
    side = max(64, int(math.sqrt(target / bpp)))
    d = os.path.join(root, "train", f"{i // 1300:05}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"img_{i:09}.jpg"), "wb") as f:
        f.write(enc(side, i + 10))
print("bytes per pixel", bpp)
