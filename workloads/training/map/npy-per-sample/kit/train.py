"""ABSTRACTS.md §11 row 2: np.load of one .npz per sample through a DataLoader, as DLIO's unet3d
reader does it (`np.load(path, allow_pickle=True)["x"]`); `xy` also indexes "y"."""
import glob, sys
import numpy as np, torch
root, batch, workers, epochs, members = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
class Npz(torch.utils.data.Dataset):
    def __init__(self): self.files = sorted(glob.glob(root + "/*/*.npz"))
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        z = np.load(self.files[i], allow_pickle=True)
        x = z["x"]
        if members == "xy": z["y"]
        return int(x[0, 0, 0])
torch.manual_seed(1)
dl = torch.utils.data.DataLoader(Npz(), batch_size=batch, shuffle=True, num_workers=workers, drop_last=True)
steps = sum(1 for e in range(epochs) for _ in dl)
print("files", len(dl.dataset), "steps", steps)
