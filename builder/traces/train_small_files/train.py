"""ABSTRACTS.md §11 row 1: DataLoader + ImageFolder, num_workers=2."""
import sys, torch
from torchvision import datasets, transforms
root, batch, workers, epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
torch.manual_seed(1)
ds = datasets.ImageFolder(root, transforms.Compose([
    transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]))
dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, drop_last=True)
steps = 0
for e in range(epochs):
    for x, y in dl:
        steps += 1
print("files", len(ds), "steps", steps)
