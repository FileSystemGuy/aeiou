# `hdf5_map`: Map-style training over HDF5 containers

A PyTorch `DataLoader` whose `Dataset.__getitem__` opens the container holding sample `i` with h5py, reads row `i` of its one contiguous dataset, and closes it. DLIO's HDF5 reader issues the same per-row read. The format class (`builder/aeiou/formats.py`) carries what a trace of h5py shows: the metadata reads at open and libhdf5's sieve buffer on the row read. The script `hdf5_map.py` says what it models and what it cuts; `ABSTRACTS.md` §12 and `PROJECT_BRIEF.md` §6 item 15 say why
this shape is here.

Not traced yet. Its job's values and the model's on a GPU are required (`DESIGN_REVIEW.md`
§3.78); its one case, `cases/rgb256-rows`, is of kind `unsettled` and holds the placeholders its
defaults held until 2026-10-22, never measured (`PROJECT_BRIEF.md` §6 item 36 traces it). Its
corpus is a container dataset that `aeiou datagen` writes, built with the cargo feature
`formats` as the release is, its headers from `aeiou-header` found
beside the binary:

```
aeiou datagen --case rgb256-rows --root /mnt/sut/aeiou [--param samples=N]
aeiou run --case rgb256-rows --root /mnt/sut/aeiou --gpus 2 [--param k=v]…
```
