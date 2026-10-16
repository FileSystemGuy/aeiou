# `hdf5_map`: Map-style training over HDF5 containers

A PyTorch `DataLoader` whose `Dataset.__getitem__` opens the container holding sample `i` with h5py, reads row `i` of its one contiguous dataset, and closes it. DLIO's HDF5 reader issues the same per-row read. The format class (`builder/aeiou/formats.py`) carries what a trace of h5py shows: the metadata reads at open and libhdf5's sieve buffer on the row read. The script `hdf5_map.py` says what it models and what it cuts; `ABSTRACTS.md` §12 and `PROJECT_BRIEF.md` §6 item 15 say why
this shape is here.

Not traced yet: there is no case, and the abstract's defaults (`aeiou dry-run hdf5_map.ast.json
--gpus 1`) stand for it. Its corpus is a container dataset that `aeiou datagen` writes, built
with the cargo feature `formats` as the release is, its headers from `aeiou-header` found
beside the binary (`aeiou-datagen`, the Python writer, writes it too):

```
aeiou datagen hdf5_map.ast.json --root /mnt/sut/aeiou [--param k=v]…
aeiou run hdf5_map.ast.json --root /mnt/sut/aeiou --gpus 2 [--param k=v]…
```
