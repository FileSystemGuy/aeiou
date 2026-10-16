"""TEST FAKE of `faiss.contrib.ondisk` (see `standins/standin.py`): `merge_ondisk` adds the shards'
lists into the trained index, stores the lists file's name as given, and writes that file (the
lists back to back, `code_size + 8` bytes a vector)."""
import faiss
from standin import record


def merge_ondisk(trained_index, shard_fnames, ivfdata_fname):
    record("merge_ondisk", shards=shard_fnames, ivfdata=ivfdata_fname)
    for name in shard_fnames:
        trained_index.counts = [a + b for a, b in zip(trained_index.counts, faiss.read_index(name).counts)]
    trained_index.ivfdata = ivfdata_fname
    with open(ivfdata_fname, "wb") as f:
        f.write(bytes(trained_index.ntotal * (trained_index.code_size + 8)))
