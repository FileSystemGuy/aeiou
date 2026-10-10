# Workloads

Every committed abstract, with what was learned from the application it stands for: one
directory per I/O mechanism, under its category (`DESIGN_REVIEW.md` §3.66).

```
workloads/<category>/[<access>/]<mechanism>/
  README.md            what the application does, what storage sees, what its traces taught
  <ast_name>.py        the authoring script (aeiou-build writes the AST beside it)
  <ast_name>.ast.json  the abstract, generated, never edited by hand
  kit/                 the capture and fitting scripts of the traced application
  params/              test-sized parameter sets: fixtures, not workloads
  cases/<case>/
    README.md          the load, its kind, the GPU counts its parameters hold at
    params.json        the parameter set (writer.params.json and reader.params.json for a pair)
    captures/<capture>/
      trace.metrics.json, trace.tolerances.json, logs, fitted.params.json
```

- **A directory is a mechanism, named for what storage sees, never for a use case.** Image
  and audio classification over one file per sample through a map-style loader are one
  mechanism with two cases; a new directory means a new I/O shape. The AST's name is the
  directory's last part with underscores.
- **A case is a workload a user would ask for**: an AST and a parameter set. An *example
  case* is small enough to trace and shows the abstract issues the application's calls; a
  *use case* describes a real job and holds at the GPU counts it declares. Every case so far
  is an example case.
- **A capture** is one traced run under its case: the metrics of the `strace`, not the trace
  (the raw traces are scratch captures, not kept), the tolerance file `aeiou-trace compare
  --judge` reads, the application's logs a fit reads, and the parameters fitted to that run
  when they differ from the case's. No file over about 1 MiB.
- Categories: `training`, `inference`, `checkpointing`, `vdb`, `kvcache`, and `synthetic`
  (benchmarks that are not applications, such as IO500; none yet).

## What is here

| Mechanism | AST | Application | Cases |
|---|---|---|---|
| `training/map/file-per-sample` | `file_per_sample` | PyTorch `DataLoader`, one file per sample (`ImageFolder`) | `imagefolder-jpeg` |
| `training/map/npy-per-sample` | `npy_per_sample` | `np.load(path)["x"]` per sample through a `DataLoader` | `npy-8mib`, `npy-140mib` |
| `training/map/hdf5` | `hdf5_map` | h5py, one row of a container per sample | none yet |
| `training/stream/tfrecord` | `tfrecord_stream` | `tf.data` over TFRecord shards | none yet |
| `training/stream/parquet` | `parquet_stream` | pyarrow over Parquet shards | none yet |
| `checkpointing/dcp-save` | `dcp_save` | `torch.distributed.checkpoint.save` | `toy-6-items` |
| `checkpointing/dcp-load` | `dcp_load` | `torch.distributed.checkpoint.load` | `mixed-9-items`, `small-last-6-items`, `name-order-17-items` |
| `inference/safetensors-load` | `safetensors_load` | `from_pretrained` over safetensors shards (`mmap`) | `gpt2-random` |
| `vdb/faiss-ivf-ondisk` | `faiss_ivf_ondisk` | FAISS `IndexIVFPQ` over `OnDiskInvertedLists` (`mmap`) | `sift1m-ivf1024-pq32` |
| `vdb/diskann-search` | `diskann_search` | DiskANN `PQFlashIndex` (`libaio`, `O_DIRECT`) | `sift1m-nocache`, `sift1m-cache` |
| `vdb/diskann-build` | `diskann_build` | DiskANN `build_disk_index` | `sift1m-13-shards` |
| `kvcache/lmcache-local-disk` | `lmcache_local_disk` | vLLM with LMCache's local-disk backend | `sharegpt-chat`, `agentx`, `agentx-odirect` |
| `kvcache/lmcache-shared-fs` | `lmcache_shared_fs`, `lmcache_shared_fs_reader` | vLLM with LMCache's `fs://` backend: a writer, then a cold reader | `sharegpt-chat` |

## Adding one

A new mechanism: a directory under its category, the script, `aeiou-build --hermetic --twice`
on it, a README. A new case: `cases/<case>/` with its `params.json` and README. A capture:
`captures/<capture>/` under the case it was traced as, and a row in `CAPTURE_PAIRS` in
`builder/tests/test_trace.py` with the verdict `compare --judge` gives. CI rebuilds every AST
and checks every parameter file against its abstract.

Traces we may not publish stay out of this tree, pointers to them included.
