# Paper Abstracts for the Target Workloads

Status: first drafts, 2026-09-29. This is open item 1 of `PROJECT_BRIEF.md` §6 and Spike 4 of
`NAPKIN_MATH.md` §6: write the target abstracts by hand, on paper, before any parser or VM
exists, to find out whether the semantic model of `GRAMMAR_OPTIONS.md` §2 and §5.2 can express
them (risk R4). The constructs the exercise surfaced are collected in §9; ~~they are proposals,
not decisions~~ **all nine were accepted on 2026-09-30** (two with qualifications, §9.6 and
§9.7) and are carried by the AST schema in `schema/abstract-ast.schema.json`.

**Provenance of the skeletons.** No `strace` of a real application has been captured for this
project yet. The syscall skeletons below are derived from how the applications are built
(CPython's `io` module, PyTorch's `DataLoader`, `torch.save`/`torch.load`,
`torch.distributed.checkpoint`, `safetensors`, DiskANN, FAISS, LMCache/vLLM). Details a trace
must confirm are tagged **[verify]**; quantities that must be measured on the real system are
tagged **[measure]**. §11 gives the capture plan. Until the traces exist, these abstracts are
shapes with named slots (`GRAMMAR_OPTIONS.md` Option D, "three sources fill the AST's slots").

## 0. Notation

Paper notation, in the style of `GRAMMAR_OPTIONS.md` Option A. It is not a committed syntax:
the AST is the contract and the Python builder (Option D) is the expected authoring form. §10
translates one abstract into builder form as a check.

```
param x = v                 named parameter, CLI-overridable
param xs = [ … ]            parameter array, indexed by a loop index: xs[t]
dataset d = files(pattern, count, size, seed)     files named by pattern, sizes by distribution
  … , chunk = c              each file realized as ceil(size/c) block objects (§9.7)
dataset d = regions(file, count, slot, size, seed) fixed-slot regions inside one file (§9.6)
namespace n = objects(pattern, size)              workload-created objects, no count (§9.7)
for i in N { … }            loop; binds i, which is part of every RNG key inside
for i in a .. b { … }       loop over [a, b)
parallel(W) { … }           W concurrent sub-actors; join at the closing brace
loader(...) { … }           PyTorch-style: workers × prefetch, ordered channel, finite
take(ch)                    wait for the next item of a channel/loader
barrier(global)             all GPUs of the run
every N { … }               when the enclosing loop index is a multiple of N
when (cond) { … }           cond over loop indices, params, and the actor id only
let x = expr                binding
consume(d) / pick(d, dist)  draw a file without / with replacement
draw(dist)                  a positional random value
file("pattern")             a named object outside any dataset (checkpoint, KV block)
size(f)                     the file's size, known from its dataset definition
op(args)[n]                 repeat n times (anonymous loop index)
read(f, len)[until_eof]     read from the current offset until exhausted, including the
                            terminating short or zero-length read the app would issue (§9.2)
op(..., expect = [E])       the op may return errno E; counted, not fatal (§9.3)
compute(t)                  emulated compute, t a constant, distribution, or expression
phase("name") { … }         label for statistics buckets (§9.8)
x @ i                       the value binding x takes at index i of its loop (§9.5)
gpu                         this actor's global GPU id
```

Ops are POSIX-shaped and identical for every backend (`PROJECT_BRIEF.md` §4). Local-only ops
(`lseek`, `ioctl`, and usually `fstat`) never reach the wire, but they cost client CPU and are
part of the real application's per-file sequence, so they are kept. §8 lists the op vocabulary
and what each op costs on NFS.

---

## 1. Small-file training (one sample per file)

**Real application.** PyTorch `DataLoader` over an `ImageFolder`-style dataset: `W` worker
processes, each building whole batches of `B` samples by reading `B` files sequentially with
blocking calls; `prefetch_factor` batches in flight per worker; the main process takes batches
in order and runs the step. Each sample is a JPEG opened with PIL through Python's `open()`.

**Per-sample syscall skeleton** (one worker process, one file) ~~**[verify]**~~ **(traced
2026-10-01, see "Trace" below; the second `lseek` was added from it)**:

```
openat(AT_FDCWD, "…/img_000123.jpg", O_RDONLY|O_CLOEXEC) = 5
fstat(5, {st_size=131072, st_blksize=…})     # FileIO: not-a-directory check; BufferedReader buffer size
ioctl(5, TCGETS, …) = -1 ENOTTY               # io.open isatty() check; local
lseek(5, 0, SEEK_CUR) = 0                     # BufferedReader.tell(); local
lseek(5, 0, SEEK_SET) = 0                     # PIL Image.open: fp.seek(0); local   (added 2026-10-01)
read(5, …, st_blksize) = 131072               # first buffer fill; short at EOF
read(5, …, st_blksize) = 0                    # decoder asks for more; EOF   (present on every file)
close(5) = 0
```

Two details matter on NFS. First, `st_blksize` on an NFS file is the mount's I/O size, typically
1 MiB, not 4 KiB (traced: `st_blksize` 1048576 at rsize 1 MiB, and every `read` asks for 1048576), so Python's "small" buffered reads request 1 MiB and a 128 KiB
file is read with one short read. Second, which of these reach the wire (LOOKUP, OPEN, GETATTR,
READ ×⌈S/rsize⌉, CLOSE) depends on the attribute cache, delegations, and directory-entry cache;
the abstract does not assume, the run reports `mountstats`.

**Abstract.**

```
workload train_small_files {
  param batch      = 32
  param workers    = 8
  param prefetch   = 2
  param steps      = 500
  param sync_every = 500          # real DDP all-reduces every step (=1); see Cuts
  param step_time  = 105ms        # [measure] GPU step time on the target accelerator
  param hdr_read   = 1MiB         # Python BufferedReader request = st_blksize (traced 2026-10-01)
  param enumerate  = false        # include the startup directory walk (§9.8)

  dataset train = files("train/{id div 1300:05}/img_{id:09}.jpg",
                        count = 50_000_000,
                        size  = lognormal(median = 110KiB, sigma = 0.45),   # [measure] corpus
                        seed  = 0x5eed_da7a)

  per gpu {
    when ($enumerate) {
      phase("enumerate") {                                # every actor walks: ImageFolder has no broadcast
        for d in dirs(train) {                            # 38,462 class directories
          stat(d),                                        # make_dataset: os.path.isdir   (2026-10-01)
          open(d, RDONLY|CLOEXEC|DIRECTORY), fstat(d),    # os.scandir                    (2026-10-01)
          readdir(d)[until_end], close(d)
        }
      }
    }

    loader batches(workers = $workers, prefetch = $prefetch, order = in_order,
                   batches = $steps) {
      for j in $batch {
        let f = consume(train)                            # position = gpu + G·(b·batch + j)
        open(f, RDONLY|CLOEXEC), fstat(f), ioctl(f, TCGETS, expect = [ENOTTY]),
        lseek(f, 0, CUR), lseek(f, 0, SET),
        read(f, $hdr_read)[until_eof],                    # 1 short read + 1 EOF read for S < hdr_read
        close(f)
      }
    }

    phase("train") {
      for step in $steps {
        take(batches),
        compute($step_time),
        every $sync_every { barrier(global) }
      }
    }
  }
}
```

**Parameters and sources.**

| Slot | Source |
|---|---|
| `batch`, `workers`, `prefetch` | configuration of the real job |
| `step_time` | measured on the accelerator; a distribution if it varies |
| `hdr_read` | `strace`: the length argument of the first `read` (1 MiB on the traced mount) |
| size distribution | the corpus (fit from `find -printf %s` or the manifest of the reference dataset) |
| `sync_every` | 1 for DDP; 500 keeps the brief's reference workload |

**Cuts (stated fidelity losses).**
- Worker→main IPC through `/dev/shm` (the collated batch tensor is a shared-memory file) is
  client-local I/O that never reaches the SUT. Dropped; its memory-bandwidth cost is folded
  into the sink-buffer rule of `NAPKIN_MATH.md` §2.2.
- Decode CPU time in the worker is not modeled; the worker is I/O-bound in the abstract. If a
  trace shows the decode gap between `close` and the next `openat` matters for the offered
  concurrency, add `compute($decode)` after `close`. **The trace (2026-10-01) shows the gap:
  about 3 ms of CPU per file, most of it between the EOF read and `close`** (the file is
  open during the decode). It does not change the offered rate while the step time is the
  limit (prefetch back-pressure holds the workers), and it does when the loader is: a
  worker then issues a file every 3 ms, not back to back. Not added (decided
  2026-10-01, `DESIGN_REVIEW.md` §3.43).
- The `enumerate` phase is what the real application does at startup. **Every actor runs the
  walk** (added 2026-09-29): `ImageFolder` is a plain constructor that `os.walk`s the tree in
  the process that builds the dataset, and under DDP every rank builds its own; nothing is
  broadcast, and `DistributedSampler` shards indices on the assumption that every rank holds the
  same sorted list. DataLoader workers are forked after the list exists and inherit it, so they
  issue no walk. Hence G full walks of 38K directories, one per actor, which is why the phase is
  inside `per gpu` and not under `when (gpu == 0)`. On one client node the G walks race over the
  same directories in the same order, so the server sees roughly one walk per node while the
  client pays dentry/inode slab and the file list G times; the runner reproduces the former by
  issuing real `getdents64` per actor and does not reproduce the list's RAM. A pipeline that
  lists on rank 0 and broadcasts is modeled by wrapping the phase in `when (gpu == 0)` (§9.1).
  Whether the walk is part of the CLOSED measurement is a WG policy question (`PROJECT_BRIEF.md` §8); here it is a
  separately reported phase, off by default.

**Trace (2026-10-01).** `builder/traces/train_small_files`: `ImageFolder` + `DataLoader`,
batch 16, 2 workers, 2 epochs over 3,200 real JPEGs (median 111 KiB) on the loopback NFS
mount (v4.2, rsize 1 MiB); torch 2.14.1, torchvision 0.29.1, Pillow 12.3.0, Python 3.12.3.
Reasoning in `DESIGN_REVIEW.md` §3.43.

- **The skeleton holds, with one call missing from the draft:** Pillow's `Image.open` seeks
  to 0 before it reads, so there are two `lseek`s per file. Added to the abstract. The read
  request is `st_blksize` = 1 MiB and the EOF read happens on every file.
- **The walk** is, per class directory, `stat` (path), `open`, `fstat`, `getdents64` until
  empty, `close`; the draft had no `stat` and no `fstat`. Added. The real `open` also sets
  `O_NONBLOCK`, which a directory ignores and the contract's flag list does not have. One
  listing of the dataset root (`find_classes`) precedes the walk: 4 ops per actor, not
  modeled.
- **After those two changes `aeiou-trace compare` at the fitted parameters** shows every op
  share, the request sizes, and popularity equal; run length in bytes at distance 0.015
  (the JPEG corpus is near the lognormal, not drawn from it) and reuse distance at 0.048
  (the order). The op counts are equal but for the root listing (51,219 against 51,215).
- **On the wire** (`mountstats` delta, 6,400 opens of 3,200 files): READ 3,200, so one READ
  per file and none in the second epoch (the corpus fits the page cache: the dataset-size
  rule, not the abstract, decides whether an epoch reaches the server); OPEN + OPEN_NOATTR
  5,214, CLOSE 4,028, GETATTR 3,212, READDIR 8. The opens the server did not see were
  covered by read delegations.
- **The worker is not I/O-bound.** The file stays open while Pillow decodes: `close` comes
  after `convert("RGB")`. Untraced, the run took 13.4 s wall and 22.4 s user for 6,400
  files on two workers, about 3 ms of CPU per file, against about 0.3 ms inside the file's
  calls. The transform and collate fall between `close` and the next `openat`. See Cuts.

**What it stresses.** Metadata: LOOKUP/OPEN/CLOSE per 110 KiB, client dentry/inode slab growth
(`NAPKIN_MATH.md` §2.3), attribute-cache and delegation behaviour, and how the client copes when
each read request is 8× the file. Client CPU per op is the number to watch.

**Constructs used.** loader sugar, consume with positional sharding, `until_eof` with a short
read, `expect` on `ioctl`, `dirs(ds)` and `readdir` for the optional walk, `phase`, `when` on a
boolean parameter.

---

## 2. Large-sample training (one large file per sample)

**Real application.** The MLPerf Storage `unet3d` shape: `np.load` on a `.npz` per sample,
about 140 MiB. A `.npz` is a zip archive, so the read pattern is not sequential from offset 0:
the tail is read first (end-of-central-directory record), then the central directory, then each
member. NumPy reads a non-seekable member in fixed chunks.

**Per-sample syscall skeleton** (**traced 2026-10-01** and rewritten from the trace; the
drafted skeleton, which read two equal members by header and had no read at offset 0, is
in the history before that date. See "Trace" below.):

```
openat(…, O_RDONLY|O_CLOEXEC); fstat; ioctl(TCGETS)=ENOTTY; lseek(0, SEEK_CUR)
read(fd, …, blksize) = blksize                                    # np.load peeks the magic: a buffer fill at 0
lseek(fd, 0, SEEK_END); lseek(fd, 0, SEEK_CUR)                    # zipfile._EndRecData
lseek(fd, -22, SEEK_END); read(fd, …, blksize) = 22               # EOCD record
lseek(fd, size-42, SEEK_SET); read(fd, …, blksize) = 42           # zip64 locator probe
lseek(fd, size-22-cd_len, SEEK_SET); read(fd, …, blksize) = cd_len+22   # central directory
lseek(fd, 0, SEEK_SET)                                            # local header of "x"
read(fd, …, blksize) × ⌈size / blksize⌉                           # front to back; the last is short, no EOF read
lseek(fd, 0, SEEK_CUR) × (⌈(size − framing) / 256 KiB⌉ + 4)       # between the reads: one tell per NumPy chunk
close(fd)
```

NumPy asks the zip member for 256 KiB at a time; the `BufferedReader` beneath fills
`st_blksize` (1 MiB on the traced mount), so each fill serves four chunks, and each chunk
costs one `lseek(0, SEEK_CUR)` (the buffered seek asks the raw position). The first
megabyte is read twice, once for the magic and once as data. Upstream DLIO's reader indexes only
`"x"`; `"y"` (a label list, 191 bytes with its headers) sits between `x` and the central
directory and is fetched only as part of the last buffer fill. `framing` is the 498 bytes of
the archive that are not `x`'s data.

**Abstract.**

```
workload train_large_samples {
  param batch = 7, workers = 4, prefetch = 2, steps = 500, sync_every = 500
  param step_time = 323ms                 # [measure]
  param hdr_read  = 1MiB                  # st_blksize (traced)
  param xfer      = 1MiB                  # buffer fill under the member read: st_blksize (traced)
  param np_chunk  = 256KiB                # NumPy's read chunk; one tell each
  param cd_len    = 102                   # central directory: "x.npy", "y.npy" (traced)
  param framing   = 498                   # archive bytes that are not x's data (traced)

  dataset train = files("train/{id div 10000:05}/sample_{id:09}.npz",
                        count = 50_000,
                        size  = normal(140MiB, 4MiB),        # [measure]
                        seed  = 0x5eed_da7b)

  per gpu {
    loader batches(workers = $workers, prefetch = $prefetch, order = in_order,
                   batches = $steps) {
      for j in $batch {
        let f = consume(train)
        open(f, RDONLY|CLOEXEC), fstat(f), ioctl(f, TCGETS, expect = [ENOTTY]), lseek(f, 0, CUR),
        read(f, $hdr_read),                                         # magic: buffer fill at 0
        lseek(f, 0, END), lseek(f, 0, CUR),
        lseek(f, -22, END), read(f, $hdr_read),                     # EOCD
        lseek(f, size(f) - 42, SET), read(f, $hdr_read),            # zip64 locator probe
        lseek(f, size(f) - 22 - $cd_len, SET), read(f, $hdr_read),  # central directory
        lseek(f, 0, SET),
        for k in chunks(f) / per_fill {                             # fills NumPy drains whole
          read(f, $xfer), for t in per_fill { lseek(f, 0, CUR) }
        }
        read(f, $xfer)[ceil(size(f) / $xfer) - chunks(f) / per_fill],   # the rest, to EOF
        for t in chunks(f) % per_fill + 4 { lseek(f, 0, CUR) },
        close(f)
      }
    }
    for step in $steps {
      take(batches), compute($step_time), every $sync_every { barrier(global) }
    }
  }
}
```

`per_fill = $xfer / $np_chunk` and `chunks(f) = ceil((size(f) − $framing) / $np_chunk)` are
builder-side helpers over `size(f)` (§9.4). The counts are exact per file: `4 + ⌈size/xfer⌉`
reads and `11 + chunks(f)` seeks, checked against every file of both traced corpora.

**Parameters and sources.** As §1, plus `xfer`, `np_chunk`, `cd_len`, and `framing` from
the trace; `step_time` from the accelerator. ~~`members` and `lh_len`~~ are gone (2026-10-01):
the reader takes one member and never reads a local header as a request of its own.

**Trace (2026-10-01).** `builder/traces/train_large_samples`: upstream DLIO's read call (argonne-lcf
`npz_reader.py`; the MLCommons fork reads differently and is not modeled, `DESIGN_REVIEW.md`
§3.46) through a
`DataLoader`, 2 workers, 2 epochs, on the loopback NFS mount, over corpora written the way
DLIO's generator writes them; NumPy 2.5.2, Python 3.12.3. Reasoning in `DESIGN_REVIEW.md`
§3.44.

- **16 files of about 140 MiB:** every op share, request size, run length, and popularity
  within 0.002 of the abstract at the fitted parameters; reuse distance at 0.27, which
  sixteen files cannot settle (two seeds of the abstract differ by 0.12).
- **256 files of about 8 MiB:** largest distance 0.040, the reuse distance, inside the 0.062
  between two seeds of the abstract.
- **80 % of the calls are `lseek`**, all local. They cost client CPU and no wire op.
- **On the wire** (140 MiB corpus, `mountstats` delta): 2,260 READs for 2.35 GB, one pass
  over each file at about 1 MiB per READ; the second epoch came from the page cache. 2,247
  GETATTRs, about one per READ; the cause was not looked into. 16 OPEN_NOATTR, no CLOSE.

**Cuts.** None specific. The `.npz` may be stored uncompressed (`np.savez`) or deflated
(`np.savez_compressed`); only the uncompressed layout has this shape, and the reference dataset
must use it.

**What it stresses.** Bandwidth, and the client's readahead heuristics: a tail read followed by
a jump to the front and a long sequential run is exactly the pattern that makes a naive
readahead window thrash and a good one detect the run late. Under O_DIRECT the wire pattern is
the abstract's `xfer`, which is the intended way to make the wire pattern explicit
(`NAPKIN_MATH.md` §8.B).

**Constructs used.** `lseek` with `END` and with an offset expression over `size(f)`; repeat
counts from file metadata; builder-side helper functions.

---

## 3. Checkpoint write

**Real application.** Two shapes are in use. (a) `torch.save(state_dict, path)` per rank: one
zip archive per rank, written with many small metadata writes and one large write per tensor
storage, **no `fsync`** (durability is whatever `close` gives, which on NFS is a flush plus
COMMIT under close-to-open). (b) `torch.distributed.checkpoint.save` (DCP) with
`FileSystemWriter`: every rank writes `__{rank}_0.distcp`, each write item serialized as a small
`torch.save` stream into that file, `fsync` per file (`sync_files=True`), then a collective
gathers the plan and the coordinator writes `.metadata` via a temp file and `rename`. Both run
inside the training loop, so the loader keeps prefetching while the checkpoint is written.

**Size.** With Adam in mixed precision the state is ≈14 B/param (bf16 weights + fp32 master +
two fp32 moments), so an 8B model is ≈112 GB per checkpoint across the job, ≈14 GB per GPU at
G = 8. The per-rank write-item list is the model's parameter table: FSDP2/DTensor gives one item
per parameter and per optimizer state (≈900 items for an 8B model), each `1/G` of the tensor;
FSDP1 flat-params give a handful of very large items. Both are configuration, not trace.

**DCP per-rank syscall skeleton** (**traced 2026-10-01**, torch 2.14.1; the drafted one had
a guessed 64 KiB of small records per item and no seeks; see "Trace" below):

```
stat(parent); access(parent, W_OK)                      # not modeled
mkdir("ckpt/step_000100", 0777) = 0 | -1 EEXIST          # every rank; the loser then stats the directory
stat("ckpt/step_000100/.metadata") = -1 ENOENT           # is there a checkpoint here already
openat("ckpt/step_000100/__3_0.distcp", O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC, 0666)
fstat; ioctl(TCGETS)=ENOTTY; lseek(0, SEEK_CUR)
for each write item:                                    # torch.save(tensor, stream)
    lseek(0, SEEK_CUR)                                  # tell: the item's offset
    item above the buffer:   write(fd, …, 704); write(fd, …, item_bytes); write(fd, …, 873)
    item within the buffer:  write(fd, …, 704 + item_bytes + 873)
    lseek(0, SEEK_CUR)                                  # tell: the item's length
fsync(fd)
close(fd)
-- collective: gather WriteResults to coordinator (network, not storage) --
rank 0: openat(".metadata.tmp", O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC); fstat; ioctl; lseek(0, SEEK_CUR);
        write(…, meta_bytes); fsync; close;
        stat(".metadata") = -1 ENOENT; rename(".metadata.tmp", ".metadata")
barrier
```

Each item is its own `torch.save` stream: 704 bytes before the storage (zip local headers,
`data.pkl`, padding to 64), 873 after it (version, byte order, central directory, end
record). Python's `BufferedWriter` holds `st_blksize` (1 MiB on the traced mount): an item
larger than that goes to the kernel in one `write`, with its header and trailer as two
small writes around it; a smaller one leaves as one coalesced write. The stream is flushed
at every item, so nothing is coalesced across items.

**Abstract** (DCP shape; inserted into the training loop of §1 or §2):

```
  param ckpt_every = 100
  param items      = 900                         # write items per rank [config]
  param item_bytes = [ … ]                       # per-item bytes, 1/G of each tensor [config]
  param hdr        = 704                         # per item, before the storage (traced)
  param trailer    = 873                         # per item, after the storage (traced)
  param buf        = 1MiB                        # BufferedWriter = st_blksize (traced)
  param meta_bytes = 2MiB                        # .metadata size [measure]; about 217 B per item and rank

  namespace ckpt = objects("ckpt/step_{step:06}/{name}", size = as_written)

  # inside `for step in $steps { … }`, after compute:
  every $ckpt_every {
    barrier(global)                                         # collective before save
    phase("ckpt_write") {
      let dir = file("ckpt/step_{step:06}")
      mkdir(dir, expect = [EEXIST]),
      stat(file("ckpt/step_{step:06}/.metadata"), expect = [ENOENT]),
      let c = file("ckpt/step_{step:06}/__{gpu}_0.distcp"),
      open(c, WRONLY|CREAT|TRUNC|CLOEXEC), fstat(c), ioctl(c, TCGETS, expect = [ENOTTY]),
      lseek(c, 0, CUR),
      for t in $items {
        lseek(c, 0, CUR),
        when ($item_bytes[t] > $buf) { write(c, $hdr), write(c, $item_bytes[t]), write(c, $trailer) }
        else                         { write(c, $hdr + $item_bytes[t] + $trailer) },
        lseek(c, 0, CUR)
      },
      fsync(c), close(c),
      barrier(global),                                      # gather WriteResults
      when (gpu == 0) {
        let m = file("ckpt/step_{step:06}/.metadata.tmp"),
        open(m, WRONLY|CREAT|TRUNC|CLOEXEC), fstat(m), ioctl(m, TCGETS, expect = [ENOTTY]), lseek(m, 0, CUR),
        write(m, $meta_bytes), fsync(m), close(m),
        stat(file("ckpt/step_{step:06}/.metadata"), expect = [ENOENT]),
        rename(m, file("ckpt/step_{step:06}/.metadata"))
      },
      barrier(global)
    }
  }
```

**Trace (2026-10-01).** `builder/traces/ckpt_write_dcp`: `dcp.save` on two ranks (gloo, CPU),
six `DTensor` items of 16 MiB down to 2 KiB per rank, two checkpoints, on the loopback NFS
mount; torch 2.14.1. Reasoning in `DESIGN_REVIEW.md` §3.45.

- **Every call on the shard files and on `.metadata` is in the abstract**, with equal
  counts, equal bytes written (304,138,284), and equal write-size buckets. Largest distance
  0.046, the share of `stat`.
- **Not modeled, 10 of 164 calls:** per rank and checkpoint a `stat` and an `access` of the
  step directory's parent (no `access` op; no handle for a namespace's parent); the `stat`
  of the directory by the rank that lost the `mkdir` (who loses is timing); once in the
  job, the `mkdir` of the absent parent.
- **Each rank is an instance.** With the whole trace as one instance the other rank's
  writes fall between a rank's own and the reuse distance is off by 0.92; with
  `--instance-root` per rank it is equal.
- **On the wire** (an untraced repeat, `mountstats` delta): 294 WRITEs for 304 MB, about
  1 MiB each; 4 COMMITs, one per shard `fsync` (the two `.metadata` fsyncs sent none);
  6 OPEN, 3 CREATE, 2 RENAME, no CLOSE.
- **The writes are not aligned** (704, the item, 873), so the abstract at its traced
  parameters does not run under a `-direct` backend: the runner refuses unaligned direct
  writes, as the kernel would refuse Python's. See Cuts.
- **`torch.save` on one rank** (`save.py --torch-save`, the first variant below) is another
  writer: a C++ stream, `open` without `O_CLOEXEC` and with no `fstat`/`ioctl`/`lseek`, one
  `writev` per tensor of a 64-byte record header and the storage (1,088 bytes before the
  first), small tensors and the central directory in one last `write` (5,373 bytes), no
  `fsync`.

Variants, each a small edit:
- **`torch.save` per rank:** drop `mkdir` and the metadata step, drop `fsync`, the `fstat`,
  the `ioctl`, and the seeks; ~~the per-tensor loop becomes `write(c, $hdr),
  write(c, $item_bytes[t]), write(c, pad)`~~ the per-tensor loop is one `writev` of 64 bytes
  and the storage (traced 2026-10-01; a `write` of `64 + $item_bytes[t]` in the abstract,
  which has no vectored op) with the archive tail once at the end. The measured quantity is then the `close` latency (flush + COMMIT).
- **Asynchronous checkpoint** (`torch.distributed.checkpoint.async_save`): the write runs on a
  background thread while training continues, and the next checkpoint waits for the previous
  one. Expressible as `parallel(1)` forked at the checkpoint step with the join at the next
  `every $ckpt_every` (a `channel(capacity = 1)` between the trainer and a writer sub-actor).
  The loader keeps running in both variants; only the trainer's `compute` overlaps differently.
- **Multiple files per rank** (`thread_count > 1`): `parallel($threads) { … __{gpu}_{i}.distcp }`.
- **Read-back** (revised 2026-09-29; off by default since 2026-09-30): after the final
  barrier, each rank reads its own file back with `read(c, 1MiB)[until_eof]`, under
  `param readback = false`. It is a durability check in the sense that the bytes come back,
  not a content check (content verification is the separate tool, `PROJECT_BRIEF.md` §5), and
  on the writing node it measures the page cache, so a benchmark run leaves it off: the
  restore is the separate §4 run on other nodes (`DESIGN_REVIEW.md` §3.24). Reported as its
  own phase when on.

**Cuts.** The collective (NCCL/Gloo) that gathers the write plan is network, not storage, and
becomes a `barrier`. ~~`O_DIRECT` checkpoint writers (some vendor plugins) are a backend choice,
not an abstract change.~~ **Revised 2026-10-01:** an `O_DIRECT` checkpoint writer is another
application. The traced writer's requests are 704 bytes, the item, and 873 bytes, none
aligned; a writer that uses `O_DIRECT` has to buffer and pad differently, and that call
stream is its own abstract (or this one with `hdr` and `trailer` set to aligned values,
which is then a statement about that writer, not about PyTorch's).

**What it stresses.** Write bandwidth with G concurrent large sequential streams, `fsync`/COMMIT
latency, and the `rename` and `mkdir` metadata path. With buffered writes, how much of the
14 GB the client can absorb as dirty pages before the writer blocks decides the shape of the
stall; with `sync-direct`, each `write` waits for a stable write.

**Constructs used.** `mkdir`, `rename`, `fsync` ops; `expect = [EEXIST]`; `when (gpu == 0)`
(conditional on the actor id, §9.1); parameter arrays; `{step:06}` and `{gpu}` in names;
workload-created `namespace` (§9.7); `phase`.

---

## 4. Checkpoint restore and model load

Two read shapes, distinguished by how many ranks read the same bytes.

**4a. Restore into the same topology** (DCP `load`, or `torch.load` per rank). Each rank reads
`.metadata` (everyone reads the same small file), then its own shard file. With data-parallel
replication whose state is saved once (DDP, HSDP with DCP's deduplication), the `replicas`
ranks that share a shard all read the same file, so each file has fan-in `replicas`; fully
sharded training (FSDP, ZeRO-3) has `replicas = 1`. The files are the ones a `ckpt_write_dcp`
run wrote: the namespaces are declared `input` and the run that reads them may not modify them
(`schema/README.md` V14, `DESIGN_REVIEW.md` §3.24).

**Syscall skeleton, DCP `FileSystemReader`** **[verify]**:

```
openat(".metadata", O_RDONLY); fstat; ioctl; lseek; read(…, blksize)…; close      # every rank
openat("__3_0.distcp", O_RDONLY); fstat; ioctl; lseek
for each read item (this rank's slice):
    lseek(fd, item_offset, SEEK_SET)
    read(fd, …, blksize)…                            # torch.load on a file view: EOCD, central dir
    lseek(…); read(fd, …, item_bytes)                # storage; one large read
close(fd)
```

**Abstract.**

```
workload ckpt_restore {
  param items = 900, item_bytes = [ … ], item_off = [ … ]      # [config], prefix sums of §3
  param replicas = 1                                          # ranks reading the same shard file
  param meta_bytes = 2MiB, hdr_read = 1MiB, lh_len = 40
  param restore_step = 100

  per gpu {
    phase("restore") {
      let m = file("ckpt/step_{restore_step:06}/.metadata"),
      open(m, RDONLY), fstat(m), read(m, $hdr_read)[until_eof], close(m),
      let c = file("ckpt/step_{restore_step:06}/__{gpu div $replicas}_0.distcp"),   # fan-in
      open(c, RDONLY), fstat(c), ioctl(c, TCGETS, expect = [ENOTTY]), lseek(c, 0, CUR),
      for t in $items {
        read(c, offset = $item_off[t], $hdr_read),                         # zip tail of the item
        read(c, offset = $item_off[t] + $lh_len, $item_bytes[t])
      },
      close(c),
      barrier(global)
    }
  }
}
```

**4b. Model load for serving** (Hugging Face `safetensors` shards, vLLM/TGI). Every process
reads the small `config.json` and `model.safetensors.index.json`, then for **every** shard file:
`open`, `read(8)` header length, `read(hdr)` JSON header, `mmap` the file, and touch the byte
range of each tensor it needs. Tensor-parallel rank `tp` needs a `1/TP` slice of each tensor:
contiguous for column-parallel weights, strided for row-parallel ones. The result is that all G
processes read all shards (fan-in G), each touching a different subset of pages.

```
workload model_load {
  param shards = 4, shard_bytes = 5GiB                          # [config]
  param tensors = [ (shard, off, bytes, split, rows, row_bytes) … ]   # table from the index json [config];
                                                # columns referenced as $off[t], $bytes[t], …; tensors_in(s) filters by shard
  param tp = 8

  dataset model = files("model-{id+1:05}-of-{shards:05}.safetensors", count = $shards,
                        size = const($shard_bytes), seed = 0x5eed_da7c)

  per gpu {
    phase("load") {
      for s in $shards {
        let f = file(model, s),
        open(f, RDONLY|CLOEXEC), fstat(f), read(f, 8), read(f, $hdr_len[s]),
        for t in tensors_in(s) {
          when ($split[t] == column) {
            read(f, offset = $off[t] + (gpu mod $tp) * $bytes[t] / $tp, $bytes[t] / $tp)
          }
          when ($split[t] == row) {
            for r in $rows[t] {                                   # strided: one piece per row
              read(f, offset = $off[t] + r * $row_bytes[t] + (gpu mod $tp) * $row_bytes[t] / $tp,
                   $row_bytes[t] / $tp)
            }
          }
        },
        close(f)
      },
      barrier(global)
    }
  }
}
```

Run with `--io-backend mmap`, each `read(f, off, len)` becomes a populate of that range and the
kernel's fault-around decides the RPC sizes, which is the point of having the backend: the
abstract states what the application touched, the run reports what went over the wire.

**Added 2026-09-30.** The builder form has a third `split`, `full`: norms and biases are
replicated, and every rank reads them whole. The table is a parameter file built from the real
shards by `aeiou-params safetensors` (`schema/README.md` §8); the five-tensor defaults in the
script are a stand-in. Shards are modeled at the largest shard's size (`shard_bytes`), and the
dataset pattern is `model-{id:05}.safetensors` (the names are the dataset's, not the model's).

**Cuts.** The JSON header parse and the tensor copies to the GPU are compute; `compute` nodes
can be added per tensor if a trace shows they gate the I/O. Replication of the same read by all
G processes is the real behaviour, not a cut.

**What it stresses.** Same-file fan-in: G clients reading identical bytes at the same moment
tests NFSv4 delegations, server-side caching, and (for pNFS) whether the layout spreads the hot
file. 4b with strided row-parallel slices is a page-fault storm of 4 KiB–64 KiB pieces inside
multi-GiB files. This is the "many ranks read the same files" case the design review asked for.

**Constructs used.** `gpu div dp`, `gpu mod tp` in names and offsets; `file(ds, index)` to
address a dataset file by id; parameter arrays of tuples (a table); `when` on a table entry.

---

## 5. Vector-database search, DiskANN-style (graph index on storage)

**Real application.** DiskANN keeps PQ-compressed vectors in memory and the graph plus
full-precision vectors on storage, one node (vector + neighbour list) packed into 4 KiB sectors
(several small nodes per sector, or several sectors per large node). A query is a beam search:
each round takes up to `beam` unvisited frontier nodes and reads their sectors concurrently
(`io_submit` of `beam` requests, then wait for all **[verify]**), then expands. The first rounds
hit an in-memory cache of the nodes nearest the medoid (`num_nodes_to_cache`), so their reads
never reach storage. Search threads run independent queries in parallel.

**Per-query I/O skeleton** **[verify]**: `open(index)` once per thread at startup; per query,
`H` dependent rounds of `beam` concurrent `pread(fd, 4 KiB, off)` at sector-aligned offsets; no
other I/O. Typical totals are 50–150 sector reads per query at L = 100 **[measure]**.

**Abstract.**

```
workload vdb_search_diskann {
  param threads     = 32                       # search threads per node ("gpu" = a search node)
  param queries     = 100_000                  # per thread; finite
  param beam        = 4
  param cached_hops = 2                        # rounds served from the node cache [measure]; documentation only:
  param hops        = empirical([ … ])         # `hops` holds the trace's rounds minus cached_hops [measure]
  param hot         = hotset(fraction = 0.01, weight = 0.30)   # hub nodes [measure from trace]
  param rerank      = 120us                    # [measure]

  dataset index = regions(file = "diskann/index.bin", count = 200_000_000,   # sectors
                          slot = 4KiB, size = const(4KiB), seed = 0x5eed_da7d)

  per gpu {
    parallel(threads) {
      let f = file(index), open(f, RDONLY|DIRECT),
      for q in $queries {
        for hop in draw($hops) {
          parallel($beam) {
            let s = pick(index, dist = when (hop < 1) { $hot } else { uniform })   # §9.9
            read(f, offset = offset(s), 4KiB)
          }
        }
        compute($rerank)
      },
      close(f)
    }
  }
}
```

`hop < 1` (after the cached rounds are removed) puts the hub bias on the first storage round,
where the frontier is still near the medoid; later rounds are close to uniform over sectors,
with the residual popularity skew of the hot set. The exact split is what the locality check
(§7) tunes.

**Cuts.** In-memory PQ distance computation and the node cache are compute/absent; both are
configuration of the real system. Query popularity (repeated identical queries) is not modeled;
DiskANN has no result cache.

**What it stresses.** Random 4 KiB reads at a dependency depth of `hops` with fan-out `beam`
per thread: latency-bound, not bandwidth-bound. `--io-backend libaio` is the fidelity reference
here (what DiskANN uses on Linux); `io_uring` and `sync-direct` are the comparison rows.

**Constructs used.** `regions` dataset (§9.6); `parallel` inside `for` inside `parallel`;
a `when` expression choosing a distribution by loop index; `pick` with `hotset`; a file handle
bound outside a `parallel` and shared by its sub-actors.

---

## 6. Vector-database search, IVF (inverted lists on storage)

**Real application.** FAISS `IndexIVF` with `OnDiskInvertedLists`: the coarse quantizer is in
memory; a query selects `nprobe` lists, and each list's codes and ids are read as one contiguous
region of the inverted-lists file (FAISS `mmap`s the file; a prefetch mode issues `pread`s
from a thread pool **[verify]**). List sizes are skewed (imbalance factors of 1.2–3 are common),
and list popularity is skewed because queries follow the data distribution.

**Abstract.**

```
workload vdb_search_ivf {
  param threads = 32, queries = 100_000
  param nprobe  = 64
  param list_pop = zipf(s = 0.8)               # list popularity [measure]
  param distance = 400us                        # [measure]

  dataset lists = regions(file = "ivf/invlists.bin", count = 1_000_000,
                          slot = 256KiB,                       # ≥ p99 list size
                          size = lognormal(median = 40KiB, sigma = 0.9),   # codes+ids [measure]
                          seed = 0x5eed_da7e)

  per gpu {
    parallel(threads) {
      let f = file(lists), open(f, RDONLY),
      for q in $queries {
        parallel($nprobe) {
          let c = pick(lists, dist = $list_pop)
          read(f, offset = offset(c), size(c))                  # one sequential run per list
        }
        compute($distance)
      },
      close(f)
    }
  }
}
```

**Cuts.** The real file packs lists back to back; `regions` with fixed slots pads each list to
`slot`, so addresses are spaced differently while sizes and popularity are identical (§9.6). For
`nprobe` random lists out of a million this changes nothing a prefetcher could use; it would
matter only for a scan.

**What it stresses.** Medium-sized (tens to hundreds of KiB) reads with a Zipf popularity, so
server and client caching are legitimately rewarded; fan-out `nprobe` with no dependency
between the reads. With `--io-backend mmap` it is FAISS's real path.

**Constructs used.** `regions` with a size distribution; `pick` with `zipf`; `size(c)` and
`offset(c)` on a region.

---

## 7. Index build (DiskANN)

**Real application.** Training-shaped. Phases: (1) sample the base vectors for PQ training
(random rows: many small random reads **[verify]**); (2) for each of `K` shards, read the shard's
rows sequentially, build a graph in memory (long compute), write the shard index sequentially;
(3) merge the shard indexes into one graph (sequential read of all, sequential write); (4) write
the disk layout sector by sector (`ofstream` 4 KiB writes, coalesced by the kernel) plus the PQ
files. Single process, multi-threaded compute; the I/O is one stream per phase.

**Abstract.**

```
workload vdb_build_diskann {
  param dim = 128, n = 1_000_000_000, shards = 40
  param sample = 256_000                         # PQ training rows [config]
  param build_time = normal(1800s, 60s)          # per shard, [measure]; scaled by --time-scale
  param xfer = 1MiB
  param shard_index_bytes = 12GiB, index_bytes = 480GiB, sectors = 200_000_000   # [config]

  dataset base = regions(file = "base.fbin", count = $n, slot = $dim * 4, size = const($dim * 4),
                         seed = 0x5eed_da7f)
  namespace out = objects("diskann/{name}", size = as_written)

  per gpu {                                       # one builder; --gpus 1
    let b = file(base), open(b, RDONLY),
    phase("pq_sample")  { for i in $sample { let r = pick(base); read(b, offset = offset(r), size(r)) } }
    phase("shard_build") {
      for k in $shards {
        read(b, offset = k * $n / $shards * $dim * 4, $xfer)[($n / $shards * $dim * 4) / $xfer],
        compute($build_time),
        let s = file("diskann/shard_{k:03}.index"),
        open(s, WRONLY|CREAT|TRUNC), write(s, $xfer)[$shard_index_bytes / $xfer], close(s)
      }
    }
    phase("merge") {
      for k in $shards {
        let s = file("diskann/shard_{k:03}.index"),
        open(s, RDONLY), read(s, $xfer)[until_eof], close(s)
      },
      let g = file("diskann/merged.index"),
      open(g, WRONLY|CREAT|TRUNC), write(g, $xfer)[$index_bytes / $xfer], close(g)
    }
    phase("layout") {
      read(b, $xfer)[until_eof],                                 # base re-read for full vectors
      let d = file("diskann/index.bin"),
      open(d, WRONLY|CREAT|TRUNC), write(d, 4KiB)[$sectors], fsync(d), close(d)
    }
    close(b)
  }
}
```

**Cuts.** Build compute is hours; `--time-scale` shrinks every `compute` by a factor for
benchmark runs, which is allowed because compute never gates a dependency across actors here.
Peak-memory-driven spill behaviour of the real builder is not modeled.

**What it stresses.** Large sequential read and write streams from one client, small random
reads in the sample phase, and a 4 KiB write stream that tests write coalescing. This is the
least storage-sensitive of the set; it exists so a VDB submission covers build as well as
search.

**Constructs used.** Sequential loops over phases; `pick` without replacement is not needed
(sample with replacement is what the real code does).

---

## 8. KV-cache serving (prefix cache offloaded to storage)

**Real application.** An inference engine (vLLM, SGLang) with a KV-cache offload tier
(LMCache, Mooncake Store, SGLang HiCache, Dynamo KVBM) that stores KV blocks on storage keyed by
the hash of the token prefix. Per request: the engine hashes the prompt in fixed-size chunks
(`chunk_tokens` = 16-token vLLM blocks, 64-token HiCache pages, or 256-token LMCache chunks),
looks each chunk up, **reads** the blocks of the longest cached prefix, computes the prefill
for the rest, **writes** the blocks it computed, then decodes and writes the blocks of the
generated tokens too. Hits come from three populations: shared system prompts (a small hot set
present from the start), returning conversations (recency, within one engine when the router
uses session affinity), and cross-engine sharing (a second engine's writes).

**Bytes per token** are configuration: llama-3-8B (32 layers, 8 KV heads × 128, bf16)
= 128 KiB/token, so a 16-token block is 2 MiB and a 256-token chunk is 32 MiB; llama-3-70B
= 320 KiB/token; a DeepSeek-V3 MLA layout ≈ 70 KiB/token. The object granularity is also
configuration: one object per chunk, or one object per chunk per layer (vLLM's
`SharedStorageConnector` writes a `safetensors` file per layer per block **[verify]**, i.e. 32
files of 64 KiB for one 8B block).

**Per-request I/O skeleton** (LMCache-style shared-filesystem backend) **[verify]**:

```
for each chunk of the prompt:      stat("kv/{hash}") = 0 | ENOENT          # lookup
for each hit chunk, in order:      openat("kv/{hash}", O_RDONLY); read(…, chunk_bytes); close
-- prefill compute for the miss tail --
for each new chunk:                openat("kv/{hash}", O_WRONLY|O_CREAT|O_TRUNC); write(…, chunk_bytes); close
-- decode; every chunk_tokens of output: another write --
```

**Abstract.** The conversation identity is a *chain*: request `r` either continues the
conversation of an earlier request `r − d` or starts a new one (§9.5). Everything about the
conversation (its id, how many blocks it had) is recomputed positionally from the earlier index,
so no history is stored and any request is computable in isolation.

```
workload kv_cache_serving {
  param concurrency  = 64                       # request slots per engine ("gpu" = one engine)
  param warm         = 2_000                    # requests per slot before measurement (§9.5)
  param requests     = 20_000                   # per slot, measured
  param chunk_tokens = 256
  param chunk_bytes  = 32MiB                    # 128 KiB/token × 256 [config]
  param files_per_chunk = 1                     # or layers, for per-layer objects [config]
  param sys_prompts  = 50                       # hot set, pre-populated by datagen
  param sys_pop      = zipf(s = 1.1)            # [measure]
  param sys_len      = lognormal(median = 1500, sigma = 0.3)    # tokens [measure]
  param reuse        = mixture(0.55: none,      # new conversation
                               0.45: lognormal(median = 40, sigma = 1.2))  # requests ago [measure]
  param retain       = 5_000                    # requests; older chunks are evicted [config: capacity]
  param turn_in      = lognormal(median = 300, sigma = 0.8)     # new prompt tokens [measure]
  param turn_out     = lognormal(median = 250, sigma = 0.6)     # generated tokens [measure]
  param prefill_per_token = 40us, decode_per_token = 12ms       # [measure]

  dataset  sysp = files("kv/sys/{id:04}/blk_{k:04}", count = $sys_prompts,
                        size = $sys_len * 128KiB, chunk = $chunk_bytes,   # realized as chunks(sp) block objects
                        seed = 0x5eed_da80)
  namespace kv = objects("kv/{conv:016x}/blk_{k:04}", size = const($chunk_bytes))

  per gpu {
    parallel($concurrency) {
      for r in $warm + $requests {                                  # one index space (§9.5)
        let d      = draw($reuse)                                   # none, or requests ago
        let cont   = d != none && d <= r
        let conv   = when (cont) { conv @ (r - d) } else { draw(uniform64) }      # chain
        let sp     = when (cont) { sp @ (r - d) } else { pick(sysp, dist = $sys_pop) }
        let prev   = when (cont) { total @ (r - d) } else { 0 }    # blocks the conversation had
        let hit    = when (cont && d <= $retain) { prev } else { 0 }
        let inn    = draw($turn_in), out = draw($turn_out)         # one draw each; reused below
        let total  = prev + ceil((inn + out) / $chunk_tokens)
        let sysblk = chunks(sp)                                     # block objects of this system prompt

        phase(when (r < $warm) { "warm" } else { "serve" }) {
          for k in sysblk        { stat(file(sp, k)) }                           # lookups
          for k in total         { stat(file("kv/{conv:016x}/blk_{k:04}"), expect = [ENOENT]) }
          for k in sysblk        { let b = file(sp, k), open(b, RDONLY), read(b, $chunk_bytes), close(b) }
          for k in hit           { let b = file("kv/{conv:016x}/blk_{k:04}"),
                                   open(b, RDONLY), read(b, $chunk_bytes), close(b) }
          compute($prefill_per_token * (total - hit) * $chunk_tokens)
          when (hit < total)     { mkdir("kv/{conv:016x}", expect = [EEXIST]) }   # the conversation directory
          for k in hit .. total  { let b = file("kv/{conv:016x}/blk_{k:04}"),
                                   open(b, WRONLY|CREAT|TRUNC), write(b, $chunk_bytes), close(b) }
          compute($decode_per_token * out)
        }
      }
    }
  }
}
```

Notes on the shape:
- The `warm` prefix of the index space exists so that `conv @ (r − d)` has something to reach
  back to when measurement starts; its writes populate the cache, its statistics are discarded.
  A chain that reaches an index below 0 is a new conversation by definition.
- Eviction is an input (`retain`), as `GRAMMAR_OPTIONS.md` §5.3 requires: a conversation older
  than `retain` requests is a miss, the engine recomputes, and the blocks are rewritten under
  the same names (real caches re-store on miss). No simulated cache exists in the runner.
- Decode-time writes are folded into the tail of the write loop; a finer model writes one block
  every `chunk_tokens × decode_per_token` during decode, which is a `for` with a `compute`
  inside and needs nothing new.
- `files_per_chunk > 1` wraps each block access in `for l in $files_per_chunk` with `/l_{l:02}`
  appended to the name.

**Cuts (stated).**
- **Cross-engine sharing.** A block written by engine A and read by engine B is a hit only
  because of timing (§5.3 of `GRAMMAR_OPTIONS.md`, R21). Not modeled in this shape. If a trace
  shows it matters, add a statistical term: a fraction of "new" conversations instead draw a
  conversation from a pre-populated shared namespace written by `datagen` (a hit by
  construction), or run a barrier-separated two-phase variant.
- **Lookups are metadata.** Real LMCache keeps an in-memory index and only touches storage for
  data; the `stat` loop models a shared-filesystem backend with no index. Keep or drop per the
  system being modeled; it doubles the op count, so the choice must be stated.
- Router behaviour (session affinity) is assumed perfect; imperfect affinity is a smaller
  `reuse` hit fraction, i.e. a parameter.

**What it stresses.** Write-then-read with a lag measured in minutes, a hot set that every
engine reads continuously, multi-MiB whole-object reads and writes with an open/close per
object, and a `stat` storm if lookups go to storage. Client-side caching of the hot set is a
legitimate win and the workload rewards it; the `retain` window sets how much storage-side
capacity matters.

**Constructs used.** `@` (positional evaluation at an earlier index) and the chain recurrence
it enables (§9.5); `none` in a mixture; `stat` and `expect = [ENOENT]`; a workload-created
`namespace`; arithmetic on drawn values in `compute`; `phase` chosen by an expression.

---

## 9. Constructs the exercise surfaced

Each item is a proposed addition or refinement to the semantic model of `GRAMMAR_OPTIONS.md`
§2 and §5.2. None weakens the invariants: every value is still a pure function of
`(seed, actor, site, loop indices)`, producers stay finite, and the op multiset is fixed before
the run.

**Decided 2026-09-30.** All nine are accepted: §9.1–§9.5, §9.8, and §9.9 as written; §9.6
with the naive slot layout and no slot-group refinement; §9.7 with the constraint stated there
on `as_written`. §9.5 replaces `recent(site, d)` in `GRAMMAR_OPTIONS.md` §5.2. Each has a node
or an expression form in the AST schema (`schema/README.md` maps them). Reasoning in
`DESIGN_REVIEW.md` §3.18.

### 9.1 Conditionals on the actor id

`when (gpu == 0)` (§3), `gpu div dp` and `gpu mod tp` in names and offsets (§4). The guardrail
in §4 of `GRAMMAR_OPTIONS.md` allows conditionals on loop indices, epoch, and parameters; the
actor id is a static property of the actor and belongs on that list. It does not introduce
timing dependence. **Proposal:** add `actor id` to the allowed operands, with integer
arithmetic (`div`, `mod`, `+`, `*`).

### 9.2 `until_eof` includes the terminating read, and short reads are expected there

Python's `readall` and PIL's decoder both issue a read that returns fewer bytes than requested,
and then one that returns 0. `NAPKIN_MATH.md` §8.D treats short reads as run failures, which is
right for a mid-file read. **Proposal:** `read(f, len)[until_eof]` issues reads of `len` until
the known size is exhausted, and the verifier checks each returned count against the size from
the dataset definition, not against `len`. Any other read must return exactly `len`. Ops on the
verifier's view: expected bytes, not requested bytes.

### 9.3 `expect = [errno…]` on any op

`ioctl(TCGETS)` returns `ENOTTY` for every file (§1); `mkdir` with `exist_ok` returns `EEXIST`
on every rank but one (§3); `stat` for a cache lookup returns `ENOENT` on a miss (§8). This is
the fail-soft op of `GRAMMAR_OPTIONS.md` §5.3, generalized. The listed errnos are counted per
op site and reported; any other error, or a listed error where the abstract needed success, is a
run failure. The fingerprint hashes the op, not its result, so `expect` does not affect it.

### 9.4 Offsets and repeat counts as expressions over dataset metadata

`read(f, offset = size(f) − 22, …)` for the zip tail (§2), `offset(c)` and `size(c)` for a
region (§5, §6), and `item_off[t]` from a parameter array (§4). The current model has
`size(f)` and `until_eof`; **proposal:** allow integer expressions over `size(f)`, `offset(r)`,
parameters, parameter arrays, and loop indices wherever an offset or a count is expected. For
archive layouts (`.npz`), the helper functions `member_off`/`member_len` are expressions the
generator and the abstract share: `datagen` writes members whose proportions are parameters, so
the abstract computes their offsets from `size(f)`.

### 9.5 `x @ i`: evaluate a binding at another index of its loop, and chains

`recent(site, d)` from `GRAMMAR_OPTIONS.md` §5.2 refers to the object drawn `d` iterations ago.
Writing the KV-cache abstract showed two things. First, the natural primitive is one level down:
`x @ i` is the value binding `x` takes at index `i` of its enclosing loop, for any binding whose
definition is positional. `recent(x, d)` is `x @ (r − d)`. Second, the reference must be
*recursive* for the model to be self-consistent: request `r` continues the conversation of
request `r − d`, but that request may itself have been a continuation, and the blocks it wrote
carry the conversation's original id. So `conv` is defined by a recurrence,

```
conv @ r  =  when (cont @ r) { conv @ (r − d @ r) } else { fresh draw at r }
```

which terminates because the index strictly decreases, and is still a pure function of the
position: evaluating it walks the chain, one hash per link, with no stored history. The same
recurrence gives `total @ r` (how many blocks the conversation had) and therefore the hit
length, without a separate `hit_len` distribution: hit length is *derived*, which is what makes
the reads land on blocks that were actually written. Costs: `--dry-run` evaluation of one
request is O(chain length) rather than O(1), bounded in practice by the reuse distribution's
tail, and `conv @ i` for `i < 0` is defined as a fresh draw, which is why the `warm` prefix of
the index space exists.

**Proposal:** replace `recent(site, d)` in §5.2 with `x @ i` plus the rule that a binding may
refer to itself at a strictly smaller index. The validator checks that every self-reference
decreases the index (syntactically: the index expression is `i − e` with `e ≥ 1`).
**Decided 2026-09-30**; `recent` is gone from the model. AST form: `{at: {ref: x, index: e}}`.

### 9.6 `regions`: a dataset of fixed-slot regions inside one file

DiskANN's index, FAISS's inverted lists, and a raw vector file are one file with many
addressable pieces. Modeling each piece as a file would change the op pattern (an open per
piece). Modeling them as computed offsets needs each region's offset to be computable in O(1),
which a size *distribution* prevents (prefix sums are a per-region table). **Proposal:**
`regions(file, count, slot, size, seed)`: region `c` starts at `c × slot`, has `size(c)` drawn
from the distribution capped at `slot`, and `datagen` writes it there, leaving the slot's tail
unwritten (sparse) or padded. The read sizes and popularity match the real system; only the
address spacing differs, which matters for scans and not for random probes. When `size` is
constant (sectors, rows), `slot = size` and the layout is exact.

**Decided 2026-09-30, with the naive layout kept deliberately.** The tail of each slot is left
unwritten and the filesystem is assumed to handle unwritten ranges the ordinary way (delayed
allocation lays the written regions out contiguously; extents break at holes of a block or
more, a bounded metadata cost rather than a per-read one). A slot-group layout (K regions per
group, prefix sums computed on the fly) was considered on 2026-09-29 to shrink the slack and the
size cap, and declined: real HDF5 and NetCDF files with alignment already contain unwritten
ranges, and the simpler layout is easier to reason about. The one thing to watch: capping
`size(c)` at `slot` truncates the request-size distribution, which is an acceptance metric
(`PROJECT_BRIEF.md` §6 item 14); revisit if a trace shows the tail matters.

### 9.7 `namespace`: workload-created objects without a count

Checkpoints and KV blocks are created by the run, named positionally, with no fixed count and
sizes known only from the writes. **Proposal:** `namespace n = objects(pattern, size)`, where
`size = as_written` for objects the run creates, or a constant. Its objects are not in any
manifest; the runner's startup check confirms the directory exists and is writable. Written
content for these objects is generated positionally from the workload's namespace seed and the
hash of the path, with the same dedupe/compression controls as `datagen` (`PROJECT_BRIEF.md`
§5; no block headers, revised 2026-09-29). Op-count computability holds because every loop over
a namespace is bounded by a drawn or parameterized count.

**Decided 2026-09-30, with one constraint: object sizes are never observed, only computed.**
`size = as_written` read literally would mean the runner records how many bytes each object
received, which is per-object state and a dependence on history; it would also make a read of
the object depend on which writer ran first. The accepted meaning is narrower. Every `write` to
a namespace object has a length that is a positional expression (a constant, a parameter, a
parameter-array element, a drawn value, or arithmetic over these and loop indices), so the
object's size is the sum of those lengths in the sequence that created it, and is computable
at the creator's position without I/O. A reader at the *same* position (the read-back of §3,
same handle binding) may use `until_eof`, which the runner resolves from that sum. A reader at
a *different* position (a KV block written by request `r − d`, read by `r`) states its length
from the same expression the writer used, reached through `@` if it involves drawn values, or
declares the namespace with a `size` expression instead of `as_written`. `fstat` on a
namespace object is issued when the application issues it, and its result is checked
structurally when the size is computable and ignored otherwise; it is never consumed by the
abstract. The validator rule: `until_eof` on an `as_written` object requires the read to use
the handle binding that the creating writes used.

A related small addition for *datasets*: `chunk = c` on a `files` dataset realizes each file of
the size distribution as `ceil(size / c)` block objects named by the pattern's `{k}`, so a
chunked prefix cache pre-populated by `datagen` (the system prompts of §8) is declared with the
same pattern the workload's own writes use, and `chunks(f)` is computable from the dataset seed.

### 9.8 `phase("name")` and `dirs(ds)` / `readdir`

Startup enumeration, warm-up, the checkpoint write, and read-back verification must be
reported separately from the training or serving steady state. `phase` is a statistics label
carried by every op inside it; it has no effect on execution or on the fingerprint (it is not
hashed). `dirs(ds)` enumerates the directories a file pattern generates (computable from the
pattern and the count), and `readdir(d)[until_end]` issues `getdents64` until exhausted. These
make the ImageFolder walk expressible; whether the walk belongs in CLOSED is for the WG.

### 9.9 `when` as an expression selecting a distribution

`pick(index, dist = when (hop < 1) { $hot } else { uniform })` (§5). Choosing a distribution by
loop index is already implied by conditionals on indices; stating it as an expression form
keeps the abstract short. No new semantics.

### 9.10 Checked, no change needed

- Fan-in reads (§4) need only actor-id arithmetic in names.
- Asynchronous checkpointing (§3) is `parallel` plus a `channel(capacity = 1)`.
- A file handle bound outside `parallel` and used inside it (§5, §6) is a binding-scope rule
  the VM needs anyway.
- `--time-scale` for the index build (§7) is a run parameter that divides every `compute`; it
  is legal because no compute gates a cross-actor dependency.
- Mixtures with a `none` outcome (§8) are a distribution over an option type; the builder can
  represent this as a `choose` with a null branch.

---

## 10. Builder-form check (Option D)

The small-file abstract of §1 in the Python builder of `GRAMMAR_OPTIONS.md` Option D, to
confirm the builder covers what the paper form uses.

```python
from aeiou import Workload, lognormal, KiB, MiB, ms

w = Workload("train_small_files")
P = w.params(batch=32, workers=8, prefetch=2, steps=500, sync_every=500,
             step_time=105 * ms, hdr_read=1 * MiB, enumerate=False)
train = w.dataset("train", pattern="train/{id div 1300:05}/img_{id:09}.jpg",
                  count=50_000_000, size=lognormal(median=110 * KiB, sigma=0.45),
                  seed=0x5eed_da7a)

with w.actor("gpu") as gpu:
    with gpu.when(P.enumerate), gpu.phase("enumerate"):
        with gpu.loop("d", train.dirs()) as d:
            gpu.open(d, "RDONLY|DIRECTORY"); gpu.readdir(d, repeat="until_end"); gpu.close(d)

    with gpu.loader("batches", workers=P.workers, prefetch=P.prefetch,
                    batches=P.steps, ordered=True) as worker:
        with worker.loop("j", P.batch):
            f = worker.consume(train)
            worker.open(f, "RDONLY|CLOEXEC")
            worker.fstat(f)
            worker.ioctl(f, "TCGETS", expect=["ENOTTY"])
            worker.lseek(f, 0, "CUR")
            worker.read(f, P.hdr_read, repeat="until_eof")
            worker.close(f)

    with gpu.phase("train"), gpu.loop("step", P.steps) as step:
        gpu.take("batches")
        gpu.compute(P.step_time)
        with gpu.every(P.sync_every):
            gpu.barrier("global")

w.write("train_small_files.ast.json")
```

The KV-cache chain (§9.5) in builder form, to check that a self-referencing binding is
expressible without Python evaluating it:

```python
with slot.loop("r", P.warm + P.requests) as r:
    d    = slot.draw("d", P.reuse)                       # Draw node; may be `none`
    cont = (d != None) & (d <= r)                        # Expr node
    conv = slot.let("conv", when(cont, slot.ref("conv").at(r - d), slot.draw("conv0", uniform64())))
    prev = slot.let("prev", when(cont, slot.ref("total").at(r - d), 0))
    ...
```

`slot.ref("conv").at(r - d)` builds an `At` node whose index is `r − d`; the builder rejects
an `At` on the binding being defined unless the index expression is provably smaller than the
loop index (here `d ≥ 1` when `cont` holds), which is the validator rule of §9.5 applied at
construction time. Everything else in §1–§8 (`when`, `phase`, `regions`, `namespace`,
parameter arrays, `expect`) is a node type plus a builder method; nothing needs Python to run
the workload.

---

## 12. Container workloads (added 2026-09-30)

Three workloads over container datasets, written straight in builder form (no paper form;
the shapes are the loaders' and the format classes carry the readers' protocols,
`GRAMMAR_OPTIONS.md` §6, `DESIGN_REVIEW.md` §3.28):

- **`train_stream_tfrecord`, `train_stream_parquet`** (`builder/abstracts/train_stream_shards.py`,
  `PROJECT_BRIEF.md` §6 item 15): the tf.data shape, and HF `streaming=True`, Ray Data, DALI
  over shards. The input pipeline shuffles the shard list and interleaves `cycle` shards; each
  is streamed whole by its reader library; the training loop takes a shard every
  `per_shard / batch` steps. `consume` over a `stream` dataset draws shard ids. Cuts: decode
  CPU is one `compute` per shard; the shuffle buffer and in-shard batch boundaries are
  application memory. What it stresses: large sequential reads (TFRecord: positioned 256 KiB
  reads; Parquet: `fadvise(WILLNEED)` readahead then multi-MiB `pread`s), few opens.
- **`train_map_hdf5`** (`builder/abstracts/train_map_hdf5.py`): a DataLoader whose
  `__getitem__` opens the container holding sample `i`, reads row `i` through libhdf5's
  sieve, and closes it. Cut: an open-file cache (the per-item open is the fork-safe shape).
  What it stresses: an open, eight small metadata reads, and one 192 KiB positioned read per
  sample across many 96 MiB files.

Capture for these: `tf.data` and `pyarrow` pipelines on an NFS mount of the target class
(`strace -f -e trace=%file,%desc`), compare the per-shard syscall sequence with the class's
protocol, and fit `xfer`, `cycle`, `decode`, and `per_shard`.

## 11. Trace capture plan (to replace [verify] and fill [measure])

One small run per workload on an NFS mount of the target class, one process, a few hundred
iterations, captured with:

```
strace -f -ttt -T -yy -e trace=%file,%desc,%network,%process -o trace.txt <command>
nfsstat -c > before.txt; <run>; nfsstat -c > after.txt; cat /proc/self/mountstats
```

| Workload | Command to trace | What to read off the trace |
|---|---|---|
| §1 small files (**traced 2026-10-01**, §1 "Trace") | PyTorch DataLoader + ImageFolder, `num_workers=2`, 200 steps | per-file syscall sequence and order; `read` length argument (`st_blksize` on NFS); whether the EOF read occurs; wire ops per file from `nfsstat` deltas |
| §2 large samples (**traced 2026-10-01**, §2 "Trace") | DLIO `unet3d` reader or `np.load` loop on `.npz` | tail-read offsets; member header reads; chunk size of the member reads; `cd_len`, `lh_len` |
| §3 checkpoint write (**traced 2026-10-01**, §3 "Trace") | `torch.distributed.checkpoint.save` on 2 ranks; `torch.save` on 1 | `mkdir` result per rank; write sizes per item and the coalesced small-record size; `fsync` presence; `.metadata` size and the `rename` |
| §4 restore / load | `dcp.load`; `AutoModel.from_pretrained` with `safetensors` | per-item `lseek`/`read` pairs; header read lengths; `mmap` and fault pattern (`perf trace -F` or `/proc/PID/smaps` deltas) |
| §5 DiskANN | `search_disk_index` on a 10M-point index, `beam_width=4` | `io_submit` batch sizes (= beam), rounds per query (hops), offset histogram (hub concentration), node-cache size |
| §6 IVF | FAISS `IndexIVFPQ` with `OnDiskInvertedLists`, `nprobe=64` | list size distribution (from the index), list popularity over a query set, `pread` vs page-fault path |
| §7 index build | DiskANN `build_disk_index` at 1M points | phase boundaries, read chunk sizes, sample-phase read pattern, layout write sizes |
| §8 KV cache | vLLM + LMCache with the local-disk or shared-fs backend, a chat replay (ShareGPT) | objects per chunk, object size, `stat` lookups, reuse-distance and prompt-length distributions, hit-length vs derived-length agreement |

(`%process` added 2026-10-01: `aeiou-trace` follows `clone` to know which threads share
descriptors and which processes are one instance; `builder/README.md` §7.)

For each: fit the distributions into the parameter file, run `--dry-run --metrics`, and compare
reuse distance, run length, popularity, request size, dependency depth, and read/write mix with
the same metrics computed from the trace (`GRAMMAR_OPTIONS.md` §5.4). A metric that no parameter
setting can match means the shape is missing a construct, and this document gets a §9 entry.
