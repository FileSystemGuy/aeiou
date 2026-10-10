# Quick start from the release tarball

This is the fastest way to a first run and a first bug report. It needs no Rust toolchain
and, for the runner alone, no Python.

## What is in the tarball

| Path | What |
|---|---|
| `bin/aeiou` | The runner: `check`, `dry-run`, `datagen`, `run`. x86_64 Linux, glibc 2.34 or newer (RHEL 9, Ubuntu 22.04, Debian 12 and later). Links nothing but libc. |
| `bin/aeiou-launch` | One `aeiou run` or `aeiou datagen` rank per host over ssh. A POSIX shell script. |
| `share/man/` | The manual pages as roff: `man -l share/man/man1/aeiou.1`, or `MANPATH=$PWD/share/man man aeiou`. |
| `man/` | The same pages as Markdown. |
| `examples/` | The committed abstracts (`*.ast.json`), the only thing the runner executes, and `params/`, example parameter files. |
| `wheel/` | The Python builder, for those who want to author or change workloads, generate container datasets, or compare an `strace` of a real application with an abstract. Python 3.12 or newer. |

## First run

```
tar xzf aeiou-0.1.0-x86_64-linux.tar.gz && cd aeiou-0.1.0-x86_64-linux
export PATH=$PWD/bin:$PATH

# 1. the runner validates every shipped abstract and prints its hash and op counts
aeiou check examples/*.ast.json

# 2. walk the small-file training workload without I/O: op counts, bytes, the fingerprint
aeiou dry-run examples/train_small_files.ast.json --gpus 2 --seed 1 \
    --params-file examples/params/train_small_files.smoke.params.json

# 3. write its dataset (4000 files, ~470 MiB) under the storage to test
aeiou datagen examples/train_small_files.ast.json --root /mnt/sut/aeiou \
    --params-file examples/params/train_small_files.smoke.params.json

# 4. run it: 2 emulated GPUs, 50 steps, the JSON report in run.json
aeiou run examples/train_small_files.ast.json --root /mnt/sut/aeiou --gpus 2 --seed 1 \
    --params-file examples/params/train_small_files.smoke.params.json --report-json run.json
```

`--root` is the storage under test: a directory on the mount you want to measure. The
dataset lands under it as `train/`; `aeiou run` refuses a dataset whose manifest does not
match the abstract's definition, so regenerate after changing a dataset parameter. The
fingerprint printed by `dry-run` must equal the one `run` prints: the same workload was
executed. `--time-scale 0` drops the emulated GPU compute and runs the I/O back to back.

Each command prints the options it resolved and where each came from (command line,
`AEIOU_*` environment, `--config` file, default) before it does anything, so the report
and the command line of a run never disagree.

## Other workloads

Every `examples/*.ast.json` runs the same way. `aeiou dry-run ABSTRACT --gpus 1` with no
parameter file uses the abstract's defaults, which for the training and checkpoint
workloads are the sizes of a real job (thousands of files, hundreds of steps); scale them
down with `--param k=v` or a parameter file. The `train_stream_*` and `train_map_hdf5`
workloads read container files (TFRecord, Parquet, HDF5) that the Python side writes:

```
python3 -m venv .venv && .venv/bin/pip install 'aeiou[formats] @ ./wheel/aeiou-0.1.0-py3-none-any.whl'
# or: uv tool install 'aeiou[formats] @ ./wheel/aeiou-0.1.0-py3-none-any.whl'
.venv/bin/aeiou-datagen examples/train_stream_tfrecord.ast.json --root /mnt/sut/aeiou
```

`aeiou-build`, `aeiou-params`, `aeiou-datagen`, and `aeiou-trace` each have a manual page
in `share/man/man1/`.

## Reporting a problem

Send the command line, everything the tool printed (the options block at the top
included), `run.json` when there is one, and `uname -r`. For a run on a network mount,
`mount | grep sut` and the API and cache mode you used (`--posix`, `--cache`, printed in the options block)
decide what the numbers mean, so include them.

## From source instead

To patch and rebuild, clone the repository and follow the quick start in its `README.md`:
`cargo build --release` in `runner/` and `uv sync --extra test --extra formats` in
`builder/`.
