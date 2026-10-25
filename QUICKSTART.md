# Quick start from the release tarball

This is the fastest way to a first run and a first bug report. It needs no Rust toolchain
and, for the runner alone, no Python.

Not sure which path is yours (run a shipped workload, run one at your job's sizes, model a
new application from an `strace`, or play one capture back)? The repository README's
[Where to start](https://github.com/FileSystemGuy/aeiou#where-to-start) lays out each,
with the recommended sequence of tools.

A clone of the repository is for changing aeiou, not for running it: its README's
[Building from source](https://github.com/FileSystemGuy/aeiou#building-from-source) is
`./dev.sh setup`, after rustup and uv.

## What is in the tarball

| Path | What |
|---|---|
| `bin/aeiou` | The runner: `check`, `dry-run`, `datagen`, `run`. x86_64 or aarch64 Linux (one tarball each, `-x86_64-linux` and `-aarch64-linux`), glibc 2.34 or newer (RHEL 9, Ubuntu 22.04, Debian 12 and later). Links nothing but libc. Reads and writes S3 object stores as well as directories (**aeiou**(1), OBJECT STORES). |
| `bin/aeiou-header` | Made by `aeiou datagen`, never by you: the header of an HDF5 file, from libhdf5 (one file carrying Python, numpy, h5py). It must stay beside `bin/aeiou`; only the host of rank 0 runs it, so client hosts need `aeiou` alone. |
| `bin/aeiou-launch` | One `aeiou run`, `aeiou datagen`, or `aeiou-datagen` rank per host over ssh. A POSIX shell script. |
| `share/man/` | The manual pages as roff: `man -l share/man/man1/aeiou.1`, or `MANPATH=$PWD/share/man man aeiou`. |
| `man/` | The same pages as Markdown. |
| `workloads/` | The committed abstracts (`*.ast.json`), the only thing the runner executes, one directory per I/O mechanism under its category (`input-pipeline/map/file-per-sample/`, `kvcache/lmcache-local-disk/`, …), each with its README and catalog record (`mechanism.toml`), its cases (`cases/<case>/params.json`: a parameter set describing one workload, with its records, its description in prose (`DESCRIPTION.md`), measured `shape.json`, and, where fitted, the parameters fitted to its captures), and, for some, test-sized parameter files in `params/`; the catalog, `CATALOG.md`, and its vocabulary, `attributes.toml`. The authoring scripts, capture kits, and trace metrics stay in the repository. |
| `wheel/` | The Python builder, for those who want to author workloads, describe one in prose, or compare an `strace` of a real application with an abstract, with `requirements.txt` (the versions and hashes of what it needs, from the lock it was tested with) and `python-version` (that Python); installed below. |
| `QUICKSTART.md`, `LICENSE` | This guide; the Apache License 2.0. |

## First run

```
tar xzf aeiou-0.3.0-x86_64-linux.tar.gz && cd aeiou-0.3.0-x86_64-linux     # or -aarch64-linux
export PATH=$PWD/bin:$PATH

# 1. the runner validates every shipped abstract and prints its hash and op counts
aeiou check $(find workloads -name '*.ast.json')

# 2. walk the small-file training workload without I/O: op counts, bytes, the fingerprint
W=workloads/input-pipeline/map/file-per-sample
aeiou dry-run $W/file_per_sample.ast.json --gpus 2 --seed 1 \
    --params-file $W/params/smoke.params.json

# 3. write its dataset (4000 files, ~470 MiB) under the storage to test
aeiou datagen $W/file_per_sample.ast.json --root /mnt/sut/aeiou \
    --params-file $W/params/smoke.params.json

# 4. run it: 2 emulated GPUs, 50 steps, the JSON report in run.json
aeiou run $W/file_per_sample.ast.json --root /mnt/sut/aeiou --gpus 2 --seed 1 \
    --params-file $W/params/smoke.params.json --report-json run.json
```

`--root` is the storage under test: a directory on the mount you want to measure. The
dataset lands under it as `train/`; `aeiou run` refuses a dataset whose manifest does not
match the abstract's definition, so regenerate after changing a dataset parameter. The
fingerprint printed by `dry-run` must equal the one `run` prints: the same workload was
executed. `--time-scale 0` drops the emulated GPU compute and runs the I/O back to back.

Each command prints the options it resolved and where each came from (command line,
`AEIOU_*` environment, `--config` file, default) before it does anything, so the report
and the command line of a run never disagree.

## Use cases

A case is an abstract with the values of one job, run by its name. `aeiou cases` lists every
case this release can run, and `--where` narrows the list by the catalog's attributes
(`aeiou cases --help` gives them all; `workloads/CATALOG.md` says what each means):

```
aeiou cases --where category=kvcache --where 'kind!=example'
aeiou cases sift1m-nocache            # one case, and the commands that run it
aeiou datagen --case diskann-search/sift1m-nocache --root /mnt/sut/aeiou-diskann
aeiou run --case diskann-search/sift1m-nocache --gpus 1 --root /mnt/sut/aeiou-diskann
```

## Other workloads

Every `workloads/**/*.ast.json` runs the same way. Most abstracts require the values of a
job, which a case gives (above) or a parameter file does; `aeiou dry-run ABSTRACT --gpus 1`
alone names every value it still lacks, and `--param k=v` gives one. The `tfrecord_stream`, `parquet_stream`, and
`hdf5_map` workloads read container files (TFRecord, Parquet, HDF5) that `aeiou datagen`
writes as it writes any other. Each abstract's dataset lands under its `--root` (`train/`
for `file_per_sample` and `parquet_stream` alike), so give each workload a root of its own:

```
aeiou datagen workloads/input-pipeline/stream/parquet/parquet_stream.ast.json --root /mnt/sut/aeiou-parquet \
    --param samples=2048
```

The Python builder in `wheel/` is needed only to author workloads (to change a shipped one,
clone the repository, which holds the authoring scripts) or to compare a trace with an
abstract:

```
uv venv --python "$(cat wheel/python-version)" --managed-python .venv
uv pip install --python .venv/bin/python --require-hashes -r wheel/requirements.txt
uv pip install --python .venv/bin/python --no-deps wheel/aeiou-0.3.0-py3-none-any.whl
```

That is the environment the release was tested in: the Python in `wheel/python-version`, one
[uv](https://docs.astral.sh/uv/) downloads (never the system's), and every library at the
version and hash in `wheel/requirements.txt`. Without uv, `python3 -m venv .venv &&
.venv/bin/pip install --require-hashes -r wheel/requirements.txt && .venv/bin/pip install
--no-deps wheel/aeiou-0.3.0-py3-none-any.whl` installs the same libraries under whatever
Python 3.12 or later the system has.

The manual pages, in `share/man/` and as Markdown in `man/`: **aeiou**(1), the runner;
**aeiou-launch**(1); **aeiou-build**(1), **aeiou-params**(1), **aeiou-trace**(1),
**aeiou-catalog**(1), **aeiou-datagen**(1) (deprecated: `aeiou datagen` writes every
format), and **aeiou-header**(1), which `aeiou datagen` runs; **aeiou-config**(5), the
configuration file; **aeiou-abstract**(7), the abstract language.

## Reporting a problem

Send the command line, everything the tool printed (the options block at the top
included), `run.json` when there is one, and `uname -r`. For a run on a network mount,
`mount | grep sut` and the API and cache mode you used (`--posix`, `--cache`, printed in the options block)
decide what the numbers mean, so include them.

## From source instead

To patch and rebuild, clone the repository and follow the quick start in its `README.md`:
`cargo build --release` in `runner/` (the default features, the release binary's: the container writers and the S3 engine) and `uv sync --extra test --extra formats` in
`builder/`.
