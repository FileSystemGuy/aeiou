# aeiou-datagen(1)

## NAME

aeiou-datagen - write the container datasets an abstract declares, through their format classes

## SYNOPSIS

```
aeiou-datagen [OPTIONS] --root DIR ABSTRACT
aeiou-datagen --help | --version
```

## DESCRIPTION

**aeiou-datagen** is the Python twin of `aeiou datagen` (**aeiou**(1)). It writes every
dataset of the abstract that declares a container **format class**: real Parquet, HDF5,
TFRecord, or tar (WebDataset) files, written with the format's own library, uncompressed
and plainly encoded, so that the real reader library issues against them the operation
protocol the abstract models. Datasets without a format class are left to `aeiou datagen`,
which refuses the ones with a class unless built with the cargo feature `formats`, under
which it writes TFRecord, tar, and HDF5 files byte for byte as this tool does (HDF5's
header through **aeiou-header**(1)); a corpus with both
kinds is written by both tools, each dataset by one of them.

Names come from the dataset's pattern, sizes from the dataset seed (the Python port of the
runner's sampler), and bytes from the same positional payload generator as the Rust writer:
1 MiB blocks of an SFC64 stream (numpy's bit generator) keyed by dataset seed, unit, and
block and laid out in 4 KiB pieces, bit-identical to the Rust writer's, with the dedupe and
compression ratios given. Every file is checked
against the geometry its declared layout predicts (file and unit headers and footers,
column chunks, row framing, alignment), since the runner computes every offset from that
layout and never reads a container's metadata. The manifest `.aeiou-dataset.json` is
written last at each dataset root, its state `complete`, with the same normative content as the Rust writer's
plus a `format` block naming the class and the writer library; `aeiou run` compares it
before the gate.

A dataset declared `protocol: s3` is written to the object store its `--endpoint`
names, with no copy on a local file system: each writer streams the file into an upload
(one `PUT`, or a multipart upload in 8 MiB parts, each sent before the next is filled),
the check runs before the upload is completed so a file that fails it leaves no object,
and the manifest is one `PUT` of `.aeiou-dataset.json` at the prefix. The client is
`obstore`, the Python bindings of the runner's `object_store`, and the store comes from the
same `AWS_*` environment the runner reads. Every writer streams, whatever the destination,
so a writer holds a piece of its file, never the file: Parquet a row group at a time, and
HDF5 as the header libhdf5 writes for the file's row count (taken from a sparse scratch
file in `/dev/shm`, a few hundred KiB however large the file) followed by the rows.

The format classes and their reader protocols are in the `aeiou.formats` module of the
builder: `tfrecord`, `parquet`, `hdf5`, `webdataset`. Their libraries are the `formats`
extra of the Python package; `obstore` is the `s3` extra.

On several hosts (`--ranks`, `--rank`, `--coordinator`, as for `aeiou datagen`;
**aeiou-launch**(1) starts them) rank *r* writes its contiguous slice of every dataset's
file ids. Every rank checks the roots are empty before the start gate, writes after it,
brings the file systems it wrote to stable storage, and sends its counts; rank 0 writes the
manifests once every rank has reported, so a manifest means the whole corpus is there, and
records every rank's host and counts in the provenance (`ranks`, `hosts`). The coordinator
is `aeiou datagen`'s protocol, spoken by rank 0 in process: it refuses a host whose
abstract, parameters, `--gpus`, `--dedupe`, `--compress`, or `--dataset` list differ from
rank 0's, naming each difference, and a failure on any host stops every host. The corpus is
the one a single host writes, byte for byte, with the same dataset ids.

`--root`, `--endpoint`, `--writers`, `--rank`, `--ranks`, and `--coordinator` are resolved through the suite's option layers
(**aeiou-config**(5)): the command line, else the environment, else the `[datagen]` table
of the config file, else the default. Every other option is the command line's alone. The
block of what was resolved, with each value's source, is printed as `aeiou datagen` prints
it, and a wrong command line is reported in the frame every tool of the suite shares, with
every missing argument listed at once.

## OPTIONS

- *ABSTRACT*

  The abstract (`.ast.json`).
- **--params-file** *FILE*

  A parameter file (`.params.json`, **aeiou-params**(1)), applied over the defaults and
  under `--param`. Repeatable, in order.
- **--param** *NAME=VALUE*

  Override a parameter. The value is JSON; a bare word is a string. Repeatable.
- **--gpus** *GPUS*

  Instance count, for dataset definitions that reference `gpus`. Default 1.
- **--root** *DIR*

  Required, from some layer. The directory the abstract's paths are relative to. From the
  command line, else `$AEIOU_ROOT`, else `root` in the `[datagen]` table of the config
  file. *Layered.*
- **--endpoint** *NAME=DIR* | *NAME=s3://BUCKET[/PREFIX]*

  Puts the root directory of the abstract's dataset NAME at DIR instead of under `--root`,
  or, for a name declared `protocol: s3`, at a prefix of a bucket (repeatable), as
  **aeiou**(1) ENDPOINTS describes: an object name has no default place, an endpoint is of
  its name's protocol's kind, and a `posix` name never falls under a root placed in a
  store. From the command line, else
  `$AEIOU_ENDPOINT`, else `endpoint` in the `[datagen]` table of the config file.
  *Layered.*
- **--writers** *N*

  Files written at once, each writer a process: the writers' Python holds its interpreter
  lock between the library calls, so threads would write no faster than one. The name and
  its layers (`AEIOU_WRITERS`, `writers` in the `[datagen]` table) are `aeiou datagen`'s,
  whose writers are threads. A file is a pure function of its id, so the writers share
  nothing; one writer, or one file, runs in the process itself. Default all cores.
  *Layered.*
- **--dedupe** *DEDUPE*

  Dedupe ratio: files share their content *DEDUPE* by *DEDUPE*, each byte at its own file
  offset (`aeiou-positional/4`, as `aeiou datagen`: aeiou(1)). Default 1. Recorded in the
  manifest's payload block.
- **--compress** *COMPRESS*

  Compression ratio: in every 4 KiB, the first ⌈4096/C⌉ bytes are random and the rest
  0x5A, so every file and every piece compresses at C. Default 1. Recorded in the manifest's
  payload block.
- **--dataset** *NAME*

  Write only these datasets; repeatable. Default all.
- **--rank** *R*

  This host's index among `--ranks` hosts. Default 0. *Layered.*
- **--ranks** *N*

  Hosts the datagen is spread over: each writes its slice of every dataset's file ids, and
  rank 0 writes the manifests once every rank has reported. Default 1. *Layered.*
- **--coordinator** *HOST:PORT*

  With `--ranks` above 1, required: the coordinator's address. Rank 0 listens on it (in
  process); every rank connects to it. *Layered.*
- **--config** *FILE*

  A TOML config file, else the one `$AEIOU_CONFIG` names, else none; never searched for.
  One table per subcommand, keys spelled as the long flags; this tool reads the
  `[datagen]` table.
- **-h**, **--help**

  Print the help.
- **-V**, **--version**

  Print the version.

## ENVIRONMENT

- **AEIOU_ROOT**, **AEIOU_ENDPOINT**, **AEIOU_WRITERS**, **AEIOU_RANK**, **AEIOU_RANKS**, **AEIOU_COORDINATOR**

  The layered options, when not on the command line.
- **AEIOU_CONFIG**

  The config file, when `--config` is not given.
- **AWS_ENDPOINT_URL**, **AWS_ACCESS_KEY_ID**, **AWS_SECRET_ACCESS_KEY**, **AWS_REGION**, **AWS_ALLOW_HTTP**, **AWS_VIRTUAL_HOSTED_STYLE_REQUEST**

  The S3 store of an object endpoint, as **aeiou**(1) reads them; path-style requests
  unless the last is set.
- **AEIOU_SCHEMA_DIR**

  The directory holding the schemas, when the package is not run from the repository.

## FILES

- `.aeiou-dataset.json`

  The manifest at each dataset root, an object at the prefix of one in a store. Its id is
  printed.

## EXIT STATUS

- **0**

  Every selected dataset was written and checked against its layout.
- **1**

  A failure during the work: an invalid abstract, a dataset with no format class, a
  non-empty root or prefix, a file whose geometry differs from its layout, a library that
  is not installed, a request the store refused, a host the coordinator refused or lost
  (on every host).
- **2**

  A usage error.

## EXAMPLES

```
aeiou-datagen workloads/input-pipeline/stream/parquet/parquet_stream.ast.json --root /mnt/sut --param shards=64
AEIOU_ROOT=/mnt/sut aeiou-datagen workloads/input-pipeline/map/hdf5/hdf5_map.ast.json --dataset train
AWS_ENDPOINT_URL=http://store:9000 aeiou-datagen my_s3_variant.ast.json --root /unused --endpoint shards=s3://bench/train
aeiou-launch -f hosts -- aeiou-datagen workloads/input-pipeline/stream/tfrecord/tfrecord_stream.ast.json --root /mnt/sut
```

## SEE ALSO

**aeiou**(1), **aeiou-build**(1), **aeiou-launch**(1), **aeiou-params**(1), **aeiou-config**(5),
**aeiou-abstract**(7).

The reference: `builder/REFERENCE.md` §6 (the format classes and this tool),
`runner/REFERENCE.md` §5 (the payload and the manifest), `schema/README.md` §2 (the
container layout) and §6 (the manifest).
