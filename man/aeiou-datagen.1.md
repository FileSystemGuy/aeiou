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
which in turn refuses the ones with a class; a corpus with both kinds is written by both
tools, each dataset by one of them.

Names come from the dataset's pattern, sizes from the dataset seed (the Python port of the
runner's sampler), and bytes from the same positional payload generator as the Rust writer:
1 MiB blocks of a `dgen-py` stream keyed by dataset seed, unit, and block, bit-identical to
the Rust crate's, with the dedupe and compression ratios given. Every file is checked
against the geometry its declared layout predicts (file and unit headers and footers,
column chunks, row framing, alignment), since the runner computes every offset from that
layout and never reads a container's metadata. The manifest `.aeiou-dataset.json` is
written last at each dataset root, with the same normative content as the Rust writer's
plus a `format` block naming the class and the writer library; `aeiou run` compares it
before the gate.

The format classes and their reader protocols are in the `aeiou.formats` module of the
builder: `tfrecord`, `parquet`, `hdf5`, `webdataset`. Their libraries are the `formats`
extra of the Python package.

`--root` and `--threads` are resolved through the suite's option layers
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
- **--endpoint** *NAME=DIR*

  Puts the root directory of the abstract's dataset NAME at DIR instead of under `--root`
  (repeatable), as **aeiou**(1) ENDPOINTS describes. From the command line, else
  `$AEIOU_ENDPOINT`, else `endpoint` in the `[datagen]` table of the config file.
  *Layered.*
- **--threads** *THREADS*

  Writer threads. Default all cores. *Layered.*
- **--dedupe** *DEDUPE*

  Dedupe ratio: every *DEDUPE* files share content. Default 1. Recorded in the manifest's
  payload block.
- **--compress** *COMPRESS*

  Compression ratio: the last (C-1)/C of every 1 MiB block is zeros. Default 1. Recorded in
  the manifest's payload block.
- **--dataset** *NAME*

  Write only these datasets; repeatable. Default all.
- **--config** *FILE*

  A TOML config file, else the one `$AEIOU_CONFIG` names, else none; never searched for.
  One table per subcommand, keys spelled as the long flags; this tool reads the
  `[datagen]` table.
- **-h**, **--help**

  Print the help.
- **-V**, **--version**

  Print the version.

## ENVIRONMENT

- **AEIOU_ROOT**, **AEIOU_THREADS**

  The layered options, when not on the command line.
- **AEIOU_CONFIG**

  The config file, when `--config` is not given.
- **AEIOU_SCHEMA_DIR**

  The directory holding the schemas, when the package is not run from the repository.

## FILES

- `.aeiou-dataset.json`

  The manifest at each dataset root. Its id is printed.

## EXIT STATUS

- **0**

  Every selected dataset was written and checked against its layout.
- **1**

  A failure during the work: an invalid abstract, a dataset with no format class, a
  non-empty root, a file whose geometry differs from its layout, a library that is not
  installed.
- **2**

  A usage error.

## EXAMPLES

```
aeiou-datagen schema/examples/train_stream_parquet.ast.json --root /mnt/sut --param shards=64
AEIOU_ROOT=/mnt/sut aeiou-datagen schema/examples/train_map_hdf5.ast.json --dataset train
```

## SEE ALSO

**aeiou**(1), **aeiou-build**(1), **aeiou-params**(1), **aeiou-config**(5),
**aeiou-abstract**(7).

The reference: `builder/REFERENCE.md` §6 (the format classes and this tool),
`runner/REFERENCE.md` §5 (the payload and the manifest), `schema/README.md` §2 (the
container layout) and §6 (the manifest).
