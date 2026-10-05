# aeiou-params(1)

## NAME

aeiou-params - parameter files of an abstract: write its defaults as a set, validate sets, build a set from real files

## SYNOPSIS

```
aeiou-params defaults [-o FILE] [--doc TEXT] [--pin] ABSTRACT
aeiou-params check ABSTRACT FILES...
aeiou-params safetensors [--tp TP] [--column REGEX] [--row REGEX] [--doc TEXT] [--pin] -o FILE ABSTRACT SHARDS...
aeiou-params npz [-o FILE] [--member NAME] [--doc TEXT] [--pin] ABSTRACT ARCHIVES...
aeiou-params --help | --version
```

## DESCRIPTION

An abstract is a *shape*: named parameter slots whose defaults are placeholders or one
reference set. A **parameter file** (`<abstract>.<set>.params.json`) fills the slots with
values from configuration, measurement, or a trace, so one published shape carries several
parameter sets, and a fitted set is reviewed, hashed, and published on its own. The runner
applies a file with `aeiou run --params-file FILE` (repeatable, in order; `--param` applies
last), and so do `aeiou dry-run`, `aeiou datagen`, and **aeiou-datagen**(1).

The file is `{"params_version": 1, "abstract": NAME, "ast_sha256"?: HEX, "doc"?: TEXT,
"params": {name: value, ...}, "provenance"?: {...}}`. A value has the form of a parameter
default: a scalar, a distribution, or an array of these, and it must keep the **kind** of
the default it replaces, because the script fixed at build time how each slot is consumed
(a distribution is drawn, an array is indexed). `abstract` must be the document's name;
`ast_sha256`, when present, must be its hash; every name must be declared; `gpus` may not
appear.

**aeiou-params** is the helper for these files. Four subcommands:

- **defaults** writes every default of an abstract as a set, to edit or to fit.
- **check** validates files against an abstract with the runner's rules.
- **safetensors** reads real safetensors shard headers and writes the tensor table the
  `model_load` abstract takes as parameters.
- **npz** reads the archive framing the `train_large_samples` abstract takes as parameters
  from real `.npz` files and compares it with the abstract's defaults.

## OPTIONS

### aeiou-params defaults

- *ABSTRACT*

  The abstract (`.ast.json`).
- **-o**, **--out** *FILE*

  The output file. Default `<abstract>.defaults.params.json`.
- **--doc** *TEXT*

  The set's `doc` line.
- **--pin**

  Record the abstract's hash as `ast_sha256`, so the set is for this exact shape.

### aeiou-params check

- *ABSTRACT*

  The abstract (`.ast.json`).
- *FILES...*

  The parameter files (`.params.json`) to validate. Prints `ok` or `FAIL` per file with the
  rule that failed.

### aeiou-params safetensors

- *ABSTRACT*

  The `model_load` abstract; its parameter names are checked.
- *SHARDS...*

  The shard files, in dataset id order.
- **-o**, **--out** *FILE*

  Required. The parameter file to write.
- **--tp** *TP*

  The tensor-parallel degree. Default 8.
- **--column** *REGEX*

  A regex of the tensors split column-parallel. Default: Llama-shaped names.
- **--row** *REGEX*

  A regex of the tensors split row-parallel. Default: Llama-shaped names. Tensors matching
  neither are replicated.
- **--doc** *TEXT*

  The set's `doc` line.
- **--pin**

  Record the abstract's hash as `ast_sha256`.

The table written has the parallel arrays `shard`, `off`, `bytes`, `split`, `rows`,
`row_bytes`, and the scalars `hdr_len`, `shards`, `shard_bytes`, `tp`.

### aeiou-params npz

- *ABSTRACT*

  The `train_large_samples` abstract.
- *ARCHIVES...*

  Archives of the corpus. A few are enough; all must agree with each other.
- **-o**, **--out** *FILE*

  Write a parameter file with the two values, `framing` and `cd_len`.
- **--member** *NAME*

  The member the application reads. Default `x`.
- **--doc** *TEXT*

  The set's `doc` line.
- **--pin**

  Record the abstract's hash as `ast_sha256`.

The subcommand reads the archives with the standard library alone, prints both the archives'
values and the abstract's defaults, and refuses archives that disagree with each other, a
compressed member, a member that is not first in the archive, and a zip64 end record, since
the abstract reads none of those shapes.

### Common

- **-h**, **--help**

  Print the help of the command or subcommand.
- **-V**, **--version**

  Print the version.

## ENVIRONMENT

- **AEIOU_SCHEMA_DIR**

  The directory holding the schemas, when the package is not run from the repository.

## FILES

- `<abstract>.<set>.params.json`

  A parameter file. The schema is `schema/params.schema.json`; the committed examples are
  `schema/examples/params/`, each checked against its abstract in CI.

## EXIT STATUS

- **0**

  Success. For `check`, every file fits; for `npz` without `-o`, the archives agree with the
  abstract's defaults.
- **1**

  A file that does not fit (`check`), an abstract that is not the one the subcommand serves,
  archives or shards that cannot be read as the abstract reads them, or, for `npz` without
  `-o`, archive values that differ from the defaults.
- **2**

  A usage error, in the frame every tool of the suite shares.

## EXAMPLES

```
aeiou-params defaults schema/examples/train_small_files.ast.json -o tsf.params.json --pin
aeiou-params check schema/examples/train_small_files.ast.json tsf.params.json
aeiou-params safetensors schema/examples/model_load.ast.json model-0000?-of-00004.safetensors \
    -o llama.params.json --tp 8
aeiou-params npz schema/examples/train_large_samples.ast.json corpus/train/00000/sample_0000000?.npz
aeiou run schema/examples/model_load.ast.json --gpus 8 --root /mnt/sut --params-file llama.params.json
```

## SEE ALSO

**aeiou**(1), **aeiou-build**(1), **aeiou-datagen**(1), **aeiou-trace**(1),
**aeiou-abstract**(7).

The reference: `schema/README.md` §8 (the form and the rules), `builder/REFERENCE.md` §5
(the tool). `python -m aeiou.params` is the same program.
