# aeiou-params(1)

## NAME

aeiou-params - parameter files of an abstract: write its defaults as a set, validate sets, build a set from real files, lock a use case

## SYNOPSIS

```
aeiou-params defaults [-o FILE] [--doc TEXT] [--pin] ABSTRACT
aeiou-params check ABSTRACT FILES...
aeiou-params safetensors [--tp TP] [--column REGEX] [--row REGEX] [--doc TEXT] [--pin] -o FILE ABSTRACT SHARDS...
aeiou-params npz [-o FILE] [--member NAME] [--doc TEXT] [--pin] ABSTRACT ARCHIVES...
aeiou-params lock [--rules FILE] [-o FILE] [--workloads DIR] CASE
aeiou-params --help | --version
```

## DESCRIPTION

An abstract is a *shape*: named parameter slots whose defaults are placeholders or one
reference set. A **parameter file** (`<abstract>.<set>.params.json`) fills the slots with
values from configuration, measurement, or a trace, so one published shape carries several
parameter sets, and a fitted set is reviewed, hashed, and published on its own. The runner
applies a file with `aeiou run --params-file FILE` (repeatable, in order; `--param` applies
last), and so do `aeiou dry-run`, `aeiou datagen`, and **aeiou-datagen**(1).

The file is `{"params_version": 2, "abstract": NAME, "ast_sha256"?: HEX, "doc"?: TEXT,
"params": {name: entry, ...}, "run"?: {option: entry, ...}, "provenance"?: {...}}`, each
entry a value with its **state** (contract 0.11): `{"default": v}`, a value a later layer
may replace; `{"locked": v}`, final; or `{"required": true}`, no value, which a later layer
must give. A value has the form of a parameter default: a scalar, a distribution, or an
array of these, and it must keep the **kind** of the parameter it is for (its default's, or
a required parameter's declared `kind`), because the script fixed at build time how each
slot is consumed (a distribution is drawn, an array is indexed). `abstract` must be the
document's name; `ast_sha256`, when present, must be its hash; every name must be declared;
`gpus` may not appear. Under `aeiou run --params-file` a file never replaces what an earlier
one locked, and `--param` neither; a `run` table, the runner's identity options (`seed`,
`posix`, `cache`, `s3`, `time_scale`, `rank_rotate`, `write_compress`), is applied only from
a lock.

A **use case** is a case directory under `workloads/` (`cases/<case>/`) whose `case.toml`
lists each run's abstract and the fragments (parameter files) that set the job and the
accelerator; a **lock** (`case.lock.json`) is the use case resolved, the one file a run of
the use case names (`DESIGN_REVIEW.md` §3.78).

**aeiou-params** is the helper for these files. Five subcommands:

- **defaults** writes every default of an abstract as a set, to edit or to fit.
- **check** validates files against an abstract with the runner's rules.
- **safetensors** reads real safetensors shard headers and writes the tensor table the
  `safetensors_load` abstract takes as parameters.
- **npz** reads the archive framing the `npy_per_sample` abstract takes as parameters
  from real `.npz` files and compares it with the abstract's defaults.
- **lock** resolves a use case, and a benchmark's rules, into its lock.

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

  The `safetensors_load` abstract; its parameter names are checked.
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

  The `npy_per_sample` abstract.
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

### aeiou-params lock

- *CASE*

  The case directory, with its `case.toml`. Each `[[run]]` names its abstract (`ast`, in the
  mechanism's directory two levels up) and its fragments (`params`, one file or a list,
  relative to the case). The fragments of one run combine by **agreement**, in any order:
  two that set a name to different entries are refused, both named. A fragment may replace
  the abstract's default or require a value; nothing replaces a lock.
- **--rules** *FILE*

  A benchmark's rules file (`schema/rules.schema.json`): it names no parameter. Its
  `freeze` makes every default locked (`"defaults": "locked"`), every `scale` parameter
  required (`"scale": "required"`; one the use case locked is refused), and every identity
  option locked (`"identity_options": "locked"`); its `constraints`, infix expressions over
  the runner's `fact.` and `result.` names, are parsed into the abstract's expression form
  and copied into the lock with their text. A lock made with rules is a benchmark entry's,
  kept outside aeiou's tree.
- **-o**, **--out** *FILE*

  The lock to write. Default *CASE*`/case.lock.json`.
- **--workloads** *DIR*

  The `workloads/` directory the lock's paths are relative to. Default: the one above
  *CASE*.

The lock pins each run's abstract by its path below `workloads/` and its `ast_sha256`,
holds every parameter with its state and where it came from (`abstract`, `rules`, or a
fragment's path), and records the case, the fragments, and the rules file with their
hashes as provenance (a file outside `workloads/` by its name), so it is byte-identical on
every site. It is generated and never edited; **aeiou-build**(1) remakes the lock of every
committed case of an abstract it rebuilt, and `aeiou-catalog check` refuses a committed case
whose lock is missing or stale.

### Common

- **-h**, **--help**

  Print the help of the command or subcommand.
- **-V**, **--version**

  Print the version.

## ENVIRONMENT

- **AEIOU_SCHEMA_DIR**

  The directory holding the schemas, when the package is not run from the repository.

## FILES

- `<abstract>.<set>.params.json`, `params.json`

  A parameter file. The schema is `schema/params.schema.json`; the committed ones are under
  `workloads/` (a mechanism's `params/`, a case's `params.json`, a capture's
  `fitted.params.json`), each checked against its abstract in CI.
- `cases/<case>/case.lock.json`

  A use case's lock. The schema is `schema/lock.schema.json`; a rules file's is
  `schema/rules.schema.json`.

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
aeiou-params defaults workloads/input-pipeline/map/file-per-sample/file_per_sample.ast.json -o tsf.params.json --pin
aeiou-params check workloads/input-pipeline/map/file-per-sample/file_per_sample.ast.json tsf.params.json
aeiou-params safetensors workloads/model-loading/safetensors-load/safetensors_load.ast.json model-0000?-of-00004.safetensors \
    -o llama.params.json --tp 8
aeiou-params npz workloads/input-pipeline/map/npy-per-sample/npy_per_sample.ast.json corpus/train/00000/sample_0000000?.npz
aeiou run workloads/model-loading/safetensors-load/safetensors_load.ast.json --gpus 8 --root /mnt/sut --params-file llama.params.json
aeiou-params lock workloads/input-pipeline/map/npy-per-sample/cases/unet3d-h100
aeiou-params lock workloads/input-pipeline/map/npy-per-sample/cases/unet3d-h100 --rules bench.rules.json -o bench-unet3d-h100.lock.json
```

## SEE ALSO

**aeiou**(1), **aeiou-build**(1), **aeiou-datagen**(1), **aeiou-trace**(1),
**aeiou-abstract**(7).

The reference: `schema/README.md` §8 (the form and the rules), `builder/REFERENCE.md` §5
(the tool). `python -m aeiou.params` is the same program.
