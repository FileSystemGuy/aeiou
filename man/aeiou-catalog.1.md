# aeiou-catalog(1)

## NAME

aeiou-catalog - the workload catalog: check the records under workloads/ against their dictionary, measure each case's shape, render CATALOG.md and each case's description

## SYNOPSIS

```
aeiou-catalog check [--workloads DIR] [--runner PATH] [--no-shapes]
aeiou-catalog write [--workloads DIR] [--runner PATH] [--no-shapes]
aeiou-catalog describe [--params-file FILE] [--gpus N] [--workloads DIR] [--runner PATH] [--no-shapes] AST
aeiou-catalog --help | --version
```

## DESCRIPTION

`workloads/` holds one directory per I/O mechanism (an authoring script and the ASTs it
emits) and, under `cases/`, one directory per **case**: an AST and a parameter set, a
workload a user would ask for, with the captures traced under it. **aeiou-catalog** keeps
the records of that tree consistent with their dictionary and with the abstracts they
describe, and renders the catalog a reader searches. Building an abstract is
**aeiou-build**(1)'s; this tool never writes an AST or a parameter file.

The records are TOML files a person writes:

- `<category>/[<class>/]<mechanism>/mechanism.toml`: the curated attributes that hold for
  every case (`category`; `class` where the category has classes; `application`, `format`;
  `phase` and `domain`, each one value or a list), and a `[defaults]` table
  (`ast`, `gpus`, `length`) when the mechanism has no case yet, so that its row shows the
  abstract at its defaults.
- `cases/<case>/case.toml`: `kind`, `modality`, and `model-family`; `gpus`, the GPU counts its parameters
  hold at (the shape is measured at the first); `length`, the parameter that only makes the
  run longer (`requests`, `queries`, `calls`), absent when none does (a training job's `steps`
  is not one: fewer steps than an epoch drop the next epoch's re-reads);
  `metrics_block`, the unit of the locality metrics (default 4096); and one `[[run]]`
  (`ast`, `params`) per abstract it runs, a writer and its reader being two.
- `captures/<capture>/capture.toml`: `raw_trace` (where the trace is, or `"not retained"`)
  and one `[[pair]]` per trace: `trace` (the stem of its `.metrics.json`), `ast`, `params`
  (a file of the capture, when its fit differs from the case's), and `verdict`, what
  **aeiou-trace**(1) `compare --judge` gives on it (the builder's tests check it).

`workloads/attributes.toml` is the dictionary: every attribute with its meaning, its source
(curated, derived, measured), its level (mechanism, case), and its values or classes. An
attribute marked `multiple` takes a list of distinct values (a mechanism serving two phases
lives in one directory); every other takes one. A record may state only curated attributes
of its level, with defined values; an unknown key, an unknown value, a missing required
attribute, a category or class that does not match the directory, a parameter file no run
names, a trace no pair judges, and two ASTs of one name are refused.

The tree follows the dictionary. Every value of `category` is a directory of the tree's first
level and nothing else is; a value of `class` names its category, and a category with classes
holds only their directories. Every category and class has its directory from the start,
with a `README.md` **write** generates from the dictionary (what belongs there, the
mechanisms there now), so that the tree shows what the catalog means to cover and where a
contribution goes.

The **derived** attributes are read from the AST (`api` from its `posix` key, `writes` and
`inputs` from its namespaces) and the captures (`status`, the best verdict). The
**measured** ones come from `aeiou dry-run --metrics-json` of each run, written to the
case's `shape.json` (`shape.defaults.json` in a mechanism with no case), at the case's own
parameters and first GPU count: the median of seeds 1, 2, and 3, their spread, and the
numbers on which the seeds differ by more than the attribute's tolerance (the case's own
statistical error, shown in the catalog). A shorter run is never taken for the case: on a
fixed dataset a reuse share grows with the run (a DiskANN search over SIFT1M re-reads 24 % of
its blocks at 500 queries and 97 % at 32,000), and a fixed load is amortized only over a long
one. For a case with a length parameter, **write** also halves it, down to 1/64, while the
numbers stay the case's within their tolerances, and records the smallest such length;
**check** re-measures there, so a long case costs its full length three times only when it is
written, and once (the first seed, for the counts below) when it is checked.
Sizes are byte-weighted medians (half the bytes move in runs or requests at least this
large), read from the dry run's quarter-octave histograms, so a size is a bucket's lower
bound. The shape also keeps the first seed's op counts by kind, its bytes read and written,
and each dataset's projection (samples, files, directories, bytes; `shape_version` 2 since
2026-10-14). `workloads/CATALOG.md` is rendered from the records, the shapes, and the
dictionary: the cases, a paragraph on each, their shapes, and the glossary.

Each case also has a **description**, `DESCRIPTION.md` in its directory
(`DESCRIPTION.defaults.md` in a mechanism with no case): the abstract in prose, for readers
who will not read the JSON. It is generated, not written: a deterministic walk of the AST at
the case's parameters, joined with the records and the shape, so a parameter change shows up
as a prose diff and **check** fails a description that drifted. Two lengths: a paragraph
(what the AST's `doc` says, where the catalog files it, what each dataset holds, what each
actor does in order with the parameters' values in place and the branches they switch off
left out, and the measured shape), and an account (every parameter with its value, default,
unit, and `doc`; every dataset and namespace; every actor's body in order, each op named, a
branch marked taken or not here; the shape in full). An expression over parameters and
literals is worked out and shown with its formula, one that depends on a loop index, a
binding, or a draw stays a formula, and a distribution is given in words with percentiles
computed from its parameters. It says what the abstract does, not why the application does
it (the mechanism's README) or what a storage system should do about it.

## OPTIONS

### aeiou-catalog check

Validate the records, re-measure every shape at its recorded check length and render
`CATALOG.md`, and fail when a committed file differs from what is derived now, a shape was
measured on another AST, parameter file, or case record than the one committed, or the dry run
no longer gives its measures (within their tolerances) or, exactly, its op counts, bytes, and
dataset projections (the first seed's at the case's own length, which the descriptions print).

- **--workloads** *DIR*

  The workload tree. Default `workloads`.
- **--runner** *PATH*

  The runner binary the shapes are measured with. Default `aeiou`, found on the `PATH`.
- **--no-shapes**

  No dry run and no runner: check only that each shape file is there, of this tool's
  `shape_version`, and was measured on the AST and the parameter file committed now (their
  SHA-256 in `shape.json`) and on the case's GPU count, block, and length parameter, and
  render `CATALOG.md` and the descriptions from the shapes.

### aeiou-catalog write

Make any category or class directory the dictionary defines and the tree lacks, validate
the records, measure every shape at the case's own parameters and find its check length,
and write the shapes, the descriptions, `CATALOG.md`, and the category and class READMEs
where they changed.

- **--workloads** *DIR*

  The workload tree. Default `workloads`.
- **--runner** *PATH*

  The runner binary. Default `aeiou`, found on the `PATH`.
- **--no-shapes**

  Write `CATALOG.md`, the descriptions, and the READMEs alone, from the committed shapes; a
  shape measured on another AST or parameter file is reported.

### aeiou-catalog describe

Print the description of any abstract, inside the catalog or not, at its defaults or a
parameter file, to standard output: the same paragraph and account as a case's
`DESCRIPTION.md`, without the records' context, and with its shape measured as **write**
measures a case's (seeds 1, 2, and 3), no length halved.

- *AST*

  The abstract (`*.ast.json`).
- **--params-file** *FILE*

  A parameter file of it (**aeiou-params**(1)); its defaults when absent.
- **--gpus** *N*

  The GPU count. Default 1.
- **--workloads** *DIR*

  The workload tree whose dictionary classes the shape. Default `workloads`.
- **--runner** *PATH*

  The runner binary the shape is measured with. Default `aeiou`, found on the `PATH`.
- **--no-shapes**

  No dry run: the description without its measured shape, op counts, or dataset
  projections.

### Common

- **-h**, **--help**

  Print the help of the command or subcommand.
- **-V**, **--version**

  Print the version.

## FILES

- `workloads/attributes.toml`

  The dictionary.
- `mechanism.toml`, `case.toml`, `capture.toml`

  The records, curated.
- `shape.json`, `shape.defaults.json`, `DESCRIPTION.md`, `DESCRIPTION.defaults.md`,
  `workloads/CATALOG.md`, the `README.md` of every category and class directory

  Generated; never edited by hand.

## EXIT STATUS

- **0**

  The records are valid and every generated file is what is derived now (`check`), was
  written (`write`), or the description was printed (`describe`).
- **1**

  A record that breaks a rule, a generated file that drifted or a shape the dry run no longer
  gives (`check`), a stale shape under `--no-shapes`, a dry run that failed, or (`describe`)
  an abstract or parameter file the schema rejects.
- **2**

  A usage error, in the frame every tool of the suite shares: among them `--gpus` below 1,
  or an abstract that is not there or not JSON.

## EXAMPLES

```
aeiou-catalog check --no-shapes
aeiou-catalog check --runner runner/target/release/aeiou
aeiou-catalog write --runner runner/target/release/aeiou
aeiou-catalog describe --params-file cases/npy-8mib/params.json --no-shapes npy_per_sample.ast.json
```

## SEE ALSO

**aeiou**(1), **aeiou-build**(1), **aeiou-params**(1), **aeiou-trace**(1).

The layout and the reasoning: `workloads/README.md`, `DESIGN_REVIEW.md` §3.66 (the
descriptions: §3.76); the tool:
`builder/REFERENCE.md`. `python -m aeiou.catalog` is the same program.
