# aeiou-catalog(1)

## NAME

aeiou-catalog - the workload catalog: check the records under workloads/ against their dictionary, measure each case's shape, render CATALOG.md

## SYNOPSIS

```
aeiou-catalog check [--workloads DIR] [--runner PATH] [--no-shapes]
aeiou-catalog write [--workloads DIR] [--runner PATH] [--no-shapes]
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

- `<mechanism>/mechanism.toml`: the curated attributes that hold for every case
  (`category`, `access` under training, `application`, `format`), and a `[defaults]` table
  (`ast`, `gpus`, `length`) when the mechanism has no case yet, so that its row shows the
  abstract at its defaults.
- `cases/<case>/case.toml`: `kind` and `modality`; `gpus`, the GPU counts its parameters
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
(curated, derived, measured), its level (mechanism, case), and its values or classes. A
record may state only curated attributes of its level, with defined values; an unknown key,
an unknown value, a missing required attribute, a category or access that does not match the
directory, a parameter file no run names, a trace no pair judges, and two ASTs of one name
are refused.

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
**check** re-measures there, so a long case costs its full length once, when it is written.
Sizes are byte-weighted medians (half the bytes move in runs or requests at least this
large), read from the dry run's quarter-octave histograms, so a size is a bucket's lower
bound. `workloads/CATALOG.md` is rendered from the records, the shapes, and the
dictionary: the cases, their shapes, and the glossary.

## OPTIONS

### aeiou-catalog check

Validate the records, re-measure every shape at its recorded check length and render
`CATALOG.md`, and fail when a committed file differs from what is derived now, or a shape was
measured on another AST, parameter file, or case record than the one committed.

- **--workloads** *DIR*

  The workload tree. Default `workloads`.
- **--runner** *PATH*

  The runner binary the shapes are measured with. Default `aeiou`, found on the `PATH`.
- **--no-shapes**

  No dry run and no runner: check only that each shape file is there and was measured on
  the AST and the parameter file committed now (their SHA-256 in `shape.json`) and on the
  case's GPU count, block, and length parameter, and render `CATALOG.md` from the shapes.

### aeiou-catalog write

Validate the records, measure every shape at the case's own parameters and find its check
length, and write the shapes and `CATALOG.md` where they changed.

- **--workloads** *DIR*

  The workload tree. Default `workloads`.
- **--runner** *PATH*

  The runner binary. Default `aeiou`, found on the `PATH`.
- **--no-shapes**

  Write `CATALOG.md` alone, from the committed shapes; a shape measured on another AST or
  parameter file is reported.

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
- `shape.json`, `shape.defaults.json`, `workloads/CATALOG.md`

  Generated; never edited by hand.

## EXIT STATUS

- **0**

  The records are valid and every generated file is what is derived now (`check`), or was
  written (`write`).
- **1**

  A record that breaks a rule, a generated file that drifted or a shape the dry run no longer
  gives (`check`), a stale shape under `--no-shapes`, or a dry run that failed.
- **2**

  A usage error, in the frame every tool of the suite shares.

## EXAMPLES

```
aeiou-catalog check --no-shapes
aeiou-catalog check --runner runner/target/release/aeiou
aeiou-catalog write --runner runner/target/release/aeiou
```

## SEE ALSO

**aeiou**(1), **aeiou-build**(1), **aeiou-params**(1), **aeiou-trace**(1).

The layout and the reasoning: `workloads/README.md`, `DESIGN_REVIEW.md` §3.66; the tool:
`builder/REFERENCE.md`. `python -m aeiou.catalog` is the same program.
