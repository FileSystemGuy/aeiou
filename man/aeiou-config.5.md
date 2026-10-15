# aeiou-config(5)

## NAME

aeiou-config - the option layers of the aeiou tools: command line, environment, TOML config file, default

## SYNOPSIS

```
aeiou --config FILE SUBCOMMAND ...
aeiou-datagen --config FILE ...
AEIOU_CONFIG=FILE aeiou SUBCOMMAND ...
AEIOU_<FLAG>=VALUE aeiou SUBCOMMAND ...
```

## DESCRIPTION

Every option of every subcommand of **aeiou**(1), and of **aeiou-datagen**(1), is resolved
through four layers, a higher one replacing a lower one's value:

1. the command line;
2. the environment: `AEIOU_<FLAG>`, the long flag upper-cased with `_` for `-`
   (`AEIOU_RANK`, `AEIOU_BUFFER_MIB`);
3. the config file: TOML, named by `--config FILE` or `$AEIOU_CONFIG`, **never searched
   for**: there is no working-directory, home, or XDG path, because a file nobody remembers
   is how two hosts of one run come to differ;
4. the compiled default.

The layers exist for one case above all: `--rank` from the environment. A launcher sets
`AEIOU_RANK` from its own rank variable and the same command line runs on every host.
"Replace" is the only merge rule. The one list, `--endpoint`, is replaced whole: a list
from a higher layer is the list in effect, never merged with a lower one's. In the
environment its values are separated by whitespace (`AEIOU_ENDPOINT="kv=/a sysp=/b"`), in
the file it is an array of strings (`endpoint = ["kv=/a", "sysp=/b"]`).

**Two kinds of option.** A *fixed* option is the command line's alone, and the environment
and the file are refused when they name it: nothing the workload's identity, a dataset id,
or a safety check depends on may come from a layer the command line does not show. A
*layered* option may come from any layer. The man page of each tool marks its layered
options; the lists are in OPTIONS BY KIND below.

**The block.** Every invocation prints, after the `abstract` line, one line per option with
its value in effect and its source, fixed options first, the config file last, then any
warning, then an empty line. There is no flag to print the configuration: the block is always there.

```
options (cli > env > config > default)
  abstract = ../workloads/input-pipeline/map/file-per-sample/file_per_sample.ast.json   [cli]
  param = samples=4000 steps=50                              [cli]
  gpus = 8                                                   [cli]
  seed = 1                                                   [cli]
  posix = the abstract's                                     [default]
  cache = the abstract's                                     [default]
  s3 = the abstract's                                        [default]
  root = /mnt/sut                                            [config /etc/aeiou.toml]
  threads = 8                                                [env AEIOU_THREADS]
  rank = 2                                                   [env AEIOU_RANK]
  config = /etc/aeiou.toml (sha256 68468a873b884760...)      [env AEIOU_CONFIG]
  WARNING: AEIOU_BOGUS is set and is no option of any subcommand; ignored
```

A refusal or cross-check of a layered option names the layer that set it when that was not
the command line: `--aio-depth 0: a context needs room for a request (--aio-depth from
config /etc/aeiou.toml)`. A value the user typed gets no suffix. The checks made during the
run name the flag only; its source is in the block above them.

**A knob the run cannot use.** A backend knob the run's API has no use for (`--threads`
under a thread-per-actor API, the io_uring knobs without a ring, `--aio-depth` without
libaio, the mmap knobs without `mmap`, any POSIX knob when every name is `s3`) and an
object-engine option with no name in an object store (`--object-threads`,
`--object-part-mib`) are refused from the command line, which asked for something that
would change nothing. From the environment or the file, which serve every run a host or a
cluster makes, the value is passed over: it leaves the block, and a warning says why
(`WARNING: --threads from config /etc/aeiou.toml is not used: this run has one thread per
actor`). So one file can carry the knobs of every backend.

**The defaults in effect.** A default the program resolves is shown as what it comes to:
`writers = 20 (all cores)`, `threads = one per actor`, `object-threads = 2`. The report's
`layers` keeps `null` for it, the default as given.

**Values** (**aeiou**(1)). An empty value (`AEIOU_ROOT=`, `root = ""`), where the command line would refuse
an empty argument, is refused naming the variable or the key; an empty path is never the
working directory. A value that is not UTF-8 is taken as it is for a path and refused for
text, naming the variable; so is such an argument on the command line. A variable outside
the family that is not UTF-8 is none of the tools' business.

**The record.** `aeiou run` writes the block to its JSON report as `layers` (the config
file's path and sha256, the environment variables that contributed, every option with its
value and source), sends it to the coordinator, and, on rank 0 of several hosts, records
every host's block as `hosts` and prints after the gate the options whose values differ
between hosts. The most likely way a layer goes wrong across hosts is ssh itself: a
non-interactive shell does not read the profile an interactive one does.

## FILE FORMAT

TOML. One table per subcommand, keys spelled as the long flags, values typed as the flag's
value: integers, strings, booleans.

```toml
[run]
root = "/mnt/sut"
threads = 8
buffer-mib = 16
require-cold = true
report-json = "/var/tmp/aeiou-run.json"

[datagen]
root = "/mnt/sut"
writers = 16

[dry-run]
threads = 4
```

Refused, with the message naming the key: a key at the top level; an unknown key; a key
that is a fixed option of the table's subcommand; a `no-x` key (a boolean is a value under
its own name, `require-cold = false`, so the double negative cannot be written; the negation
of a fixed boolean, `no-metrics`, is the command line's alone); a value of the wrong type,
named as TOML has it (`expected an integer, got the string "3"`); an empty string. A key that
is another subcommand's option is left to that subcommand. The `[datagen]` table is read by
both `aeiou datagen` and **aeiou-datagen**(1); its `root` and `endpoint` are passed over, with
a warning, by `--resume`, `--grow`, `--shrink`, and `--settle`, which take the place from the
dataset's manifest, so one file serves the datagen and the changes of what it wrote.

## ENVIRONMENT

- **AEIOU_CONFIG**

  The config file, when `--config` is not given.
- **`AEIOU_<FLAG>`**

  The value of a layered option, the long flag upper-cased with `_` for `-`. Booleans take
  `true`/`false`, `1`/`0`, `yes`/`no`, `on`/`off`. `AEIOU_NO_X` is refused with the
  spelling to use, `AEIOU_X=false` for a layered boolean, `--no-x` on the command line for a
  fixed one. A variable naming a fixed option is refused, as is an empty one. A variable that is no option
  of any subcommand is warned about and ignored; one that is another subcommand's option is
  left alone.
- **AEIOU_SCHEMA_DIR**, **AEIOU_RUNNER**, **AEIOU_RSH**

  The family's own, not options: the schema directory for the Python tools, the runner
  binary for the Python test suite, the remote shell for **aeiou-launch**(1). They draw no
  warning.

## NEGATION

Every boolean of every subcommand reads `--[no-]x`: `--x` turns it on, `--no-x` turns it
off, and the last one on the line wins, so the command line can turn off what the file
turned on, and a wrapper script can append to a line it did not write. The help shows the
pair as one row, `--[no-]x`, and says so once at its foot. In the environment and the file
a boolean is a value under its own name. The cross-checks between booleans and the options
they need (`--sqpoll-shared` needs `--sqpoll`, `--defer-taskrun` excludes it,
`--report-takes` needs `--report-json`) are made on the resolved values, since the two
sides may come from different layers.

## OPTIONS BY KIND

**aeiou run.** Fixed: the abstract, `--gpus`, `--seed`, `--param`, `--params-file`, `--subset`,
`--posix`, `--cache`, `--s3`, `--expect-fingerprint`, `--expect-dataset-id`, `--clean-namespaces`,
`--ignore-limits`. Layered: `--root`, `--endpoint`, `--threads`, `--buffer-mib`, `--write-compress`,
`--time-scale`, the io_uring knobs (`--iowq-max-workers`, `--sqpoll`, `--sqpoll-shared`,
`--defer-taskrun`, `--coop-taskrun`), `--aio-depth`, `--mmap-mode`, `--mmap-consume`,
`--object-threads`, `--object-part-mib`, `--rank`, `--ranks`, `--coordinator`, `--rank-rotate`, `--max-gap`, `--require-cold`,
`--drop-caches`, `--report-json`, `--report-takes`.

**aeiou datagen** and **aeiou-datagen.** Fixed: the abstract, `--gpus`, `--param`,
`--params-file`, `--dedupe`, `--compress`, `--dataset` (the payload is part of what the
run compares), and for **aeiou datagen** `--ignore-space` (a safety check) and `--resume`,
`--grow`, `--shrink`, `--settle` (the dataset root changed). Layered: `--root`,
`--endpoint`, `--writers`, `--rank`, `--ranks`, `--coordinator`, and for **aeiou datagen**
`--object-threads` and `--progress-secs`.

**aeiou dry-run.** Fixed: the identity (the abstract, `--gpus`, `--seed`, `--param`,
`--params-file`) and the output and metrics flags. Layered: `--threads`, `--ranks`.

**aeiou check.** Only `--config` itself; the block shows the files and the config file.

`--root` is required from some layer for `run` and `datagen`; what no layer supplies is
listed with everything else the command lacks, in one usage error.

## EXAMPLES

One file for a cluster, one command line for every host:

```
# /etc/aeiou.toml on every host
[run]
root = "/mnt/sut"
threads = 8
require-cold = true

# the launcher sets AEIOU_RANK per host; AEIOU_CONFIG is in the service environment
AEIOU_CONFIG=/etc/aeiou.toml AEIOU_RANK=$SLURM_PROCID \
    aeiou run x.ast.json --gpus 64 --seed 1 --ranks 8 --coordinator n1:7311 --no-require-cold
```

The last line turns off, for this run, the cold requirement the file turned on.

## SEE ALSO

**aeiou**(1), **aeiou-datagen**(1), **aeiou-launch**(1).

The reference: `runner/REFERENCE.md` §14 (the layers) and §15 (usage errors);
`DESIGN_REVIEW.md` §3.60 for the reasoning.
