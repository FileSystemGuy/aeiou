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
"Replace" is the only merge rule; no layered option is a list.

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
  abstract = ../schema/examples/train_small_files.ast.json   [cli]
  param = files=4000 steps=50                                [cli]
  gpus = 8                                                   [cli]
  seed = 1                                                   [cli]
  io-backend = the abstract's                                [default]
  root = /mnt/sut                                            [config /etc/aeiou.toml]
  threads = 8                                                [env AEIOU_THREADS]
  rank = 2                                                   [env AEIOU_RANK]
  config = /etc/aeiou.toml (sha256 68468a873b884760...)      [env AEIOU_CONFIG]
  WARNING: AEIOU_BOGUS is set and is no option of any subcommand; ignored
```

A refusal or cross-check of a layered option names the layer that set it when that was not
the command line: `--aio-depth is a libaio knob; --io-backend sync has no AIO context
(--aio-depth from config /etc/aeiou.toml)`. A value the user typed gets no suffix. The
checks made during the run name the flag only; its source is in the block above them.

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
threads = 16

[dry-run]
threads = 4
```

Refused, with the message naming the key: a key at the top level; an unknown key; a key
that is a fixed option of the table's subcommand; a `no-x` key (a boolean is a value under
its own name, `require-cold = false`, so the double negative cannot be written). A key that
is another subcommand's option is left to that subcommand. The `[datagen]` table is read by
both `aeiou datagen` and **aeiou-datagen**(1).

## ENVIRONMENT

- **AEIOU_CONFIG**

  The config file, when `--config` is not given.
- **`AEIOU_<FLAG>`**

  The value of a layered option, the long flag upper-cased with `_` for `-`. Booleans take
  `true`/`false`, `1`/`0`, `yes`/`no`, `on`/`off`. `AEIOU_NO_X` is refused with the
  spelling to use. A variable naming a fixed option is refused. A variable that is no option
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

**aeiou run.** Fixed: the abstract, `--gpus`, `--seed`, `--param`, `--params-file`,
`--io-backend`, `--expect-fingerprint`, `--expect-dataset-id`, `--clean-namespaces`,
`--ignore-limits`. Layered: `--root`, `--threads`, `--buffer-mib`, `--write-compress`,
`--time-scale`, the io_uring knobs (`--iowq-max-workers`, `--sqpoll`, `--sqpoll-shared`,
`--defer-taskrun`, `--coop-taskrun`), `--aio-depth`, `--mmap-mode`, `--mmap-consume`,
`--rank`, `--ranks`, `--coordinator`, `--rank-rotate`, `--max-gap`, `--require-cold`,
`--drop-caches`, `--report-json`, `--report-takes`.

**aeiou datagen** and **aeiou-datagen.** Fixed: the abstract, `--gpus`, `--param`,
`--params-file`, `--dedupe`, `--compress`, `--dataset` (the payload is part of what the
run compares). Layered: `--root`, `--threads`.

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
