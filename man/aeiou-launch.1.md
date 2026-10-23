# aeiou-launch(1)

## NAME

aeiou-launch - start one `aeiou run`, `aeiou datagen`, or `aeiou-datagen` rank per host over ssh

## SYNOPSIS

```
aeiou-launch [-p PORT] [-f FILE] [HOSTS...] -- aeiou run ARGS...
aeiou-launch [-p PORT] [-f FILE] [HOSTS...] -- aeiou datagen ARGS...
aeiou-launch [-p PORT] [-f FILE] [HOSTS...] -- aeiou-datagen ARGS...
aeiou-launch --help
```

## DESCRIPTION

**aeiou-launch** is the multi-host launcher of the suite, a POSIX shell script. It starts
the `aeiou run`, `aeiou datagen`, or `aeiou-datagen` command line given after `--` on every
host named, rank 0 first, appending `--ranks`, `--rank`, and `--coordinator` to each: the *i*-th host is rank
*i*, and `HOST0:PORT` is the coordinator address rank 0 listens on and every rank connects
to, so the first host must be reachable from the others under that name. Each host's output
is prefixed with its rank and host, a line at a time as it arrives, every line whole even when
the output is a file or a pipe. The exit status is non-zero if any rank's is.

The hosts are the command line's, then the file's (`-f`). A name may carry one bracket range
as **pdsh**(1) writes it: `n[1-4]` is `n1` to `n4`, `n[01-10]` keeps the zero padding,
`n[1,3,5-7]` lists and ranges, `n[1-4]-ib` a suffix. (The shell's own `n{1..4}` works too;
it is expanded before this script runs.) Two hundred hosts are a short line or a file.

A `datagen` on several hosts writes the corpus once, each host its slice (**aeiou**(1),
datagen); it is the same loop with the same three flags appended. **aeiou-datagen**(1), the
writer of container datasets, takes the same three flags and speaks the same coordinator
protocol.

There is no MPI. Cross-host coordination is the runner's own TCP coordinator
(**aeiou**(1), SEVERAL HOSTS); this script is the ssh loop around it. The remote shell is
non-interactive, so a variable set in the operator's terminal is absent on the remote
hosts; the options block every rank prints, and the differences table rank 0 prints after
the gate, show what each host resolved.

## OPTIONS

- *HOSTS...*

  The hosts, one rank each, bracket ranges allowed. The first (of the command line's and
  then the file's) is rank 0 and the coordinator.
- *ARGS...*

  After `--`: the `aeiou run`, `aeiou datagen`, or `aeiou-datagen` command line every host
  runs. `--ranks`,
  `--rank`, and `--coordinator` are appended.
- **-p** *PORT*

  The coordinator's port. Default 7311.
- **-f** *FILE*

  Hosts from a file too: one or more per line, `#` to the end of a line is a comment,
  bracket ranges as above.
- **-h**, **--help**

  Print the help.

## SIGNALS

Ctrl-C (`SIGINT`) on **aeiou-launch** stops every host: it sends `SIGINT` to the command on
each host, prints `aeiou-launch: SIGINT: stopping every rank`, waits for every rank, names
each rank that exited non-zero or was not started, and exits 130; what each command does on `SIGINT` is its own
(**aeiou**(1): a `datagen` stops every host and leaves its manifests `stopped`). A second
Ctrl-C sends a second `SIGINT`, which makes an `aeiou` exit at once. `SIGTERM` is forwarded
the same way, and the exit status is then 143. A rank not yet started when the signal came
is not started.

A command run over ssh without a terminal gets no signal when the session ends, so the
signal travels over the session's standard input instead: each remote command line is
preceded by a small watcher, in the remote shell, that reads that input and sends the
signal named by each line, `INT` or `TERM`, to the command (which the remote shell `exec`s,
so it keeps the shell's pid); the end of the input with no line, when the command has
exited or ssh has gone, signals nothing. The command's own standard input is `/dev/null`. So
the remote login shell must be a POSIX shell, and the ssh of `AEIOU_RSH` must pass its
standard input to the remote command (no `ssh -n`); no terminal (`ssh -t`) is needed or
wanted. If **aeiou-launch** itself is killed (`SIGKILL`), the commands are not stopped.

## ENVIRONMENT

- **AEIOU_RSH**

  Replaces `ssh`. It is called as `$AEIOU_RSH HOST COMMAND`, with the launcher's signals
  arriving as lines on its standard input, which it must pass to COMMAND (SIGNALS).

## EXIT STATUS

- **0**

  Every rank exited 0.
- **1**

  Some rank failed, or was not started; each is named (`aeiou-launch: rank 3 (n4) exited
  with status 1`). The run's verdict is the same on every host, so this is the run's
  failure.
- **2**

  A usage error, in the frame every tool of the suite shares.
- **3**

  The run of a use case's lock completed and did not pass a constraint known after it:
  rank 0 exited 3 and no rank failed otherwise (**aeiou**(1), EXIT STATUS).
- **130**, **143**

  Stopped by `SIGINT` (Ctrl-C) or `SIGTERM`, forwarded to every rank (SIGNALS).

## EXAMPLES

The corpus written by the fifty hosts that will run against it, then a run on them:

```
aeiou-launch 'client[01-50]' -- aeiou datagen --case jpeg-50m --gpus 400 --root /mnt/sut
aeiou-launch 'client[01-50]' -- aeiou run --case jpeg-50m --gpus 400 --root /mnt/sut --drop-caches
```

A write run on four hosts, then the restore on the same hosts with each reading what another
wrote; `save.lock.json` and `load.lock.json` are the locks of a `dcp_save` case and of the
`dcp_load` case that reads its checkpoint, both holding at 32 GPUs:

```
aeiou-launch n1 n2 n3 n4 -- aeiou run --case-file save.lock.json --gpus 32 --root /mnt/sut --seed 7
aeiou-launch n1 n2 n3 n4 -- aeiou run --case-file load.lock.json --gpus 32 --root /mnt/sut --seed 7 \
    --rank-rotate 1 --max-gap 30 --require-cold
```

Two hundred hosts from a file:

```
aeiou-launch -f hosts.txt -- aeiou run safetensors_load.ast.json --gpus 1600 --root /mnt/sut
```

## SEE ALSO

**aeiou**(1), **aeiou-datagen**(1), **aeiou-config**(5), **pdsh**(1).

The reference: `runner/REFERENCE.md` §6 (the coordinator), §14 (the option layers across
hosts).
