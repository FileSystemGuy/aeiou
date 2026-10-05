# aeiou-launch(1)

## NAME

aeiou-launch - start one `aeiou run` or `aeiou datagen` rank per host over ssh

## SYNOPSIS

```
aeiou-launch [-p PORT] [-f FILE] [HOSTS...] -- aeiou run ARGS...
aeiou-launch [-p PORT] [-f FILE] [HOSTS...] -- aeiou datagen ARGS...
aeiou-launch --help
```

## DESCRIPTION

**aeiou-launch** is the multi-host launcher of the suite, a POSIX shell script. It starts
the `aeiou run` or `aeiou datagen` command line given after `--` on every host named, rank
0 first, appending `--ranks`, `--rank`, and `--coordinator` to each: the *i*-th host is rank
*i*, and `HOST0:PORT` is the coordinator address rank 0 listens on and every rank connects
to, so the first host must be reachable from the others under that name. Each host's output
is prefixed with its rank and host. The exit status is non-zero if any rank's is.

The hosts are the command line's, then the file's (`-f`). A name may carry one bracket range
as **pdsh**(1) writes it: `n[1-4]` is `n1` to `n4`, `n[01-10]` keeps the zero padding,
`n[1,3,5-7]` lists and ranges, `n[1-4]-ib` a suffix. (The shell's own `n{1..4}` works too;
it is expanded before this script runs.) Two hundred hosts are a short line or a file.

A `datagen` on several hosts writes the corpus once, each host its slice (**aeiou**(1),
datagen); it is the same loop with the same three flags appended.

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

  After `--`: the `aeiou run` or `aeiou datagen` command line every host runs. `--ranks`,
  `--rank`, and `--coordinator` are appended.
- **-p** *PORT*

  The coordinator's port. Default 7311.
- **-f** *FILE*

  Hosts from a file too: one or more per line, `#` to the end of a line is a comment,
  bracket ranges as above.
- **-h**, **--help**

  Print the help.

## ENVIRONMENT

- **AEIOU_RSH**

  Replaces `ssh`. It is called as `$AEIOU_RSH HOST COMMAND`.

## EXIT STATUS

- **0**

  Every rank exited 0.
- **1**

  Some rank failed; the run's verdict is the same on every host, so this is the run's
  failure.
- **2**

  A usage error, in the frame every tool of the suite shares.

## EXAMPLES

The corpus written by the fifty hosts that will run against it, then a run on them:

```
aeiou-launch 'client[01-50]' -- aeiou datagen train_small_files.ast.json --gpus 400 --root /mnt/sut
aeiou-launch 'client[01-50]' -- aeiou run train_small_files.ast.json --gpus 400 --root /mnt/sut --drop-caches
```

A write run on four hosts, then the restore on the same hosts with each reading what another
wrote:

```
aeiou-launch n1 n2 n3 n4 -- aeiou run ckpt_write_dcp.ast.json --gpus 32 --root /mnt/sut --seed 7
aeiou-launch n1 n2 n3 n4 -- aeiou run ckpt_restore.ast.json --gpus 32 --root /mnt/sut --seed 7 \
    --rank-rotate 1 --max-gap 30 --require-cold
```

Two hundred hosts from a file:

```
aeiou-launch -f hosts.txt -- aeiou run model_load.ast.json --gpus 1600 --root /mnt/sut
```

## SEE ALSO

**aeiou**(1), **aeiou-config**(5), **pdsh**(1).

The reference: `runner/REFERENCE.md` §6 (the coordinator), §14 (the option layers across
hosts).
