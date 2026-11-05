# Capturing a trace of your application

This guide is for the owner of a real application whose storage workload aeiou is to
model: a training job, a checkpointer, a vector database, a KV cache. It needs the release
tarball's `bin/aeiou-trace` (no Python), and `strace` 5.3 or later from your distribution
or, to trace without slowing the application, the eBPF source (below).
With an object store, it also needs the tarball's `lib/libaeiou_capture.so`, which the tool
finds beside its `bin/`. The reference is **aeiou-trace**(1), capture (`man/aeiou-trace.1.md`); the reasoning is
`DESIGN_REVIEW.md` §3.96 in the repository.

A capture starts your command under a trace, as you, in your environment, and writes one
directory: `trace.txt`, the trace, and `capture.json`, the record of what was captured and
how (the host, the kernel, the mounts of your data, the clocks, the exact trace line, your
command, its exit status). Keep the directory whole: the record is what lets someone who
was not there read the trace.

## Before you start

- Run the capture as the user the application runs as. Never under `sudo`: that resets
  `PATH`, `LD_LIBRARY_PATH`, and your virtualenv, and the application you trace is no longer
  the one you run. The tool warns when it runs as root and records it.
- Know where your data is: each directory the application reads or writes is a `--root`.
  Only calls on paths under a root are measured, and each root's mount (file system, server,
  export, options) is recorded.
- Start under the capture everything that does the job's I/O. Only the command and the
  processes it starts are traced: other programs on the node (daemons, cron jobs, agents) are
  left out, which is what you want, but so is a server you started apart (a vector database,
  an inference server with its KV cache), work handed to `systemd-run` or `docker exec`, and a
  FUSE or cache daemon, and nothing in the capture can tell you it missed them. Programs that
  are not the job still load the storage while you trace: stop what you can (a backup, an
  `updatedb` that crawls the mount).
- Put the capture directory (`-o`) on a disk local to the node, outside every root. A trace
  grows by hundreds of megabytes a minute on a busy job; written to the storage under test,
  it is load on it, and written under a root, it traces itself. The checks refuse both.
- Under strace the trace slows the application: strace stops each traced thread at each
  traced call (88 µs a call on the measurements of §3.96, with `--seccomp-bpf`). A capture
  under strace is for the shape of the workload, never for its speed. The eBPF source costs
  a fraction of a microsecond a call.

## The eBPF source

Where an administrator can grant it once per node, `--source ebpf` traces the same calls
with probes in the kernel instead of stopping your threads. It needs root or two file
capabilities on a copy of the tool on the node's local disk (not over NFS):

```
cp bin/aeiou-trace /usr/local/bin/aeiou-trace-ebpf
sudo setcap cap_bpf,cap_perfmon+ep /usr/local/bin/aeiou-trace-ebpf
```

Limit who can run that copy (a group, `chmod 750`): it can watch any process on the host.
The tool gives up both capabilities before it starts your command, which runs as you, in
your environment. The kernel needs BTF (`/sys/kernel/btf/vmlinux`, every recent
distribution's). `--source auto`, the default, takes eBPF when the copy can load its probes
and strace when it cannot, and the `source` check says which and why.

It also sees what strace cannot: the requests an application hands the kernel through
`io_uring` (LMCache does for some of its operations), each
recorded when the kernel takes it and when it completes. Under strace a ring's I/O is only its
`io_uring_enter` calls, so for such an application use the eBPF source. The record's `uring`
says whether this kernel's `io_uring` was traced, and why not.

And it sees reads and writes through a mapping (`mmap`), which strace cannot: each page fault
on a mapped file, where the application first touched a page, written as runs of faults.
A page beside a faulting one is mapped with it where the kernel's fault-around is above one
page, and the capture checks that before it starts (the `fault-around` check). (A fault also
maps the whole folio its page is in, up to 2 MiB, however the file came to be cached; no
setting changes that, and the record's `not_seen` says so.) As root it sets it for the capture and
puts it back. Under the file capabilities it cannot, so, where the check fails, have root run
before the capture

```
cat /sys/kernel/debug/fault_around_bytes          # note it: 65536 by default
echo 4096 | sudo tee /sys/kernel/debug/fault_around_bytes
```

and put the noted value back after. Every program on the host takes more faults while it is
set. If you cannot, `--accept fault-around` captures anyway, and the record says the faults
missed some first touches. The record's `faults` says whether faults were traced, and why not.

For an application that uses libaio (`io_submit`), it records when the kernel completed each
request, where strace shows only when the application collected the result, and sees results
the application reads from the ring's memory with no call. The record's `aio` says whether
libaio's completions were traced, and why not.

The probes cannot make your application wait, so when the application makes calls faster
than the tool writes them down, the ring buffer fills and calls are lost. Any loss ends the
capture with a box saying how many, where, and what to do: usually `--buffer 1GiB` (or what
the box suggests), or `-o` on a faster disk, or `--source strace`. The record says
`complete: false`; the trace is still readable, and marks each gap. The capture directory
also keeps `syscalls.bin`, the probes' own records; send it with the rest.

## One node

```
aeiou-trace capture --root /mnt/data -o /local/cap-%h -- python train.py /mnt/data/train
```

The tool checks the node first and prints the table: the source can trace a child here, each
root has a mount, the clock is synchronized, `-o` is local, outside the roots, with a
gigabyte free, and no other capture runs on the host. A check that fails starts nothing and
says why. If you know why and the capture is still worth taking, waive that check by name
and it is recorded: `--accept clock`. `--check` runs the checks alone, to plan.

The tool exits with your command's status. Ctrl-C, or a `SIGTERM` sent to the tool (a
scheduler's time limit), reaches the command; the record says the capture was interrupted,
and the trace up to then is kept.

`%h` in `-o` is the host name and `%p` the pid, so one line serves every node.

## Several nodes

Over password-less ssh, `aeiou-launch` starts one capture per host:

```
aeiou-launch n[1-4] -- aeiou-trace capture --source strace --root /mnt/data -o /local/cap-%h -- python train.py
```

Every node checks itself, rank 0 (the first host) compares them: one release of the tool,
one strace version, every root mounted alike (a node mounted with another `vers`, `rsize`,
or `nconnect` is the classic bad node), every clock within 10 ms of rank 0's. A check failing
on any node starts no node, and every node prints which node and which check. Then every
command starts together, after every node is ready. Rank 0 writes `capture-set.json` beside
its capture: each node's directory, source, clock offset, exit status, and trace hash. The
traces stay on each node's disk; collect them from there.

When your application has its own launcher (`mpirun`, `srun`), put the capture inside its
line and give each node `--ranks`, `--rank`, and `--coordinator HOST:PORT` from the
launcher's environment. Several captures on one node (one per `mpirun` rank) share a
`--capture-id`; under Slurm it defaults to `$SLURM_JOB_ID`.

## Objects

Name each object store the application talks to with `--endpoint`, as its URL
(`http://HOST:PORT/BUCKET`, `https://...`, or `s3://BUCKET/PREFIX`):

```
aeiou-trace capture --endpoint https://store.example:9000/train -o /local/cap-%h -- python train.py
```

The command then runs with the capture's library preloaded, which writes down the head of
each HTTP request your object client sends and of each response, into `requests.jsonl` in
the capture directory: the method, the object, the range, the status, the sizes, the time,
the process and thread. It keeps only the headers that describe a request: never your
credentials, the request's signature, a session token, an encryption key, a cookie, or your
objects' metadata. It sees requests made through a shared libssl or plain HTTP: the system's
Python and boto3, curl, and most C and C++ clients. It cannot see TLS built into the program
(a standalone Python build, the AWS CRT, Rust's rustls, Java), Go clients, or HTTP/2; the
record names every process it missed, and `--requests none` leaves it out.

For an `http://` or `https://` endpoint the tool also sends two requests of its own, at the
start and the end: an unsigned `HEAD` of `aeiou-anchor-...`, which your store refuses (a 403
in its log). Those two lines in the store's log tie its clock to the capture's; send that
part of the store's log too if you can (MinIO audit, S3 server access: `aeiou-trace metrics`
reads both).

## What to send

The capture directory, or, if the trace must stay with you, its digest:

```
aeiou-trace digest /local/cap-n1/trace.txt --root /mnt/data -o app.digest.json
```

The digest carries the structure a fit is made from, path templates and counts, never the
raw paths unless you add `--paths` (**aeiou-trace**(1), digest); on a terminal it asks you
what the trace cannot decide, and records your answers as yours.

No file of a capture holds a secret of yours (**aeiou-trace**(1), capture, No secret in a
capture): `trace.txt` has no `execve` argument list or environment and none of the bytes the
application read, wrote, or sent; `capture.json` has the command line with the value of every
option or variable named as a secret (`--password=...`, `--token ...`,
`AWS_SECRET_ACCESS_KEY=...`) and every URL's password replaced by `<redacted>`. A secret your
command takes by position is not recognized: capture with `--argv none`, and the record keeps
the program's name alone. Look at `capture.json` before you send it. A trace you took by hand
with another line may print all of these, and `aeiou-trace metrics` and `digest` warn when it
does: send its digest, not the trace.

## Not implemented yet

- For the eBPF source on hosts where file capabilities cannot be granted, a password-less
  `sudo -n` helper (phase 3b of brief item 54).
- Object requests over TLS inside the program, Go, or HTTP/2 (libssl uprobes come with the
  eBPF source, phase 4): the record's `not_seen` lists what this capture could not see.
  `io_uring`'s requests, page faults on mappings, and libaio's completions are traced by the
  eBPF source (2026-11-04) where the kernel allows; under strace they are not.
- One node's interruption stopping the others (each node stops on its own signal, which
  `aeiou-launch` sends to every node), and comparing per-node records afterward for captures
  taken without a coordinator.
