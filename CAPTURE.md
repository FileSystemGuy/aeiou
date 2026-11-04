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

## Not implemented yet

- For the eBPF source on hosts where file capabilities cannot be granted, a password-less
  `sudo -n` helper (phase 3b of brief item 54).
- Object requests over TLS inside the program, Go, or HTTP/2 (libssl uprobes come with the
  eBPF source, phase 4), and page faults on mappings and `io_uring` operations (phase 4): the
  record's `not_seen` lists what this capture could not see.
- One node's interruption stopping the others (each node stops on its own signal, which
  `aeiou-launch` sends to every node), and comparing per-node records afterward for captures
  taken without a coordinator.
