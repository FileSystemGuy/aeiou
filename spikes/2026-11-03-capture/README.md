# The capture spike of 2026-11-03

What a capture costs and reaches: strace, a minimal binary ptrace tracer, an `LD_PRELOAD`
shim of HTTP heads, and eBPF (bpftrace), on a syscall microbenchmark and two S3 clients
against MinIO. The results and what they decide are `DESIGN_REVIEW.md` §3.96 and brief item
54; this directory keeps the programs and the raw results so the measurements can be repeated,
first on a real Linux client node (every number here is WSL2's, whose ptrace wakeups across
CPUs cost about 50 µs). Spike code: not tested, not covered, not part of any tool.

| File | What |
|---|---|
| `setup.sh` | builds the programs, makes the two venvs, a certificate, `data.bin`, starts MinIO on 19000 (HTTP) and 19443 (HTTPS), data in `/dev/shm`, and preloads the GET objects |
| `bench.c` | N cached 4 KiB `pread`s of `data.bin`, each followed by K untraced `getppid`s; prints µs per `pread` |
| `mtrace.c` | the minimal ptrace tracer: `mtrace [-s] -o OUT -- CMD`, binary records (time, tid, number, six arguments at entry; the return at exit; a path read at entry), `-s` a seccomp filter on the calls it traces |
| `shim.c` | the `LD_PRELOAD` shim: request and response heads seen through OpenSSL and through libc on sockets, with tid and `CLOCK_REALTIME`, to `AEIOU_SHIM_LOG`; finds a `libssl` loaded `RTLD_LOCAL` by walking the loaded objects |
| `pshim.c` | an in-process recorder of `pread64`, binary records, for the cost of interposition itself |
| `s3load.py` | `s3load.py boto3\|crt N M`: PUT N objects of 4 KiB and M of 8 MiB, GET them, a ranged GET of each large one; `EP` the endpoint |
| `s3get.py` | `s3get.py boto3\|crt N M REP`: GETs only over the objects `setup` wrote |
| `s3bench.sh` | the unprivileged S3 comparison, medians of 7 interleaved runs → `s3bench-results.txt` |
| `incg.sh` | puts itself in the cgroup `aeiou-spike` and becomes the command (the eBPF scripts' filter) |
| `ebpf-bench.sh` | root: bpftrace (count only, every call recorded) against strace and `mtrace` on both loads, and a libssl uprobe against the shim over TLS → `ebpf-results.txt` |
| `ebpf-buf.sh` | root: bpftrace's buffer at 64 to 16,384 pages per CPU, two records per call or one → `ebpf-buf-results.txt` |
| `ebpf-sustain.sh` | root: 1 and 5 million `pread`s, the sustained loss → `ebpf-sustain-results.txt` |
| `ptrace-results.txt` | the unprivileged microbenchmark's numbers and strace's own calls, as reported in the session |

Order: `./setup.sh`, then `./s3bench.sh`, then `sudo ./ebpf-bench.sh` (it writes the `.bt`
programs the other two read), `sudo ./ebpf-buf.sh`, `sudo ./ebpf-sustain.sh`. The
microbenchmark lines of `ptrace-results.txt`, for example:

```
E=%file,%desc,%process,sync,msync,madvise,munmap,mremap,mlock,mlock2,io_setup,io_submit,io_getevents,io_pgetevents,io_cancel,io_destroy
./bench data.bin 100000 4
strace -f --seccomp-bpf -ttt -T -yy -v -e trace=$E -o t.txt ./bench data.bin 100000 4
./mtrace -s -o m.bin -- ./bench data.bin 100000 4
taskset -c 5 ./mtrace -s -o m.bin -- ./bench data.bin 100000 4
taskset -c 5 strace -c -o outer.txt strace -f --seccomp-bpf -ttt -T -yy -v -e trace=$E -o t.txt ./bench data.bin 100000 4
```

What WSL2 needed, likely not elsewhere: bpftrace's `pid` is the kernel's and WSL2 runs the
distribution in a PID namespace, so the targets are filtered by cgroup; reading a tracepoint's
`args` made bpftrace 0.20 look for kernel headers WSL2 does not ship, so the programs use raw
tracepoints and read `pt_regs` at x86_64's offsets; `bpftrace -c` takes an ELF, so `incg.sh`
runs under `/bin/sh`; WSL2's `perf` has no `trace`.
