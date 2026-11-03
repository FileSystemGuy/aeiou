#!/bin/bash
# Run as root (sudo). In-kernel capture (bpftrace raw tracepoints, a libssl uprobe) against
# ptrace (strace, the minimal binary tracer) and the untraced run. Targets run in the cgroup aeiou-spike,
# the probes' filter (bpftrace's pid is the kernel's, not WSL2's namespace's). Results in ebpf-results.txt.
set -u
cd "$(dirname "$0")"; S=$PWD; OUT=$S/ebpf-results.txt; exec > >(tee "$OUT") 2>&1
U=$(stat -c %U "$S"); CG=/sys/fs/cgroup/aeiou-spike
mkdir -p $CG
trap 'chown -R "$U" "$S"' EXIT
bpftrace --version; uname -r
IDS="0 1 2 3 5 8 16 17 18 19 20 32 33 41 42 44 45 46 47 56 57 58 59 72 74 75 77 82 83 87 217 231 257 258 262 263 285 292 316 332 435"
F='cgroup == cgroupid("/sys/fs/cgroup/aeiou-spike")'
BEGIN_MAP="BEGIN { $(for i in $IDS; do printf '@t[%d]=1; ' $i; done) }"
# x86_64 pt_regs: di 112, si 104, dx 96, r10 56, orig_ax 120
cat > rec.bt <<EOB
$BEGIN_MAP
rawtracepoint:sys_enter /$F && @t[arg1]/ { \$r = arg0; printf("%llu %d e %d %lx %lx %lx %lx\\n", nsecs, tid, arg1, *(uint64 *)(\$r + 112), *(uint64 *)(\$r + 104), *(uint64 *)(\$r + 96), *(uint64 *)(\$r + 56)); }
rawtracepoint:sys_exit /$F/ { \$id = *(uint64 *)(arg0 + 120); if (@t[\$id]) { printf("%llu %d x %d %ld\\n", nsecs, tid, \$id, arg1); } }
END { clear(@t); }
EOB
cat > cnt.bt <<EOB
$BEGIN_MAP
rawtracepoint:sys_enter /$F && @t[arg1]/ { @n = count(); }
rawtracepoint:sys_exit /$F/ { if (@t[*(uint64 *)(arg0 + 120)]) { @n = count(); } }
END { clear(@t); }
EOB
E="%file,%desc,%process,sync,msync,madvise,munmap,mremap,mlock,mlock2,io_setup,io_submit,io_getevents,io_pgetevents,io_cancel,io_destroy"
med() { sort -n | awk '{a[NR]=$1} END{if (NR) printf "%.3f", a[int((NR+1)/2)]; else printf "FAILED"}'; }

echo "== sanity: each probe once on 1000 preads (expect about 2000 events: 1000 pread entries + exits)"
bpftrace -q -c "/bin/sh $S/incg.sh ./bench data.bin 1000 0" cnt.bt
bpftrace -q -o b-s.txt -c "/bin/sh $S/incg.sh ./bench data.bin 1000 0" rec.bt; echo "rec lines: $(wc -l < b-s.txt), pread entries: $(grep -c ' e 17 ' b-s.txt)"; grep ' e 17 ' b-s.txt | head -2
[ "$(grep -c ' e 17 ' b-s.txt)" -ge 900 ] || { echo "sanity FAILED: bpftrace recorded too few events; stopping"; exit 1; }

echo "== microbenchmark: 100000 preads of 4 KiB + 4 untraced getppid each; us per pread, best of 3"
B="./bench data.bin 100000 4"
mb() { label=$1; shift; best=99999; for r in 1 2 3; do x=$("$@" 2>&1 | grep -o 'us_per_pread=[0-9.]*' | cut -d= -f2); [ -n "$x" ] && best=$(python3 -c "print(min($x,$best))"); done; printf "%-44s %8.2f us\n" "$label" "$best"; }
mb "untraced" $S/incg.sh $B
mb "bpftrace count only" bpftrace -q -o /dev/null -c "/bin/sh $S/incg.sh $B" cnt.bt
mb "bpftrace record every call (text, async)" bpftrace -q -o $S/b.txt -c "/bin/sh $S/incg.sh $B" rec.bt
echo "  recorded: $(grep -c ' e 17 ' b.txt) pread entries of 100000 (fewer = lost)"; grep -i lost b.txt | head -2
mb "strace man line --seccomp-bpf" strace -f --seccomp-bpf -ttt -T -yy -v -e trace=$E -o $S/s.txt $B
mb "minimal binary ptrace tracer -s" ./mtrace -s -o $S/m.bin -- $B

echo "== S3 GET load (200 x 4 KiB + 8 x 8 MiB, twice); median of 5, seconds"
for c in boto3 crt; do
  L="venv/bin/python s3get.py $c 200 8 2"
  for k in untraced bpf-count bpf-record strace-seccomp mtrace; do
    for r in 1 2 3 4 5; do
      case $k in
        untraced) $S/incg.sh $L 2>&1 >/dev/null ;;
        bpf-count) bpftrace -q -o /dev/null -c "/bin/sh $S/incg.sh $L" cnt.bt 2>&1 ;;
        bpf-record) bpftrace -q -o $S/b3.txt -c "/bin/sh $S/incg.sh $L" rec.bt 2>&1 ;;
        strace-seccomp) strace -f --seccomp-bpf -ttt -T -yy -v -e trace=$E -o $S/s.txt $L 2>&1 >/dev/null ;;
        mtrace) ./mtrace -s -o $S/m.bin -- $L 2>&1 >/dev/null ;;
      esac | grep -o 'get=[0-9.]*' | cut -d= -f2
    done | med | xargs printf "%-6s %-16s median %s s\n" $c $k
  done
  echo "  last bpf-record of $c: $(wc -l < b3.txt) lines; lost: $(grep -ci lost b3.txt)"
done

SSL=/usr/lib/x86_64-linux-gnu/libssl.so.3
echo "== uprobe on $SSL SSL_write_ex (plaintext head) + SSL_read_ex count, boto3 over TLS (system python, 100 x 4 KiB + 4 x 8 MiB put and get); median of 5"
cat > ssl.bt <<EOB
uprobe:$SSL:SSL_write_ex /$F/ { printf("%llu %d w %s\\n", nsecs, tid, str(arg1, 64)); }
uprobe:$SSL:SSL_read_ex /$F/ { @r = count(); }
EOB
T="env EP=https://127.0.0.1:19443 PYTHONWARNINGS=ignore sysvenv/bin/python s3load.py boto3 100 4"
for k in untraced uprobe shim; do
  for r in 1 2 3 4 5; do
    case $k in
      untraced) $S/incg.sh $T 2>&1 >/dev/null ;;
      uprobe) bpftrace -q -o $S/u.txt -c "/bin/sh $S/incg.sh $T" ssl.bt 2>&1 ;;
      shim) env LD_PRELOAD=$S/shim.so AEIOU_SHIM_LOG=$S/u-shim.log $T 2>&1 >/dev/null ;;
    esac | grep -o 'put=[0-9.]*s get=[0-9.]*' | sed 's/put=//;s/s get=/ /' | awk '{print $1+$2}'
  done | med | xargs printf "boto3-tls %-10s median %s s\n" $k
done
echo "  uprobe lines: $(grep -c ' w ' u.txt)"; grep -m2 ' w GET\| w PUT' u.txt
echo done
