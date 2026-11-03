#!/bin/bash
# Run as root. Does a larger output buffer stop bpftrace's lost events on the 100000-pread microbenchmark?
# Each buffer size x {two records per call, one record per call}: time, pread records kept, events lost.
set -u
cd "$(dirname "$0")"; S=$PWD; OUT=$S/ebpf-buf-results.txt; exec > >(tee "$OUT") 2>&1
U=$(stat -c %U "$S"); CG=/sys/fs/cgroup/aeiou-spike; mkdir -p $CG
trap 'chown -R "$U" "$S"' EXIT
bpftrace --version; nproc; bpftrace --info 2>&1 | grep -iE "ringbuf|perf|ring" | head -8
IDS="0 1 2 3 5 8 16 17 18 19 20 32 33 41 42 44 45 46 47 56 57 58 59 72 74 75 77 82 83 87 217 231 257 258 262 263 285 292 316 332 435"
F='cgroup == cgroupid("/sys/fs/cgroup/aeiou-spike")'
BEGIN_MAP="BEGIN { $(for i in $IDS; do printf '@t[%d]=1; ' $i; done) }"
# one record per call: arguments kept per thread at entry, printed with the return at exit
cat > rec1.bt <<EOB
$BEGIN_MAP
rawtracepoint:sys_enter /$F && @t[arg1]/ { \$r = arg0; @t0[tid] = nsecs; @a0[tid] = *(uint64 *)(\$r + 112); @a1[tid] = *(uint64 *)(\$r + 104); @a2[tid] = *(uint64 *)(\$r + 96); @a3[tid] = *(uint64 *)(\$r + 56); }
rawtracepoint:sys_exit /$F/ { \$id = *(uint64 *)(arg0 + 120); if (@t[\$id]) { printf("%llu %llu %d c %d %lx %lx %lx %lx %ld\\n", @t0[tid], nsecs, tid, \$id, @a0[tid], @a1[tid], @a2[tid], @a3[tid], arg1); } }
END { clear(@t); clear(@t0); clear(@a0); clear(@a1); clear(@a2); clear(@a3); }
EOB
B="./bench data.bin 100000 4"
printf "%-8s %-22s %10s %10s %10s\n" pages records us/pread kept lost
for pages in 64 1024 4096 16384; do
  for prog in rec.bt rec1.bt; do
    for r in 1 2 3; do
      rm -f bo.txt
      us=$(BPFTRACE_PERF_RB_PAGES=$pages bpftrace -q -o $S/bo.txt -c "/bin/sh $S/incg.sh $B" $prog 2>&1 | grep -o 'us_per_pread=[0-9.]*' | cut -d= -f2)
      kept=$(grep -cE ' (e|c) 17 ' bo.txt); lost=$(grep -i 'lost' bo.txt | grep -o '[0-9]*' | head -1)
      printf "%-8s %-22s %10s %10s %10s\n" $pages "$([ $prog = rec.bt ] && echo 'entry+exit' || echo 'one per call')" "${us:-FAILED}" "$kept" "${lost:-0}"
    done
  done
done
echo done
