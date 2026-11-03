#!/bin/bash
# Run as root. Sustained rate: does bpftrace's text reader keep up when the burst outlasts the buffer?
# 1M and 5M preads (about 1.2 s and 6 s untraced at ~1.6M events/s), 1024 and 16384 pages per CPU,
# two records per call. Also the reader's CPU time, from bpftrace's own rusage.
set -u
cd "$(dirname "$0")"; S=$PWD; OUT=$S/ebpf-sustain-results.txt; exec > >(tee "$OUT") 2>&1
U=$(stat -c %U "$S"); CG=/sys/fs/cgroup/aeiou-spike; mkdir -p $CG
trap 'rm -f $S/bo.txt; chown -R "$U" "$S"' EXIT
printf "%-9s %-7s %10s %10s %10s %8s %12s %12s\n" preads pages us/pread kept lost lost% reader_cpu_s out_MiB
for n in 1000000 5000000; do
  for pages in 1024 16384; do
    for r in 1 2; do
      rm -f bo.txt
      res=$( { /usr/bin/time -f 'CPU %U %S' env BPFTRACE_PERF_RB_PAGES=$pages bpftrace -q -o $S/bo.txt -c "/bin/sh $S/incg.sh ./bench data.bin $n 0" rec.bt; } 2>&1 )
      us=$(echo "$res" | grep -o 'us_per_pread=[0-9.]*' | cut -d= -f2)
      # bpftrace's CPU time includes the child's; the child's is about n * 0.6 us of syscalls, reported separately below
      cpu=$(echo "$res" | awk '/^CPU/{printf "%.2f", $2+$3}')
      kept=$(grep -c ' e 17 ' bo.txt); lost=$(grep -i 'lost' bo.txt | grep -o '[0-9]*' | awk '{s+=$1} END{print s+0}')
      pct=$(awk -v l=$lost -v n=$n 'BEGIN{printf "%.1f", 100*l/(2*n)}')
      mib=$(du -m bo.txt | cut -f1)
      printf "%-9s %-7s %10s %10s %10s %8s %12s %12s\n" $n $pages "${us:-FAILED}" "$kept" "$lost" "$pct" "$cpu" "$mib"
    done
  done
done
echo "untraced CPU for reference:"
for n in 1000000 5000000; do /usr/bin/time -f "n=$n CPU %U %S" ./bench data.bin $n 0; done
echo done
