#!/bin/bash
# median GET time over 7 interleaved runs per condition and client
cd "$(dirname "$0")"; S=$PWD
E="%file,%desc,%process,sync,msync,madvise,munmap,mremap,mlock,mlock2,io_setup,io_submit,io_getevents,io_pgetevents,io_cancel,io_destroy"
declare -A CMD=(
 [1-untraced]=""
 [2-shim]="env LD_PRELOAD=$S/shim.so AEIOU_SHIM_LOG=$S/o.log"
 [3-strace-seccomp]="strace -f --seccomp-bpf -ttt -T -yy -v -e trace=$E -o o.txt"
 [4-strace-seccomp+net]="strace -f --seccomp-bpf -ttt -T -yy -v -e trace=$E,%network -o o.txt"
 [5-strace-noseccomp]="strace -f -ttt -T -yy -v -e trace=$E -o o.txt"
 [6-strace-noseccomp+net]="strace -f -ttt -T -yy -v -e trace=$E,%network -o o.txt"
 [7-mtrace-seccomp(+net)]="./mtrace -s -o o.bin --"
)
for c in boto3 crt; do for k in "${!CMD[@]}"; do : > "res-$c-$k"; done; done
for r in 1 2 3 4 5 6 7; do for c in boto3 crt; do for k in "${!CMD[@]}"; do
  rm -f o.log; ${CMD[$k]} venv/bin/python s3get.py $c 200 8 2 2>&1 >/dev/null | grep -o 'get=[0-9.]*' | cut -d= -f2 >> "res-$c-$k"
done; done; done
for c in boto3 crt; do for k in $(printf '%s\n' "${!CMD[@]}" | sort); do
  printf "%-6s %-26s median %.3f s  (min %.3f max %.3f)\n" $c "$k" $(sort -n "res-$c-$k" | awk '{a[NR]=$1} END{print a[int((NR+1)/2)], a[1], a[NR]}'); done; done
