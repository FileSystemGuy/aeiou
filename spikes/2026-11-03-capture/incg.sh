#!/bin/sh
# put this process in the spike's cgroup, then become the command (so the probes' filter sees it and its threads)
echo $$ > /sys/fs/cgroup/aeiou-spike/cgroup.procs || exit 125
exec "$@"
