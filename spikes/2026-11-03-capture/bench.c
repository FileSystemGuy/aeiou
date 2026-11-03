// Syscall microbenchmark: N preads of 4 KiB from a cached file, each followed by K
// untraced syscalls (getppid), so a filter that skips untraced calls shows its worth.
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "bench FILE N K\n"); return 2; }
    long n = atol(argv[2]), k = atol(argv[3]);
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    char buf[4096];
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (long i = 0; i < n; i++) {
        if (pread(fd, buf, sizeof buf, (i % 16384) * 4096) != sizeof buf) { perror("pread"); return 1; }
        for (long j = 0; j < k; j++) syscall(SYS_getppid);
    }
    clock_gettime(CLOCK_MONOTONIC, &b);
    double s = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
    fprintf(stderr, "preads=%ld k=%ld secs=%.3f us_per_pread=%.3f\n", n, k, s, s * 1e6 / n);
    return 0;
}
