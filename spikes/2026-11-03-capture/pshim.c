// in-process recorder: pread64 intercepted, a binary record per call into a per-thread buffer
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
struct rec { uint64_t t0, t1; uint32_t tid; uint16_t nr; uint16_t pad; uint64_t a[3]; int64_t ret; };
static __thread struct rec buf[4096]; static __thread int used; static int fd = -2;
static void flush(void) { if (fd == -2) { const char *p = getenv("AEIOU_SHIM_LOG"); fd = p ? (int)syscall(SYS_openat, AT_FDCWD, p, O_WRONLY|O_CREAT|O_APPEND, 0644) : -1; }
  if (fd >= 0) syscall(SYS_write, fd, buf, used * sizeof *buf);
  used = 0; }
static uint64_t now(void) { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (uint64_t)ts.tv_sec * 1000000000u + ts.tv_nsec; }
ssize_t pread64(int f, void *b, size_t n, off_t o) {
  static ssize_t (*real)(int, void *, size_t, off_t); if (!real) real = dlsym(RTLD_NEXT, "pread64");
  uint64_t t0 = now(); ssize_t r = real(f, b, n, o);
  struct rec *x = &buf[used++]; x->t0 = t0; x->t1 = now(); x->tid = (uint32_t)syscall(SYS_gettid); x->nr = SYS_pread64; x->a[0] = f; x->a[1] = n; x->a[2] = o; x->ret = r;
  if (used == 4096) flush();
  return r; }
ssize_t pread(int f, void *b, size_t n, off_t o) __attribute__((alias("pread64")));
