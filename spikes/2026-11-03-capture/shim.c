// Spike LD_PRELOAD shim: logs HTTP request and response heads seen through OpenSSL
// (SSL_write/SSL_read, _ex too) and through libc on sockets (send*, recv*, write*, read*),
// with tid and CLOCK_REALTIME, to AEIOU_SHIM_LOG. Only heads at a buffer's start are parsed.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>

static int logfd = -1;
static __thread int busy;
static unsigned char sock[65536];   // 0 unknown, 1 socket, 2 not
static pthread_once_t once = PTHREAD_ONCE_INIT;

static void init(void) {
    const char *p = getenv("AEIOU_SHIM_LOG");
    if (p) logfd = (int)syscall(SYS_openat, AT_FDCWD, p, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
}
static int is_sock(int fd) {
    if (fd < 0 || fd >= 65536) return 0;
    if (!sock[fd]) { struct stat s; sock[fd] = (syscall(SYS_fstat, fd, &s) == 0 && S_ISSOCK(s.st_mode)) ? 1 : 2; }
    return sock[fd] == 1;
}
static const char *methods[] = {"GET ", "PUT ", "HEAD ", "POST ", "DELETE ", NULL};
// one line per head: t tid via chan dir first-line [range|content-range|content-length]
static void head(const char *via, long chan, char dir, const void *buf, long n) {
    pthread_once(&once, init);
    if (logfd < 0 || n <= 0) return;
    const char *b = buf; int hit = 0;
    if (dir == '>') { for (int i = 0; methods[i]; i++) if (n >= (long)strlen(methods[i]) && !memcmp(b, methods[i], strlen(methods[i]))) hit = 1; }
    else hit = n >= 9 && !memcmp(b, "HTTP/1.1 ", 9);
    if (!hit) return;
    long eol = 0; while (eol < n && eol < 300 && b[eol] != '\r') eol++;
    char extra[160] = "";
    static const char *keys[] = {"\r\nRange: ", "\r\nrange: ", "\r\nContent-Range: ", "\r\ncontent-range: ", "\r\nContent-Length: ", "\r\ncontent-length: ", NULL};
    long lim = n < 4096 ? n : 4096;
    for (int i = 0; keys[i]; i++) {
        const char *k = memmem(b, (size_t)lim, keys[i], strlen(keys[i]));
        if (!k) continue;
        k += strlen(keys[i]);
        const char *e = memmem(k, (size_t)(b + lim - k), "\r\n", 2);
        if (!e) continue;
        snprintf(extra + strlen(extra), sizeof extra - strlen(extra), " %.*s=%.*s", (int)(strlen(keys[i]) - 4), keys[i] + 2, (int)(e - k), k);
    }
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    char line[640];
    int m = snprintf(line, sizeof line, "%ld.%09ld %ld %s %ld %c %.*s%s\n", (long)ts.tv_sec, ts.tv_nsec, (long)syscall(SYS_gettid), via, chan, dir, (int)eol, b, extra);
    if (m > (int)sizeof line) m = sizeof line;
    syscall(SYS_write, logfd, line, m);
}

#include <link.h>
// RTLD_NEXT misses a library loaded RTLD_LOCAL (CPython's _ssl and its libssl): look in every loaded object but this one
struct find { const char *name; void *self; void *hit; };
static int one(struct dl_phdr_info *i, size_t sz, void *d) {
    struct find *f = d; (void)sz;
    if (!i->dlpi_name || !*i->dlpi_name || f->hit) return f->hit != 0;
    void *h = dlopen(i->dlpi_name, RTLD_LAZY | RTLD_NOLOAD);
    if (!h) return 0;
    void *s = dlsym(h, f->name);
    if (s && s != f->self) f->hit = s;
    dlclose(h);
    return f->hit != 0;
}
static void *next(const char *name, void *self) {
    void *s = dlsym(RTLD_NEXT, name);
    if (s) return s;
    struct find f = {name, self, NULL};
    dl_iterate_phdr(one, &f);
    return f.hit;
}
#define REAL(name, ret, ...) static ret (*real_##name)(__VA_ARGS__); if (!real_##name) real_##name = next(#name, (void *)name)

int SSL_write(void *s, const void *buf, int num) {
    REAL(SSL_write, int, void *, const void *, int);
    if (!busy) { busy = 1; head("ssl", (long)s, '>', buf, num); busy = 0; }
    return real_SSL_write(s, buf, num);
}
int SSL_write_ex(void *s, const void *buf, size_t num, size_t *w) {
    REAL(SSL_write_ex, int, void *, const void *, size_t, size_t *);
    if (!busy) { busy = 1; head("ssl", (long)s, '>', buf, (long)num); busy = 0; }
    return real_SSL_write_ex(s, buf, num, w);
}
int SSL_read(void *s, void *buf, int num) {
    REAL(SSL_read, int, void *, void *, int);
    int r = real_SSL_read(s, buf, num);
    if (!busy) { busy = 1; head("ssl", (long)s, '<', buf, r); busy = 0; }
    return r;
}
int SSL_read_ex(void *s, void *buf, size_t num, size_t *got) {
    REAL(SSL_read_ex, int, void *, void *, size_t, size_t *);
    int r = real_SSL_read_ex(s, buf, num, got);
    if (r > 0 && !busy) { busy = 1; head("ssl", (long)s, '<', buf, (long)*got); busy = 0; }
    return r;
}
// libc: only on sockets, and not while inside an SSL call (that's ciphertext)
#define SOCK_OUT(fd, buf, n) do { if (!busy && is_sock(fd)) { busy = 1; head("sock", fd, '>', buf, (long)(n)); busy = 0; } } while (0)
#define SOCK_IN(fd, buf, r) do { if (r > 0 && !busy && is_sock(fd)) { busy = 1; head("sock", fd, '<', buf, (long)(r)); busy = 0; } } while (0)
ssize_t send(int fd, const void *buf, size_t n, int fl) { REAL(send, ssize_t, int, const void *, size_t, int); SOCK_OUT(fd, buf, n); return real_send(fd, buf, n, fl); }
ssize_t sendto(int fd, const void *buf, size_t n, int fl, const struct sockaddr *a, socklen_t l) { REAL(sendto, ssize_t, int, const void *, size_t, int, const struct sockaddr *, socklen_t); SOCK_OUT(fd, buf, n); return real_sendto(fd, buf, n, fl, a, l); }
ssize_t write(int fd, const void *buf, size_t n) { REAL(write, ssize_t, int, const void *, size_t); SOCK_OUT(fd, buf, n); return real_write(fd, buf, n); }
ssize_t sendmsg(int fd, const struct msghdr *m, int fl) { REAL(sendmsg, ssize_t, int, const struct msghdr *, int); if (m->msg_iovlen) SOCK_OUT(fd, m->msg_iov[0].iov_base, m->msg_iov[0].iov_len); return real_sendmsg(fd, m, fl); }
ssize_t writev(int fd, const struct iovec *v, int c) { REAL(writev, ssize_t, int, const struct iovec *, int); if (c) SOCK_OUT(fd, v[0].iov_base, v[0].iov_len); return real_writev(fd, v, c); }
ssize_t recv(int fd, void *buf, size_t n, int fl) { REAL(recv, ssize_t, int, void *, size_t, int); ssize_t r = real_recv(fd, buf, n, fl); SOCK_IN(fd, buf, r); return r; }
ssize_t recvfrom(int fd, void *buf, size_t n, int fl, struct sockaddr *a, socklen_t *l) { REAL(recvfrom, ssize_t, int, void *, size_t, int, struct sockaddr *, socklen_t *); ssize_t r = real_recvfrom(fd, buf, n, fl, a, l); SOCK_IN(fd, buf, r); return r; }
ssize_t read(int fd, void *buf, size_t n) { REAL(read, ssize_t, int, void *, size_t); ssize_t r = real_read(fd, buf, n); SOCK_IN(fd, buf, r); return r; }
ssize_t recvmsg(int fd, struct msghdr *m, int fl) { REAL(recvmsg, ssize_t, int, struct msghdr *, int); ssize_t r = real_recvmsg(fd, m, fl); if (m->msg_iovlen) SOCK_IN(fd, m->msg_iov[0].iov_base, r < (ssize_t)m->msg_iov[0].iov_len ? r : (ssize_t)m->msg_iov[0].iov_len); return r; }
int close(int fd) { REAL(close, int, int); if (fd >= 0 && fd < 65536) sock[fd] = 0; return real_close(fd); }
