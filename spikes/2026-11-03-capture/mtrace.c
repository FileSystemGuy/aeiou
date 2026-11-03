// Minimal ptrace tracer: the lower bound of a ptrace design that writes raw binary records
// (time, tid, nr, args at entry; time, ret at exit; up to 256 bytes of a path for path calls)
// and leaves all formatting to after the run.
//   mtrace [-s] -o OUT -- CMD...    -s: a seccomp filter so only the listed calls stop
#define _GNU_SOURCE
#include <errno.h>
#include <linux/filter.h>
#include <fcntl.h>
#include <linux/seccomp.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

// the calls traced: a stand-in for %file,%desc (enough for the benchmarks)
static const int traced[] = {SYS_openat, SYS_open, SYS_close, SYS_read, SYS_write, SYS_pread64,
    SYS_pwrite64, SYS_readv, SYS_writev, SYS_lseek, SYS_fstat, SYS_newfstatat, SYS_statx,
    SYS_unlink, SYS_unlinkat, SYS_rename, SYS_renameat2, SYS_fsync, SYS_fdatasync, SYS_getdents64,
    SYS_mkdir, SYS_mkdirat, SYS_ftruncate, SYS_fallocate, SYS_dup, SYS_dup2, SYS_dup3,
    SYS_fcntl, SYS_ioctl, SYS_sendto, SYS_recvfrom, SYS_sendmsg, SYS_recvmsg, SYS_connect,
    SYS_socket, SYS_clone, SYS_clone3, SYS_fork, SYS_vfork, SYS_execve, SYS_exit_group};

struct rec { uint64_t t; uint32_t tid; uint16_t nr; uint16_t kind; uint64_t a[6]; };

static char out[1 << 20];
static size_t used;
static int ofd;
static void put(const void *p, size_t n) {
    if (used + n > sizeof out) { (void)!write(ofd, out, used); used = 0; }
    memcpy(out + used, p, n); used += n;
}
static uint64_t now(void) {
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + ts.tv_nsec;
}
static int is_path(int nr) {
    return nr == SYS_openat || nr == SYS_open || nr == SYS_newfstatat || nr == SYS_statx || nr == SYS_unlink
        || nr == SYS_unlinkat || nr == SYS_rename || nr == SYS_renameat2 || nr == SYS_mkdir || nr == SYS_mkdirat || nr == SYS_execve;
}

int main(int argc, char **argv) {
    int filt = 0, i = 1; const char *op = NULL;
    for (; i < argc; i++) {
        if (!strcmp(argv[i], "-s")) filt = 1;
        else if (!strcmp(argv[i], "-o")) op = argv[++i];
        else if (!strcmp(argv[i], "--")) { i++; break; }
    }
    if (!op || i >= argc) { fprintf(stderr, "mtrace [-s] -o OUT -- CMD...\n"); return 2; }
    ofd = creat(op, 0644);
    pid_t child = fork();
    if (child == 0) {
        ptrace(PTRACE_TRACEME, 0, 0, 0);
        if (filt) {
            int n = sizeof traced / sizeof *traced;
            struct sock_filter f[128]; int k = 0;
            f[k++] = (struct sock_filter)BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr));
            for (int j = 0; j < n; j++) f[k++] = (struct sock_filter)BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, traced[j], n - j, 0);
            f[k++] = (struct sock_filter)BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
            f[k++] = (struct sock_filter)BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRACE);
            struct sock_fprog prog = {.len = (unsigned short)k, .filter = f};
            prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
            if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog)) { perror("seccomp"); _exit(1); }
        }
        raise(SIGSTOP);
        execvp(argv[i], argv + i);
        _exit(127);
    }
    int st; waitpid(child, &st, 0);
    long opts = PTRACE_O_TRACESYSGOOD | PTRACE_O_TRACECLONE | PTRACE_O_TRACEFORK | PTRACE_O_TRACEVFORK | PTRACE_O_EXITKILL;
    if (filt) opts |= PTRACE_O_TRACESECCOMP;
    ptrace(PTRACE_SETOPTIONS, child, 0, opts);
    int cont = filt ? PTRACE_CONT : PTRACE_SYSCALL;
    ptrace(cont, child, 0, 0);
    long live = 1, rc = 0;
    while (live > 0) {
        pid_t p = waitpid(-1, &st, __WALL);
        if (p < 0) { if (errno == EINTR) continue; break; }
        if (WIFEXITED(st) || WIFSIGNALED(st)) { live--; if (p == child) rc = WIFEXITED(st) ? WEXITSTATUS(st) : 128; continue; }
        int sig = WSTOPSIG(st), ev = st >> 16, next = cont, deliver = 0;
        if (ev == PTRACE_EVENT_CLONE || ev == PTRACE_EVENT_FORK || ev == PTRACE_EVENT_VFORK) live++;
        else if (ev == PTRACE_EVENT_SECCOMP || sig == (SIGTRAP | 0x80)) {
            struct __ptrace_syscall_info si;
            ptrace(PTRACE_GET_SYSCALL_INFO, p, sizeof si, &si);
            struct rec r = {.t = now(), .tid = (uint32_t)p};
            if (si.op == PTRACE_SYSCALL_INFO_ENTRY || si.op == PTRACE_SYSCALL_INFO_SECCOMP) {
                r.nr = (uint16_t)si.entry.nr; r.kind = 0;
                memcpy(r.a, si.entry.args, sizeof r.a);
                put(&r, sizeof r);
                if (is_path((int)si.entry.nr)) {   // the path, read at entry (strace reads it too)
                    char path[256];
                    int ai = (si.entry.nr == SYS_openat || si.entry.nr == SYS_newfstatat || si.entry.nr == SYS_statx
                        || si.entry.nr == SYS_unlinkat || si.entry.nr == SYS_mkdirat || si.entry.nr == SYS_renameat2) ? 1 : 0;
                    struct iovec l = {path, sizeof path}, rm = {(void *)si.entry.args[ai], sizeof path};
                    ssize_t got = process_vm_readv(p, &l, 1, &rm, 1, 0);
                    uint16_t len = got > 0 ? (uint16_t)strnlen(path, (size_t)got) : 0;
                    put(&len, 2); put(path, len);
                }
                if (filt) next = PTRACE_SYSCALL;   // stop again at its exit
            } else if (si.op == PTRACE_SYSCALL_INFO_EXIT) {
                r.kind = 1; r.a[0] = (uint64_t)si.exit.rval;
                put(&r, sizeof r);
            }
        } else if (ev == 0 && sig != SIGTRAP && sig != SIGSTOP) deliver = sig;
        ptrace(next, p, 0, deliver);
    }
    (void)!write(ofd, out, used);
    return (int)rc;
}
