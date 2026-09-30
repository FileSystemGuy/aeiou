"""Hermetic generation: what `abstract-build --hermetic` installs in the child process before
it runs the author's script (GRAMMAR_OPTIONS.md Option D, technique 6).

- PEP 578 audit hooks deny sockets, subprocesses, and file opens outside the interpreter, the
  installed packages, the schema, and the script's own directory. Writes are denied
  altogether: the child returns the AST on stdout and the parent writes the file.
- Audit hooks do not see clock reads, so the clocks, `os.urandom`, `uuid`, and the unseeded
  `random` / `numpy.random` entry points are replaced with functions that raise.

The build-twice check (`--twice`) is what proves this worked; the hooks only make the failure
immediate and named.
"""
from __future__ import annotations

import os
import pathlib
import sys


class HermeticViolation(RuntimeError):
    pass


_DENIED_EVENTS = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.fork",
                  "os.forkpty", "os.spawn", "os.startfile", "os.putenv", "os.unsetenv",
                  "os.kill", "os.killpg", "os.truncate", "os.remove", "os.rename", "os.mkdir",
                  "os.rmdir", "os.link", "os.symlink", "os.chmod", "os.chown", "shutil.rmtree",
                  "shutil.move", "shutil.copyfile", "shutil.copymode", "shutil.copystat",
                  "shutil.chown", "shutil.make_archive", "shutil.unpack_archive", "tempfile.mkstemp",
                  "tempfile.mkdtemp", "webbrowser.open", "ctypes.dlopen", "ctypes.call_function",
                  "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname",
                  "socket.gethostbyaddr", "socket.gethostname", "socket.sendto", "socket.sendmsg",
                  "socket.getnameinfo", "socket.__new__")

_WRITE_MODES = set("wax+")


def _deny(what):
    def f(*a, **k):
        raise HermeticViolation(f"{what} is not available under --hermetic: an abstract must "
                                f"not depend on time, entropy, the environment, or the network")
    f.__name__ = f"denied_{what.replace('.', '_')}"
    return f


def _under(path: pathlib.Path, roots) -> bool:
    return any(path == r or r in path.parents for r in roots)


def install(script: os.PathLike | str, extra_read_roots=()):
    """Install the hooks. Call once, in the child, before running the script."""
    script = pathlib.Path(script).resolve()
    roots = {pathlib.Path(p).resolve() for p in (sys.prefix, sys.base_prefix, sys.exec_prefix,
                                                 *[p for p in sys.path if p])}
    roots.add(script.parent)
    roots.update(pathlib.Path(p).resolve() for p in extra_read_roots)
    roots.update(pathlib.Path(p) for p in ("/usr/lib", "/usr/share/zoneinfo", "/proc/self"))

    def hook(event, args):
        if event.startswith("socket.") or event in _DENIED_EVENTS:
            raise HermeticViolation(f"{event} is denied under --hermetic")
        if event == "open":
            path, mode, flags = args
            if isinstance(path, int):
                return
            if isinstance(path, bytes):
                path = os.fsdecode(path)
            path = pathlib.Path(path)
            writing = (mode is not None and bool(set(mode) & _WRITE_MODES)) or \
                      (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing:
                raise HermeticViolation(f"write to {path} is denied under --hermetic; the harness "
                                        f"writes the AST")
            if not path.is_absolute():
                path = pathlib.Path.cwd() / path
            if not _under(path, roots):
                raise HermeticViolation(f"read of {path} is denied under --hermetic (not the "
                                        f"interpreter, its packages, or the script's directory)")

    sys.addaudithook(hook)
    _stub_entropy_and_clocks()


def _stub_entropy_and_clocks():
    import datetime
    import random
    import time
    import uuid

    for name in ("time", "time_ns", "monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns",
                 "process_time", "process_time_ns", "thread_time", "thread_time_ns", "localtime",
                 "gmtime", "ctime", "asctime", "strftime"):
        setattr(time, name, _deny(f"time.{name}"))
    os.urandom = _deny("os.urandom")
    os.getrandom = _deny("os.getrandom")
    uuid.uuid1 = _deny("uuid.uuid1")
    uuid.uuid4 = _deny("uuid.uuid4")

    class _Datetime(datetime.datetime):
        now = classmethod(_deny("datetime.now"))
        utcnow = classmethod(_deny("datetime.utcnow"))
        today = classmethod(_deny("datetime.today"))

    class _Date(datetime.date):
        today = classmethod(_deny("date.today"))

    datetime.datetime = _Datetime
    datetime.date = _Date

    for name in ("random", "randint", "randrange", "choice", "choices", "shuffle", "sample", "uniform",
                 "gauss", "getrandbits", "seed", "betavariate", "expovariate", "gammavariate",
                 "lognormvariate", "normalvariate", "paretovariate", "triangular", "vonmisesvariate",
                 "weibullvariate", "randbytes", "binomialvariate"):
        if hasattr(random, name):
            setattr(random, name, _deny(f"random.{name} (module-level, unseeded)"))
    random.SystemRandom.__init__ = _deny("random.SystemRandom")
    orig_init = random.Random.__init__

    def seeded_init(self, x=None):
        if x is None:
            raise HermeticViolation("random.Random() without a seed is not available under --hermetic")
        orig_init(self, x)
    random.Random.__init__ = seeded_init

    try:
        import numpy.random as npr
    except ImportError:
        return
    for name in ("seed", "random", "rand", "randn", "randint", "choice", "shuffle", "permutation",
                 "uniform", "normal", "random_sample", "bytes", "lognormal", "zipf"):
        if hasattr(npr, name):
            setattr(npr, name, _deny(f"numpy.random.{name} (module-level, unseeded)"))
    orig_rng = npr.default_rng

    def default_rng(seed=None, *a, **k):
        if seed is None:
            raise HermeticViolation("numpy.random.default_rng() without a seed is not available under --hermetic")
        return orig_rng(seed, *a, **k)
    npr.default_rng = default_rng
