"""aeiou: the Python builder for abstract-driven I/O benchmark workloads (layer 1 of
GRAMMAR_OPTIONS.md Option D). It constructs the AST of schema/abstract-ast.schema.json; it
never runs a workload.

    from aeiou import *
    w = Workload("train_small_files")
    P = w.params(batch=32, steps=500, step_time=105 * ms)
    train = w.dataset("train", pattern="train/{id div 1300:05}/img_{id:09}.jpg",
                      count=50_000_000, size=lognormal(110 * KiB, 0.45), seed=0x5eedda7a)
    with w.actor("gpu") as gpu:
        with gpu.loader("batches", workers=8, prefetch=2, batches=P.steps) as worker:
            with worker.loop("j", P.batch):
                f = worker.let("f", train.consume())
                worker.open(f, "RDONLY|CLOEXEC"); worker.read(f, 1 * MiB, repeat="until_eof"); worker.close(f)
        with gpu.loop("step", P.steps) as step:
            gpu.take("batches"); gpu.compute(P.step_time)
            with gpu.every(500): gpu.barrier("global")
    w.write()
"""

__version__ = "0.1.0"

from .nodes import BuildError, Expr, Dist, Handle, Ref, Index, Param  # noqa: E402
from .dists import (const, uniform, uniform64, normal, lognormal, empirical, zipf, hotset,  # noqa: E402
                    mixture, none, draw, when, ceil_div, min_, max_)
from .builder import Workload, Cursor, Dataset, Namespace, gpu_id, actor_count  # noqa: E402
from .units import KiB, MiB, GiB, TiB, KB, MB, GB, ns, us, ms, s, minute  # noqa: E402

__all__ = [
    "Workload", "Cursor", "Dataset", "Namespace", "BuildError", "Expr", "Dist", "Handle", "Ref",
    "Index", "Param", "gpu_id", "actor_count",
    "const", "uniform", "uniform64", "normal", "lognormal", "empirical", "zipf", "hotset", "mixture",
    "none", "draw", "when", "ceil_div", "min_", "max_",
    "KiB", "MiB", "GiB", "TiB", "KB", "MB", "GB", "ns", "us", "ms", "s", "minute",
]
