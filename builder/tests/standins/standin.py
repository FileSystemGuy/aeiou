"""TEST FAKES, never libraries: `builder/tests/standins/` holds stand-ins for the libraries the
workloads' kit scripts (`workloads/**/kit/`) call that are not installed in the builder's
environment: torch, torchvision, transformers, safetensors, faiss, diskannpy, PIL. Each one
implements only what a kit script calls, returns small plausible objects, and records every call
that matters as one JSON line in the file `AEIOU_STANDIN_LOG` names, so that
`builder/tests/test_kits.py` can run the real script and assert what it asked of the library
(paths, sizes, parameters, order) against what the kit's README and case files say the traced
application did. A test checks the script, not the fake.

The directory is on the path only of the kit scripts those tests run (`PYTHONPATH`), never of the
suite itself; nothing here is the library's behaviour beyond what a comment says it models.
"""
import json
import os


def record(call, **kw):
    """One line in the log: the call's name, the process that made it, and what it was asked."""
    with open(os.environ["AEIOU_STANDIN_LOG"], "a") as f:
        f.write(json.dumps({"call": call, "pid": os.getpid(), **kw}, default=str) + "\n")
