# aeiou-build(1)

## NAME

aeiou-build - run builder scripts and write the abstracts they author

## SYNOPSIS

```
aeiou-build [-o DIR] [--hermetic] [--twice] [--check] [--no-provenance] SCRIPTS...
aeiou-build --help | --version
```

## DESCRIPTION

**aeiou-build** runs each Python *builder script* once and writes the abstract it constructs
as `<name>.ast.json`, one file per `Workload` the script creates, next to the script or under
`-o`. A builder script imports the `aeiou` package and declares a workload: parameters,
datasets, namespaces, actors, and a body of operations on cursors. The calls that look like
I/O are declarations; Python runs at build time and generates *structure*, the runner
(**aeiou**(1)) runs the result and generates *iterations*. Nothing a script does runs on a
client node.

Each emitted document is validated against the published schema and the semantic rules of
**aeiou-abstract**(7) before it is written, and carries a `provenance` block: the script's
path and SHA-256, the git revision (`-dirty` when the script has uncommitted changes), the
builder and Python versions, the lock file's hash when one is present, whether the build
was done twice identically, and the document's canonical hash. The canonical hash, the
identity of the abstract, is the SHA-256 of the JSON with keys sorted, no whitespace, and the
`provenance` block removed; it is what an author publishes.

The builder enforces one discipline: a symbolic value has no truth value, cannot be
iterated, and cannot be converted to an int. `for j in P.batch`, `if step % 500 == 0`, and
`range(P.steps)` raise a `TypeError` naming `cursor.loop` or `cursor.when`. A lint refuses
three or more identical statements in a row (a hand-unrolled iteration);
`Workload(..., lint=False)` allows it. `/` on nodes is refused: say `//` or `ceil_div`.

## OPTIONS

- *SCRIPTS...*

  Builder scripts (`.py`), each constructing one or more `Workload`s.
- **-o**, **--out** *DIR*

  Output directory. Default: the script's own directory.
- **--hermetic**

  Build each script in a fresh, isolated child: `python -s -B` with a scrubbed environment
  (`PATH`, `PYTHONHASHSEED`, `LANG` only) and audit hooks that deny sockets, subprocesses,
  every write, and reads outside the interpreter, its packages, the schema, and the script's
  directory. Clocks, entropy, and the unseeded `random` and `numpy.random` entry points
  raise `HermeticViolation`. The child returns the document on stdout; the parent validates
  it again and writes it.
- **--twice**

  Build in two children with different `PYTHONHASHSEED` values and refuse to write unless the
  canonical hashes agree. This catches set-iteration order and `id()`-based ordering without
  anticipating them.
- **--check**

  Do not write. Rebuild, compare the canonical hash with the existing file's, print `ok` or
  `FAIL` per workload, and exit 1 on any difference. CI runs it on every push so the
  committed abstracts cannot drift from their scripts.
- **--no-provenance**

  Write the abstract without its `provenance` block.
- **-h**, **--help**

  Print the help.
- **-V**, **--version**

  Print the version.

## ENVIRONMENT

- **AEIOU_SCHEMA_DIR**

  The directory holding `abstract-ast.schema.json`, `params.schema.json`, and `check.py`,
  when the package is not run from the repository.

## FILES

- `<name>.ast.json`

  The written abstract, pretty-printed with two-space indentation in the builder's key
  order; **aeiou-abstract**(7).
- `schema/abstract-ast.schema.json`, `schema/check.py`

  The schema and the reference validator the builder loads; it keeps no second copy of the
  rules.

## EXIT STATUS

- **0**

  Every script built and every file was written, or, with `--check`, every hash matched.
- **1**

  A script failed to build (the Python traceback names the construction error, or the
  validator names the rule and the JSON pointer), a hermetic violation, two builds whose
  hashes differ under `--twice`, or a hash that differs from the committed file under
  `--check`.
- **2**

  A usage error, in the frame every tool of the suite shares.

## EXAMPLES

Build every committed abstract hermetically, twice, into the examples directory:

```
cd builder
aeiou-build --hermetic --twice -o ../schema/examples abstracts/*.py
```

The drift check CI runs:

```
aeiou-build --hermetic --twice --check -o ../schema/examples abstracts/*.py
```

A minimal script:

```python
from aeiou import *

w = Workload("tiny", doc="one file per step")
P = w.P
w.param("steps", 100)
data = w.dataset("data", pattern="data/{id:06}.bin", count=10_000,
                 size=lognormal(median=110 * KiB, sigma=0.45), seed=0x5eed, access="map")
with w.actor("gpu") as gpu:
    with gpu.loop("step", P.steps):
        f = gpu.let("f", data.consume())
        gpu.open(f, "RDONLY|CLOEXEC")
        gpu.read(f, 128 * KiB, repeat="until_eof")
        gpu.close(f)

if __name__ == "__main__":
    w.write()
```

## SEE ALSO

**aeiou**(1), **aeiou-params**(1), **aeiou-datagen**(1), **aeiou-trace**(1),
**aeiou-abstract**(7).

The reference: `builder/REFERENCE.md` (writing an abstract, §1; the discipline, §2; what
fails where, §3; reproducible builds, §4). The committed scripts: `builder/abstracts/*.py`.
