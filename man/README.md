# Manual pages

The manual of the aeiou suite, in Markdown with the sections and conventions of a man page,
so that GitHub renders them and `pandoc -s -t man` can produce roff pages from the same
files. Option definitions are list items beginning with the bold flag; the test suites of the
runner (`runner/aeiou/tests/man.rs`) and the builder (`builder/tests/test_man.py`) compare
every tool's `--help` with its page, so a flag added, removed, or renamed in the code fails CI
until its page follows, and a flag named in any page must exist in some tool.

| Page | Section | What |
|---|---|---|
| [aeiou(1)](aeiou.1.md) | 1 | The Rust runner: `check`, `dry-run`, `datagen`, `run`; the backends; several hosts |
| [aeiou-build(1)](aeiou-build.1.md) | 1 | The builder: a Python script to an abstract, hermetically, twice |
| [aeiou-params(1)](aeiou-params.1.md) | 1 | Parameter files: defaults, check, from safetensors shards, from `.npz` archives |
| [aeiou-datagen(1)](aeiou-datagen.1.md) | 1 | Container datasets through their format classes (deprecated: `aeiou datagen` writes every format) |
| [aeiou-trace(1)](aeiou-trace.1.md) | 1 | The metrics of an strace, their comparison with an abstract's, the `trace` node's file, the digest a fit is made from |
| [aeiou-catalog(1)](aeiou-catalog.1.md) | 1 | The workload catalog: records against their dictionary, each case's shape, `CATALOG.md`, each case's description in prose, `describe` for any abstract |
| [aeiou-header(1)](aeiou-header.1.md) | 1 | The header bytes of a container file from the format's own library (HDF5), for `aeiou datagen` |
| [aeiou-launch(1)](aeiou-launch.1.md) | 1 | One `aeiou run`, `aeiou datagen`, or `aeiou-datagen` rank per host over ssh |
| [aeiou-config(5)](aeiou-config.5.md) | 5 | The option layers: command line, `AEIOU_*`, the TOML file, default |
| [aeiou-abstract(7)](aeiou-abstract.7.md) | 7 | The workload abstract: model, JSON form, rules, parameter files, manifests |

To render as roff:

```
pandoc -s -t man man/aeiou.1.md -o aeiou.1 && man -l aeiou.1
```

The release tarball's `share/man/` is made the same way, with every heading shifted up one
level so that a page's first heading becomes its title and `NAME` a section, and the manual's
header and footer set (the exact command is in `.github/workflows/release.yml`).
