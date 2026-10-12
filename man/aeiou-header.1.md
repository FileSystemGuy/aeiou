# aeiou-header(1)

## NAME

aeiou-header - the header bytes of a container file, made by the format's own library, for aeiou datagen

## SYNOPSIS

```
aeiou-header --format JSON --rows N [--rows N]...
aeiou-header --help | --version
```

## DESCRIPTION

**aeiou-header** writes to its standard output the first bytes of a container file whose
layout the format's own library decides, and nothing else. `aeiou datagen` (**aeiou**(1))
runs it; a user never needs to. Today the one format is HDF5: the bytes before a file's first
row, made by libhdf5 through h5py, for a file of `N` rows of the dataset (they depend on the
dataset's name, shape, dtype, and row count, and on nothing the rows hold). `aeiou datagen`
then writes the rows after them itself.

`aeiou datagen` looks for **aeiou-header** in the directory of its own binary, the real file
with symlinks resolved, and never on the `PATH`, so the two always come from one release; the
release ships both in `bin/`, this one as a single file carrying Python, numpy, h5py, and
libhdf5. Only the host running rank 0 runs it: rank 0 makes the headers before the start gate
and the coordinator sends them to every other rank, so client hosts need `aeiou` alone. Each
dataset takes one run, for the row counts its files have (the full files and a short last
one); the headers come back to back, each the layout's `file_header` bytes long.

The tool refuses when its h5py and libhdf5 differ from the version the abstract records in
its `format` entry, the one the layout's data offset was probed with when the abstract was
built: another library could lay the header out otherwise.

## OPTIONS

- **--format** *JSON*

  The dataset's resolved `format` entry: the abstract's, parameters substituted, as the
  dataset manifest records it. Required.
- **--rows** *N*

  The rows a file holds. Repeatable: one header per occurrence, in order. Required.
- **-h**, **--help**

  Print the help.
- **-V**, **--version**

  Print the version.

## EXIT STATUS

- **0**

  The headers were written.
- **1**

  A format class with no header, a library version other than the abstract's, or a header
  whose length is not the layout's `file_header`; the reason on standard error.
- **2**

  A usage error, in the frame every tool of the suite shares.

## EXAMPLES

```
aeiou-header --format "$(jq -c .datasets.train.files.format train.ast.json)" --rows 16 | xxd | head
```

## SEE ALSO

**aeiou**(1), **aeiou-datagen**(1), **aeiou-abstract**(7).

The decision and its reasoning: `DESIGN_REVIEW.md` §3.70 (revised 2026-10-12); the HDF5
format class: `builder/REFERENCE.md` §6.
