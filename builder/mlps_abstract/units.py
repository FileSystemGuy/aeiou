"""Unit spellings. Sizes are bytes and durations are nanoseconds in the AST; these are the
builder-side multipliers, plain ints so `105 * ms` and `64 * MiB` are ints."""

KiB = 1024
MiB = 1024 ** 2
GiB = 1024 ** 3
TiB = 1024 ** 4
KB = 1000
MB = 1000 ** 2
GB = 1000 ** 3

ns = 1
us = 1000
ms = 1000 ** 2
s = 1000 ** 3
minute = 60 * s
