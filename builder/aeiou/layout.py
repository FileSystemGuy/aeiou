"""Container geometry, ported from `runner/aeiou/src/eval.rs` (`Layout`, the `DsMeta` unit and
column formulas; `schema/README.md` §2 *Container layout*). The writer computes where every
unit, column chunk, and sample payload must land and checks the real file against it."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def align_up(x: int, a: int) -> int:
    return x if a <= 1 else (x + a - 1) // a * a


@dataclass
class Column:
    header: int = 0
    fixed: int = 0
    weight: float = 0.0
    row_header: int = 0
    row_footer: int = 0
    row_align: int = 1
    align: int = 1
    last_weighted: bool = False


@dataclass
class Layout:
    unit: int = 1
    file_header: int = 0
    file_header_per_sample: int = 0
    file_header_per_unit: int = 0
    file_footer: int = 0
    file_footer_per_sample: int = 0
    file_footer_per_unit: int = 0
    file_align: int = 1
    unit_header: int = 0
    unit_header_per_sample: int = 0
    unit_footer: int = 0
    unit_footer_per_sample: int = 0
    unit_align: int = 1
    columns: list = field(default_factory=lambda: [Column(weight=1.0, last_weighted=True)])

    @classmethod
    def from_spec(cls, spec: dict | None, spf: int, ev) -> "Layout":
        """`spec` is the resolved `format.layout` (parameters substituted); `ev` evaluates an
        expression to an int."""
        if not spec:
            return cls(unit=spf)
        g = lambda k, d: ev(spec[k]) if k in spec else d  # noqa: E731
        cols = spec.get("columns")
        if cols:
            last = max((i for i, c in enumerate(cols) if float(c.get("weight", 0)) > 0), default=None)
            columns = [Column(header=ev(c.get("header", 0)), fixed=ev(c.get("fixed", 0)), weight=float(c.get("weight", 0)),
                              row_header=ev(c.get("row_header", 0)), row_footer=ev(c.get("row_footer", 0)),
                              row_align=max(1, ev(c.get("row_align", 1))), align=max(1, ev(c.get("align", 1))),
                              last_weighted=(i == last)) for i, c in enumerate(cols)]
        else:
            columns = [Column(weight=1.0, last_weighted=True)]
        unit = g("unit", spf)
        if unit < 1 or unit > spf:
            raise ValueError(f"layout unit {unit} must be in [1, samples_per_file = {spf}]")
        return cls(unit=unit, file_header=g("file_header", 0), file_header_per_sample=g("file_header_per_sample", 0),
                   file_header_per_unit=g("file_header_per_unit", 0), file_footer=g("file_footer", 0),
                   file_footer_per_sample=g("file_footer_per_sample", 0), file_footer_per_unit=g("file_footer_per_unit", 0),
                   file_align=max(1, g("file_align", 1)), unit_header=g("unit_header", 0),
                   unit_header_per_sample=g("unit_header_per_sample", 0), unit_footer=g("unit_footer", 0),
                   unit_footer_per_sample=g("unit_footer_per_sample", 0), unit_align=max(1, g("unit_align", 1)), columns=columns)

    def part(self, c: int, size: int) -> int:
        col = self.columns[c]
        if col.weight <= 0.0:
            return 0
        if col.last_weighted:
            others = sum(math.floor(size * k.weight) for i, k in enumerate(self.columns) if i != c and k.weight > 0.0)
            return size - others
        return math.floor(size * col.weight)

    def row_bytes(self, c: int, size: int) -> int:
        col = self.columns[c]
        return align_up(col.row_header + col.fixed + self.part(c, size) + col.row_footer, col.row_align)


class FileGeometry:
    """The geometry of one container file: `sizes` are its samples' payload sizes in order."""

    def __init__(self, layout: Layout, sizes: list):
        self.layout, self.sizes = layout, sizes
        self.n = len(sizes)
        self.units = (self.n + layout.unit - 1) // layout.unit
        self.col_lens = []          # per unit: list of column chunk lengths
        self.unit_lens = []
        for u in range(self.units):
            rows = sizes[u * layout.unit:(u + 1) * layout.unit]
            lens = [c.header for c in layout.columns]
            for s in rows:
                for c in range(len(lens)):
                    lens[c] += layout.row_bytes(c, s)
            lens = [align_up(x, layout.columns[c].align) for c, x in enumerate(lens)]
            self.col_lens.append(lens)
            n = len(rows)
            self.unit_lens.append(align_up(layout.unit_header + layout.unit_header_per_sample * n + sum(lens)
                                           + layout.unit_footer + layout.unit_footer_per_sample * n, layout.unit_align))
        self.data_start = layout.file_header + layout.file_header_per_sample * self.n + layout.file_header_per_unit * self.units
        self.unit_offsets = []
        off = self.data_start
        for ul in self.unit_lens:
            self.unit_offsets.append(off)
            off += ul
        self.data_end = off
        self.file_size = align_up(off + layout.file_footer + layout.file_footer_per_sample * self.n
                                  + layout.file_footer_per_unit * self.units, layout.file_align)

    def samples_in_unit(self, u: int) -> int:
        return min(self.layout.unit, self.n - u * self.layout.unit)

    def col_offset(self, u: int, c: int) -> int:
        n = self.samples_in_unit(u)
        return self.unit_offsets[u] + self.layout.unit_header + self.layout.unit_header_per_sample * n + sum(self.col_lens[u][:c])

    def sample_offset(self, i: int) -> int:
        """Start of sample `i`'s payload in a one-column layout."""
        if len(self.layout.columns) != 1:
            raise ValueError("offset of a sample split over several columns")
        u = i // self.layout.unit
        off = self.col_offset(u, 0) + self.layout.columns[0].header
        for s in self.sizes[u * self.layout.unit:i]:
            off += self.layout.row_bytes(0, s)
        return off + self.layout.columns[0].row_header
