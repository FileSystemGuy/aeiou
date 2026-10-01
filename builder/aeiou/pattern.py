"""Path patterns, ported from `runner/aeiou/src/pattern.rs`: `train/{id div 1300:05}/img_{id:09}.jpg`.
Fields are `{name}`, `{name div N}`, `{name mod N}`, each with an optional `:[0][width][x|d]`."""
from __future__ import annotations

import re

_FIELD = re.compile(r"^([a-z_][a-z0-9_]*)(?:\s+(div|mod)\s+(\d+))?(?::([0-9]*[xd]?))?$")


class Pattern:
    def __init__(self, source: str):
        self.source = source
        self.segments: list = []          # ("lit", text) | ("field", name, op, n, zero, width, hex)
        rest = source
        lit = ""
        while "{" in rest:
            open_, after = rest.split("{", 1)
            lit += open_
            if "}" not in after:
                raise ValueError(f"pattern `{source}`: unclosed `{{`")
            spec, rest = after.split("}", 1)
            if lit:
                self.segments.append(("lit", lit))
                lit = ""
            m = _FIELD.match(spec.strip())
            if not m:
                raise ValueError(f"pattern `{source}`: cannot parse field `{{{spec}}}`")
            name, op, n, fmt = m.groups()
            fmt = fmt or ""
            zero = fmt.startswith("0")
            digits = fmt[1:] if zero else fmt
            hexa = digits.endswith("x")
            digits = digits.rstrip("xd")
            width = int(digits) if digits else 0
            self.segments.append(("field", name, op, int(n) if n else None, zero, width, hexa))
        lit += rest
        if lit:
            self.segments.append(("lit", lit))

    def root(self) -> str:
        prefix = self.source.split("{", 1)[0]
        return prefix.rpartition("/")[0]

    def format(self, **fields) -> str:
        out = []
        for seg in self.segments:
            if seg[0] == "lit":
                out.append(seg[1])
                continue
            _, name, op, n, zero, width, hexa = seg
            if name not in fields:
                raise ValueError(f"pattern `{self.source}`: no value for field `{name}`")
            v = fields[name]
            if isinstance(v, str):
                out.append(v)
                continue
            if op == "div":
                v = v // n
            elif op == "mod":
                v = v % n
            if hexa:
                s = format(v & ((1 << 64) - 1), "x")        # `n as u64`, as the runner prints it
                if width:
                    s = s.rjust(width, "0" if zero else " ")
            elif width and zero and v < 0:
                s = "-" + str(-v).rjust(width - 1, "0")      # Rust's `{:0w}` keeps the sign in the width
            elif width:
                s = str(v).rjust(width, "0" if zero else " ")
            else:
                s = str(v)
            out.append(s)
        return "".join(out)
