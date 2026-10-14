"""Path patterns, ported from `runner/aeiou/src/pattern.rs`: `train/{id div 1300:05}/img_{id:09}.jpg`.
Fields are `{name}`, `{name div N}`, `{name mod N}`, each with an optional `:[0][width][x|d]`;
the divisor `N` is a positive number or (contract 0.10) the name of an integer parameter, bound to its
value by `bind` before any name is formatted (V23)."""
from __future__ import annotations

import re

_FIELD = re.compile(r"^([a-z_][a-z0-9_]*)(?:\s+(div|mod)\s+(\d+|[a-z_][a-z0-9_]*))?(?::([0-9]*[xd]?))?$")


class Pattern:
    def __init__(self, source: str):
        self.source = source
        self.segments: list = []          # ("lit", text) | ("field", name, op, n, zero, width, hex, spec)
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
            if n is not None and n.isdigit() and int(n) <= 0:
                raise ValueError(f"pattern `{source}`: `{op} {n}` needs a positive divisor")
            self.segments.append(("field", name, op, (int(n) if n.isdigit() else n) if n else None, zero, width, hexa, spec))
        lit += rest
        if lit:
            self.segments.append(("lit", lit))

    def params(self) -> list:
        """The parameters naming divisors, in pattern order."""
        return [seg[3] for seg in self.segments if seg[0] == "field" and isinstance(seg[3], str)]

    def bind(self, values: dict) -> "Pattern":
        """The pattern with each parameter divisor replaced by its value in `values`; its source is the
        original text with the names replaced, so a pattern without parameters binds to itself."""
        out = []
        for seg in self.segments:
            if seg[0] == "lit":
                out.append(seg[1])
                continue
            _, name, op, n, _z, _w, _h, spec = seg
            if isinstance(n, str):
                v = values.get(n)
                if not isinstance(v, int) or isinstance(v, bool):
                    raise ValueError(f"pattern `{self.source}`: divisor `{n}` has no integer value")
                if v <= 0:
                    raise ValueError(f"pattern `{self.source}`: `{op} {n}` is {v}: it needs a positive divisor")
                fmt = spec.partition(":")
                spec = f"{name} {op} {v}" + (":" + fmt[2] if fmt[1] else "")
            out.append("{" + spec + "}")
        return Pattern("".join(out))

    def root(self) -> str:
        prefix = self.source.split("{", 1)[0]
        return prefix.rpartition("/")[0]

    def format(self, **fields) -> str:
        out = []
        for seg in self.segments:
            if seg[0] == "lit":
                out.append(seg[1])
                continue
            _, name, op, n, zero, width, hexa, _spec = seg
            if isinstance(n, str):
                raise ValueError(f"pattern `{self.source}`: divisor `{n}` is not bound to a value")
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
