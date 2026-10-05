"""The option layers of the runner, for the Python tools that share them (`aeiou-datagen`;
`runner/aeiou/src/options.rs` is the definition, `runner/REFERENCE.md` §14 the description).

A value comes from the command line, else from the environment (`AEIOU_<FLAG>`, the long flag
upper-cased with underscores), else from the config file (TOML, named by `--config FILE` or
`AEIOU_CONFIG`, never searched for; one table per subcommand, keys spelled as the long flags),
else from the compiled default; a higher layer replaces a lower one's value. A *fixed* option
is the command line's alone, and the environment and the file are refused when they name it.
Every invocation prints the block of what it resolved and where each value came from, in the
runner's format, so the two writers of one corpus read alike. Every refusal here is a usage
error (`usage.py`): the invocation is wrong and nothing has run, so it is printed in the
suite's frame and exits 2.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import sys
import tomllib

from .usage import UsageError

ENV_PREFIX = "AEIOU_"
ENV_CONFIG = "AEIOU_CONFIG"
RESERVED_ENV = (ENV_CONFIG, "AEIOU_SCHEMA_DIR", "AEIOU_RUNNER", "AEIOU_RSH")
SUBCOMMANDS = ("check", "dry-run", "datagen", "run")
# every option name of every subcommand of the runner (options.rs ALL_OPTIONS), so an AEIOU_*
# for another subcommand's option is left alone and anything else warns
ALL_OPTIONS = frozenset("""files abstract param params-file gpus seed io-backend root threads buffer-mib write-compress time-scale
iowq-max-workers sqpoll sqpoll-shared defer-taskrun coop-taskrun aio-depth mmap-mode mmap-consume rank ranks coordinator
rank-rotate expect-fingerprint expect-dataset-id max-gap require-cold drop-caches clean-namespaces ignore-limits report-json
report-takes dedupe compress dataset gpu steps limit metrics metrics-block metrics-sample metrics-json config""".split())

FIXED_RULE = ("nothing the fingerprint, a dataset id, or a safety check depends on may come from the environment or the "
              "config file")


def env_name(option: str) -> str:
    return ENV_PREFIX + option.upper().replace("-", "_")


def _shown(v) -> str:
    if v is None or v == []:
        return "none"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return " ".join(_shown(x) for x in v)
    return str(v)


def _json(v):
    if isinstance(v, pathlib.Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_json(x) for x in v]
    return v


def _from_env(text: str, kind):
    text = text.strip()
    if kind is bool:
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"{text!r}: expected true or false")
    if kind is pathlib.Path:
        return pathlib.Path(text)
    return kind(text)


def _from_toml(v, kind):
    if kind is bool:
        if isinstance(v, bool):
            return v
        raise ValueError(f"expected true or false, got {v!r}")
    if kind is int:
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        raise ValueError(f"expected an integer, got {v!r}")
    if kind is float:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        raise ValueError(f"expected a number, got {v!r}")
    if kind is pathlib.Path:
        if isinstance(v, str):
            return pathlib.Path(v)
        raise ValueError(f"expected a path string, got {v!r}")
    if kind is str:
        if isinstance(v, str):
            return v
        raise ValueError(f"expected a string, got {v!r}")
    raise TypeError(kind)


class ConfigFile:
    def __init__(self, path: pathlib.Path):
        self.path = path
        try:
            data = path.read_bytes()
        except OSError as e:
            raise UsageError(f"config {path}: {e.strerror}") from None
        self.sha256 = hashlib.sha256(data).hexdigest()
        try:
            self.tables = tomllib.loads(data.decode("utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
            raise UsageError(f"config {path}: {e}") from None
        for k, v in self.tables.items():
            if k not in SUBCOMMANDS:
                raise UsageError(f"config {path}: `{k}` at the top level; options live in a subcommand's table "
                                 f"([run], [dry-run], [datagen], [check]), spelled as the long flags")
            if not isinstance(v, dict):
                raise UsageError(f"config {path}: [{k}] must be a table")
            for n in v:
                if n.startswith("no-"):
                    raise UsageError(f"config {path}: [{k}] {n}: a boolean is written as its name with true or false "
                                     f"(`{n[3:]} = false`); `--no-x` is the command line's negation")

    def table(self, sub: str) -> dict:
        return self.tables.get(sub, {})


class Layers:
    """The resolver of one invocation of subcommand `sub`."""

    def __init__(self, sub: str, config_cli: pathlib.Path | None, env: dict | None = None):
        if sub not in SUBCOMMANDS:
            raise UsageError(f"no subcommand `{sub}`")
        self.sub = sub
        self.env = {k: v for k, v in (os.environ if env is None else env).items() if k.startswith(ENV_PREFIX)}
        self.entries: list[tuple[str, str, object, str]] = []   # name, shown, json value, source
        self.env_used: set[str] = set()
        self.keys_used: set[str] = set()
        self.warnings: list[str] = []
        if config_cli is not None:
            path, self.config_source = pathlib.Path(config_cli), "cli"
        elif self.env.get(ENV_CONFIG):
            path, self.config_source = pathlib.Path(self.env[ENV_CONFIG]), f"env {ENV_CONFIG}"
            self.env_used.add(ENV_CONFIG)
        else:
            path, self.config_source = None, "default"
        self.config = ConfigFile(path) if path is not None else None

    def _table_value(self, name: str):
        if self.config is None:
            return None
        return self.config.table(self.sub).get(name)

    def fixed(self, name: str, value, given: bool) -> None:
        env = env_name(name)
        if env in self.env:
            raise UsageError(f"{env} is set, but --{name} is the command line's alone: {FIXED_RULE}")
        if self._table_value(name) is not None:
            raise UsageError(f"config {self.config.path}: [{self.sub}] {name} is set, but --{name} is the command line's alone: {FIXED_RULE}")
        self.entries.append((name, _shown(value), _json(value), "cli" if given else "default"))

    def layered(self, name: str, cli, kind, default=None):
        env = env_name(name)
        self.keys_used.add(name)
        if cli is not None:
            value, source = cli, "cli"
        elif env in self.env:
            self.env_used.add(env)
            try:
                value = _from_env(self.env[env], kind)
            except ValueError as e:
                raise UsageError(f"{env}: {e}") from None
            source = f"env {env}"
        elif self._table_value(name) is not None:
            try:
                value = _from_toml(self._table_value(name), kind)
            except ValueError as e:
                raise UsageError(f"config {self.config.path}: [{self.sub}] {name}: {e}") from None
            source = f"config {self.config.path}"
        else:
            value, source = default, "default"
        self.entries.append((name, _shown(value), _json(value), source))
        return value

    def flag(self, name: str, cli, default: bool = False) -> bool:
        v = self.layered(name, cli, bool, default)
        return default if v is None else v

    def finish(self) -> None:
        if self.config is not None:
            for k in self.config.table(self.sub):
                if k not in self.keys_used:
                    if k in ALL_OPTIONS:
                        raise UsageError(f"config {self.config.path}: [{self.sub}] {k}: not an option of `aeiou {self.sub}` (or the command line's alone)")
                    raise UsageError(f"config {self.config.path}: [{self.sub}] {k}: unknown option (keys are spelled as the long flags)")
        for k in sorted(self.env):
            if k in RESERVED_ENV or k in self.env_used:
                continue
            if k.startswith("AEIOU_NO_") and k[len("AEIOU_NO_"):].lower().replace("_", "-") in ALL_OPTIONS:
                raise UsageError(f"{k}: a boolean is set in the environment as {env_name(k[len('AEIOU_NO_'):].lower().replace('_', '-'))}=true or false; "
                                 f"`--no-x` is the command line's negation")
            if k[len(ENV_PREFIX):].lower().replace("_", "-") not in ALL_OPTIONS:
                self.warnings.append(f"{k} is set and is no option of any subcommand; ignored")
        if self.config is not None:
            shown = f"{self.config.path} (sha256 {self.config.sha256[:16]}…)"
            value = {"path": str(self.config.path), "sha256": self.config.sha256}
        else:
            shown, value = "none", None
        self.entries.append(("config", shown, value, self.config_source))

    def print(self, out=None) -> None:
        out = sys.stdout if out is None else out
        print("options (cli > env > config > default)", file=out)
        width = min(72, max((len(n) + 3 + len(s) for n, s, _, _ in self.entries), default=0))
        for name, shown, _, source in self.entries:
            print(f"  {f'{name} = {shown}':<{width}}  [{source}]", file=out)
        for w in self.warnings:
            print(f"  WARNING: {w}", file=out)
        print(file=out)

    def json(self) -> dict:
        config = None if self.config is None else {"path": str(self.config.path), "sha256": self.config.sha256}
        return {"config": config, "env": sorted(self.env_used),
                "options": {n: {"value": v, "source": s} for n, _, v, s in self.entries}, "warnings": list(self.warnings)}
