"""Usage errors: the command line is wrong, and nothing has run (`runner/README.md` §15,
`DESIGN_REVIEW.md` §3.61; the definition is `runner/aeiou/src/usage.rs`, this is its mirror
for the Python tools). Every tool of the suite prints them in one frame, so a new user meets
the same message whichever tool spoke:

    aeiou-datagen: the following required arguments were not provided:
      <ABSTRACT_PATH>  the abstract (`.ast.json`, written by aeiou-build from a builder script)
      --root <DIR>     directory the abstract's paths are relative to
                       (also $AEIOU_ROOT, or `root` in the [datagen] table of the config file)

    Usage: aeiou-datagen [OPTIONS] --root <DIR> <ABSTRACT_PATH>

    For more information, try 'aeiou-datagen --help'.

The first line names the command as typed, then the message; the usage line; the pointer to
the help. The exit status is 2 (the parsers' own convention), a failure during the work is 1.
A missing requirement is never reported alone: the command collects every argument it still
lacks, from any layer, and lists them together (`Missing`), so the next attempt is the last
one. `Parser` is argparse in the runner's clothes: the same section titles, `<METAVAR>`
spelling, usage prefix, and help wording as the Rust binary, and its own errors raised as
`UsageError` so the frame is the same whether argparse or the tool found the mistake.
"""
from __future__ import annotations

import argparse
import shutil
import sys

from .nodes import BuildError

MISSING = "the following required arguments were not provided:"
# the abstract every tool but aeiou-build takes, and --root, as `Missing` lists them (the
# same words as usage.rs, so `aeiou datagen` and `aeiou-datagen` print the same list)
ABSTRACT = ("<ABSTRACT_PATH>", "the abstract (`.ast.json`, written by aeiou-build from a builder script)")
WHAT_AN_ABSTRACT = "an abstract is the `.ast.json` file aeiou-build writes from a builder script"


def root(sub: str) -> tuple[str, str, str]:
    return ("--root <DIR>", "directory the abstract's paths are relative to", f"also $AEIOU_ROOT, or `root` in the [{sub}] table of the config file")


class UsageError(BuildError):
    """The command line is wrong. `parser` is the one whose usage the frame shows, when the
    error knows it (argparse's own, a `Missing` list); else the caller's default."""

    def __init__(self, message: str, parser: Parser | None = None):
        super().__init__(message)
        self.parser = parser


def render(prog: str, message: str, usage: str) -> str:
    """The frame: `prog` the command as typed, `usage` its usage text beginning `Usage: `."""
    return f"{prog}: {message}\n\n{usage.rstrip()}\n\nFor more information, try '{prog} --help'.\n"


def fail(e: UsageError, default: Parser) -> int:
    """Print a usage error in the frame to stderr; the exit status to return."""
    p = e.parser or default
    sys.stderr.write(render(p.prog, str(e), p.format_usage()))
    return 2


def failed(prog: str, e: BaseException) -> int:
    """Print a failure during the work (`prog: message`) to stderr; the exit status to return."""
    print(f"{prog}: {e}", file=sys.stderr)
    return 1


class Formatter(argparse.RawDescriptionHelpFormatter):
    """At most 120 columns (clap's `max_term_width`), the description kept as written, an
    option's names before its one value (`-o, --out <FILE>`, as clap prints them)."""

    def __init__(self, prog, indent_increment=2, max_help_position=30, width=None):
        if width is None:
            width = min(120, shutil.get_terminal_size().columns - 2)
        super().__init__(prog, indent_increment, max_help_position, width)

    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        return ", ".join(action.option_strings) + " " + self._format_args(action, self._get_default_metavar_for_optional(action))


class Parser(argparse.ArgumentParser):
    """argparse with the runner's look: `Usage:`, `Arguments:` and `Options:`, `<METAVAR>`,
    `-h, --help  Print help`, the description above the usage, and every error raised as a
    `UsageError` (nothing exits from inside the parser but `--help` and `--version`).
    Subcommands made with `add_subparsers` are `Parser`s too, named `prog sub`."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", Formatter)
        kwargs["add_help"] = False
        super().__init__(*args, **kwargs)
        self._positionals.title = "Arguments"
        self._optionals.title = "Options"
        self._version_text: str | None = None
        self._tail_added = False

    def add_argument(self, *names, **kwargs):
        action = kwargs.get("action", "store")
        if action in ("store", "append", "extend") and kwargs.get("nargs") != 0:
            metavar = kwargs.get("metavar")
            if metavar is None:
                dest = kwargs.get("dest") or (names[0] if not names[0].startswith("-") else names[-1].lstrip("-").replace("-", "_"))
                metavar = dest.upper()
            if not metavar.startswith("<"):
                metavar = f"<{metavar}>"
            kwargs["metavar"] = metavar
        return super().add_argument(*names, **kwargs)

    def add_subparsers(self, **kwargs):
        """Subcommands named `prog sub`, listed under `<COMMAND>`; a subcommand's one-line
        `help` is also its `--help`'s description (as clap's `about` is both)."""
        kwargs.setdefault("prog", self.prog)
        kwargs.setdefault("metavar", "<COMMAND>")
        action = super().add_subparsers(**kwargs)
        add_parser = action.add_parser

        def add_parser_described(name, **kw):
            if "help" in kw:
                kw.setdefault("description", kw["help"].replace("%%", "%"))
            return add_parser(name, **kw)

        action.add_parser = add_parser_described
        return action

    def version(self, version: str) -> None:
        """`-V, --version` prints this, as clap's does; shown after `--help`, last."""
        self._version_text = version

    def _tail(self) -> None:
        """`-h, --help` and `-V, --version` as the last rows of the options, as clap shows them."""
        if not self._tail_added:
            self._tail_added = True
            super().add_argument("-h", "--help", action="help", help="Print help")
            if self._version_text is not None:
                super().add_argument("-V", "--version", action="version", version=self._version_text, help="Print version")

    def parse_known_args(self, args=None, namespace=None):
        self._tail()
        for action in self._subparsers._group_actions if self._subparsers else ():
            for sub in getattr(action, "choices", {}).values():
                if isinstance(sub, Parser):
                    sub._tail()
        return super().parse_known_args(args, namespace)

    def error(self, message):
        raise UsageError(message, self)

    def format_usage(self):
        self._tail()
        f = self._get_formatter()
        f.add_usage(self.usage, self._actions, self._mutually_exclusive_groups, prefix="Usage: ")
        return f.format_help()

    def format_help(self):
        self._tail()
        f = self._get_formatter()
        f.add_text(self.description)
        f.add_usage(self.usage, self._actions, self._mutually_exclusive_groups, prefix="Usage: ")
        for group in self._action_groups:
            f.start_section(group.title)
            f.add_text(group.description)
            f.add_arguments(group._group_actions)
            f.end_section()
        f.add_text(self.epilog)
        return f.format_help()


class Missing:
    """The requirements a command still lacks, collected before anything is done so that one
    message lists them all; `parser` is the (sub)command whose usage the frame shows."""

    def __init__(self, parser: Parser):
        self.parser = parser
        self.items: list[tuple[str, str, str | None]] = []

    def want(self, value, shown: str, what: str, also: str | None = None):
        """A value that must be present: records it when None (or an empty list), hands it back."""
        if value is None or value == []:
            self.need(False, shown, what, also)
        return value

    def need(self, present: bool, shown: str, what: str, also: str | None = None) -> None:
        """A requirement stated as a condition: records it when `present` is false."""
        if not present:
            self.items.append((shown, what, also))

    def message(self) -> str:
        width = max((len(s) for s, _, _ in self.items), default=0)
        lines = [MISSING]
        for shown, what, also in self.items:
            lines.append(f"  {shown:<{width}}  {what}")
            if also:
                lines.append(f"  {'':<{width}}  ({also})")
        return "\n".join(lines)

    def check(self) -> None:
        if self.items:
            raise UsageError(self.message(), self.parser)
