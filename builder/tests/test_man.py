"""The manual pages (`man/*.md`) against the tools' `--help`: every long option of every
Python tool (and of every subcommand of `aeiou-params` and `aeiou-trace`) is defined in its
page's section, spelled as the help spells it, and the page defines nothing the tool does not
have; `aeiou-launch` likewise; every `--flag` any page names, in prose included, is an option
of some tool of the suite (the runner's from the definitions of `man/aeiou.1.md`, which
`runner/aeiou/tests/man.rs` holds to the binary); every page has the sections of a man page
and `man/README.md` lists it."""
import pathlib
import re
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BUILDER = HERE.parent
ROOT = BUILDER.parent
MAN = ROOT / "man"

# tool -> (module, subcommands); a flat tool has the one section `### <tool>`
TOOLS = {
    "aeiou-build": ("cli", []),
    "aeiou-params": ("params", ["defaults", "check", "safetensors", "npz"]),
    "aeiou-datagen": ("datagen", []),
    "aeiou-trace": ("trace", ["metrics", "export", "compare"]),
}
PAGES = ["aeiou.1.md", "aeiou-build.1.md", "aeiou-params.1.md", "aeiou-datagen.1.md", "aeiou-trace.1.md",
         "aeiou-launch.1.md", "aeiou-config.5.md", "aeiou-abstract.7.md"]

OPTION_LINE = re.compile(r"^ {2,6}(?:-\w, )?--([a-z0-9\[\]-]+)")
DEFINED_LINE = re.compile(r"^- \*\*(?:-\w\*\*, \*\*)?--([a-z0-9\[\]-]+)\*\*")
MENTION = re.compile(r"(?<!-)--(\[no-\])?([a-z][a-z0-9-]*)")


def help_text(module, *sub):
    r = subprocess.run([sys.executable, "-m", f"aeiou.{module}", *sub, "--help"], capture_output=True, text=True, cwd=str(BUILDER),
                       env={"PYTHONPATH": str(BUILDER), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    return r.stdout


def help_flags(text):
    return {m.group(1) for line in text.splitlines() if (m := OPTION_LINE.match(line))} - {"help", "version"}


def sections(page):
    """(heading, body) for every `### ` heading of a page."""
    out = []
    for line in page.splitlines():
        if line.startswith("### "):
            out.append([line[4:].strip(), []])
        elif out:
            out[-1][1].append(line)
    return {h: "\n".join(b) for h, b in out}


def defined(body):
    return {m.group(1) for line in body.splitlines() if (m := DEFINED_LINE.match(line))} - {"help", "version"}


def mentioned(page):
    return {m.group(2).rstrip("-") for m in MENTION.finditer(page)}


def read(name):
    return (MAN / name).read_text()


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_python_tool_options_are_the_pages(tool):
    module, subs = TOOLS[tool]
    secs = sections(read(f"{tool}.1.md"))
    if not subs:
        in_help, in_page = help_flags(help_text(module)), defined_flat(read(f"{tool}.1.md"))
        assert in_help == in_page, f"{tool}: --help {sorted(in_help - in_page)} missing from the page; page {sorted(in_page - in_help)} not in --help"
        return
    top = help_text(module)
    for sub in subs:
        assert re.search(rf"^\s+{sub}\s", top, re.M), f"`{tool} --help` lists {sub}"
        in_help = help_flags(help_text(module, sub))
        assert f"{tool} {sub}" in secs, f"no `### {tool} {sub}` section"
        in_page = defined(secs[f"{tool} {sub}"])
        assert in_help == in_page, f"{tool} {sub}: --help {sorted(in_help - in_page)} missing from the page; page {sorted(in_page - in_help)} not in --help"


def defined_flat(page):
    """A flat tool's definitions: the list items of its OPTIONS section."""
    body = page.split("\n## OPTIONS\n", 1)[1].split("\n## ", 1)[0]
    return defined(body)


def test_launch_options_are_the_pages():
    r = subprocess.run([str(ROOT / "runner" / "aeiou-launch"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert help_flags(r.stdout) == set() == defined_flat(read("aeiou-launch.1.md"))
    assert re.search(r"^- \*\*-p\*\* \*PORT\*", read("aeiou-launch.1.md"), re.M), "the one option, -p PORT"


def runner_flags():
    """The runner's options as `man/aeiou.1.md` defines them (held to the binary by man.rs)."""
    out = set()
    for h, body in sections(read("aeiou.1.md")).items():
        if h == "Global options" or h.startswith("aeiou "):
            out |= defined(body)
    return out


def test_every_flag_named_in_any_page_exists():
    flags = runner_flags()
    for tool, (module, subs) in TOOLS.items():
        for sub in subs or [None]:
            flags |= help_flags(help_text(module, *([sub] if sub else [])))
    booleans = {f[5:] for f in flags if f.startswith("[no-]")}
    allowed = {f.removeprefix("[no-]") for f in flags} | {f"no-{b}" for b in booleans} | {"x", "no-x", "help", "version"}
    for name in PAGES + ["README.md"]:
        unknown = mentioned(read(name)) - allowed
        assert not unknown, f"man/{name} names options no tool has: {sorted(unknown)}"


@pytest.mark.parametrize("name", PAGES)
def test_page_has_the_sections_of_a_man_page(name):
    page = read(name)
    title, section = name.rsplit(".md", 1)[0].rsplit(".", 1)
    assert page.startswith(f"# {title}({section})\n"), "the title line is `# name(section)`"
    assert f"\n{title} - " in page, "the NAME line is `name - description`"
    required = ["NAME", "DESCRIPTION", "SEE ALSO"]
    if section == "1":
        required += ["SYNOPSIS", "OPTIONS", "EXIT STATUS", "EXAMPLES"]
    for h in required:
        assert f"\n## {h}\n" in page, f"no `## {h}` section"
    # headings are man-page headings: upper case, level 2; the level-3 ones are the per-tool groups
    for m in re.finditer(r"^## (.+)$", page, re.M):
        assert m.group(1) == m.group(1).upper(), f"section heading not upper case: {m.group(1)}"
    assert f"]({name})" in read("README.md"), f"man/README.md does not list {name}"
