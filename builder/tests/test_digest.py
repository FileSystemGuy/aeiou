"""`aeiou-trace digest` (`PROJECT_BRIEF.md` item 47, `DESIGN_REVIEW.md` §3.86, §3.87): the
structural summary a fit is made from, with a verdict per site on what the trace decides and an
`open` list, ranked by projected divergence, of what it does not; and the answers given to it
(§3.88), every answer checked against the trace and recorded under `asserted` with provenance,
never as a measurement. The questions themselves, the interview, are the binary's since
`DESIGN_REVIEW.md` §3.89 stage 3 and tested in Rust (`runner/aeiou/tests/digest_corpus.rs`
holds every interview these tests made, `trace_tool.rs` the terminal)."""
import json
import os
import math
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BUILDER = HERE.parent
sys.path.insert(0, str(BUILDER))

from aeiou import digest, trace  # noqa: E402
from aeiou.nodes import BuildError  # noqa: E402

RUNNER = pathlib.Path(os.environ.get("AEIOU_RUNNER", BUILDER.parent / "runner" / "target" / "release" / "aeiou"))
TRACE_BIN = pathlib.Path(os.environ.get("AEIOU_TRACE") or RUNNER.with_name("aeiou-trace"))
# the digest reads a trace's calls as the lines the runner's aeiou-trace exports (DESIGN_REVIEW.md §3.89)
pytestmark = pytest.mark.skipif(not TRACE_BIN.exists(), reason="needs the runner binary aeiou-trace")


def write_export(path, header, events):
    """An exported trace written as `aeiou-trace export` writes it."""
    with open(path, "w", encoding="utf-8") as out:
        for doc in (header, *events):
            out.write(json.dumps(doc, separators=(",", ":")) + "\n")


def strace_of(opens, root="/d", tid=100, t0=1700000000.0, size=4096, reads=None, per_open=None):
    """An strace of `np.load`-shaped opens: open, fstat, `reads` reads of `size`, close; one
    line per call, 1 ms apart. `opens` is the paths (under the root) in order; `per_open`
    gives (path, bytes) pairs instead, every open one read of its bytes."""
    lines, t, fd = [], t0, 3
    items = per_open or [(p, size * (reads or 1)) for p in opens]
    for path, total in items:
        n = reads or 1
        each = total // n
        lines.append(f'{tid} {t:.6f} openat(AT_FDCWD</d>, "{root}/{path}", O_RDONLY|O_CLOEXEC) = {fd}<{root}/{path}> <0.000010>')
        t += 0.001
        lines.append(f"{tid} {t:.6f} fstat({fd}<{root}/{path}>, {{st_mode=S_IFREG|0644, st_size={total}, ...}}) = 0 <0.000005>")
        t += 0.001
        for _ in range(n):
            lines.append(f'{tid} {t:.6f} read({fd}<{root}/{path}>, "x"..., {each}) = {each} <0.000100>')
            t += 0.001
        lines.append(f"{tid} {t:.6f} close({fd}<{root}/{path}>) = 0 <0.000005>")
        t += 0.010
    return "\n".join(lines) + "\n"


def digest_of(text, root="/d", **kw):
    return digest.digest_of(text.splitlines(True), root, **kw)


def report_of(text, root="/d", **kw):
    """The binary's text report (`runner/aeiou/tests/digest_corpus.rs` holds every report the
    Python digest wrote; these say what a reader looks for)."""
    return digest.report(digest.analyze(text.splitlines(True), root), **kw)


# ---------------------------------------------------------------- templates


def test_path_templates_turn_every_number_into_a_field():
    paths = [f"train/{i // 10:05}/sample_{i:09}.npz" for i in range(12)]
    d = digest_of(strace_of(paths))
    (t,) = d["templates"]
    assert t["template"] == "train/{a:05}/sample_{b:09}.npz"
    assert t["paths"] == 12 and t["opens"] == 12
    a, b = t["fields"]
    assert a == {"name": "a", "kind": "int", "width": 5, "values": 2, "min": 0, "max": 1}
    assert b == {"name": "b", "kind": "int", "width": 9, "values": 12, "min": 0, "max": 11}
    assert t["ops"] == {"open": 12, "fstat": 12, "read": 12, "close": 12}
    assert t["bytes_read"] == 12 * 4096 and t["bytes_written"] == 0
    assert t["read_size"]["buckets"] == [[4096, 12]]
    assert "raw_paths" not in t


def test_unpadded_numbers_hex_hashes_and_literals():
    paths = ["kv/3f2a9c0e1b7d4e6f8a9b0c1d2e3f4a5b.bin", "kv/0000111122223333444455556666777a.bin", "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "idx/7.lst", "idx/12.lst"]
    d = digest_of(strace_of(paths))
    by = {t["template"]: t for t in d["templates"]}
    assert set(by) == {"kv/{a:x32}.bin", "model-{a:05}-of-{b:05}.safetensors", "idx/{a}.lst"}
    assert by["kv/{a:x32}.bin"]["fields"] == [{"name": "a", "kind": "hex", "width": 32, "values": 2}]
    assert by["idx/{a}.lst"]["fields"][0] == {"name": "a", "kind": "int", "width": None, "values": 2, "min": 7, "max": 12}
    assert by["model-{a:05}-of-{b:05}.safetensors"]["fields"][1]["values"] == 1


def test_raw_paths_only_on_request():
    paths = ["a/1.npz", "a/2.npz"]
    assert "raw_paths" not in digest_of(strace_of(paths))["templates"][0]
    t = digest_of(strace_of(paths), paths=True)["templates"][0]
    assert t["raw_paths"] == ["a/1.npz", "a/2.npz"]


# ---------------------------------------------------------------- lanes and phases


def test_phases_are_runs_of_one_motif_with_repeats_collapsed():
    paths = [f"s/{i:03}.npz" for i in range(5)]
    d = digest_of(strace_of(paths, reads=7))
    (lane,) = d["lanes"]
    assert lane["lane"] == 0 and lane["ops"] == 5 * 10
    (ph,) = lane["phases"]
    assert ph["motif"] == "open fstat read+ close" and ph["template"] == 0 and ph["units"] == 5
    assert ph["repeats"]["read+"] == {"n": 5, "mean": 7.0, "max": 7, "buckets": [[7, 5]]}
    assert ph["ops"] == 50 and ph["bytes_read"] == 5 * 7 * 4096
    # 10 ms between one unit's close and the next open; 4 gaps
    assert ph["think"]["n"] == 4 and abs(ph["think"]["mean"] - 10_000_000) < 20_000
    assert ph["first"] == 0 and ph["last"] > ph["first"]


def test_alternating_pairs_collapse_as_a_cycle_and_a_phase_change_is_a_new_phase():
    t = 1700000000.0
    lines = ['100 %.6f openat(AT_FDCWD</d>, "/d/big.bin", O_RDONLY) = 3</d/big.bin> <0.000010>' % t]
    for i in range(6):
        t += 0.001
        lines.append(f"100 {t:.6f} lseek(3</d/big.bin>, {i * 8192}, SEEK_SET) = {i * 8192} <0.000005>")
        t += 0.001
        lines.append(f'100 {t:.6f} read(3</d/big.bin>, "x"..., 4096) = 4096 <0.000050>')
    t += 0.001
    lines.append(f"100 {t:.6f} close(3</d/big.bin>) = 0 <0.000005>")
    for p in ("out/0.bin", "out/1.bin"):
        t += 0.001
        lines.append(f'100 {t:.6f} openat(AT_FDCWD</d>, "/d/{p}", O_WRONLY|O_CREAT|O_TRUNC, 0644) = 4</d/{p}> <0.000010>')
        t += 0.001
        lines.append(f'100 {t:.6f} write(4</d/{p}>, "y"..., 1024) = 1024 <0.000050>')
        t += 0.001
        lines.append(f"100 {t:.6f} close(4</d/{p}>) = 0 <0.000005>")
    t += 0.001
    lines.append(f'100 {t:.6f} rename("/d/out/0.bin", "/d/out/final.bin") = 0 <0.000005>')
    d = digest_of("\n".join(lines) + "\n")
    (lane,) = d["lanes"]
    motifs = [(p["motif"], p["units"]) for p in lane["phases"]]
    assert motifs == [("open (lseek read)+ close", 1), ("open write close", 2), ("rename", 1)]
    assert lane["phases"][0]["repeats"]["(lseek read)+"]["buckets"] == [[6, 1]]
    by = {x["template"]: x for x in d["templates"]}
    assert by["big.bin"]["offsets"] == {"positioned": 1}
    assert by["out/{a}.bin"]["offsets"] == {"whole": 2}
    assert by["out/{a}.bin"]["ops"] == {"open": 2, "write": 2, "close": 2, "rename": 1}
    assert by["out/final.bin"]["ops"] == {"renamed_to": 1}


def test_lanes_are_traced_tasks():
    a = strace_of([f"x/{i}.npz" for i in range(3)], tid=100)
    b = strace_of([f"x/{i}.npz" for i in range(3, 6)], tid=101, t0=1700000000.0005)
    d = digest_of(a + b)
    assert [l["lane"] for l in d["lanes"]] == [0, 1] and d["contexts"] == 2
    assert d["notes"].get("threads_without_clone") == 1


# ---------------------------------------------------------------- the selection verdict


def permutation(n, seed):
    import random
    r = random.Random(seed)
    xs = list(range(n))
    r.shuffle(xs)
    return xs


def test_one_pass_with_no_repeat_is_undecided_and_says_what_decides_it():
    ids = permutation(28, 1)
    d = digest_of(strace_of([f"s/{i:03}.npz" for i in ids]))
    (s,) = d["sites"]
    assert s["template"] == 0 and s["touches"] == 28 and s["distinct"] == 28 and s["epochs"] == [28]
    sel = s["selection"]
    assert sel["verdict"] == "undecided"
    # no repeat in 28 draws with replacement needs a corpus of at least 28·27/(2 ln 20)
    assert sel["candidates_for_choose"] == math.ceil(28 * 27 / (2 * math.log(20)))
    assert sel["consume"] == {"epoch": 28}
    assert "consume" in sel["consistent"] and "choose" in sel["consistent"]
    assert sel["settles"].startswith("a second pass")


def test_a_small_corpus_refutes_choose():
    ids = permutation(28, 1)
    d = digest_of(strace_of([f"s/{i:03}.npz" for i in ids]), candidates=100)
    sel = d["sites"][0]["selection"]
    assert sel["verdict"] == "consume" and sel["consistent"] == ["consume"]
    assert sel["p_no_repeat_under_choose"] < 0.05
    assert "choose" in sel["refuted"]


def test_a_large_corpus_leaves_it_open_and_counts_the_touches_needed():
    ids = permutation(28, 1)
    d = digest_of(strace_of([f"s/{i:03}.npz" for i in ids]), candidates=2000)
    sel = d["sites"][0]["selection"]
    assert sel["verdict"] == "undecided"
    k = sel["touches_to_decide"]
    assert math.exp(-k * (k - 1) / 4000) <= 0.05 < math.exp(-(k - 1) * (k - 2) / 4000)


def test_two_equal_passes_over_the_same_set_are_consume():
    ids = permutation(20, 1) + permutation(20, 2)
    d = digest_of(strace_of([f"s/{i:03}.npz" for i in ids]))
    s = d["sites"][0]
    assert s["epochs"] == [20, 20] and s["distinct"] == 20
    sel = s["selection"]
    assert sel["verdict"] == "consume" and sel["consume"] == {"epoch": 20} and sel["refuted"] == {"choose": sel["refuted"]["choose"]}
    assert "equal length" in sel["refuted"]["choose"]


def test_drop_last_leaves_a_few_untouched_per_pass():
    # 23 files, batch 4: each pass touches 20, a different 3 left out each time
    a, b = permutation(23, 3), permutation(23, 4)
    ids = a[:20] + b[:20] + permutation(23, 5)[:20]
    d = digest_of(strace_of([f"s/{i:03}.npz" for i in ids]))
    s = d["sites"][0]
    assert s["epochs"] == [20, 20, 20] and s["distinct"] == 23
    sel = s["selection"]
    assert sel["verdict"] == "consume" and sel["consume"] == {"epoch": 20, "seen_not_every_pass": 3}


def test_repeats_before_the_set_is_exhausted_are_choose():
    import random
    r = random.Random(7)
    ids = [r.randrange(50) for _ in range(120)]
    d = digest_of(strace_of([f"s/{i:03}.npz" for i in ids]))
    s = d["sites"][0]
    sel = s["selection"]
    assert sel["verdict"] == "choose" and len(s["epochs"]) > 3
    assert "consume" in sel["refuted"] and "unequal" in sel["refuted"]["consume"]
    # the implied corpus from the mean first-repeat distance, 2·mean²/π, within a factor of 3 of 50
    assert 50 / 3 < sel["choose"]["candidates_implied"] < 50 * 3
    assert sel["choose"]["touches_per_path"]["max"] >= 3
    assert 0 <= sel["choose"]["top_10pct_share"] <= 1


def test_one_full_pass_then_a_repeat_is_undecided_between_both():
    ids = permutation(20, 1) + permutation(20, 2)[:5]
    d = digest_of(strace_of([f"s/{i:03}.npz" for i in ids]))
    s = d["sites"][0]
    assert s["epochs"] == [20, 5]
    sel = s["selection"]
    assert sel["verdict"] == "undecided" and set(sel["consistent"]) == {"consume", "choose"}
    assert "second full pass" in sel["settles"]


def test_a_site_needs_two_touches():
    d = digest_of(strace_of(["one.bin"]))
    assert d["sites"] == []


# ---------------------------------------------------------------- order and size


def test_order_within_a_pass_is_sequential_or_permuted():
    seq = digest_of(strace_of([f"s/{i:03}.npz" for i in range(30)]))["sites"][0]["order"]
    assert seq["verdict"] == "sequential" and seq["ascending"] == 1.0 and seq["step_of_one"] == 1.0
    per = digest_of(strace_of([f"s/{i:03}.npz" for i in permutation(200, 9)]))["sites"][0]["order"]
    assert per["verdict"] == "permuted" and 0.35 < per["ascending"] < 0.65 and per["step_of_one"] < 0.1
    few = digest_of(strace_of([f"s/{i:03}.npz" for i in [3, 1, 2, 0]]))["sites"][0]["order"]
    assert few["verdict"] == "undecided" and few["pairs"] == 3


def test_sizes_are_constant_per_path_or_drawn():
    const = digest_of(strace_of([f"s/{i}.npz" for i in range(4)] * 2))["sites"][0]["size"]
    assert const == {"verdict": "constant", "bytes": 4096}
    per_path = digest_of(strace_of([], per_open=[("s/0.npz", 100), ("s/1.npz", 200), ("s/0.npz", 100), ("s/1.npz", 200)]))["sites"][0]["size"]
    assert per_path["verdict"] == "per-path" and per_path["repeated_paths"] == 2
    drawn = digest_of(strace_of([], per_open=[("s/0.npz", 100), ("s/1.npz", 200), ("s/0.npz", 300)]))["sites"][0]["size"]
    assert drawn["verdict"] == "drawn" and drawn["paths_with_differing_bytes"] == 1
    one_pass = digest_of(strace_of([], per_open=[("s/0.npz", 100), ("s/1.npz", 200)]))["sites"][0]["size"]
    assert one_pass["verdict"] == "undecided" and "repeat" in one_pass["settles"]


# ---------------------------------------------------------------- the open list


def test_open_items_are_ranked_by_projected_divergence():
    ids = permutation(28, 1)
    d = digest_of(strace_of([(f"s/{i:03}.npz") for i in ids]), epochs=2)
    kinds = [(o["site"], o["question"]) for o in d["open"]]
    assert kinds == [(0, "selection"), (0, "size"), (0, "order")] or kinds[0] == (0, "selection")
    sel = d["open"][0]
    assert sel["rank"] == 1 and set(sel["candidates"]) == {"consume", "choose"}
    assert sel["divergence"] == pytest.approx(math.exp(-2), abs=1e-9)
    assert sel["at_scale"]["epochs"] == 2
    assert sel["at_scale"]["choose"]["never_read_share"] == pytest.approx(math.exp(-2))
    assert sel["at_scale"]["consume"]["never_read_share"] == 0.0
    assert "revisit" in sel["ask"]
    assert sel["settles"] == d["sites"][0]["selection"]["settles"]
    assert d["projection"] == {"epochs": 2, "candidates": None}


def test_nothing_open_when_the_trace_decides():
    ids = permutation(20, 1) + permutation(20, 2)
    d = digest_of(strace_of([f"s/{i:03}.npz" for i in ids] * 1))
    questions = {o["question"] for o in d["open"]}
    assert "selection" not in questions and "size" not in questions


# ---------------------------------------------------------------- inputs and the document


def test_an_exported_trace_file_digests_as_its_strace(tmp_path):
    text = strace_of([f"s/{i:03}.npz" for i in permutation(12, 1)], reads=3)
    header, events = trace.export_of(text.splitlines(True), "/d")
    path = tmp_path / "app.jsonl"
    write_export(str(path), header, events)
    with open(path) as f:
        from_file = digest.digest_of(f, None)
    from_strace = digest_of(text)
    for key in ("templates", "sites", "open"):
        assert from_file[key] == from_strace[key]
    assert from_file["source"] == "trace" and from_strace["source"] == "strace"
    assert [p["motif"] for l in from_file["lanes"] for p in l["phases"]] == [p["motif"] for l in from_strace["lanes"] for p in l["phases"]]


def test_the_document_names_its_format_and_root():
    d = digest_of(strace_of(["a.bin", "b.bin"]))
    assert d["aeiou_digest"] == 2 and d["root"] == "/d" and d["source"] == "strace"
    assert d["metrics"]["aeiou_metrics"] == 2 and d["metrics"]["total"]["ops"] == 8
    assert d["ops"] == 8 and d["contexts"] == 1
    assert isinstance(d["notes"], dict)


def test_nothing_under_the_root_is_an_error():
    with pytest.raises(BuildError, match="no call under --root /nowhere"):
        digest_of(strace_of(["a.bin"]), root="/nowhere")


def test_text_report_reads_like_a_summary():
    ids = permutation(28, 1)
    text = report_of(strace_of([f"s/{i:03}.npz" for i in ids]), epochs=2)
    assert "s/{a:03}.npz" in text and "28 paths" in text
    assert "open fstat read close" in text
    assert "selection" in text and "undecided" in text
    assert "open questions" in text and "1." in text


# ---------------------------------------------------------------- the command


def run(*args, stdin=None):
    # stdin is always a pipe here, never the terminal pytest may run on: the digest asks nothing
    return subprocess.run([str(TRACE_BIN), "digest", *args], capture_output=True, text=True, cwd=BUILDER, input=stdin or "")


def test_cli_writes_json_or_text(tmp_path):
    text = strace_of([f"s/{i:03}.npz" for i in permutation(12, 1)])
    tr = tmp_path / "t.txt"
    tr.write_text(text)
    r = run(str(tr), "--root", "/d")
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["aeiou_digest"] == 2 and doc["trace"] == str(tr)
    out = tmp_path / "d.json"
    r = run(str(tr), "--root", "/d", "-o", str(out), "--epochs", "3", "--candidates", "500", "--paths")
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())
    assert doc["projection"] == {"epochs": 3, "candidates": 500} and doc["templates"][0]["raw_paths"]
    assert "open questions" in r.stdout and str(out) in r.stdout
    r = run(str(tr), "--root", "/d", "--text")
    assert r.returncode == 0 and r.stdout.startswith("digest of ") and "{" not in r.stdout.splitlines()[0]
    r = run("-", "--root", "/d", stdin=text)
    assert r.returncode == 2 and "never stdin" in r.stderr  # the trace is a file: the comparison and a re-run with answers read it again (§3.88)


def test_cli_refusals(tmp_path):
    tr = tmp_path / "t.txt"
    tr.write_text(strace_of(["a.bin"]))
    r = run(str(tr))
    assert r.returncode == 2 and "--root <DIR>" in r.stderr
    r = run()
    assert r.returncode == 2 and "<TRACE>" in r.stderr
    r = run(str(tr), "--root", "/nowhere")
    assert r.returncode == 1 and r.stderr.startswith("aeiou-trace digest: ") and "no call under --root /nowhere" in r.stderr
    r = run(str(tmp_path / "absent.txt"), "--root", "/d")
    assert r.returncode == 1 and "absent.txt" in r.stderr
    r = run(str(tr), "--root", "/d", "--epochs", "0")
    assert r.returncode == 2 and "--epochs" in r.stderr
    r = run(str(tr), "--root", "/d", "--candidates", "0")
    assert r.returncode == 2 and "--candidates" in r.stderr
    r = run(str(tr), "--root", "/d", "-o", str(tmp_path / "no" / "dir" / "x.json"))
    assert r.returncode == 1 and "x.json" in r.stderr
    # an exported file carries its root, and refuses a second
    header, events = trace.export_of(strace_of(["a.bin"]).splitlines(True), "/d")
    ex = tmp_path / "e.jsonl"
    write_export(str(ex), header, events)
    r = run(str(ex))
    assert r.returncode == 0 and json.loads(r.stdout)["root"] == "/d"
    r = run(str(ex), "--root", "/d")
    assert r.returncode == 0
    r = run(str(ex), "--root", "/other")
    assert r.returncode == 1 and "carries its root" in r.stderr
    many = tmp_path / "many.txt"
    many.write_text(strace_of([f"s/{i:03}.npz" for i in permutation(12, 1)]))
    r = run(str(many), "--root", "/d", "--candidates", "3")
    assert r.returncode == 0  # fewer candidates than paths seen: noted, the seen count used
    doc = json.loads(r.stdout)
    assert "12 distinct paths" in doc["notes"]["candidates"] and doc["sites"][0]["selection"]["verdict"] == "consume"


# ---------------------------------------------------------------- the rest of the branches


AIO = """
300 1700000003.000000 openat(AT_FDCWD</d>, "idx", O_RDONLY|O_DIRECT) = 3</d/idx> <0.000010>
300 1700000003.000100 io_submit(0x7f00, 2, [{aio_data=0x1, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=3</d/idx>, aio_buf=0x7f, aio_nbytes=4096, aio_offset=0}, {aio_data=0x2, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=3</d/idx>, aio_buf=0x7f, aio_nbytes=4096, aio_offset=40960}]) = 2 <0.000010>
300 1700000003.000200 io_getevents(0x7f00, 2, 2, [{data=0x2, obj=0x7f10, res=4096, res2=0}, {data=0x1, obj=0x7f20, res=4096, res2=0}], NULL) = 2 <0.000050>
300 1700000003.000300 io_submit(0x7f00, 1, [{aio_data=0x1, aio_lio_opcode=IOCB_CMD_PREAD, aio_fildes=3</d/idx>, aio_buf=0x7f, aio_nbytes=4096, aio_offset=8192}]) = 1 <0.000010>
300 1700000003.000400 io_getevents(0x7f00, 1, 1, [{data=0x1, obj=0x7f10, res=4096, res2=0}], NULL) = 1 <0.000050>
"""


def test_io_submit_is_a_unit_of_its_own_and_an_open_left_open_is_noted():
    d = digest_of(AIO)
    (t,) = d["templates"]
    assert t["template"] == "idx" and t["ops"] == {"read": 3, "open": 1} and t["bytes_read"] == 3 * 4096
    assert t["offsets"] == {"positioned": 1}       # 0, 40960, 8192: three runs, closed at the end for it
    assert d["notes"]["opens_not_closed"] == 1 and d["notes"]["submit_members"] == 3
    (lane,) = d["lanes"]
    assert [(p["motif"], p["units"]) for p in lane["phases"]] == [("open", 1), ("submit member+", 1), ("submit member", 1)] or \
        [(p["motif"], p["units"]) for p in lane["phases"]] == [("open", 1), ("submit member+", 2)]
    text = report_of(AIO)
    assert "idx" in text and "none: the trace decides every verdict" in text


def test_exported_file_refusals_and_unknown_opens(tmp_path):
    hdr = json.dumps({"aeiou_trace": 1, "source": "strace", "root": "/d", "lanes": 1, "lines": 0, "opens": 0, "creates": [], "notes": {"x": 1}})
    with pytest.raises(BuildError, match="line 2: .*"):
        digest.digest_of([hdr + "\n", "not json\n"], None)
    with pytest.raises(BuildError, match="line 2: not a JSON object"):
        digest.digest_of([hdr + "\n", "[1]\n"], None)
    with pytest.raises(BuildError, match="not an aeiou_trace: 1 file"):
        digest.digest_of(['{"aeiou_trace": 9}\n'], None)
    with pytest.raises(BuildError, match="no call in it"):
        digest.digest_of([hdr + "\n", "\n"], None)
    with pytest.raises(BuildError, match="--root <DIR>"):
        digest.digest_of(strace_of(["a.bin"]).splitlines(True), None)
    with pytest.raises(BuildError, match="carries its root"):
        digest.digest_of([hdr + "\n", json.dumps({"lane": 0, "t": 0, "op": "stat", "path": "a"}) + "\n"], "/other")
    # ops on opens the file never made: noted, not counted against a template
    events = [
        {"lane": 0, "t": 0, "dur": 1, "op": "read", "fd": 7, "len": 10, "ret": 10},
        {"lane": 0, "t": 1, "dur": 1, "op": "close", "fd": 7},
        {"lane": 0, "t": 2, "dur": 1, "op": "fstat", "fd": 7},
        {"lane": 0, "t": 3, "dur": 1, "op": "open", "fd": 0, "path": "w/1.bin", "flags": ["WRONLY", "CREAT"]},
        {"lane": 0, "t": 4, "dur": 1, "op": "write", "fd": 0, "len": 100, "ret": 100, "offset": 0},
        {"lane": 0, "t": 5, "dur": 1, "op": "write", "fd": 0, "len": 100, "ret": 100, "offset": 100},   # a pwrite leaves the position where it was
        {"lane": 0, "t": 6, "dur": 1, "op": "close", "fd": 0, "ret": "EIO"},
        {"lane": 0, "t": 7, "dur": 1, "op": "submit", "ops": [{"op": "read", "fd": 9, "offset": 0, "len": 1, "ret": 1}]},
    ]
    lines = [hdr + "\n", *(json.dumps(e) + "\n" for e in events)]
    d = digest.digest_of(lines, "/d")
    assert d["notes"]["data_on_unknown_open"] == 2 and d["notes"]["close_of_unknown_open"] == 1 and d["notes"]["fstat_on_unknown_open"] == 1
    assert d["notes"]["x"] == 1 and d["source"] == "trace"
    (t,) = d["templates"]
    assert t["template"] == "w/{a}.bin" and t["bytes_written"] == 200 and t["offsets"] == {"whole": 1}
    assert d["metrics"]["notes"]["lines_not_counted"] == 3     # what the runner refuses: a read, a close, and a submit on opens no line made
    text = digest.report(digest.analyze(lines, "/d"))
    assert "wrote 200 B" in text and "submit" in text


def test_order_other_and_the_rendering_of_every_verdict():
    down = digest_of(strace_of([f"s/{i:03}.npz" for i in range(29, -1, -1)]), paths=True)
    s = down["sites"][0]
    assert s["order"]["verdict"] == "other" and s["order"]["ascending"] == 0.0
    text = report_of(strace_of([f"s/{i:03}.npz" for i in range(29, -1, -1)]), paths=True)
    assert "other" in text and "interleaved scans" in text and "s/000.npz" in text
    # a refuted candidate, and an undecided size, each rendered
    two_text = strace_of([], per_open=[(f"s/{i}.npz", 100 * (i + 1)) for i in range(12)])
    two = digest_of(two_text, candidates=12)
    text = report_of(two_text, candidates=12)
    assert "choose refuted" in text and "size: undecided; settles: a repeat" in text
    assert any(o["question"] == "size" for o in two["open"]) and "at scale: the op multiset is the same in distribution" in text
    assert two["projection"]["candidates"] == 12 and ", 12 candidates):" in text


def test_fewer_candidates_than_paths_seen_is_noted_and_the_seen_count_used():
    d = digest_of(strace_of([f"s/{i:03}.npz" for i in permutation(12, 1)]), candidates=3)
    assert "12 distinct paths of template 0" in d["notes"]["candidates"]
    assert d["sites"][0]["selection"]["p_no_repeat_under_choose"] == pytest.approx(math.exp(-12 * 11 / 24))


# ---------------------------------------------------------------- the answers (§3.88)


BY, AT = "someone@host", "2026-10-29T12:00:00+00:00"


def one_pass(n=28, seed=1):
    return strace_of([f"s/{i:03}.npz" for i in permutation(n, seed)])


def two_passes(n=20):
    return strace_of([f"s/{i:03}.npz" for i in permutation(n, 1) + permutation(n, 2)])


def open_all():
    """Seven files once each, every one a different size, shuffled: all three verdicts undecided
    (no repeat; no repeated path to compare bytes on; six adjacent pairs, under the eight order needs)."""
    return strace_of([], per_open=[(f"s/{i:03}.npz", 4096 * (i + 1)) for i in permutation(7, 1)])


def answered(text, answers, **kw):
    return digest_of(text, answers=answers, by=BY, at=AT, **kw)


def test_a_fact_about_the_corpus_decides_a_site_and_is_recorded_as_asserted():
    d = answered(one_pass(), [{"template": "s/{a:03}.npz", "fact": "candidates", "value": 100, "note": "the dataset has 100 archives"}])
    (s,) = d["sites"]
    assert s["selection"]["verdict"] == "consume" and "choose" in s["selection"]["refuted"]
    assert s["projection"] == {"candidates": 100, "passes": 1, "asserted": ["candidates"]}
    assert d["asserted"] == [{"site": 0, "template": "s/{a:03}.npz", "fact": "candidates", "value": 100, "note": "the dataset has 100 archives", "by": BY, "at": AT}]
    assert d["open"] == [] and d["rejected"] == []


def test_a_fact_about_the_passes_changes_the_divergence_of_that_site_only():
    text = one_pass() + strace_of([f"t/{i:02}.bin" for i in permutation(10, 3)], tid=101)
    d = answered(text, [{"template": "s/{a:03}.npz", "fact": "passes", "value": 4}], epochs=1)
    by = {it["site"]: it for it in d["open"] if it["question"] == "selection"}
    assert by[0]["at_scale"]["epochs"] == 4 and by[0]["divergence"] == pytest.approx(math.exp(-4))
    assert by[1]["at_scale"]["epochs"] == 1 and by[1]["divergence"] == pytest.approx(math.exp(-1))
    assert d["sites"][0]["projection"] == {"candidates": None, "passes": 4, "asserted": ["passes"]}
    assert d["sites"][1]["projection"] == {"candidates": None, "passes": 1, "asserted": []}
    assert by[1]["rank"] == 1 and by[0]["rank"] == 2  # ranked by divergence, the larger first


def test_a_fact_the_trace_contradicts_is_rejected_with_the_evidence():
    d = answered(one_pass(), [{"template": "s/{a:03}.npz", "fact": "candidates", "value": 20}])
    assert d["sites"][0]["selection"]["verdict"] == "undecided" and d["asserted"] == []
    (r,) = d["rejected"]
    assert r["fact"] == "candidates" and r["value"] == 20 and "28 distinct" in r["why"]
    d = answered(two_passes(), [{"template": "s/{a:03}.npz", "fact": "passes", "value": 1}])
    (r,) = d["rejected"]
    assert r["fact"] == "passes" and "2 full passes" in r["why"]
    d = answered(one_pass(), [{"template": "s/{a:03}.npz", "fact": "candidates", "value": "many"}, {"template": "s/{a:03}.npz", "fact": "sizes", "value": 3}])
    assert [r["why"] for r in d["rejected"]] == ["not a whole number of at least 1", "no such fact for a set site: candidates, passes"]
    assert d["notes"]["rejected_answers"] == 2


def test_an_answer_to_an_open_question_moves_it_to_asserted_and_the_verdict_stays_the_traces():
    d = answered(open_all(), [{"template": "s/{a:03}.npz", "question": "selection", "answer": "consume", "note": "DataLoader with shuffle=True"}])
    (s,) = d["sites"]
    assert s["selection"]["verdict"] == "undecided"   # the measurement is the trace's; the answer sits beside it
    assert s["selection"]["asserted"] == {"answer": "consume", "by": BY, "at": AT, "note": "DataLoader with shuffle=True"}
    assert [it["question"] for it in d["open"]] == ["size", "order"] and [it["rank"] for it in d["open"]] == [1, 2]
    (a,) = d["asserted"]
    assert a == {"site": 0, "template": "s/{a:03}.npz", "question": "selection", "answer": "consume", "note": "DataLoader with shuffle=True", "by": BY, "at": AT}


def test_dont_know_is_recorded_and_leaves_the_item_open():
    d = answered(one_pass(), [{"template": "s/{a:03}.npz", "question": "selection", "answer": "unknown"}])
    assert d["asserted"][0]["answer"] == "unknown" and "note" not in d["asserted"][0]
    it = d["open"][0]
    assert it["question"] == "selection" and it["answered"] == "unknown"
    assert d["sites"][0]["selection"]["asserted"]["answer"] == "unknown"


def test_an_answer_against_a_decided_verdict_is_rejected_and_one_agreeing_is_redundant():
    d = answered(two_passes(), [{"template": "s/{a:03}.npz", "question": "selection", "answer": "choose"}])
    (r,) = d["rejected"]
    assert r["answer"] == "choose" and r["why"].startswith("the trace decided consume") and "equal length" in r["why"]
    assert d["asserted"] == []
    d = answered(two_passes(), [{"template": "s/{a:03}.npz", "question": "selection", "answer": "consume"}])
    assert d["asserted"] == [] and d["rejected"] == [] and d["notes"]["redundant_answers"] == 1


def test_answers_that_name_nothing_in_the_trace_are_rejected():
    d = answered(one_pass(), [
        {"template": "elsewhere/{a}.bin", "question": "selection", "answer": "consume"},
        {"template": "s/{a:03}.npz", "question": "colour", "answer": "blue"},
        {"template": "s/{a:03}.npz", "question": "selection", "answer": "maybe"},
        {"template": "s/{a:03}.npz", "answer": "consume"},
    ])
    assert [r["why"] for r in d["rejected"]] == [
        "no such template in the trace",
        "no such question for a set site: selection, size, order",
        "not one of consume, choose, unknown",
        "neither a fact nor a question",
    ]


def test_a_site_the_trace_decides_takes_no_facts_either():
    d = answered(two_passes(), [{"template": "s/{a:03}.npz", "fact": "candidates", "value": 500}])
    assert d["sites"][0]["selection"]["verdict"] == "consume" and d["asserted"][0]["fact"] == "candidates"
    assert d["sites"][0]["projection"]["candidates"] == 500   # recorded; the verdict needed no help


ANSWERS_SHOWN = [
    {"template": "s/{a:03}.npz", "fact": "passes", "value": 4},
    {"template": "s/{a:03}.npz", "question": "selection", "answer": "consume", "note": "shuffle=True"},
    {"template": "s/{a:03}.npz", "question": "order", "answer": "unknown"},
    {"template": "s/{a:03}.npz", "fact": "candidates", "value": 3},
]


def test_render_shows_the_assertions_and_the_rejections():
    text = digest.report(digest.analyze(open_all().splitlines(True), "/d"), answers=ANSWERS_SHOWN, by=BY, at=AT)
    assert "3 asserted" in text.splitlines()[0] and "1 rejected" in text.splitlines()[0]
    assert "asserted:" in text and "; by someone@host, 2026-10-29T12:00:00+00:00" in text
    assert "site 0 passes: 4" in text and "site 0 selection: consume (shuffle=True)" in text and "site 0 order: unknown" in text
    assert "rejected:" in text and "candidates 3: " in text and "7 distinct" in text
    assert "selection: undecided" in text and "asserted consume" in text   # the verdict stays the trace's, the assertion beside it
    assert "answered: unknown" in text


def test_the_rest_of_the_answer_branches(tmp_path):
    # a question about a template opened once: not a site
    d = answered(one_pass() + strace_of(["solo.bin"], tid=101), [{"template": "solo.bin", "question": "size", "answer": "drawn"}])
    assert d["rejected"][0]["why"].startswith("not a site: the template was opened once")
    # the evidence named against an answer the trace decided otherwise, per verdict kind
    d = answered(one_pass(), [{"template": "s/{a:03}.npz", "question": "order", "answer": "sequential"}, {"template": "s/{a:03}.npz", "question": "size", "answer": "drawn"}])
    assert d["rejected"][0]["why"].startswith("the trace decided permuted: ascending 0.48 of 27 adjacent pairs")
    assert d["rejected"][1]["why"] == "the trace decided constant: every open moved 4096 B"
    per_path = strace_of([], per_open=[(f"s/{i:03}.npz", 4096 * (i + 1)) for i in permutation(10, 1) + permutation(10, 2)])
    d = answered(per_path, [{"template": "s/{a:03}.npz", "question": "size", "answer": "drawn"}])
    assert d["rejected"][0]["why"] == "the trace decided per-path: 10 repeated path(s) always moved the same bytes"
    drawn = strace_of([], per_open=[("s/001.npz", 4096), ("s/002.npz", 4096), ("s/001.npz", 8192)])
    d = answered(drawn, [{"template": "s/{a:03}.npz", "question": "size", "answer": "per-path"}])
    assert d["rejected"][0]["why"] == "the trace decided drawn: 1 of 1 repeated path(s) moved differing bytes"


# ---------------------------------------------------------------- the application in the document (§3.88)


def test_the_document_carries_the_application_and_its_blind_spots_as_notes():
    app = {"name": "qdrant", "display": "Qdrant", "version": "1.20", "record": "1.19", "exact": True, "options": {"vectors_memory": "cold"},
           "blind_spots": [{"kind": "mmap", "note": "vectors are mapped", "source": "x"}]}
    d = answered(open_all(), [], application=app)
    assert d["application"] == app and d["notes"]["blind_spot_mmap"] == "vectors are mapped"
    text = digest.report(digest.analyze(open_all().splitlines(True), "/d"), application=app, by=BY, at=AT)
    assert "application: Qdrant 1.20; record 1.19" in text and "  vectors_memory: cold" in text and "blind spot (mmap): vectors are mapped" in text


# ---------------------------------------------------------------- the site kinds (§3.88 step 4, revised 2026-10-30)


def calls_strace(seq, root="/d", t0=1700000000.0, gap=0.001):
    """An strace of arbitrary opens: `seq` is (tid, path, ops) or (tid, path, ops, t), ops a list
    of ("read", n) | ("write", n) | ("rw", n) | ("pread", n, offset) | ("pwrite", n, offset) |
    ("fsync",), each open one line per call, `gap` seconds apart, closed at the end; `t` places
    the open at t0 + t seconds, else it follows the one before; ops [("unlink",)] is an unlink of
    the path and no open. The lines are written in time order, lanes interleaving."""
    lines, t, fd = [], t0, 3
    for entry in seq:
        tid, path, ops = entry[:3]
        if len(entry) == 4:
            t = t0 + entry[3]
        if ops == [("unlink",)]:
            lines.append((t, f'{tid} {t:.6f} unlink("{root}/{path}") = 0 <0.000010>'))
            t += gap
            continue
        flags = "O_RDONLY" if all(o[0] in ("read", "pread") for o in ops) else ("O_RDWR|O_CREAT" if any(o[0] == "rw" for o in ops) else "O_WRONLY|O_CREAT")
        lines.append((t, f'{tid} {t:.6f} openat(AT_FDCWD</d>, "{root}/{path}", {flags}|O_CLOEXEC, 0644) = {fd}<{root}/{path}> <0.000010>'))
        t += gap
        for o in ops:
            if o[0] in ("read", "rw"):
                lines.append((t, f'{tid} {t:.6f} read({fd}<{root}/{path}>, "x"..., {o[1]}) = {o[1]} <0.000100>'))
                if o[0] == "rw":
                    t += gap
                    lines.append((t, f'{tid} {t:.6f} write({fd}<{root}/{path}>, "x"..., {o[1]}) = {o[1]} <0.000100>'))
            elif o[0] == "write":
                lines.append((t, f'{tid} {t:.6f} write({fd}<{root}/{path}>, "x"..., {o[1]}) = {o[1]} <0.000100>'))
            elif o[0] == "pread":
                lines.append((t, f'{tid} {t:.6f} pread64({fd}<{root}/{path}>, "x"..., {o[1]}, {o[2]}) = {o[1]} <0.000100>'))
            elif o[0] == "pwrite":
                lines.append((t, f'{tid} {t:.6f} pwrite64({fd}<{root}/{path}>, "x"..., {o[1]}, {o[2]}) = {o[1]} <0.000100>'))
            else:
                lines.append((t, f"{tid} {t:.6f} fsync({fd}<{root}/{path}>) = 0 <0.000500>"))
            t += gap
        lines.append((t, f"{tid} {t:.6f} close({fd}<{root}/{path}>) = 0 <0.000005>"))
        t += gap * 10
        fd += 1
    return "\n".join(line for _, line in sorted(lines, key=lambda x: x[0])) + "\n"


KEYS = ["3f2a9c0e1b7d4e6f8a9b0c1d2e3f4a5b", "0000111122223333444455556666777a", "deadbeefdeadbeefdeadbeefdeadbeef", "cafebabecafebabecafebabecafebabe"]


def kv_trace():
    """A KV cache: four keys written once, their reads interleaving with the writes, three read
    after their write (one of them twice), and one key read that the trace never saw written."""
    k = [f"kv/{h}.pt" for h in KEYS]
    seq = [(100, k[0], [("write", 8192)]), (100, k[1], [("write", 8192)]), (101, k[0], [("read", 8192)]), (100, k[2], [("write", 8192)]),
           (101, k[1], [("read", 8192)]), (101, "kv/9999888877776666555544443333222a.pt", [("read", 8192)]), (100, k[3], [("write", 8192)]),
           (101, k[0], [("read", 8192)]), (101, k[2], [("read", 8192)])]
    return calls_strace(seq)


def test_a_keyed_site_counts_keys_hits_and_the_lag_from_write_to_read():
    d = digest_of(kv_trace())
    (s,) = d["sites"]
    assert s["kind"] == "keyed" and s["template"] == 0 and s["phase"] is None and s["touches"] == 9 and s["distinct"] == 5 and s["hex_name"] is True
    a = s["access"]
    assert a == {"keys": 5, "written": 4, "read": 4, "reads": 5, "writes": 4, "written_once_share": 1.0, "read_after_write": 4, "read_before_write": 1,
                 "rereads": 1, "reread_share": 0.2, "top_10pct_share": 0.4, "reads_per_key": a["reads_per_key"]}
    assert a["reads_per_key"]["n"] == 4 and a["reads_per_key"]["max"] == 2
    assert s["lag"]["verdict"] == "write_then_read" and s["lag"]["ns"]["n"] == 4 and s["lag"]["touches"]["n"] == 4 and s["lag"]["touches"]["max"] == 7
    assert s["origin"]["verdict"] == "undecided" and s["origin"]["settles"].startswith("a trace from the application's start")
    assert s["reuse"]["verdict"] == "repeated" and s["reuse"]["keys_reread"] == 1
    assert s["size"]["verdict"] == "constant"
    assert "selection" not in s and "order" not in s
    assert [(it["question"], it["candidates"]) for it in d["open"]] == [("origin", ["earlier_run", "other_writer"])]
    assert "never wrote" in d["open"][0]["ask"] and d["open"][0]["divergence"] == 0.0 and "phase" not in d["open"][0]
    assert s["projection"] == {"keys": None, "requests": None, "asserted": []}


def test_keyed_is_writes_and_reads_interleaving_and_a_hex_name_only_supports_it():
    k = [f"kv/{h}.pt" for h in KEYS[:3]]
    d = digest_of(calls_strace([(1, k[0], [("write", 100)]), (2, k[0], [("read", 100)]), (1, k[1], [("write", 200)]), (2, k[1], [("read", 200)])]))
    (s,) = d["sites"]
    assert s["kind"] == "keyed" and s["origin"]["verdict"] == "traced" and s["reuse"]["verdict"] == "undecided" and s["lag"]["verdict"] == "write_then_read"
    assert [it["question"] for it in d["open"]] == ["reuse"] and d["open"][0]["candidates"] == ["once", "repeated"]
    assert s["size"]["verdict"] == "per-path"
    # hex names read and never written: a sample set, noted that a store written elsewhere fits it
    d = digest_of(calls_strace([(2, k[0], [("read", 100)]), (2, k[1], [("read", 200)])]))
    (s,) = d["sites"]
    assert s["kind"] == "set" and "written elsewhere" in s["note"]
    # hex names written and never read: a bulk write
    d = digest_of(calls_strace([(1, k[0], [("write", 100)]), (1, k[1], [("write", 200)])]))
    assert [s["kind"] for s in d["sites"]] == ["bulk_write"]
    # interleaving without a hex name is keyed, the name no support
    d = digest_of(calls_strace([(1, "spill/7.bin", [("write", 100)]), (1, "spill/7.bin", [("read", 100)]), (1, "spill/8.bin", [("write", 100)])]))
    (s,) = d["sites"]
    assert s["kind"] == "keyed" and s["hex_name"] is False
    # an open both read and written interleaves within itself
    d = digest_of(calls_strace([(1, "db/1.bin", [("rw", 100)]), (1, "db/2.bin", [("rw", 100)])]))
    assert [s["kind"] for s in d["sites"]] == ["keyed"]
    # reads before writes of the same paths: a path read and then written again
    d = digest_of(calls_strace([(1, "spill/1.bin", [("read", 100)]), (1, "spill/2.bin", [("read", 100)]), (1, "spill/1.bin", [("write", 100)])]))
    assert [s["kind"] for s in d["sites"]] == ["keyed"]
    # interleaved with no read after a write of its key
    d = digest_of(calls_strace([(2, k[0], [("read", 100)]), (1, k[1], [("write", 100)]), (2, k[2], [("read", 100)])]))
    (s,) = d["sites"]
    assert s["kind"] == "keyed" and s["lag"]["verdict"] == "no_read_after_write"


def test_keyed_facts_and_answers():
    d = answered(kv_trace(), [{"template": "kv/{a:x32}.pt", "fact": "keys", "value": 1000}, {"template": "kv/{a:x32}.pt", "fact": "requests", "value": 50000},
                              {"template": "kv/{a:x32}.pt", "question": "origin", "answer": "earlier_run", "note": "the cache was warm"},
                              {"template": "kv/{a:x32}.pt", "question": "reuse", "answer": "once"},
                              {"template": "kv/{a:x32}.pt", "fact": "keys", "value": 3},
                              {"template": "kv/{a:x32}.pt", "fact": "candidates", "value": 3},
                              {"template": "kv/{a:x32}.pt", "question": "selection", "answer": "consume"}])
    s = d["sites"][0]
    assert s["projection"] == {"keys": 1000, "requests": 50000, "asserted": ["keys", "requests"]}
    assert s["origin"]["asserted"]["answer"] == "earlier_run" and d["open"] == []
    assert [r["why"] for r in d["rejected"]] == [
        "the trace decided repeated: 1 key(s) read more than once",
        "the trace touched 5 distinct keys of this template; 3 is fewer",
        "no such fact for a keyed site: keys, requests",
        "no such question for a keyed site: origin, reuse, size",
    ]


def build_search():
    """A leaf store: three leaves written by a build, then read by a search; never interleaving."""
    seq = [(1, f"leaves/{i}.bin", [("write", 4096 * (i + 1))]) for i in range(3)]
    seq += [(2, f"leaves/{i}.bin", [("read", 4096 * (i + 1))]) for i in (2, 0, 1)]
    return calls_strace(seq)


def test_all_writes_before_all_reads_is_two_phases_a_bulk_write_and_a_reader():
    d = digest_of(build_search())
    assert [(s["template"], s["phase"], s["kind"], s["touches"]) for s in d["sites"]] == [(0, "write", "bulk_write", 3), (0, "read", "set", 3)]
    w = d["sites"][0]
    assert w["label"]["verdict"] == "output" and w["label"]["refuted"] == {"checkpoint": "read after its first write: a checkpoint is read only after a failure, in another trace"}
    assert w["saves"]["verdict"] == "one_writer" and w["saves"]["count"] == 3
    assert [(it["site"], it["phase"], it["question"]) for it in d["open"]] == [(0, "read", "selection"), (0, "write", "size"), (0, "read", "size"), (0, "read", "order")]
    t = "leaves/{a}.bin"
    d = answered(build_search(), [{"template": t, "question": "label", "answer": "checkpoint"},
                                  {"template": t, "question": "selection", "answer": "consume"},
                                  {"template": t, "question": "size", "answer": "per-path", "phase": "write"},
                                  {"template": t, "fact": "keep", "value": 1, "phase": "read"},
                                  {"template": t, "fact": "passes", "value": 2, "phase": "sideways"}])
    assert [(a["phase"], a["question"]) for a in d["asserted"]] == [("read", "selection"), ("write", "size")]
    assert [r["why"] for r in d["rejected"]] == [
        "the trace decided output: read after its first write: a checkpoint is read only after a failure, in another trace",
        "no such fact for a set site: candidates, passes",
        "no such phase here: write, read",
    ]
    assert d["sites"][1]["selection"]["asserted"]["answer"] == "consume" and "asserted" not in d["sites"][1]["size"]
    # a phase of one open is a site only when it is one read at many offsets
    idx = [("pread", 4096, 4096 * i) for i in (5, 1, 9, 2, 7, 3, 8, 0)]
    d = digest_of(calls_strace([(1, "idx/0.bin", [("write", 65536)]), (1, "idx/1.bin", [("write", 65536)]), (2, "idx/0.bin", idx)]))
    assert [(s["phase"], s["kind"]) for s in d["sites"]] == [("write", "bulk_write"), ("read", "within")]
    d = digest_of(calls_strace([(1, "idx/0.bin", [("write", 65536)]), (2, "idx/0.bin", [("read", 100)]), (2, "idx/1.bin", [("read", 100)])]))
    assert [(s["phase"], s["kind"]) for s in d["sites"]] == [("read", "set")]


def restore_save():
    """A job restarted: each rank reads its shard of step 1, then saves step 2."""
    return calls_strace([(100, "ckpt/step_000001/__0_0.distcp", [("read", 4096)], 0), (101, "ckpt/step_000001/__1_0.distcp", [("read", 8192)], 0.0001),
                         (100, "ckpt/step_000002/__0_0.distcp", [("write", 4096), ("fsync",)], 10), (101, "ckpt/step_000002/__1_0.distcp", [("write", 8192), ("fsync",)], 10.0001)])


def test_reads_before_the_first_write_of_paths_never_written_are_a_restore():
    d = digest_of(restore_save())
    assert [(s["phase"], s["kind"]) for s in d["sites"]] == [("read", "restore"), ("write", "bulk_write")]
    r, w = d["sites"]
    assert r["restore"] == {"files": 2, "bytes": 12288} and r["projection"] == {"asserted": []}
    assert [(f["name"], f["values"], f["verdict"]) for f in w["fields"]] == [("a", 1, "undecided"), ("b", 2, "varies"), ("c", 1, "undecided")]
    assert w["label"]["verdict"] == "undecided" and w["label"]["supports"] == [
        "every save starts on all its lanes together", "every file fsynced", "2 file(s) of it read before its first write and never written: a restore"]
    assert r["facts"] == []
    # a restore starts within the trace's first quarter (2026-10-31, the user's): start-up I/O, the
    # model, then the parameters and optimizer state; reads of the pattern later are a reader, no restore
    late = [(100, "config.json", [("read", 100)], 0), (100, "ckpt/step_000001/__0_0.distcp", [("read", 4096)], 6),
            (101, "ckpt/step_000001/__1_0.distcp", [("read", 8192)], 6.0001),
            (100, "ckpt/step_000002/__0_0.distcp", [("write", 4096), ("fsync",)], 10), (101, "ckpt/step_000002/__1_0.distcp", [("write", 8192), ("fsync",)], 10.0001)]
    d = digest_of(calls_strace(late))
    assert [(s["phase"], s["kind"]) for s in d["sites"]] == [("read", "set"), ("write", "bulk_write")]
    assert d["sites"][1]["label"]["supports"] == ["every save starts on all its lanes together", "every file fsynced"]
    late[1], late[2] = late[1][:3] + (2,), late[2][:3] + (2.0001,)
    assert [s["kind"] for s in digest_of(calls_strace(late))["sites"]] == ["restore", "bulk_write"]


def dcp_save(steps=1, rewrite=False, keep=None, unsafe=False, prior=False):
    """DCP saves 10 s apart, each rank's shard opened together and fsynced, then rank 0's
    `.metadata`; `keep` deletes the shards of the save `keep` back once the next is written,
    `unsafe` deletes the last one's before the next starts, `prior` deletes a save from before
    the trace after the first, and `rewrite` writes a shard of step 1 again."""
    seq = []
    for step in range(1, steps + 1):
        base = 10 * step
        if unsafe and step >= 2:
            seq += [(100, f"ckpt/step_{step - 1:06}/__{r}_0.distcp", [("unlink",)], base - 1 + 0.001 * r) for r in range(2)]
        for rank in range(2):
            seq.append((100 + rank, f"ckpt/step_{step:06}/__{rank}_0.distcp", [("write", 4096 * (rank + 1)), ("fsync",)], base + 0.0001 * rank))
        seq.append((100, f"ckpt/step_{step:06}/.metadata", [("write", 512)], base + 0.01))
        if keep and step > keep:
            seq += [(100, f"ckpt/step_{step - keep:06}/__{r}_0.distcp", [("unlink",)], base + 0.02 + 0.001 * r) for r in range(2)]
        if prior and step == 1:
            seq.append((100, "ckpt/step_000000/__0_0.distcp", [("unlink",)], base + 0.02))
    if rewrite:
        seq.append((100, "ckpt/step_000001/__0_0.distcp", [("write", 4096)], 10 * (steps + 1)))
    return calls_strace(seq)


def test_a_bulk_write_site_judges_its_saves_fields_retention_and_label():
    d = digest_of(dcp_save())
    (s,) = d["sites"]   # the shard template; .metadata is written once, one open, no site
    assert s["kind"] == "bulk_write" and s["touches"] == 2 and s["distinct"] == 2
    assert s["rewrite"] == {"verdict": "new_names", "rewritten_paths": 0}
    assert [(f["name"], f["values"], f["verdict"]) for f in s["fields"]] == [("a", 1, "undecided"), ("b", 2, "varies"), ("c", 1, "undecided")]
    assert s["fields"][0]["settles"].startswith("a second save")
    assert s["sync"] == {"fsync_share": 1.0, "opens": 2}
    sv = s["saves"]
    assert sv["verdict"] == "simultaneous" and sv["count"] == 1 and sv["lanes"]["max"] == 2 and sv["interval_ns"]["n"] == 0 and sv["start_spread_ns"]["max"] <= 200_000
    assert s["retention"] == {"verdict": "kept", "deleted": 0, "unsafe": 0}
    assert s["label"]["verdict"] == "undecided" and s["label"]["consistent"] == ["checkpoint", "output"] and s["label"]["supports"] == ["every save starts on all its lanes together", "every file fsynced"]
    assert s["size"]["verdict"] == "undecided"
    assert [(it["question"], it["candidates"]) for it in d["open"]] == [("label", ["checkpoint", "output"]), ("size", ["per-path", "drawn"])]
    assert "read back only after one" in d["open"][0]["ask"]
    assert s["projection"] == {"values:a": None, "values:b": None, "values:c": None, "passes": None, "keep": None, "asserted": []}
    # two saves: the step field varies, the item field holds one value over both, so it is constant and no fact
    d = digest_of(dcp_save(steps=2))
    s = d["sites"][0]
    assert [(f["name"], f["verdict"]) for f in s["fields"]] == [("a", "varies"), ("b", "varies"), ("c", "constant")]
    assert s["saves"]["count"] == 2 and s["saves"]["interval_ns"]["n"] == 1
    assert s["projection"] == {"values:a": None, "values:b": None, "passes": None, "keep": None, "asserted": []}
    # a rewrite is evidence against a checkpoint, which is never overwritten
    d = digest_of(dcp_save(rewrite=True))
    s = d["sites"][0]
    assert s["rewrite"]["verdict"] == "overwrite" and s["rewrite"]["rewritten_paths"] == 1 and "never overwritten" in s["rewrite"]["note"]
    assert [f["verdict"] for f in s["fields"]] == ["constant", "varies", "constant"] and s["saves"]["verdict"] == "mixed"
    assert s["label"] == {"verdict": "output", "refuted": {"checkpoint": "1 path(s) written again: a checkpoint is never overwritten"},
                          "supports": [], "settles": "decided"}
    assert s["size"]["verdict"] == "per-path" and d["open"] == []


def test_retention_is_kept_keeps_n_or_unsafe():
    s = digest_of(dcp_save(steps=3, keep=1))["sites"][0]
    assert s["retention"] == {"verdict": "keeps", "n": 1, "deleted": 4, "unsafe": 0}
    assert "earlier saves deleted only once the next was stable (keeps 1)" in s["label"]["supports"]
    s = digest_of(dcp_save(steps=2, keep=2))["sites"][0]
    assert s["retention"] == {"verdict": "kept", "deleted": 0, "unsafe": 0}
    s = digest_of(dcp_save(prior=True))["sites"][0]   # a save from before the trace, deleted once the first is stable
    assert s["retention"] == {"verdict": "keeps", "n": 1, "deleted": 1, "unsafe": 0}
    s = digest_of(dcp_save(steps=2, unsafe=True))["sites"][0]
    assert s["retention"]["verdict"] == "unsafe" and s["retention"]["unsafe"] == 2 and "contradicts what a checkpoint is for" in s["retention"]["note"]


def test_one_writing_lane_is_accepted_and_the_writers_asked():
    seq = [(100, f"ckpt/step_{s:06}.pt", [("write", 1 << 20)], 10 * s) for s in (1, 2, 3)]
    s = digest_of(calls_strace(seq))["sites"][0]
    assert [f["name"] for f in s["facts"]] == ["values:a", "passes", "keep", "writers"]
    assert s["saves"]["verdict"] == "one_writer" and s["saves"]["count"] == 3 and s["saves"]["interval_ns"]["n"] == 2
    assert s["saves"]["interval_ns"]["mean"] == pytest.approx(10e9, rel=1e-3)
    assert s["label"]["supports"] == ["one writing lane, as plain DDP's rank 0"]
    # lanes whose opens overlap but start far apart against the burst's length are no checkpoint's burst
    seq = [(100, "out/part_0.bin", [("write", 4096)], 0), (101, "out/part_1.bin", [("write", 4096)], 0.0015)]
    s = digest_of(calls_strace(seq))["sites"][0]
    assert s["saves"]["verdict"] == "staggered" and s["saves"]["count"] == 1


def test_bulk_write_facts_and_answers():
    t = "ckpt/step_{a:06}/__{b}_{c}.distcp"
    d = answered(dcp_save(), [{"template": t, "fact": "values:b", "value": 512, "note": "512 ranks"}, {"template": t, "fact": "passes", "value": 20},
                              {"template": t, "fact": "keep", "value": 2},
                              {"template": t, "question": "label", "answer": "checkpoint", "note": "torch.distributed.checkpoint"},
                              {"template": t, "fact": "values:b", "value": 1}, {"template": t, "fact": "writers", "value": 4},
                              {"template": t, "question": "field:a", "answer": "new_values"}])
    s = d["sites"][0]
    assert s["projection"] == {"values:a": None, "values:b": 512, "values:c": None, "passes": 20, "keep": 2, "asserted": ["keep", "passes", "values:b"]}
    assert s["label"]["asserted"]["answer"] == "checkpoint" and [it["question"] for it in d["open"]] == ["size"]
    assert [r["why"] for r in d["rejected"]] == [
        "field b took 2 values in the trace; 1 is fewer",
        "no such fact for a bulk_write site: values:a, values:b, values:c, passes, keep",
        "no such question for a bulk_write site: label, size",
    ]
    t2 = "ckpt/step_{a:06}/__{b}_{c}.distcp"
    d = answered(dcp_save(steps=2), [{"template": t2, "fact": "passes", "value": 1}, {"template": t2, "fact": "keep", "value": 1}])
    assert [r["why"] for r in d["rejected"]] == ["the trace shows 2 saves; 1 is fewer", "the trace kept all 2 saves; 1 is fewer"]
    d = answered(dcp_save(steps=3, keep=2), [{"template": t2, "fact": "keep", "value": 1}])
    assert [r["why"] for r in d["rejected"]] == ["the trace kept 2 saves at once; 1 is fewer"]
    d = answered(dcp_save(rewrite=True), [{"template": t, "question": "label", "answer": "checkpoint"}])
    assert d["rejected"][0]["why"] == "the trace decided output: 1 path(s) written again: a checkpoint is never overwritten"


def index_trace(reads, seed=1, hot=0):
    """One index file read at `reads` offsets: 4 KiB sectors drawn uniformly from 200, or with `hot` sectors taking half the reads."""
    import random
    r = random.Random(seed)
    ops = []
    for _ in range(reads):
        sector = r.randrange(hot) if hot and r.random() < 0.5 else r.randrange(200)
        ops.append(("pread", 4096, sector * 4096))
    return calls_strace([(7, "sift_disk.index", ops)])
def test_a_within_site_measures_reuse_and_skew_of_the_offsets_in_one_open():
    d = digest_of(index_trace(400))
    (s,) = d["sites"]
    assert s["kind"] == "within" and s["touches"] == 1 and s["reads"] == 400 and 150 < s["distinct_offsets"] <= 200
    assert s["reuse_share"] == pytest.approx(1 - s["distinct_offsets"] / 400) and s["span"] == 200 * 4096
    assert s["skew"]["verdict"] == "uniform" and s["skew"]["top_10pct_share"] <= 1.2 * s["skew"]["uniform_share"] and 0.2 < s["skew"]["uniform_share"] < 0.25
    assert 380 < s["runs"]["n"] <= 400 and s["runs"]["max"] >= 1 and s["memory"]["verdict"] == "undecided"   # a sequential pair by chance joins a run and s["dependency"]["verdict"] == "undecided"
    assert [(it["question"], it["candidates"]) for it in d["open"]] == [("memory", ["fits", "exceeds"]), ("dependency", ["dependent", "independent"])]
    assert d["open"][1]["settles"].startswith("not from a trace")
    assert s["projection"] == {"size": None, "requests": None, "asserted": []}
    d = digest_of(index_trace(400, hot=4))
    s = d["sites"][0]
    assert s["skew"]["verdict"] == "skewed" and s["skew"]["top_10pct_share"] > 0.4
    d = digest_of(index_trace(30))
    s = d["sites"][0]
    assert s["skew"]["verdict"] == "undecided" and [it["question"] for it in d["open"]] == ["skew", "memory", "dependency"]
    # a single open read in one sequential run is no site; nor is one with fewer than eight reads
    assert digest_of(calls_strace([(7, "big.bin", [("read", 4096)] * 20)]))["sites"] == []
    assert digest_of(calls_strace([(7, "idx", [("pread", 4096, 4096 * i) for i in (5, 1, 9, 2)])]))["sites"] == []


def test_within_facts_and_answers():
    d = answered(index_trace(400), [{"template": "sift_disk.index", "fact": "size", "value": 1 << 40}, {"template": "sift_disk.index", "fact": "requests", "value": 10000},
                                    {"template": "sift_disk.index", "question": "memory", "answer": "exceeds", "note": "1 TiB index, 256 GiB hosts"},
                                    {"template": "sift_disk.index", "question": "dependency", "answer": "dependent"},
                                    {"template": "sift_disk.index", "fact": "size", "value": 4096},
                                    {"template": "sift_disk.index", "question": "skew", "answer": "skewed"}])
    s = d["sites"][0]
    assert s["projection"] == {"size": 1 << 40, "requests": 10000, "asserted": ["requests", "size"]} and d["open"] == []
    sk = s["skew"]
    assert [r["why"] for r in d["rejected"]] == [
        f"the trace read up to byte {200 * 4096} of this file; 4096 is smaller",
        f"the trace decided uniform: the top tenth of the offsets took {sk['top_10pct_share']:.2f} of the reads, about {sk['uniform_share']:.2f} under a uniform spread, uniform within noise",
    ]


def test_render_shows_every_kind():
    text = report_of(kv_trace())
    assert "keyed: 5 keys, 4 written, 5 reads" in text and "lag: write_then_read" in text and "origin: undecided" in text and "reuse: repeated" in text
    text = report_of(dcp_save(steps=3, keep=1))
    assert "bulk write: 6 files in 3 save(s), 0 rewritten, fsync on 100%" in text and "field c: constant (1 value 0..0)" in text and "field b: varies (2 values 0..1)" in text
    assert "saves: simultaneous, 3 save(s), interval p50" in text and "retention: keeps 1" in text and "label: undecided; supports: every save starts on all its lanes together" in text
    text = report_of(dcp_save(rewrite=True))
    assert "label: output; checkpoint refuted: 1 path(s) written again" in text
    text = report_of(dcp_save(steps=2, unsafe=True))
    assert "retention: unsafe (2 file(s) deleted before the next save was stable)" in text
    text = report_of(restore_save())
    assert "site 0/read (template 0): restore: 2 files, 12.0 KiB read before the first write" in text and "site 0/write (template 0): bulk write" in text
    text = report_of(build_search(), answers=[{"template": "leaves/{a}.bin", "question": "selection", "answer": "consume"}], by=BY, at=AT)
    assert "1. site 0/write size" in text and "site 0/read selection: consume" in text
    text = report_of(calls_strace([(1, f"kv/{h}.pt", [("write", 100)]) for h in KEYS[:2]]))
    assert "field a: undecided (1 value)" not in text and "field a: varies (2 values)" in text
    text = report_of(index_trace(400))
    assert "within one open: 400 reads at" in text and "skew: uniform" in text and "dependency: undecided" in text


def test_the_last_branches_of_the_kinds():
    # a template that is no site has no sites
    assert digest_of(calls_strace([(7, "once.bin", [("read", 10)])]))["sites"] == []
    d = answered(calls_strace([(7, "once.bin", [("read", 10)])]), [{"template": "once.bin", "fact": "size", "value": 5}])
    assert d["rejected"][0]["why"] == "not a site"
    # an answer against a traced origin
    k = [f"kv/{h}.pt" for h in KEYS[:2]]
    d = answered(calls_strace([(1, k[0], [("write", 100)]), (2, k[0], [("read", 100)]), (2, k[1], [("read", 100)]), (1, k[1], [("write", 100)])]),
                 [{"template": "kv/{a:x32}.pt", "question": "origin", "answer": "other_writer"}])
    assert d["sites"][0]["origin"]["verdict"] == "undecided"   # k[1] was read before its write
    d = answered(calls_strace([(1, k[0], [("write", 100)]), (2, k[0], [("read", 100)]), (1, k[1], [("write", 100)])]),
                 [{"template": "kv/{a:x32}.pt", "question": "origin", "answer": "other_writer"}])
    assert d["rejected"][0]["why"] == "the trace decided traced: every key read was written earlier in the trace"
    # a skew between 1.2x and 1.5x of a uniform spread's top-tenth share stays undecided
    import random
    r = random.Random(3)
    ops = [("pread", 4096, (r.randrange(20) if r.random() < 0.22 else r.randrange(200)) * 4096) for _ in range(400)]
    s = digest_of(calls_strace([(7, "idx", ops)]))["sites"][0]
    assert s["skew"]["verdict"] == "undecided" and 1.2 < s["skew"]["top_10pct_share"] / s["skew"]["uniform_share"] < 1.5, s["skew"]
    assert "within 1.2x" in s["skew"]["settles"]
