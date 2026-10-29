"""`aeiou-trace digest` (`PROJECT_BRIEF.md` item 47, `DESIGN_REVIEW.md` §3.86, §3.87): the
structural summary a fit is made from, with a verdict per site on what the trace decides and an
`open` list, ranked by projected divergence, of what it does not; and the questions it asks
(§3.88): facts first, then the open items, every answer checked against the trace and recorded
under `asserted` with provenance, never as a measurement."""
import json
import math
import pathlib
import shutil
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BUILDER = HERE.parent
sys.path.insert(0, str(BUILDER))

from aeiou import digest, trace  # noqa: E402
from aeiou.nodes import BuildError  # noqa: E402


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
    trace.write_export(str(path), header, events)
    with open(path) as f:
        from_file = digest.digest_of(f, None)
    from_strace = digest_of(text)
    for key in ("templates", "sites", "open"):
        assert from_file[key] == from_strace[key]
    assert from_file["source"] == "trace" and from_strace["source"] == "strace"
    assert [p["motif"] for l in from_file["lanes"] for p in l["phases"]] == [p["motif"] for l in from_strace["lanes"] for p in l["phases"]]


def test_the_document_names_its_format_and_root():
    d = digest_of(strace_of(["a.bin", "b.bin"]))
    assert d["aeiou_digest"] == 1 and d["root"] == "/d" and d["source"] == "strace"
    assert d["ops"] == 8 and d["contexts"] == 1
    assert isinstance(d["notes"], dict)


def test_nothing_under_the_root_is_an_error():
    with pytest.raises(BuildError, match="no call under --root /nowhere"):
        digest_of(strace_of(["a.bin"]), root="/nowhere")


def test_text_report_reads_like_a_summary():
    ids = permutation(28, 1)
    d = digest_of(strace_of([f"s/{i:03}.npz" for i in ids]), epochs=2)
    text = digest.render(d)
    assert "s/{a:03}.npz" in text and "28 paths" in text
    assert "open fstat read close" in text
    assert "selection" in text and "undecided" in text
    assert "open questions" in text and "1." in text


# ---------------------------------------------------------------- the command


def run(*args, stdin=None):
    # stdin is always a pipe here, never the terminal pytest may run on: the digest asks nothing
    return subprocess.run([sys.executable, "-m", "aeiou.trace", "digest", *args], capture_output=True, text=True, cwd=BUILDER, input=stdin or "")


def test_cli_writes_json_or_text(tmp_path):
    text = strace_of([f"s/{i:03}.npz" for i in permutation(12, 1)])
    tr = tmp_path / "t.txt"
    tr.write_text(text)
    r = run(str(tr), "--root", "/d")
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["aeiou_digest"] == 1 and doc["trace"] == str(tr)
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
    trace.write_export(str(ex), header, events)
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
    text = digest.render(d)
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
    with pytest.raises(BuildError, match="needs --root"):
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
    d = digest.digest_of([hdr + "\n", *(json.dumps(e) + "\n" for e in events)], "/d")
    assert d["notes"]["data_on_unknown_open"] == 2 and d["notes"]["close_of_unknown_open"] == 1 and d["notes"]["fstat_on_unknown_open"] == 1
    assert d["notes"]["x"] == 1 and d["source"] == "trace"
    (t,) = d["templates"]
    assert t["template"] == "w/{a}.bin" and t["bytes_written"] == 200 and t["offsets"] == {"whole": 1}
    text = digest.render(d)
    assert "wrote 200 B" in text and "submit" in text


def test_order_other_and_the_rendering_of_every_verdict():
    down = digest_of(strace_of([f"s/{i:03}.npz" for i in range(29, -1, -1)]), paths=True)
    s = down["sites"][0]
    assert s["order"]["verdict"] == "other" and s["order"]["ascending"] == 0.0
    text = digest.render(down)
    assert "other" in text and "interleaved scans" in text and "s/000.npz" in text
    # a refuted candidate, and an undecided size, each rendered
    two = digest_of(strace_of([], per_open=[(f"s/{i}.npz", 100 * (i + 1)) for i in range(12)]), candidates=12)
    text = digest.render(two)
    assert "choose refuted" in text and "size: undecided; settles: a repeat" in text
    assert any(o["question"] == "size" for o in two["open"]) and "at scale: the op multiset is the same in distribution" in text
    assert two["projection"]["candidates"] == 12 and ", 12 candidates):" in text


def test_fewer_candidates_than_paths_seen_is_noted_and_the_seen_count_used():
    d = digest_of(strace_of([f"s/{i:03}.npz" for i in permutation(12, 1)]), candidates=3)
    assert "12 distinct paths of template 0" in d["notes"]["candidates"]
    assert d["sites"][0]["selection"]["p_no_repeat_under_choose"] == pytest.approx(math.exp(-12 * 11 / 24))


def test_touches_to_decide_is_the_least_that_refutes_no_repeat():
    for n in list(range(2, 60)) + [100, 1000, 12345]:
        k = digest.touches_to_decide(n)
        assert digest.p_no_repeat(k, n) <= 0.05
        assert k == 2 or digest.p_no_repeat(k - 1, n) > 0.05
    assert digest.candidates_for_choose(2) == 2


def test_formatting_helpers():
    assert digest._bytes(5) == "5 B" and digest._bytes(3 << 30) == "3.0 GiB"
    assert digest._ns(2.5e9) == "2.50 s" and digest._ns(3e6) == "3.0 ms" and digest._ns(4e3) == "4 us" and digest._ns(7) == "7 ns"
    assert digest._p50({"n": 0, "max": 0, "buckets": []}) == 0
    assert digest.offsets_class([]) == ("none", 0)
    assert digest.compress(["a", "b", "a", "b", "c"]) == [("a b", 2), ("c", 1)]


# ---------------------------------------------------------------- the questions (§3.88)


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
    assert [r["why"] for r in d["rejected"]] == ["not a whole number of at least 1", "no such fact: candidates, passes"]
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
        "no such question: selection, size, order",
        "not one of consume, choose, unknown",
        "neither a fact nor a question",
    ]


def test_a_site_the_trace_decides_takes_no_facts_either():
    d = answered(two_passes(), [{"template": "s/{a:03}.npz", "fact": "candidates", "value": 500}])
    assert d["sites"][0]["selection"]["verdict"] == "consume" and d["asserted"][0]["fact"] == "candidates"
    assert d["sites"][0]["projection"]["candidates"] == 500   # recorded; the verdict needed no help


class Scripted:
    """A console with the answers scripted: `None` for end of input."""

    def __init__(self, *lines):
        self.lines = list(lines)
        self.said = []
        self.asked = []

    def say(self, text):
        self.said.append(text)

    def ask(self, prompt):
        self.asked.append(prompt)
        return self.lines.pop(0) if self.lines else None


def interview(text, *lines, **kw):
    an = digest.analyze(text.splitlines(True), "/d")
    io = Scripted(*lines)
    doc, answers = digest.interview(an, io, by=BY, at=AT, **kw)
    return doc, answers, io


def test_the_interview_asks_facts_first_then_the_open_items_by_rank():
    doc, answers, io = interview(open_all(), "", "4", "1", "the loader is a DataLoader with shuffle=True", "3", "", "2", "")
    # facts: candidates (don't know), passes 4; then selection (1: consume) with a note, size (3: don't know), order (2: permuted)
    assert [a for a in answers if "fact" in a] == [{"template": "s/{a:03}.npz", "fact": "passes", "value": 4}]
    assert [(a["question"], a["answer"]) for a in answers if "question" in a] == [("selection", "consume"), ("size", "unknown"), ("order", "permuted")]
    assert answers[1]["note"] == "the loader is a DataLoader with shuffle=True" and "note" not in answers[2]
    assert doc["sites"][0]["projection"]["passes"] == 4
    assert [it["question"] for it in doc["open"]] == ["size"] and doc["open"][0]["answered"] == "unknown"
    assert len(doc["asserted"]) == 4
    screen = "\n".join(io.said)
    assert "How many files of this kind" in screen and "How many passes" in screen
    assert "1. consume" in screen and "2. choose" in screen and "3. don't know" in screen
    assert "at 4 pass(es)" in screen   # the divergence line uses the fact just given
    assert all(p.startswith("answer") or p.startswith("how do you know") or p.startswith("number") for p in io.asked)


def test_the_interview_re_asks_on_nonsense_and_on_a_contradiction_and_stops_at_end_of_input():
    doc, answers, io = interview(open_all(), "lots", "5", "2000", "", "9", "x", "2", "")
    # candidates: "lots" re-asked, 5 contradicts the 7 seen (re-asked with the evidence), 2000 taken; passes: don't know
    # selection: 9 and x re-asked, 2 = choose, no note; then end of input: size and order stay open, unasked
    assert [a for a in answers if "fact" in a] == [{"template": "s/{a:03}.npz", "fact": "candidates", "value": 2000}]
    assert [(a["question"], a["answer"]) for a in answers if "question" in a] == [("selection", "choose")]
    said = "\n".join(io.said)
    assert "7 distinct" in said and "a whole number" in said
    assert doc["rejected"] == []   # a rejected interactive answer is re-asked, never recorded
    assert [it["question"] for it in doc["open"]] == ["size", "order"] and "answered" not in doc["open"][0]
    assert doc["sites"][0]["selection"]["verdict"] == "undecided"   # 2000 candidates do not refute a draw after 7 touches at 5 %


def test_the_interview_asks_nothing_when_the_trace_decides_everything():
    doc, answers, io = interview(two_passes())
    assert answers == [] and io.asked == [] and doc["asserted"] == []


def test_answers_file_round_trip(tmp_path):
    f = tmp_path / "t.answers.json"
    answers = [{"template": "s/{a:03}.npz", "fact": "passes", "value": 4}, {"template": "s/{a:03}.npz", "question": "selection", "answer": "consume", "note": "n"}]
    digest.write_answers(str(f), answers, trace="t.txt", trace_sha256="ab" * 32, by=BY, at=AT)
    header, back = digest.read_answers(str(f))
    assert back == answers and header["aeiou_answers"] == 1 and header["trace"] == "t.txt" and header["by"] == BY and header["at"] == AT
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(BuildError, match="not an aeiou_answers"):
        digest.read_answers(str(tmp_path / "bad.json"))
    (tmp_path / "junk.json").write_text("[")
    with pytest.raises(BuildError):
        digest.read_answers(str(tmp_path / "junk.json"))


def test_render_shows_the_assertions_and_the_rejections():
    d = answered(open_all(), [
        {"template": "s/{a:03}.npz", "fact": "passes", "value": 4},
        {"template": "s/{a:03}.npz", "question": "selection", "answer": "consume", "note": "shuffle=True"},
        {"template": "s/{a:03}.npz", "question": "order", "answer": "unknown"},
        {"template": "s/{a:03}.npz", "fact": "candidates", "value": 3},
    ])
    text = digest.render(d)
    assert "3 asserted" in text.splitlines()[0] and "1 rejected" in text.splitlines()[0]
    assert "asserted:" in text and "; by someone@host, 2026-10-29T12:00:00+00:00" in text
    assert "site 0 passes: 4" in text and "site 0 selection: consume (shuffle=True)" in text and "site 0 order: unknown" in text
    assert "rejected:" in text and "candidates 3: " in text and "7 distinct" in text
    assert "selection: undecided" in text and "asserted consume" in text   # the verdict stays the trace's, the assertion beside it
    assert "answered: unknown" in text


class Tty:
    """A stdin that says it is a terminal and holds the lines typed."""

    def __init__(self, *lines):
        self.text = "".join(line + "\n" for line in lines)
        self.pos = 0

    def isatty(self):
        return True

    def readline(self):
        nl = self.text.find("\n", self.pos)
        if nl < 0:
            return ""
        line, self.pos = self.text[self.pos:nl + 1], nl + 1
        return line


def test_cli_asks_on_a_terminal_and_writes_the_answers_beside_the_document(tmp_path, monkeypatch, capsys):
    tr = tmp_path / "t.txt"
    tr.write_text(open_all())
    out = tmp_path / "t.digest.json"
    monkeypatch.setattr(sys, "stdin", Tty("8", "7", "4", "1", "shuffle=True", "3", "", "2", "because"))
    assert trace.main(["digest", str(tr), "--root", "/d", "-o", str(out), "--answered-by", "tester"]) == 0
    err = capsys.readouterr().err
    doc = json.loads(out.read_text())
    assert doc["sites"][0]["selection"]["verdict"] == "consume"   # 7 candidates refute a draw after 7 touches without a repeat (e^-3 < 0.05)
    assert [a.get("fact") or a["question"] for a in doc["asserted"]] == ["candidates", "passes", "size", "order"]
    assert all(a["by"] == "tester" for a in doc["asserted"]) and doc["asserted"][0]["at"].endswith("+00:00")
    assert len(doc["trace_sha256"]) == 64
    af = tmp_path / "t.answers.json"
    header, answers = digest.read_answers(str(af))
    assert header["trace"] == str(tr) and header["trace_sha256"] == doc["trace_sha256"] and header["by"] == "tester"
    assert len(answers) == 4 and header["application"] is None
    assert f"answers: {af}" in err and f"--answers {af}" in err and "aeiou-trace digest" in err
    # the file replays without a terminal and produces the same assertions
    monkeypatch.setattr(sys, "stdin", Tty())
    out2 = tmp_path / "again.json"
    assert trace.main(["digest", str(tr), "--root", "/d", "-o", str(out2), "--answers", str(af)]) == 0
    doc2 = json.loads(out2.read_text())
    assert doc2["asserted"] == doc["asserted"] and doc2["sites"] == doc["sites"]


def test_cli_asks_nothing_without_a_terminal_or_with_no_questions(tmp_path, monkeypatch, capsys):
    tr = tmp_path / "t.txt"
    tr.write_text(open_all())
    r = run(str(tr), "--root", "/d")
    assert r.returncode == 0 and r.stderr == ""
    doc = json.loads(r.stdout)
    assert doc["asserted"] == [] and len(doc["open"]) == 3 and not list(tmp_path.glob("*.answers.json"))
    monkeypatch.setattr(sys, "stdin", Tty("100"))
    assert trace.main(["digest", str(tr), "--root", "/d", "--no-questions"]) == 0
    cap = capsys.readouterr()
    assert cap.err == "" and json.loads(cap.out)["asserted"] == []
    # a terminal, no -o and no --answers: the answers live in the document only, and the report says so
    monkeypatch.setattr(sys, "stdin", Tty("8", "", "", "1", "", "3", "", "3", ""))
    assert trace.main(["digest", str(tr), "--root", "/d", "--text"]) == 0
    cap = capsys.readouterr()
    assert "asserted" in cap.out and "in the document only" in cap.err and not list(tmp_path.glob("*.answers.json"))


def test_cli_answers_file_refusals(tmp_path, monkeypatch):
    tr = tmp_path / "t.txt"
    tr.write_text(one_pass())
    r = run(str(tr), "--root", "/d", "--answers", str(tmp_path / "absent.json"))
    assert r.returncode == 1 and "absent.json" in r.stderr and "no terminal to ask on" in r.stderr
    (tmp_path / "bad.json").write_text("{}")
    r = run(str(tr), "--root", "/d", "--answers", str(tmp_path / "bad.json"))
    assert r.returncode == 1 and "not an aeiou_answers" in r.stderr
    # a file given on another trace is applied, and noted
    af = tmp_path / "other.answers.json"
    digest.write_answers(str(af), [{"template": "s/{a:03}.npz", "question": "selection", "answer": "consume"}], trace="elsewhere.txt", trace_sha256="00" * 32, by=BY, at=AT)
    r = run(str(tr), "--root", "/d", "--answers", str(af))
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["asserted"][0]["by"] == BY and "another trace" in doc["notes"]["answers_trace"]
    # a terminal and an --answers path that does not exist yet: asked, then written there
    monkeypatch.setattr(sys, "stdin", Tty("8", "", "", "1", "", "3", "", "3", ""))
    new = tmp_path / "new.json"
    assert trace.main(["digest", str(tr), "--root", "/d", "--answers", str(new)]) == 0
    assert digest.read_answers(str(new))[1][0]["answer"] == "consume"


def test_the_rest_of_the_answer_branches(tmp_path):
    # a question about a template opened once: not a site
    d = answered(one_pass() + strace_of(["solo.bin"], tid=101), [{"template": "solo.bin", "question": "size", "answer": "drawn"}])
    assert d["rejected"][0]["why"] == "not a site: the template was opened once"
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
    # end of input while a fact is asked, and at the note
    doc, answers, io = interview(open_all(), "lots")
    assert answers == [] and doc["asserted"] == []
    doc, answers, io = interview(open_all(), "", "", "1")
    assert [(a["question"], a["answer"]) for a in answers] == [("selection", "consume")] and "note" not in answers[0]
    assert doc["asserted"][0]["answer"] == "consume"
    # an answers file that cannot be read, and one whose answers are not a list
    with pytest.raises(BuildError, match="answers.json"):
        digest.read_answers(str(tmp_path / "absent" / "x.answers.json"))
    (tmp_path / "s.json").write_text('{"aeiou_answers": 1, "answers": "consume"}')
    with pytest.raises(BuildError, match="a list of objects"):
        digest.read_answers(str(tmp_path / "s.json"))
    assert trace._answers_beside("a/t.digest.json") == "a/t.answers.json" and trace._answers_beside("a/t.json") == "a/t.json.answers.json"
    assert "@" in trace._who() and trace._now().endswith("+00:00")


# ---------------------------------------------------------------- the application records (§3.88)


from aeiou import applications as ap  # noqa: E402

APPS = ap.load(ap.default_dir())


def dcp_load():
    """Two ranks' shards read at two checkpoint steps: the shard template is a site of four opens."""
    return strace_of([], per_open=[("ckpt/step_000001/__0_0.distcp", 4096), ("ckpt/step_000001/__1_0.distcp", 8192),
                                   ("ckpt/step_000002/__0_0.distcp", 4096), ("ckpt/step_000002/__1_0.distcp", 8192)])


def interview_with(text, *lines, apps=APPS, **kw):
    an = digest.analyze(text.splitlines(True), "/d")
    io = Scripted(*lines)
    doc, answers = digest.interview(an, io, by=BY, at=AT, apps=apps, **kw)
    return doc, answers, io


def test_the_document_carries_the_application_and_its_blind_spots_as_notes():
    app = {"name": "qdrant", "display": "Qdrant", "version": "1.20", "record": "1.19", "exact": True, "options": {"vectors_memory": "cold"},
           "blind_spots": [{"kind": "mmap", "note": "vectors are mapped", "source": "x"}]}
    d = answered(open_all(), [], application=app)
    assert d["application"] == app and d["notes"]["blind_spot_mmap"] == "vectors are mapped"
    text = digest.render(d)
    assert "application: Qdrant 1.20; record 1.19" in text and "  vectors_memory: cold" in text and "blind spot (mmap): vectors are mapped" in text


def test_the_interview_confirms_the_application_its_signatures_found_and_applies_its_record():
    # yes it is DCP; version 2.14.1; option 1 (one file per rank), option: don't know; then facts and the open items
    doc, answers, io = interview_with(dcp_load(), "1", "2.14.1", "1", "3", "", "", "3", "", "3", "")
    said = "\n".join(io.said)
    assert "This trace looks like PyTorch Distributed Checkpoint (a path matched __*_*.distcp)." in said
    assert "Does each rank write one shard file per checkpoint" in said and "(the default)" in said
    app = doc["application"]
    assert app["name"] == "pytorch-dcp" and app["version"] == "2.14.1" and app["record"] == "2.14" and app["exact"]
    assert app["options"] == {"single_file_per_rank": "one file per rank"} and app["blind_spots"] == []
    (rec,) = [a for a in doc["asserted"] if "record" in a["by"]]
    assert rec["question"] == "size" and rec["answer"] == "per-path" and rec["by"].startswith("PyTorch Distributed Checkpoint 2.14 record, read 2026-10-29 (high confidence)")
    assert rec["source"].startswith("workloads/checkpointing/dcp-load/kit/params.py") and rec["at"] == AT
    assert doc["sites"][0]["size"]["asserted"]["answer"] == "per-path" and doc["sites"][0]["size"]["verdict"] == "undecided"
    assert [it["question"] for it in doc["open"]] == ["selection", "order"]   # both answered don't know, so both stay, marked
    assert all(it["answered"] == "unknown" for it in doc["open"])
    assert answers[0]["by"].startswith("PyTorch Distributed Checkpoint 2.14 record")   # the record's assertion travels in the answers list
    assert "record" in answers[0]["by"] and "size" == answers[0]["question"]


def test_the_interview_lists_the_applications_when_the_guess_is_wrong_or_nothing_matches():
    names = sorted(APPS)
    n = names.index("pytorch-dcp") + 1
    # 2 = another application; the list; DCP at a version below every record: nothing assumed
    doc, answers, io = interview_with(dcp_load(), "2", str(n), "2.9", "", "", "3", "", "3", "", "3", "")
    said = "\n".join(io.said)
    assert "Which application produced this trace?" in said and f"{n}. PyTorch Distributed Checkpoint:" in said
    assert "no record of PyTorch Distributed Checkpoint at 2.9 or below; nothing is assumed" in said
    assert doc["application"] == {"name": "pytorch-dcp", "display": "PyTorch Distributed Checkpoint", "version": "2.9", "record": None, "exact": False, "options": {}, "blind_spots": []}
    assert not [a for a in doc["asserted"] if "record" in a["by"]]
    # nothing matches: the list, and none of these
    doc, answers, io = interview_with(open_all(), str(len(names) + 1), "", "", "3", "", "3", "", "3", "")
    said = "\n".join(io.said)
    assert "No path in the trace identifies the application." in said
    assert doc["application"] is None and answers == [] or all("question" in a for a in answers)
    # a wrong number is re-asked
    doc, answers, io = interview_with(open_all(), "99", str(len(names) + 1))
    assert "a number from 1 to 8" in "\n".join(io.said) and doc["application"] is None


def test_a_version_no_record_claims_makes_the_record_a_suggestion_to_confirm(tmp_path):
    shutil.copytree(ap.default_dir(), tmp_path / "apps")
    f = tmp_path / "apps" / "pytorch-dcp" / "2.14.toml"
    f.write_text(f.read_text().replace('from = "2.14"\n', 'from = "2.14"\nuntil = "2.15"\n'))
    apps = ap.load(tmp_path / "apps")
    # yes DCP; 2.15.2 is between records; options: 1, 3; the record's size per-path offered: 1 = yes
    doc, answers, io = interview_with(dcp_load(), "1", "2.15.2", "1", "3", "1", apps=apps)
    said = "\n".join(io.said)
    assert "no record claims PyTorch Distributed Checkpoint 2.15.2; the nearest is 2.14, read 2026-10-29: its answers are suggestions you confirm" in said
    assert "the PyTorch Distributed Checkpoint 2.14 record says, for ckpt/step_{a:06}/__{b}_{c}.distcp: size per-path" in said
    app = doc["application"]
    assert app["record"] == "2.14" and not app["exact"]
    (a,) = [a for a in doc["asserted"] if a.get("question") == "size"]
    assert a["by"] == BY and a["note"].startswith("confirmed from the PyTorch Distributed Checkpoint 2.14 record: a rank's shard")
    # 2 = no: nothing recorded
    doc, answers, io = interview_with(dcp_load(), "1", "2.15.2", "1", "3", "2", apps=apps)
    assert not [a for a in doc["asserted"] if a.get("question") == "size"]
    # end of input at the confirmation
    doc, answers, io = interview_with(dcp_load(), "1", "2.15.2", "1", "3", apps=apps)
    assert doc["application"]["record"] == "2.14" and not [a for a in doc["asserted"] if a.get("question") == "size"]
    # version unknown: the newest record, as suggestions
    doc, answers, io = interview_with(dcp_load(), "1", "", "1", "3", "3", apps=apps)
    assert "no record claims PyTorch Distributed Checkpoint of unknown version; the nearest is 2.14" in "\n".join(io.said)
    assert doc["application"]["version"] is None and doc["application"]["record"] == "2.14"


def test_a_choice_with_a_blind_spot_is_said_and_noted():
    milvus = strace_of([], per_open=[(f"files/insert_log/1/2/{i}/100/{i}", 4096) for i in (3, 4, 5)])
    # yes Milvus; 3.0.3; in-memory index; vector field memory-mapped; index loaded; eviction off; then the facts and items, all don't know
    doc, answers, io = interview_with(milvus, "1", "3.0.3", "1", "2", "1", "1", "", "", "3", "", "3", "", "3", "")
    said = "\n".join(io.said)
    assert "blind spot (object): segments and index files persist on the object store" in said
    assert "blind spot (mmap): the application maps its files" in said
    app = doc["application"]
    assert app["record"] == "3.0" and app["exact"] and [b["kind"] for b in app["blind_spots"]] == ["object", "mmap"]
    assert doc["notes"]["blind_spot_mmap"].startswith("the application maps") and "blind_spot_object" in doc["notes"]
    text = digest.render(doc)
    assert "application: Milvus 3.0.3; record 3.0" in text and "  mmap_vector_field: memory-mapped" in text
    doc["application"]["record"] = None
    assert "application: Milvus 3.0.3; no record" in digest.render(doc)


def test_the_identification_stops_at_end_of_input_wherever_it_is():
    for lines in [(), ("1",), ("1", "2.14.1"), ("2",), ("2", "1")]:
        doc, answers, io = interview_with(dcp_load(), *lines)
        assert doc["aeiou_digest"] == 1
    assert interview_with(dcp_load())[0]["application"] is None
    assert interview_with(dcp_load(), "1", "2.14.1")[0]["application"]["record"] == "2.14"


def test_a_stated_application_skips_the_identification_and_keeps_its_options():
    preset = {"name": "pytorch-dcp", "display": "PyTorch Distributed Checkpoint", "version": "2.14.1", "record": None, "exact": False, "options": {"sync_files": "fsync off"}, "blind_spots": []}
    doc, answers, io = interview_with(dcp_load(), "2", "", "", "3", "", "3", "", application=preset)
    said = "\n".join(io.said)
    assert "This trace looks like" not in said and "fsync each shard" not in said and "one shard file per checkpoint" in said
    assert doc["application"]["options"] == {"sync_files": "fsync off", "single_file_per_rank": "one file per item"} and doc["application"]["record"] == "2.14"
    preset = {"name": "nobody", "display": "Nobody", "version": None, "record": None, "exact": False, "options": {}, "blind_spots": []}
    doc, answers, io = interview_with(dcp_load(), application=preset)
    assert doc["application"] == preset   # unknown to the records: kept as stated, nothing assumed


def test_cli_states_the_application_and_replays_it(tmp_path, monkeypatch):
    tr = tmp_path / "t.txt"
    tr.write_text(dcp_load())
    r = run(str(tr), "--root", "/d", "--application", "pytorch-dcp@2.14.1")
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    app = doc["application"]
    assert app["record"] == "2.14" and app["exact"] and app["options"] == {"single_file_per_rank": "one file per rank", "sync_files": "fsync on"} and "app" not in app
    assert [a["question"] for a in doc["asserted"]] == ["size"] and "record" in doc["asserted"][0]["by"]
    r = run(str(tr), "--root", "/d", "--application", "pytorch-dcp@2.9")
    assert r.returncode == 0 and json.loads(r.stdout)["application"]["record"] is None
    r = run(str(tr), "--root", "/d", "--application", "nope")
    assert r.returncode == 2 and "no such application record" in r.stderr and "pytorch-dcp" in r.stderr
    r = run(str(tr), "--root", "/d", "--applications", str(tmp_path / "absent"))
    assert r.returncode == 1 and "no such directory of application records" in r.stderr
    # an interactive run with the application stated: the options are asked, the answers file carries the application, the replay restores it
    out = tmp_path / "t.digest.json"
    monkeypatch.setattr(sys, "stdin", Tty("2", "1", "", "", "3", "", "3", ""))
    assert trace.main(["digest", str(tr), "--root", "/d", "-o", str(out), "--application", "pytorch-dcp@2.14.1"]) == 0
    doc = json.loads(out.read_text())
    assert doc["application"]["options"] == {"single_file_per_rank": "one file per item", "sync_files": "fsync on"}
    header, answers = digest.read_answers(str(tmp_path / "t.answers.json"))
    assert header["application"] == doc["application"]
    monkeypatch.setattr(sys, "stdin", Tty())
    assert trace.main(["digest", str(tr), "--root", "/d", "-o", str(tmp_path / "again.json"), "--answers", str(tmp_path / "t.answers.json")]) == 0
    again = json.loads((tmp_path / "again.json").read_text())
    assert again["application"] == doc["application"] and again["asserted"] == doc["asserted"]
