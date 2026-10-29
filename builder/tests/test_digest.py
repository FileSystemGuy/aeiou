"""`aeiou-trace digest` (`PROJECT_BRIEF.md` item 47, `DESIGN_REVIEW.md` §3.86, §3.87): the
structural summary a fit is made from, with a verdict per site on what the trace decides and an
`open` list, ranked by projected divergence, of what it does not."""
import json
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
    return subprocess.run([sys.executable, "-m", "aeiou.trace", "digest", *args], capture_output=True, text=True, cwd=BUILDER, input=stdin)


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
    assert r.returncode == 0 and json.loads(r.stdout)["trace"] == "-"


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
