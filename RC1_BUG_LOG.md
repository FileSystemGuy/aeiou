# Release-candidate 1 review: bug log

Append-only. One entry per bug, written only once its fix is verified (a test that fails
without the fix and passes with it, or the command output before and after), and committed and
pushed together with that fix.
The review started at tag `pre-rc1-review` (644006e, 2026-10-14); `aeiou-datagen`, the
deprecated Python writer, is out of scope. Each entry: where, how it was found, what was wrong,
the fix, the evidence, and the commit, so the work can be replayed from the tag.

## 1. The description's percentile range printed inverted under a clamp

- **Where:** `builder/aeiou/describe.py`, `Describer._dist` and `_clamp` (the prose of a
  `normal` or `lognormal` with `min`/`max`).
- **Found by:** review of the Python tools (aeiou-catalog describe), checking the percentile
  prose against the runner's sampling in `runner/aeiou/src/eval.rs` (`clamp`: raise to `min`,
  then lower to `max`, every draw).
- **Defect:** the 5th percentile was clamped only by `min` and the 95th only by `max`, so a
  bound past both percentiles inverted the range: `normal(1000, 10, max=500)` printed "5th–95th
  percentile 984 bytes to 500 bytes"; `lognormal(100, 1, min=1000)` printed "1,000 bytes to 518
  bytes". The runner draws 500 (resp. 1,000) every time.
- **Fix:** `_clamp` applies both bounds to each percentile in the runner's order; when the two
  rendered percentiles coincide the prose says the value once ("5th and 95th percentiles both
  500 bytes"). Prose only: no AST, fingerprint, or committed DESCRIPTION.md changed
  (`aeiou-catalog check --no-shapes`: ok).
- **Evidence:** `builder/tests/test_describe.py::test_a_clamp_past_both_percentiles_moves_both`
  fails on the old describe.py and passes on the new; test_catalog.py + test_describe.py: 49
  passed.

## 2. `aeiou datagen` asked for free space before refusing a corpus it would refuse anyway

- **Where:** `runner/aeiou/src/datagen.rs`, `datagen()`: the order of the space check
  (`check_space`, DESIGN_REVIEW §3.74 item 4) and the structural refusals.
- **Found by:** the baseline coverage run (`cargo llvm-cov --features formats,object`): the
  test `object::object_names_need_object_endpoints` failed on this host ("space: dataset(s)
  `pq`: 29.80 GiB to write, 20.68 GiB free: the corpus does not fit") where it expected the
  `array`-in-an-object-store refusal. The test passes or fails with the host's free disk.
- **Defect:** the space check, which asks the file system, ran before the refusals the
  abstract, parameters, and endpoints decide alone (a resumed corpus's format writer, an
  `array` dataset in an object store, V18), so a datagen that would be refused anyway told the
  user to free tens of GiB first.
- **Fix:** the per-dataset loop is split: a first pass resolves each dataset's writer, root, and
  endpoint and makes the structural refusals; then rank 0's space check; then the plan (empty
  roots, directories, this rank's share). Nothing is created before either check, as before.
  `aeiou run` was checked for the same ordering defect: its storage-dependent checks already
  come after the decidable ones. runner/REFERENCE.md §5 *Space* updated.
- **Evidence:** the test now also runs the refused datagen with the posix dataset grown to 888
  PiB (`--param pq_code=1000000000`), which fits on no host: without the fix "space: … 888.18
  PiB to write … does not fit", with it the `array` refusal and no `space:` line. `cargo test
  --features formats --test all`: 75 passed; `--features formats,object` with MinIO: 79 passed.

## 3. `aeiou-build` silently dropped one of two Workloads of the same name

- **Where:** `builder/aeiou/cli.py`, `_run_script` and `main` (the writer loop), and
  `_child_main` (the `--hermetic` child).
- **Found by:** review of the builder core: a script constructing two `Workload("dup")` of
  different contents (sizes 4096 and 8192).
- **Defect:** `aeiou-build dup.py` wrote one `dup.ast.json`, exit 0, holding only the second
  workload; the first was lost without a word. Two scripts of one workload name built into one
  `-o` directory overwrote each other the same way. An AST is named for its file (CLAUDE.md), so
  one name can only be one abstract.
- **Fix:** `_run_script` refuses a second Workload of a name already constructed by the script
  (a BuildError naming it; the hermetic child now prints a BuildError as its last stderr line
  instead of a traceback, so the parent's "child build failed" ends with the reason); `main`
  refuses a script whose output path another script of the same run already wrote (the same
  script named twice is allowed). Exit 1, as other build errors. No AST or fingerprint changed;
  the CI build-twice of every committed script still passes (14 ok).
- **Evidence:** `test_builder.py::test_build_refuses_two_workloads_of_one_name` (one script,
  plain and `--hermetic`; two scripts into one `-o`; scripts in their own directories) fails
  without the cli.py change and passes with it; test_builder.py: 45 passed.
- **Noted, not changed:** the hermetic failure message names the script twice ("FAIL dup.py:
  dup.py: child build failed"), a pre-existing cosmetic issue (queued separately).

## 4. The documented strace command recorded no libaio I/O, and nothing noticed

- **Where:** the capture command in `builder/aeiou/trace.py` (docstring and `metrics --help`),
  `builder/REFERENCE.md` §7, `man/aeiou-trace.1.md`, `runner/REFERENCE.md` §13, `ABSTRACTS.md`,
  and eight workloads' READMEs; `Tracer.call`/`finish` in trace.py.
- **Found by:** review of aeiou-trace: a C program submitting 40 PREADs through raw
  `io_submit`, traced with the documented `-e trace=%file,%desc,%process`, showed no `io_submit`
  line (strace does not class the AIO calls as `%desc`); `aeiou-trace metrics` gave counts
  `{open: 1}`, no fan-out, no note. The diskann and lmcache-local-disk kits escaped it only
  because their READMEs list the calls themselves.
- **Defect:** every libaio read and write of an application traced as documented was lost
  silently: op mix, bytes, fan-out, and depth wrong, the trace judged against an abstract on
  what is left.
- **Fix:** the command names `io_setup,io_submit,io_getevents,io_pgetevents,io_destroy`
  everywhere it is written; `metrics` and `export` note `aio_contexts_without_io_submit` when a
  traced `io_setup`'s context is never named by an `io_submit` (a note, not a refusal: an unused
  context is a valid trace; it catches a hand-written `-e` list missing the submits). No
  fingerprint, dataset id, or contract change.
- **Evidence:** the 40-PREAD program traced with the old command: counts `{open: 1}`; with the
  new: `{open: 1, read: 32}`, fan-out `{32: 1}` (32, not 40, is strace's array truncation,
  queued separately). `test_trace.py::test_an_aio_context_no_io_submit_names_is_noted` fails on
  the old trace.py and passes on the new; test_trace.py + test_man.py: 77 passed.

## 5. `aeiou-params check` passed parameter files the runner refuses

- **Where:** `builder/aeiou/params.py`, `check()` (new `given_rules`).
- **Found by:** review of the builder core, comparing `aeiou-params check` with what `aeiou run
  --params-file` does to the same file (`validate::check_given`, `eval::bind_pattern`); man
  aeiou-params(1) promises "the runner's rules", and CI runs it on every committed set.
- **Defect:** `check` compared only each value's kind with its default's. It passed
  `{"files_per_directory": 2.5}` for file_per_sample (the runner: "divisor `files_per_directory`:
  parameter … is not an integer", V23) and a set making lmcache_local_disk's reuse arm `const 0`
  (the runner: 13 "`kind @` names a binding of this loop body, so its index must be … provably
  >= 1" errors, V3).
- **Fix:** after the kind checks, `given_rules` re-judges V3 with the given values as the
  runner does (the schema checker reused through `validate._load()`, `at_least` of a given
  parameter judged from its value) and binds every non-array dataset's and every namespace's
  pattern to the defaults overlaid with the given values (V23: a divisor must be a positive
  integer), with the runner's wording. The `safetensors` and `npz` subcommands go through the
  same `check`. Every committed parameter file still passes.
- **Evidence:** both reproductions now FAIL, exit 1 (the reuse case's 13 errors identical to the
  runner's); `test_params.py::test_given_values_are_judged_as_the_runner_judges_them` (2.5, True,
  0, 7 as divisors; the reuse arm) fails without the `given_rules` call and passes with it;
  test_params.py + test_usage.py: 34 passed.

## 6. aeiou-trace kept strace's escapes in `-yy` descriptor paths

- **Where:** `builder/aeiou/trace.py`, `fd_arg` (every descriptor argument, the `ret_path` of an
  open, a dirfd base, `fchdir`) and the `aio_fildes` annotations of `io_submit`.
- **Found by:** review of aeiou-trace, tracing a stat then an open of `café` and creates of
  names holding `<`, `>`, `\`, a space, and a comma.
- **Defect:** strace prints a `-yy` path as a quoted string without its quotes (octal for
  non-ASCII bytes, `\74`/`\76` for `<`/`>`, a doubled backslash); path *arguments* were
  unescaped (`c_string`) but descriptor annotations were not, so one file had two names (`stat
  café`, `open caf\303\251`): popularity, creates, and path order split, an exported trace named
  files the runner would not find, and a `--root` needing escapes matched nothing.
- **Fix:** the decoder inside `c_string` becomes `unescape`; a new `fd_path` strips the
  `<char 1:3>` device suffix and ` (deleted)` from the raw annotation and then unescapes it;
  `fd_arg` and both `io_submit` loops use it. The annotation's end detection was already right:
  strace escapes every `<`/`>` inside a path. No fingerprint or contract change.
- **Evidence:** `test_trace.py::test_a_descriptors_escaped_path_is_unescaped` and
  `test_metrics_and_export_give_an_escaped_file_one_name` (root `/r/été`; café, `c<d>e`, `x\y`,
  `a b`, `n,m`) fail on the old trace.py (`'/r/caf\\303\\251'`; counts `{}`) and pass on the
  new; test_trace.py: 61 passed.

## 7. A `trace` node's paths could reach outside `--root`, and `--clean-namespaces` deleted there

- **Where:** `runner/aeiou/src/trace.rs`, `TraceFile::load` (new `check_path`/`check_paths`).
- **Found by:** review of the runner's trace node against runner/REFERENCE.md §13 ("removes
  nothing else"; a trace's root is never emptied): op paths and the header's `creates` were
  joined to the root with `root.join(p)` and nothing refused `..` or an absolute path.
- **Defect:** a trace with `creates: ["../outside/victim", "../outside/victimdir"]` and an
  `open WRONLY|CREAT|TRUNC` of the first passed `aeiou check`, and `aeiou run --clean-namespaces`
  removed the directory outside the root (`remove_dir_all`) and truncated the file ("precious" →
  0 bytes). A safety bug: a hand-edited or buggy trace could destroy data outside the storage
  under test.
- **Fix:** at load (so `check`, `dry-run`, and `run` all refuse, before the gate), every
  `creates` entry and every op path (open, stat, unlink, mkdir, rmdir, both sides of rename,
  `submit` members) is refused if absolute, if it has a `..` component, or if empty, naming the
  file line; the root itself (`.`) is allowed only to read (an open without CREAT/TRUNC, a stat),
  never to create, truncate, or remove, so `creates: ["."]` cannot empty the root. runner
  REFERENCE §13 *Loading* updated. No fingerprint or contract change; no committed trace file
  exists.
- **Evidence:** before: `check` ok, `run` exit 0 and the files outside destroyed; after: `FAIL …
  trace t.jsonl: line 1: path ../outside/victim has a .. component` from all three commands,
  files intact. `tests/trace.rs::a_path_outside_root_is_refused_at_load_and_nothing_outside_is_touched`
  fails with the checks disabled and passes with them; `--test all`: 76 passed.
- **Queued from it:** the Python exporter's `rel()` fallback returns a path not under the root
  unchanged (the runner now refuses such a file at load).

## 8. The hermetic build let `clock_gettime` through, and `--twice` blamed the hash seed

- **Where:** `builder/aeiou/hermetic.py` (the clock stubs; the audit hook split out as
  `_audit_hook`), `builder/aeiou/cli.py` (`--twice` message, `_spawn_child`).
- **Found by:** review of the builder core: an authoring script whose default came from
  `time.clock_gettime_ns(CLOCK_REALTIME)` built under `--hermetic`; `hermetic.py` had 0% test
  coverage (the build-twice runs only as a CI step).
- **Defect:** `time.clock_gettime`/`clock_gettime_ns`, `os.times`, and `resource.getrusage`
  were not denied, so a clock could be baked into an AST under `--hermetic`; `--twice` then
  reported "not reproducible … under different PYTHONHASHSEED values (set iteration order…)",
  pointing the author at the wrong cause (the same for a pid-derived default); and a failed
  hermetic child's message named the script twice ("FAIL dup.py: dup.py: child build failed").
- **Fix:** the four clocks are denied like the others (`os.getpid` stays readable: the HDF5
  probe names a file with it; `--twice` catches a pid default); the `--twice` message names the
  real candidates (the pid, a clock or entropy read, set order, id() order, an unseeded draw);
  `_spawn_child` no longer prefixes the script name `main` already gives. builder REFERENCE
  (hermetic harness) and aeiou-build(1) `--twice` updated. Every committed abstract rebuilds
  byte-identical (14 ok).
- **Evidence:** new `builder/tests/test_hermetic.py` (32 tests: every clock denied, entropy and
  unseeded draws denied, seeded ones allowed, the audit hook's allow/deny lists, env scrub, the
  clock default refused by name, the pid default caught by `--twice` with the pid named, the
  script named once): 10 fail without the fix, 32 pass with it; hermetic.py coverage 0% → 89%.
- **Left open (noted):** an unseeded `_random.Random()` through the C type, and a default from
  a file's `st_mtime` (stable across both builds), are not caught.

## 9. aeiou-trace missed CLONE_FILES on a legacy `clone()`, so threads lost their shared descriptors

- **Where:** `builder/aeiou/trace.py`, `Tracer.call` (the CLONES branch) and `_adopt`; new
  `shares_files`.
- **Found by:** review of aeiou-trace: two traces identical but for `clone(` against `clone3(`
  (a thread opens and reads, main closes and reopens the path, the thread reads again) gave
  different exports and metrics. Every clone in the tests was a fork, so it was untested.
- **Defect:** the branch looked for CLONE_FILES in the first argument, where clone3's struct
  holds it but legacy clone's is `child_stack=`; its flags are the second argument. Under glibc
  older than 2.34 (RHEL 8, Ubuntu 20.04) every pthread is a legacy clone, so every thread was
  given a copy of the descriptor table: the close was missing from the export and the thread's
  second read went to the stale, closed open at a sequential 4096 (run lengths, first touches,
  and rereads wrong). `_adopt` (a child calling before its clone returns) searched the whole
  line instead, which could match anywhere.
- **Fix:** `shares_files(name, args)` looks for CLONE_FILES only inside a `flags=` value
  (clone's argument, clone3's field), whole word; both sites use it. No other clone flag is
  read. No fingerprint or contract change.
- **Evidence:** `test_trace.py::test_a_legacy_clone_shares_the_descriptor_table_as_clone3_does[False]`
  (glibc 2.31-style lines against clone3; metrics, header, events identical; a fork control)
  fails without the fix and passes with it; `test_shares_files_reads_the_flags_argument_only`;
  test_trace.py: 64 passed.

## 10. aeiou-trace recursed without end on a task's own unfinished clone

- **Where:** `builder/aeiou/trace.py`, `Tracer._adopt`.
- **Found by:** the fixer of bug 9, exercising `_adopt` with an unfinished clone as the parent's
  first traced call and the child's first call before `<... resumed>`.
- **Defect:** `table(parent)` called `_adopt(parent)`, which picked the parent's *own* pending
  clone as the one it was born of, then `_child(parent, parent)`, and round again: `aeiou-trace`
  died with RecursionError on such a trace.
- **Fix:** the waiting-clone filter excludes the task itself (`p != tid`: a task is never its
  own child). Same commit as bug 9.
- **Evidence:** `test_a_legacy_clone_shares_the_descriptor_table_as_clone3_does[True]` raises
  RecursionError with the guard removed and passes with it.

## 11. The builder's tools read and wrote files in the locale's encoding

- **Where:** `builder/aeiou/emit.py` (`load`, `write`, `git_describe`), `cli.py` (the AST write,
  the hermetic child's pipe), `params.py` (every read and write), `validate.py`, and
  `schema/check.py` (schema and AST reads); new `usage.console()`.
- **Found by:** review of the builder core (suspected: `read_text`/`write_text` with no
  encoding, while ASTs and docs hold "×", "−", "—", "§"); confirmed by the fixer under
  `LC_ALL=C PYTHONUTF8=0`, where Python's preferred encoding is ASCII.
- **Defect:** `aeiou-build lmcache_local_disk.py` under that locale died with
  `UnicodeEncodeError: 'ascii' codec can't encode character '\xa7'` and left a 0-byte AST;
  `--check` of a committed (UTF-8) AST failed to read it; with the file write fixed, the
  console line "wrote … sha256=…" itself raised on its `…`.
- **Fix:** every text read and write of repo files names `encoding="utf-8"` (git and child
  pipes with `errors="replace"`); `usage.console()` sets stdout/stderr to
  `errors="backslashreplace"`, called first in `aeiou-build` and `aeiou-params`. Bytes written
  under a UTF-8 locale are unchanged: no AST, fingerprint, or contract change.
- **Evidence:** `builder/tests/test_encoding.py::test_build_and_params_under_an_ascii_locale`
  (build, `--check`, `aeiou-params defaults`/`check` under C.UTF-8 and under ASCII; outputs
  byte-identical) fails without the fix (`UnicodeEncodeError … '\xd7'`) and passes with it;
  builder suite 300 passed; build-twice of the committed scripts 14 ok.
- **Queued from it:** the same gaps in `catalog.py` (CATALOG.md and DESCRIPTION.md writes) and
  in `trace.py` (stdin, console).

## 12. aeiou-trace counted an O_APPEND write from 0 on a file the trace itself made

- **Where:** `builder/aeiou/trace.py`, `Tracer` (open/write/truncate/rename/unlink, new
  `extent` and `_known_size`) and `Exporter.finish`; builder REFERENCE §7 (*Offsets*, `export`).
- **Found by:** review of aeiou-trace: four `O_CREAT|O_APPEND` opens of one file (`O_TRUNC` on
  the first), 4 KiB written each, a 16 KiB file.
- **Defect:** an append's base was "known" only when that same open truncated or exclusively
  created the file, so every later append to a file the trace had made started at 0: metrics
  noted `append_to_unknown_size: 3`, one first touch, three writes-after-write, one block (4
  and 4 in truth), and the export left the writes sequential at the runner's own position.
  REFERENCE §7 promised the unknown base only for a file *not* created in the trace.
- **Fix:** the tracer keeps each file's size while the trace knows it (from a truncating or
  exclusive create; `ftruncate`, `truncate`, `fallocate`, extending writes; moved by `rename`,
  dropped by `unlink`); an append on such a file is at its size (a `pwrite` offset ignored, as
  Linux does). The export writes it positioned at the same offset (header note
  `append_positions_resolved`), so the runner walks the same offsets `metrics` counts. The
  note now counts only appends under the roots whose base is unknown.
- **Evidence:** `test_trace.py::test_an_append_to_a_file_the_trace_made_starts_at_its_end`
  (no note, 4 first touches, 4 blocks, export offsets 0/4096/8192/12288, export metrics =
  strace metrics) and `test_an_append_whose_base_the_trace_cannot_know_is_noted` fail on the
  old trace.py and pass on the new; test_trace.py: 66 passed.

## 13. The builder accepted literal distribution arguments the runner can never draw from

- **Where:** `builder/aeiou/dists.py` (`uniform`, `normal`, `lognormal`, `empirical`, `zipf`,
  `mixture`; helpers `_literal`, `_weight`, `_bounds`) and `builder/aeiou/builder.py`
  (`choose`, `Chooser.arm`, `_divisors`).
- **Found by:** review of the builder core: generated abstracts with each argument that only
  `aeiou dry-run` refused (rules read from runner `eval.rs` `sample()` and `vm.rs`).
- **Defect:** `uniform(4096,4096)` and `uniform(8192,4096)` ("empty range"), all-zero weights
  in `empirical`/`mixture`/`choose` ("total weight is zero"), `lognormal(-5, 1)` ("negative
  length"), `normal(4096, -3)` (ran silently), `min` above `max`, and a divisor parameter
  defaulting to 0 all built and passed `aeiou check`, failing only when run; a NaN or inf weight
  failed late with a bare ValueError from json.dumps.
- **Fix:** each literal argument is refused at build with a BuildError naming the distribution
  and argument: uniform `lo >= hi` (draws are `[lo, hi)`), negative `sd`, non-positive lognormal
  median or sigma, literal `min > max`, negative or non-finite weights and zero totals (empirical,
  mixture, choose), non-finite zipf `s`, a divisor default `<= 0` (V23). Parameter references
  stay the runner's to judge. Every committed abstract builds to the same hash (14 ok).
- **Evidence:** `test_builder.py::test_literal_distribution_arguments_the_runner_cannot_draw_from_are_refused`
  and `test_choose_weights_and_divisor_defaults_the_runner_cannot_use_are_refused` ("DID NOT
  RAISE" without the fix, pass with it; edges that must still build included); builder suite
  304 passed.

## 14. `rng.rust_round` was not Rust's `round() as i64`

- **Where:** `builder/aeiou/rng.py`, `rust_round` (used by `sample` for normal/lognormal draws
  and `sample_size`, which tests/test_formats.py holds against the runner's sizes).
- **Found by:** review of the builder core, against runner `eval.rs` (`clamp(x, min,
  max).round() as i64`).
- **Defect:** `floor(x + 0.5)` loses precision: 0.49999999999999994 → 1 (Rust 0),
  4503599627370497.0 → …498 (Rust …497); no saturation (1e30 → a 31-digit int; Rust i64::MAX);
  NaN raised ValueError (Rust 0). Python's size draws could then disagree with the runner's.
- **Fix:** round half away from zero on the magnitude with an exact subtraction, restore the
  sign, saturate to [i64::MIN, i64::MAX], NaN → 0, ±inf → the bounds.
- **Evidence:** `test_formats.py::test_rust_round_is_rusts_round_then_saturating_cast` (20
  values taken from a compiled Rust program's output) fails on the old function and passes on
  the new; test_formats.py + test_payload.py: 43 passed.

## 15. aeiou-trace took strace's 32-element arrays as whole arrays

- **Where:** `builder/aeiou/trace.py` (`io_submit`, `io_getevents`, readv/writev/preadv/pwritev,
  `metrics_of`; new `truncated`), and the capture command (now `strace … -v …`) in trace.py,
  builder REFERENCE §7, aeiou-trace(1), runner REFERENCE §13, ABSTRACTS, nine workload READMEs.
- **Found by:** review of aeiou-trace: a 40-request `io_submit` traced showed `= 40` with an
  array ending `...]` after 32 elements; `kit/hops.py` of diskann-search already knew "strace
  prints 32 elements", aeiou-trace did not. The fixer found `io_getevents` cut the same way.
- **Defect:** strace prints at most 32 array elements unless `-v` (`-s N` also limits it), so
  a 40-request submit measured as fan-out 32 and 32 reads, 8 requests and their bytes gone
  without a note; cut `io_getevents` results left requests "assumed complete"; a preadv with
  more than 32 iovecs had its request size summed from the 32 shown (2,048 for a 6,400-byte
  call).
- **Fix:** the capture command adds `-v` (whole arrays; a `-v` trace parses unchanged); a cut
  array is detected (`...]`) and noted: `aio_requests_truncated` and `aio_events_truncated`
  count what is missing (the unseen requests are not invented: their file, offset, and size are
  not in the trace), `iov_arrays_truncated` counts cut iovec lists, whose request size becomes
  the larger of the shown sum and the bytes the call returned, and `metrics` adds the hint
  "trace again with strace -v". No fingerprint or contract change.
- **Evidence:** `test_trace.py::test_arrays_strace_cut_at_32_elements_are_noted` and
  `test_a_cut_iovec_list_takes_the_bytes_the_call_moved` (40 elements, complete and cut at 32)
  fail on the old trace.py and pass on the new; with `-v` the real 40-PREAD program gives
  fan-out {40: 1}; test_trace.py + test_man.py: 86 passed.
- **Left open (queued):** diskann-search's kit `hops.py` relies on the cut array to tell the
  node-cache bulk load from query hops, so its README keeps the old command; its committed
  sift1m-cache capture shows fan-out `"32": 1`, probably that cut bulk load (a recapture needs
  the application).

## 16. The coordinator declared a live host dead after 30 s when traffic flowed one way

- **Where:** `runner/aeiou/src/coord.rs` (`Liveness`, `read_full`, `client_loop`,
  `serve_loop`, every writer), `runner/aeiou/src/options.rs` (`RESERVED_ENV`); runner
  REFERENCE §6 and §14.
- **Found by:** review of the coordinator, then a 3-rank run of 4 GPUs (rank 2 owns `[4, 4)`:
  it receives a Release every step and sends nothing): every rank exited 1 after ~30 s with
  "rank 2 …: no message for 30s".
- **Defect:** a heartbeat was sent only when a side's *read* timed out (nothing received for
  5 s), so a side that kept receiving never sent one and its peer killed it at `DEAD_AFTER`.
  It hit any run where G < R or G is not a multiple of R, a host waiting for its Result while
  others release barriers for over 30 s, and datagen clients sending Progress more often than
  every 5 s; the 5 s heartbeat and 5 s read timeout also raced.
- **Fix:** each connection's writer records its last send (`Wire { s, sent }`, updated by
  every frame sent); both reader loops beat whenever `HEARTBEAT` has passed since the last
  send, after every message received and on every read timeout; the read tick is 1 s, under the
  heartbeat. Still blocking `std::net` on dedicated threads. A test-only hook,
  `AEIOU_COORD_HEARTBEAT_MS` (heartbeat N ms, dead after 6N), reserved in options.rs and named
  in REFERENCE §6 and §14. No fingerprint or contract change.
- **Evidence:** the original repro now runs its full 42 s and every rank prints the same
  fingerprint; `tests/coord.rs::a_host_that_only_receives_still_heartbeats` (3 ranks, an empty
  host, a 100 ms heartbeat over a 3 s run) fails in 1.5 s with "no message for 600ms" without
  the fix and passes with it; `--test all`: 77 passed. Fixed in an isolated worktree and applied.

## 17. `aeiou-catalog check --runner` never checked the counts the descriptions print

- **Where:** `builder/aeiou/catalog.py`, `recheck` and `stale_shapes` (new `COUNTED`,
  `_counted`, `_short`, `_case_run`); builder REFERENCE §8, aeiou-catalog(1), the CI step name.
- **Found by:** review of aeiou-catalog: in a copy of `workloads/`, 12,345 added to a shape's
  `counts.close` and 7 to its dataset's `directories`, `write --no-shapes`, then `check
  --runner`: "13 mechanisms, 16 cases: ok" while the description printed "18,748 close" (6,403
  real) and "10 directories". Commit 644006e had cited that check as evidence no projection
  changed.
- **Defect:** the re-measurement compared only the measured attributes, never the shape's
  `ops`, `counts`, `bytes_read`, `bytes_written`, and `datasets` projection, which DESCRIPTION.md
  and CATALOG.md print, so a runner change to counting or the projection left published numbers
  wrong with `check` passing.
- **Fix:** `recheck` also compares those five fields exactly with the seed-1 dry run at the
  case's own length (the run they were recorded from): free for a case checked at its full
  length; one extra dry run for a case whose check length is shorter (only kvcache agentx:
  434M ops), started first in its own pool to overlap the rest. Check with the runner takes 5m15s
  locally instead of ~40s, the price of verifying the printed numbers exactly (the alternative,
  counts recorded at the check length, would verify a runner regression but not the printed
  numbers). No shape format or fingerprint change.
- **Evidence:** the doctored copy now fails ("the dry run no longer gives the recorded counts,
  datasets"); `test_catalog.py::test_a_shapes_counts_are_checked_with_the_runner` and
  `test_counts_are_the_cases_length_when_the_check_length_is_shorter` fail on the old catalog.py
  and pass on the new; the committed tree checks ok; test_catalog.py + test_man.py: 68 passed.

## 18. aeiou-trace answered bad input with tracebacks, a late exit 1, and locale-dependent bytes

- **Where:** `builder/aeiou/trace.py`, `main` (new `_open_trace`, the usage check of
  `--max-distance`), the `-o` writes of `metrics` and `export`, and `_judged`.
- **Found by:** review of the Python tools against runner REFERENCE §15 (the usage-error frame
  every tool shares) and the encoding fix of bug 11.
- **Defect:** a missing trace (`metrics nofile`, `export nofile`) or an `-o` in a missing
  directory printed a Python traceback; `compare --judge --max-distance 0.1`, two rules that
  exclude each other, exited 1 after starting work instead of a usage error (exit 2); under an
  ASCII locale `export -` decoded stdin in the locale (é exported as two lone surrogates, a
  different sha256) and a non-ASCII `-o` path crashed the summary line after the work was done.
- **Fix:** inputs open through `_open_trace` (stdin read as UTF-8, an unopenable file a one-line
  error naming it, exit 1), `-o` write failures likewise; `--max-distance` with `--judge`,
  `--tolerances`, or `--self` is a `UsageError` before anything is read (the frame, exit 2);
  `main` calls `usage.console()`. `-o -` still names a file `-`: nothing documents it as stdout.
- **Evidence:** `test_trace.py::test_a_file_the_tool_cannot_open_is_an_error_naming_it`,
  `test_max_distance_with_a_judging_rule_is_a_usage_error`, and
  `test_stdin_is_utf8_and_the_console_escapes_what_it_cannot_spell` fail on the old trace.py
  (traceback; exit 1; byte mismatch) and pass on the new; trace, usage, man, encoding suites: 97
  passed.

## 19. `dry-run` and `run` disagreed on file positions across a `parallel` or `loader` fork

- **Where:** `runner/aeiou/src/vm.rs` (`inline_fork`, new `Cont::Fork`, its handling in
  `step`); new validity rule V24 in `runner/aeiou/src/validate.rs` and `schema/check.py`;
  schema/README §4 (V24, noted on the 0.10 entry), runner REFERENCE §4 *Threads*.
- **Found by:** review of the runner's semantic core: an AST opening a 4 KiB file, a
  `parallel` of width 4 reading 1 KiB at the file position, then a parent read.
- **Defect:** the dry run walked the sub-actors inline in the parent's VM, so open-file
  positions, as-written sums, and the open map carried from sibling to sibling and back to the
  parent (reads at 0/1024/2048/3072, the parent at 4096 → 0; fingerprint 466901655c00fe8b),
  while `run`, the io_uring loop, and metrics give every sub-actor a snapshot and never merge
  back (REFERENCE §4 *Threads*): `run --expect-fingerprint` failed ("off=0 len=1024 expected
  1024 bytes, got 0"). An AST both checkers accepted had a dry-run fingerprint that was not
  the stream run issues. Even with matching snapshots, sub-actors share the parent's descriptor
  and so the kernel's position: under `sync` which sibling reads first decides the offsets.
- **Fix:** `inline_fork` starts each sub-actor from a copy of the parent's state at the fork
  and gives the parent its own back after the last, as `snapshot`/`resume` do; V24 refuses, in
  a `parallel` or `loader` body, a read or write without `offset`, or an `lseek`, on a handle
  the innermost such body did not open (the offsets would depend on timing). Both checkers give
  the same message and path. No committed AST does this: every fingerprint and dataset id is
  unchanged (golden tests; `aeiou-catalog check --runner` ok); the contract stays 0.10.
- **Evidence:** `tests/run.rs::a_fork_leaves_the_parents_file_positions_as_run_does` (dry-run
  fingerprint, ops, and bytes equal run's under sync and io_uring; the parent's last read at
  2048; the shared-descriptor variant refused) fails on the old vm.rs and passes on the new;
  `test_builder.py::test_a_sub_actor_positions_its_ops_on_an_inherited_handle` holds check.py's
  V24; `--test all` 78 passed, `--lib` 40, MinIO object tests 4, builder 314 passed.

## 20. `aeiou-catalog describe` answered bad arguments with exit 1 and tracebacks

- **Where:** `builder/aeiou/catalog.py`, `main` and `_describe`; aeiou-catalog(1) EXIT STATUS.
- **Found by:** review of the Python tools against runner REFERENCE §15 (`aeiou dry-run --gpus
  0` is the usage frame, exit 2); the fixer then swept every subcommand's arguments.
- **Defect:** `describe --gpus 0` exited 1 outside the frame; a JSON file that is not an
  abstract (`[]`, `{}`) or a `--params-file` that is not a parameter file raised tracebacks
  (AttributeError, KeyError, TypeError) inside describe.py; another abstract's parameter file
  was not checked at all.
- **Fix:** `--gpus` below 1 is a UsageError before any read (frame, exit 2); the abstract loads
  through `emit.load` (missing or not JSON: usage error, exit 2, as the runner and aeiou-params)
  and is validated (`AST rejected: …`, exit 1); `--params-file` goes through `params.load` and
  `params.check` against the abstract (`parameter file rejected: …`, exit 1). `check` and
  `write` already behaved.
- **Evidence:** `test_catalog.py::test_describe_refuses_a_bad_option_value_in_the_frame` fails
  without the fix (exit 1 for 2) and passes with it; catalog, man, encoding suites: 71 passed.
- **Queued from it:** `aeiou-params defaults` on a JSON file that is not an abstract tracebacks
  the same way (`params._load_ast`).

## 21. `aeiou-catalog` read and wrote files in the locale's encoding

- **Where:** `builder/aeiou/catalog.py`: every text read and write, the dry-run subprocess,
  `main`, `describe`'s output.
- **Found by:** the encoding fixer of bug 11, listing what it left in catalog.py.
- **Defect:** under `LC_ALL=C PYTHONUTF8=0` (ASCII), `aeiou-catalog check --no-shapes` crashed
  at `load_dictionary` with `UnicodeDecodeError … 0xc2`, as did `describe`; `write` would have
  written CATALOG.md and DESCRIPTION.md (which hold "±", "§", "—") in the locale's encoding.
- **Fix:** `encoding="utf-8"` on every read and write; generated files compared as bytes (a
  hand-edited non-UTF-8 file is drift, not a traceback); the dry run's pipe decoded as UTF-8
  with `errors="replace"`; `usage.console()` first in `main`; `describe` writes its Markdown to
  stdout as UTF-8 bytes, so its output is the same in every locale.
- **Evidence:** `test_catalog.py::test_check_write_and_describe_under_an_ascii_locale` (check of
  the real tree, and write/check/describe of a fixture tree under C.UTF-8 and ASCII,
  byte-identical) fails without the fix and passes with it.

## 22. `aeiou-params` answered bad inputs and options with tracebacks

- **Where:** `builder/aeiou/params.py`: `_load_ast`, `read_safetensors_header`, `npz_framing`,
  every `-o` write (new `_write`), and the `--tp`, `--column`, `--row` argument types.
- **Found by:** the fixer of bug 20 (`aeiou-params defaults` on JSON that is not an abstract),
  then a sweep of all four subcommands for the same class.
- **Defect:** an AST file holding `[]` or `{}` raised AttributeError/KeyError in every
  subcommand; `safetensors` trusted a shard's 8-byte header length (MemoryError on junk), took
  any JSON header as a tensor table (TypeError), and raised on a missing shard; `npz` raised on a
  missing archive; an `-o` into a missing directory raised FileNotFoundError; `--tp 0` was
  accepted (the abstract computes `gpu_id % tp`); a bad `--column`/`--row` regex raised
  `re.error`.
- **Fix:** the AST is validated after loading ("AST rejected: …", exit 1, as aeiou-catalog);
  the safetensors header is checked against the file size and each entry for dtype, shape, and
  data_offsets; missing inputs and unwritable outputs are one-line errors (exit 1); `--tp` must
  be a positive integer and `--column`/`--row` valid regexes, as usage errors (exit 2).
- **Evidence:** `test_params.py::test_json_that_is_no_abstract_is_rejected_in_one_line` (four
  subcommands, `[]` and `{}`) and `test_bad_inputs_and_options_are_one_line_errors` (the rest):
  9 fail on the old params.py, 37 pass on the new; params, usage, man suites: 61 passed.

## 23. aeiou-trace exported trace files the runner refuses, and `--root /` counted nothing

- **Where:** `builder/aeiou/trace.py`: new module `normpath`, `Tracer.under`, `Exporter.open`,
  `Exporter.path_op`, `Exporter.rel`.
- **Found by:** the fixer of bug 7 noted that `rel()`'s fallback returned a path not under the
  root unchanged; checking whether real strace input could make the exporter emit a path the
  runner now refuses (bug 7's rules: absolute, `..`, empty, or the root itself changed) found
  three ways that did not go through that fallback (which is unreachable).
- **Defect:** (1) `os.path.normpath` keeps a leading `//`, which Linux reads as `/`: `export
  --root //` wrote `"path":"/"` (absolute, refused), and root `/d` silently dropped calls on
  `//d/x`. (2) `--root /` matched only `/` itself (the prefix test was `startswith("//")`), so
  `metrics` and `export` counted 0 ops. (3) The realistic one: a call that changes the root
  itself (`os.makedirs(root, exist_ok=True)` traces as `mkdir("/d") = -1 EEXIST`; likewise
  `rmdir`, `unlink`, `rename`, `open` with CREAT/TRUNC of the root) exported as `mkdir .`, and
  the runner refused the whole file ("path `.` is --root itself, and the trace changes it").
- **Fix:** roots and paths go through `normpath` (normpath, then `//` → `/`); `under()` builds
  its prefix as `root.rstrip("/") + "/"`, so `/` is every path; the exporter leaves out (and
  counts under `calls_not_exported`) an open that creates or truncates the root and any other
  path call that names the root; `rel()`'s fallback, and any result that is empty, absolute, or
  has `..`, is now an internal-error assertion instead of a silent bad path.
- **Evidence:** `test_trace.py::test_every_exported_path_is_one_the_runner_accepts` (five root
  spellings; mkdir/open CREAT/rmdir of the root, `..`, `//`, rename, chdir),
  `test_the_root_slash_is_every_path_and_double_slash_is_slash`,
  `test_a_path_not_under_the_exporters_root_is_an_internal_error`, and
  `test_the_runner_loads_an_export_that_mkdirs_its_root` (a real `aeiou dry-run`): 8 fail on the
  old trace.py, 8 pass on the new; test_trace.py 80 passed (+1 xfail held for a pending fix).

## 24. diskann-search's `hops.py` read its trace through strace's 32-element cut

- **Where:** `workloads/vector-search/diskann-search/kit/hops.py` (the bulk-submit test) and the
  capture command in that mechanism's README.
- **Found by:** the fix of bug 15: the kit's comment "strace prints 32 elements" showed it
  classified the node cache's bulk load by `len(io) != n` (iocbs printed fewer than requested),
  so its README could not take the `-v` every other capture command now has.
- **Defect:** on a trace taken with `-v` (whole arrays), hops.py's `bulk_submits`,
  `bulk_requests`, and `bulk_largest` fell to 0, `before_first_query` absorbed the
  breadth-first level, and `request_bytes` counted the load (hops and node_cache stayed right
  only because the marked capture drops what precedes the first query).
- **Fix:** a submit of more than `BULK = 32` requests, by the call's own count `n`, is a
  breadth-first level of the cache load (a query round submits at most the beam, 4 traced; the
  cache blocks are 8): on a cut trace that is exactly the old test, so earlier output is
  unchanged; a submit over 32 after the first query mark now exits with a message rather than
  being miscounted. The README's command adds `-v` (and `io_pgetevents`, as the others name).
- **Evidence:** new `builder/tests/test_kits.py`: `test_hops_reads_a_trace_with_and_without_v_alike`
  (one synthetic trace printed cut and whole: identical report and `--params`) and
  `test_hops_refuses_a_query_round_over_the_bulk_size` fail on the old hops.py and pass on the
  new.
- **Noted, record unchanged:** the committed sift1m-cache capture's fan-out `"32": 1` is the cut
  752-request breadth-first level (10,765 = 10,000 + 13 + 752); the record is short ~720 reads
  (~0.65%) in a part of the load the abstract does not claim to model; its `params.json`
  node_cache (10,765, taken from `n`) is right. A recapture with `-v` would correct the record.

## 25. Any stray connection to the coordinator's port aborted the run

- **Where:** `runner/aeiou/src/coord.rs`: `accept_loop`, `serve`, new `read_hello`, new
  `SState::said_hello`, the duplicate-rank check; runner REFERENCE §6.
- **Found by:** review of the coordinator: rank 0 started with `--ranks 2`, a bare connect
  and close (`exec 3<>/dev/tcp/127.0.0.1/7392; exec 3>&-`), then rank 1.
- **Defect:** a connection counted toward the expected ranks before sending Hello; one that
  closed before Hello failed the whole run ("start gate: …: connection closed"), and the
  listener, closed after `ranks` accepts or on failure, left the real rank retrying "Connection
  refused" for 120 s. A port probe or a launcher's readiness check killed a multi-host run. And
  the duplicate-rank check looked at connections registered after Welcome, so two hosts
  claiming one rank while Welcome was held could both pass.
- **Fix:** a rank counts only once its Hello passes the checks (`said_hello`); `read_hello`
  reads the first frame with a bound (`DEAD_AFTER`), and a close, garbage, an oversized or
  non-Hello frame, or silence is one stderr line ("ignored a connection from PEER that sent no
  Hello: …"), not a failure; the listener accepts until every rank has said Hello, the run fails,
  or the window ends; duplicates are checked against `said_hello`. Wire protocol unchanged.
- **Evidence:** the repro: before, rank 0 exit 1 and rank 1 still retrying at 20 s; after,
  both exit 0 with the merged fingerprint. `tests/coord.rs::a_connection_that_sends_no_hello_is_no_host`
  (four strays: silent, closing, HTTP, a valid non-Hello frame; then two hosts pass the gate)
  times out on the old coord.rs and passes in 0.22 s. Fixed in an isolated worktree, applied.

## 26. A refused host's Stop could be lost to the other hosts when rank 0 exited first

- **Where:** `runner/aeiou/src/coord.rs`, new `impl Drop for Server`; runner REFERENCE §6
  *Faults*.
- **Found by:** a CI flake in `builder/tests/test_coord.py` (the deprecated Python datagen's
  coordinator: a refused rank printed "connection closed" instead of "dedupe 2 differs from
  rank 0's 1"); checking the Rust coordinator for the same race.
- **Defect:** the refused host itself always reads its Stop (written under the state lock
  before the FIN), but the broadcast to the *other* hosts went out after rank 0's in-process
  client had its own Stop: rank 0's main thread could return its error and exit before the serve
  thread wrote the rest, and those hosts read EOF ("connection closed") with the reason lost.
- **Fix:** dropping the `Server` takes the state lock once (bounded at 2 s), so any turn in
  progress, and with it the whole broadcast, reaches the kernel before rank 0 exits.
- **Evidence:** `tests/coord.rs::a_refusal_reaches_every_host_before_rank_0_can_exit` (50
  iterations, pinned to one CPU: 16 hosts must hold the Stop the moment rank 0's ready() errors,
  and the refused host its reason before EOF; and rank 0 itself refused) fails at iteration 0
  without the fix and passes with it; `--test all` 79 passed twice.
- **Noted:** `coord::a_killed_datagen_or_grow_settles_at_whole_directories` failed once under
  load at its "may still be writing" timing assertion and passed 3/3 after (queued as a timing
  check).

## 27. A trace node's short or failed write moved the runner by its length, not its count

- **Where:** `runner/aeiou/src/trace.rs` (`build`, the Write arm), `metrics.rs` (bytes of a
  write), `run.rs` (`settle`: a write checked against its length; `issue_blocking`: the file
  position after a short traced write under a non-positional backend), `vm.rs` (`Op::bytes`
  doc); runner REFERENCE §13.
- **Found by:** review of aeiou-trace against the runner (the Python readers and the runner
  must give the same metrics document, builder REFERENCE §7 / runner REFERENCE §13); a fixer
  confirmed it with a hand-written trace (a write of 8192 returning 5000, a 3192 write failing
  ENOSPC, the 3192 retry, a read of 8192) and found the Python side already right.
- **Defect:** the runner advanced a sequential write's position and counted its bytes by the
  requested `len`, ignoring the traced count: dry-run `bytes_written` 14,576 where `aeiou-trace
  metrics` of the strace and of the export gave 8,192, the blocks and runs differed, and `aeiou
  run` left a 14,576-byte file where the application wrote 8,192 (under `sync` even after the
  position was fixed in the walk, since plain `write(2)` moves the kernel's position by what it
  wrote).
- **Fix:** a trace write's `bytes` is its traced count (0 when it failed) and a sequential one
  leaves the position at `offset + bytes`; metrics count `op.bytes` for writes as for reads (VM
  writes always have bytes == len, so abstracts are unchanged); `settle` still checks a write
  against its length; under a non-positional backend a short traced write is followed by an
  `lseek` to where the application's call left the position (a backend call, not an op, as the
  O_DIRECT read path already does). **Fingerprint:** changes only for a trace file with a
  sequential op after a short or failed write on the same open; none is committed; every
  golden and committed fingerprint is unchanged.
- **Evidence:** `test_trace.py::test_the_trace_nodes_dry_run_moves_what_a_short_write_returned`
  (written earlier as a strict xfail) and `test_the_trace_nodes_run_leaves_the_file_a_short_write_left`
  (sync, io_uring, libaio: an 8,192-byte file) fail on the old runner and pass on the new;
  `test_a_short_or_failed_write_moves_what_it_returned` pins the Python side; all backends give
  fingerprint 1ed9742a4a884314; `--lib` 40 passed, `--test all` passed (one unrelated timing
  flake, queued). Fixed in an isolated worktree and applied.

## 28. The event loops never freed a finished sub-actor: closes delayed to run end, memory grew per fork

- **Where:** `runner/aeiou/src/uring.rs` (`end_child`, `add_task`, new `Prog::Ended`,
  `Loop::free`); runner REFERENCE §8 *Tasks and loops*.
- **Found by:** review of the execution engines: tasks were only ever pushed; `end_child` set
  `done` but kept the task's VM and `ActorState`, whose frozen view held an `Arc` to the
  parent's open files.
- **Defect:** under `io_uring` and `libaio` an abstract's `close` of a file a `parallel` had read
  only dropped one reference; the real `close(2)` came at run end, so the storage (NFS: the
  CLOSE and its flush) did not see it during the run, the backend changing what reaches storage;
  descriptors piled up (`open files peak 500` for a loop of 500 open/parallel-read/close, 1 under
  `sync`), and memory grew ~0.8 MiB per fork (diskann_search: 413 MiB at 500 queries, 1.57 GiB at
  2,000, against 71 MiB under sync).
- **Fix:** when a sub-actor ends, its program becomes `Prog::Ended`, its state is replaced by an
  empty one (dropping the parent's files), and its slot goes on a free list that `add_task`
  reuses, so `tasks` holds at most the tasks alive at once. No generation counter is needed: a
  task ends only with no op in flight and no timer, so no completion can name a freed slot
  (checked by a debug assertion). Main tasks are never freed. No fingerprint change.
- **Evidence:** the repro gives `open files peak 1` under sync, io_uring, and libaio with one
  fingerprint; diskann_search's maxrss is flat (io_uring 41.4 MiB at 500 queries, 42.3 MiB at
  2,000). `tests/report.rs::a_parallel_s_file_closes_when_the_abstract_closes_it_under_every_engine`
  (peak 1 and 80 reads under each engine, equal fingerprints) fails on the old uring.rs (peak
  40) and passes on the new; `--test all` 81, `uring_knobs`, `limits`, `--lib` passed.

## 29. V12 let an abstract `mkdir` or `rmdir` a dataset's directories

- **Where:** schema/README §4 (V12), `schema/check.py`, `runner/aeiou/src/validate.rs`.
- **Found by:** review of the builder core, comparing the builder's `_read_only` and V14
  (input namespaces: mkdir and rmdir listed) with V12 (datasets are read-only, CLAUDE.md).
- **Defect:** V12 listed write, ftruncate, fallocate, unlink, and rename but not mkdir or rmdir,
  so a hand-written AST with `rmdir` (or `mkdir`) of `{"dir": {"dataset": "d", "id": 0}}`
  passed both checkers, and `aeiou run` would have removed a directory of a read-only corpus.
- **Fix:** both ops are in V12 in all three places, with the same message and path in both
  checkers ("rmdir on dataset `d`: datasets are read-only (V12)"). No committed AST changes;
  CI's diff of the two checkers' output over every committed AST is empty.
- **Evidence:** `golden.rs::validator_rejects_what_check_py_rejects` (two new cases) and
  `test_builder.py::test_check_py_v12_refuses_mkdir_and_rmdir_on_a_dataset_directory` fail
  without the fix and pass with it; test_builder.py 49 passed. Fixed in an isolated worktree.
- **Noted:** several coordinator/datagen timing tests (`a_killed_datagen_leaves_a_manifest_that_is_true`,
  `a_subset_runs_whole_directories_of_the_corpus`, `a_killed_datagen_or_grow_settles_at_whole_directories`)
  failed once each under the load of parallel builds and passed alone (queued as a timing check).

## 30. Meaningless numeric options panicked, exited 1, or were silently bent into something else

- **Where:** `runner/aeiou/src/main.rs` (`dry_run`, `resolve_run`, `datagen_cmd`),
  `runner/aeiou/src/dryrun.rs` (`write_report`).
- **Found by:** review of the runner's CLI against runner REFERENCE §15 (a bad value is a usage
  error, exit 2; a panic is never acceptable), then a sweep of every numeric option of
  dry-run, run, and datagen.
- **Defect:** `dry-run --ranks 0` (or `AEIOU_RANKS=0`) panicked dividing by zero (exit 101);
  `--steps 1..x` and `--metrics-block 0`/`--metrics-sample 0` exited 1 outside the frame;
  `--gpu 5` with two instances, `--gpu=-1`, `--steps 5..1`, `--limit 0`, `--threads 0` printed
  nothing and exited 0; `datagen --progress-secs 1e30` panicked; `--dedupe 0`/`--compress 0`
  were silently recorded as 1, `--writers 0` ran; `run --buffer-mib 0` became 1 MiB (a huge value
  wrapped on `<< 20`), `--write-compress 0` became 1, a negative or NaN `--time-scale` became 0
  and `inf` would sleep forever, `--threads 0` meant the default, `--iowq-max-workers 0` no cap,
  `--max-gap` took negatives and NaN.
- **Fix:** each is a usage error in the shared frame (exit 2) naming the flag, the value, and
  the rule (`--ranks 0: the run needs at least one host`, `--gpu 5: no actor has that instance
  (ids: gpu 0..2)`, `--steps S: expected A..B, whole numbers with 0 <= A < B`, …), with the
  layer named when the value came from the environment; `write_report` can no longer divide by
  zero. No fingerprint or contract change.
- **Evidence:** `tests/usage.rs::a_meaningless_number_is_a_usage_error` (every case above, and
  a valid `--gpu 1 --steps 0..1 --limit 3` printing exactly 3 lines) fails on the old main.rs
  (exit 101 for 2) and passes on the new; `--test all` 82 passed, man pages included.

## 31. A panic on a `sync` loader worker or actor thread hung the run forever

- **Where:** `runner/aeiou/src/run.rs` (`Ring::ensure`/`slice`, `spawn_sub`, the loader worker,
  the actor threads of `run_with`, `settle`, new `alloc_failed`, `panicked`),
  `runner/aeiou/src/uring.rs` (`Pool::get` and its callers), `aio.rs` and `object.rs` (callers).
- **Found by:** review of the execution engines: no `catch_unwind` on loader workers or actor
  threads (only the `parallel` pool caught panics), so a panic unwound past
  `aborted.store(true)` and the channel's `fail`, which the consumer and barrier peers poll.
  Reproduced with `--param hdr_read=70368744177664 --param workers=2`.
- **Defect:** under `sync` both workers panicked ("buffer allocation of 70368744177664 bytes
  failed") and the run then waited forever (killed by `timeout` at 20 s, exit 124): no exit, no
  report. Under io_uring the same input ended with a bare "an event loop panicked".
- **Fix:** a buffer that cannot be allocated is an ordinary op error (`OutOfMemory`, no errno, so
  no `expect` can admit it) naming the op and size, in the ring and the event loops' pool; and
  loader workers, sub-actors, and actor threads run under `catch_unwind`, failing the channel or
  setting `aborted` and returning the panic's message, so no panic can hang a run. No fingerprint
  change.
- **Evidence:** the repro now exits 1 at once under sync ("… take `batches`: producer failed: …
  read … len=70368744177664: buffer allocation of 70368744177664 bytes failed") and io_uring;
  with only the allocation reverted to a panic, the run still exits 1 ("producer failed:
  panicked: …"). `tests/run.rs::an_op_whose_buffer_cannot_be_allocated_ends_the_run_with_its_error`
  (sync, io_uring, libaio, each under a 20 s watchdog) fails on the old code ("did not end within
  20 s") and passes in 0.26 s. Fixed in an isolated worktree and applied over bug 28.
- **Noted:** io_uring SQEs pass a read's or write's length as `len as u32`, truncating any op
  over 4 GiB (queued).

## 32. `aeiou datagen` wrote an `array` dataset dense where the layout is sparse

- **Where:** `runner/aeiou/src/datagen.rs` (the array arm of `write_share`, new `write_slots`,
  `write_at`); runner REFERENCE §5 (*`O_DIRECT`, always*).
- **Found by:** review of datagen against DESIGN_REVIEW §3.72 (an array is "a sparse file, the
  slot's tail unwritten") and `project.rs`, which counts only slot contents: faiss_ivf_ondisk
  (sift1m-ivf1024-pq32, 256 KiB slots, lists ~36 KB) printed "38.33 MiB to write" and wrote
  255.84 MiB, all allocated.
- **Defect:** datagen filled every slot's tail with payload, so the space check (§3.74)
  undercounted by the slot/content ratio, `provenance.bytes_written` disagreed with the
  projection, and storage saw a dense file instead of the sparse layout the workload claims.
- **Fix:** the file is created and `set_len` to its full length (holes), and only each slot's
  content `[c·slot, c·slot + size(c))` is written, in the 4 KiB blocks it touches (aligned for
  `O_DIRECT`), payload still keyed by file offset, so each content's bytes and the dataset id are
  unchanged (full-slot arrays such as diskann's are byte-identical); bytes counted are the
  contents. Same 64 MiB pieces and writers, so it still streams. No abstract reads into a tail;
  grow/shrink refuse arrays; resume rewrites the array whole the same way.
- **Evidence:** faiss now writes 38.33 MiB (82,536 blocks allocated against 523,968),
  `bytes_written` 40,195,291 = the projection, same id. `tests/run.rs::array_datagen_writes_only_the_slots_contents`
  (projection = reported = manifest bytes, file length, allocation where holes exist, every
  content equal to the dense writer's bytes, every tail zeros, an mmap run = the dry run) fails
  on the old datagen.rs ((1, 13021678) vs (1, 78409175)) and passes; `--test all` 81 passed,
  builder tests touching arrays 184 passed. faiss lists=8192 datagen 1.7 s vs 1.9 s; diskann
  unchanged. Fixed in an isolated worktree and applied.

## 33. On several hosts, a trace's created paths were checked by every rank while rank 0 cleaned them

- **Where:** `runner/aeiou/src/trace.rs` (`check_root`, new `creates` argument),
  `runner/aeiou/src/main.rs` (`run_connected`); runner REFERENCE §13 *Before the gate*.
- **Found by:** review of the coordinator: a trace node run on 2 ranks with `--clean-namespaces`
  over a root holding a stale created path failed 2 of 6 tries.
- **Defect:** every rank checked that the paths the trace creates were absent, but only rank 0
  removes them under `--clean-namespaces` (it alone prepares the shared root, CLAUDE.md), and
  nothing ordered the two before the gate: rank 1 could see the stale path first and refuse
  with "out/final exists … remove it or pass --clean-namespaces", the flag the user had passed.
- **Fix:** the created-path check runs on rank 0 alone, after its clean, as output namespaces
  already work (`prepare_namespaces` is rank 0's); other ranks check the trace's inputs and say
  the created paths are rank 0's to check. No fingerprint change.
- **Evidence:** the repro passes 10 of 10 tries. `tests/trace.rs::only_rank_0_checks_what_a_trace_creates_after_its_clean`
  (two processes, each rank its own root holding the stale path, so a rank 1 that checks always
  fails) fails without the fix and passes with it; `--test all` 85 passed.
- **Noted:** failing coordinator datagen tests leave 2–4 GB each in `/tmp` and filled the disk
  during the review (cleaned; queued as a test-hygiene bug).

## 34. A failing runner test left its scratch directory (GBs for datagen tests) in /tmp

- **Where:** `runner/aeiou/tests/common/mod.rs` (new `TempDir` guard and `tmpdir`), every
  test file's `tmpdir` and its trailing `remove_dir_all`, `tests/object.rs` (MinIO's dir),
  `runner/aeiou/src/options.rs` (`RESERVED_ENV`), runner REFERENCE §14.
- **Found by:** the disk of the review's host filled to 100% ("No space left on device") with
  `/tmp/aeiou-coord-*` directories of 0.4–2.4 GB, making later tests fail in a cascade.
- **Defect:** test directories were removed only by a last-line `remove_dir_all`, so a panicking
  test (e.g. a timing flake in `a_killed_datagen_or_grow_settles_at_whole_directories`, which
  left 468 MB) leaked its corpus; `run::a_corpus_larger_than_the_free_space_is_refused_before_anything_is_written`
  leaked even when it passed (no cleanup line).
- **Fix:** `tmpdir` returns a `TempDir` that removes the directory in `Drop`, so also on a
  panic's unwind (it derefs to the path, so call sites are unchanged); every trailing removal is
  gone; `AEIOU_KEEP_TEST_DIRS` keeps them for debugging (reserved, so spawned runners do not warn).
  No assertion changed. The Python tests use pytest's `tmp_path` and needed nothing.
- **Evidence:** `tests/run.rs::a_test_directory_is_removed_when_its_test_panics` fails with a
  no-op `Drop` ("… is left behind by a panicking test") and passes; `--test all` 86 (and 90 with
  `object`), `limits`, `uring_knobs` pass, and no new `aeiou-*` directory of the test runs remains.

## 35. The object engine took any error mentioning "416" for end-of-file

- **Where:** `runner/aeiou/src/object.rs`: `not_satisfiable`, `Store::open` (new `Ranged` HTTP
  connector and `RangeNotSatisfiable` error type).
- **Found by:** review of the object engine: a ranged GET past an object's end (HTTP 416) maps
  to a 0-byte read, and it was recognised by `e.to_string().contains("416" | "Range Not
  Satisfiable" | "InvalidRange")`, before the NotFound mapping; object_store's error text holds
  the request URI and the elapsed time.
- **Defect:** a NotFound, 5xx, or timeout on a key containing "416" (`img_000000416.jpg`,
  `shard-00416`), or whose elapsed time did (`1.416ms`), became a 0-byte EOF instead of
  ENOENT/EIO: reproduced against MinIO, a deleted `img_000000416.jpg` read as "expected 91003
  bytes, got 0: the corpus does not match the dataset definition".
- **Fix:** object_store 0.14.2 exposes no error status (its RetryError is crate-private), and a
  HEAD per read would charge the backend more than its API, so the store is built with a
  connector wrapping object_store's own reqwest connector that turns a 416 response (body read,
  connection kept) into a non-retried error carrying a `RangeNotSatisfiable` type;
  `not_satisfiable` looks for that type in the source chain, never at text. No new dependency;
  no fingerprint change.
- **Evidence:** `tests/object.rs::a_missing_object_is_enoent_whatever_its_key_and_a_read_past_the_end_is_eof`
  (MinIO; both `--s3 blocking` and `async`: a full run matches the dry run with every read
  passing an object's end, then the deleted `…416.jpg` fails "not found (ENOENT)") fails on the
  old object.rs and passes on the new; object tests 5 passed. Fixed in an isolated worktree and
  applied after bug 34.
- **Noted:** with two loader workers failing, the reported error was once the secondary "run
  aborted: another actor failed" rather than the cause (queued).

## 36. `aeiou datagen --dataset` with an unknown name exited 0 and wrote nothing

- **Where:** `runner/aeiou/src/main.rs`, `datagen_cmd`.
- **Found by:** review of datagen: `aeiou datagen --root R --param samples=100 --dataset nosuch
  tfrecord_stream.ast.json` printed its options and exited 0; the root was never created.
- **Defect:** the `--dataset` filter matched nothing and datagen did nothing, successfully: a
  typo in a scripted tuning loop looked like a written corpus.
- **Fix:** every `--dataset` name not among the abstract's datasets is a usage error (exit 2, the
  frame), all unknown names listed with the known ones, before anything is printed or created
  (a namespace name is refused too: datagen writes datasets). `--endpoint` with an unknown name
  was already refused; `--subset` takes no name.
- **Evidence:** `tests/usage.rs::an_unknown_dataset_name_is_a_usage_error` (one unknown, two
  unknown among a known one, the endpoint case, no root created) fails without the fix and
  passes; `--test all` 86 passed. Fixed in an isolated worktree and applied.

## 37. A `loader` inside a loop passed `check` and `dry-run`, then `run` refused its second iteration

- **Where:** `runner/aeiou/src/run.rs` (new `retire_loader`, `join_loaders`, channel
  re-declaration), `runner/aeiou/src/uring.rs` (new `Loop::declare`); runner REFERENCE §7
  (*Loader*, *channel / put / take*).
- **Found by:** review of the execution engines: file_per_sample with its loader and training
  phase wrapped in `loop epoch<2` (a loader per epoch).
- **Defect:** channels were never removed per instance, so the second declaration of the
  loader's channel failed ("loader `batches`: a channel of that name exists") under every
  backend, while both checkers accepted the shape and the dry run gave a fingerprint: a valid
  abstract with a fingerprint that could not be run.
- **Fix:** declaring a loader again retires the earlier one when all its batches have been taken
  (its workers joined there, their stats merged; the event loops reset the channel in place);
  declaring it again with batches untaken is refused, saying how many were taken. A plain
  `channel` follows the same rule (re-declared when nothing put is left untaken; a take parked
  on the old one fails rather than hangs). The batch ordinal already counts the epoch loop, so
  epoch 2 draws `b = 12..23` in both dry run and run. No fingerprint change.
- **Evidence:** the epochs AST now gives the dry run's fingerprint (8bdc5d7547fbccd8, 48 takes)
  under sync, io_uring, libaio, posix-aio, and mmap; `tests/run.rs::a_loader_in_a_loop_runs_once_per_iteration`
  (sync and io_uring: fingerprint, ops, bytes, takes) fails without the fix and passes; all
  non-coord tests pass (the coord timing flakes are queued separately).

## 38. A malformed trace file could panic, hang, or wrap the runner instead of being refused

- **Where:** `runner/aeiou/src/trace.rs`: `TraceFile::load` (new `check_shape`, lane range,
  header `lines` as a capped hint), `index` (checked position arithmetic, a use of a failed
  open, true line numbers via `Line::no`); runner REFERENCE §13 *Loading*.
- **Found by:** review of the trace node: a nested `submit` passed `check` and panicked `dry-run`
  (`unreachable!` in `build`, exit 101); the fixer's sweep of the loader found more.
- **Defect:** any op was accepted as a `submit` member: a nested submit panicked, an `open`
  member hung `run`, a `stat` member failed with EINVAL, a `close` or unpositioned member ran
  with no meaning; `"lane": 4000000000000` hung all three commands allocating lanes; a header
  `"lines": 4e18` panicked ("capacity overflow"); a negative length gave "Bad address"; `t`/`dur`
  near i64::MAX and overflowing positions wrapped silently; a use of an open that failed in the
  trace was reported as "failed in this run"; line numbers were off after blank lines.
- **Fix:** each line's shape is checked at load (so `check`, `dry-run`, and `run` refuse before
  the gate) with its line number: a submit holds one or more positioned reads/writes only (the
  exporter never writes an empty one); lengths, `t`, `dur` non-negative and summing in range;
  lanes below the header's count; positions by checked arithmetic; a failed open's use refused;
  `header.lines` only a capped capacity hint. A huge but well-formed length still fails cleanly
  at run time (bug 31). No fingerprint change.
- **Evidence:** `tests/trace.rs::a_malformed_line_is_refused_at_load_with_its_line_number` (11
  malformed files, each refused by load, check, dry-run, and run with exit 1, "line N", no
  "panicked") fails on the old trace.rs ("a malformed trace loaded") and passes; trace tests 9
  passed. Fixed in an isolated worktree and applied.

## 39. The V3 proof trusted a `min` that a lower `max` undercuts, and the runner overflowed its stack

- **Where:** the three provers of "provably ≥ k" (`builder/aeiou/nodes.py` `Dist.lower_bound`,
  `builder.py` `_dist_at_least`, `schema/check.py` `at_least`, `runner/aeiou/src/validate.rs`
  `dist_at_least` and `check_given`), the mixture rule in both checkers, `vm.rs` `at`;
  schema/README §4 (V3).
- **Found by:** review of the builder core: `d = draw(normal(5, 1, min=1, max=0)); x = when(i >
  0, x.at(i - d) + 1, 0)`.
- **Defect:** the runner clamps to `min` then `max`, so that normal always draws 0, yet every
  prover took `min` as the lower bound and called `i - d` provably below `i`: aeiou-build wrote
  it, both checkers said ok, and `aeiou dry-run` died with "thread has overflowed its stack" (a
  core dump), the self-reference at the same index recursing forever. A mixture whose only arm
  is `none` was "≥ 1" (an empty `all()`). Bug 13 refuses a literal min > max in the builder,
  but a hand-written AST or a parameter `max` given at run time still reached the provers.
- **Fix:** a literal `min` bounds below only when `max` is absent or a known number (a literal,
  or a parameter's value: its default in the builder and check.py, the given value in the
  runner's `check_given`) not below it; both checkers refuse "normal: min 1 is above max 0" (and
  lognormal) when both bounds are known, and the runner now holds a distribution given by
  `--param` or a parameter file to every distribution rule; a mixture needs a non-null arm; and
  `at` refuses an index not below the one being evaluated (a clean error naming the binding)
  instead of recursing. No AST, hash, or fingerprint changes; the checkers' outputs over every
  committed AST are identical.
- **Evidence:** `golden.rs::the_at_rule_believes_no_minimum_a_maximum_undercuts`,
  `golden.rs::an_at_that_does_not_decrease_is_an_error_not_a_stack_overflow` (SIGABRT without
  the fix, a clean error with it), and `test_builder.py::test_at_rule_believes_no_minimum_a_maximum_undercuts`
  fail on the old code and pass on the new; golden 15 passed, builder 88 passed. Fixed in an
  isolated worktree and applied.


## 40. Three coordinator tests failed under load: a shared report path and a stepped wall clock

- **Where:** `runner/aeiou/tests/coord.rs` (`run_ranks`, `just_updated` → `noted`, and the tests
  `a_subset_runs_whole_directories_of_the_corpus`, `a_killed_datagen_leaves_a_manifest_that_is_true`,
  `a_killed_datagen_or_grow_settles_at_whole_directories`, `a_killed_grow_is_finished_by_resume`,
  `a_killed_shrink_settles_without_removing_more`); `runner/aeiou/src/datagen.rs` `check_not_live`.
- **Found by:** intermittent failures of these tests while the suite ran beside other work;
  reproduced with 16 `yes` loops raising the load to 26–39 on 20 CPUs, and a clock monitor.
- **Defect:** (1) `run_ranks` gave every rank one `--report-json` path, and a rank other than 0
  writes its own host report there (aeiou(1)), so whichever rank finished last won: every
  failure read exactly rank 1's 7 ops and 62400 bytes; the merge itself is right. (2) The killed
  datagen tests set `updated = now`, slept 60 ms, and expected the live check (two 20 ms
  windows) to accept; this WSL2 host's wall clock steps back about 2.4 s every 32 s against the
  monotonic clock, so a step in that gap made the age negative and the run was refused, and the
  refusal printed "updated 0 s ago" (`age.max(0)`), which hid the cause; the refused half also
  depended on the next process starting within 40 ms.
- **Fix:** a rank r > 0 gets `FILE.rank<r>`, and the test checks rank 0's report is `scope: run`
  and rank 1's `scope: host` with its own counts; `noted(dir, ago)` records a 30 s window at a
  chosen age, and the tests check refused at 0 s and 40 s (naming "every 30 s … wait until
  60 s") and accepted at 90 s, margins of tens of seconds; `check_not_live` now says "updated
  X s ahead of this host's clock" for a time in the future (the refusal is unchanged).
- **Evidence:** under that load the subset test failed 4/40 before and 0/60 after; the killed
  tests 0 failures in 100 runs after; the new unit test
  `datagen::tests::a_manifest_younger_than_two_windows_is_live` fails without the message fix
  ("updated 0 s ago") and passes with it; the five coord tests pass. Fixed in an isolated
  worktree and applied.

## 41. A `compute` overslept by the timer slack, by an amount that depended on the backend

- **Where:** `runner/aeiou/src/run.rs` (`run_with`, `Runner::control`, `ActorState`),
  `runner/aeiou/src/uring.rs` (`Loop::step`, the compute and trace-gap timers);
  runner/REFERENCE.md, the `compute` bullet of the run semantics.
- **Found by:** review of the execution engines: the blocking path slept
  `std::thread::sleep(scaled)` per compute and the event loop armed `now + scaled`, both under
  Linux's default 50 µs timer slack, and every compute restarted from when its sleep returned.
- **Defect:** emulated GPU time grew with every compute's overshoot and differed by backend.
  20,000 × `compute 20000` (ideal 0.400 s) took 3.08–3.14 s under `sync`, 2.57–3.26 s under
  `mmap` and `posix-aio`, 2.79–3.52 s under `libaio`, and 1.47–2.00 s under `io_uring`, so the
  same abstract's pacing, and the I/O pressure it produces, depended on the backend, against
  the rule that backends never change what the workload is.
- **Fix:** `tight_timers()` sets `PR_SET_TIMERSLACK` to 1 ns at the start of `run_with`, before
  any thread is spawned (threads inherit it). Consecutive computes are paced from absolute
  deadlines: `ActorState::pace` takes the previous compute's deadline plus `scaled`, and any
  other step (an op's completion, a take, a barrier, a fork, a trace) clears it, so a compute
  after an op still counts from that op's completion. Both the blocking path (`sleep_until`)
  and the event loop's timers use it. No AST, fingerprint, or dataset id changes.
- **Evidence:** after the fix the same runs take 0.400–0.402 s under `sync`, `posix-aio`,
  `mmap` and `io_uring`, and 0.446–0.451 s under `libaio` (a fixed ~45 ms from `io_destroy`
  at teardown, queued separately). The new test
  `run::consecutive_computes_take_their_sum_on_every_backend` (5,000 × 20 µs under `sync` and
  `io_uring`, within 50 ms of the sum) fails without the fix ("took 460.75 ms, 360.75 ms over
  their sum") and passes with it; `--test all` 92 passed in four consecutive runs.

## 42. A resumed object-store datagen left out of its totals the objects it found

- **Where:** `runner/aeiou/src/datagen.rs` `write_share` (the container and plain/chunk
  branches' `plan.present` skips); runner/REFERENCE.md §5 "Resume"; man/aeiou.1.md
  `datagen --resume`.
- **Found by:** review of datagen: on an object store, `--resume` lists the prefix once and
  skips the unfinished ids whose keys are already there, but the skip left `got` at 0.
- **Defect:** an object uploaded after the last progress window was in the store but in none
  of the counts: the windows' `files_written`/`bytes_written`, the final
  `provenance.files_written`/`bytes_written`, `provenance.resumes[i]`, and the printed
  summary. schema/README §6 and REFERENCE §5 say the totals after a resume count every file
  once, as a directory's resume does by writing the id again. Repro: 10 objects written, the
  manifest set back to `writing` with every id unfinished, 2 objects deleted, then
  `aeiou datagen --resume s3://bench/res` printed "2 file(s), 2.00 MiB" and the manifest said 2
  files and 2 MiB of a corpus of 10 files and 10 MiB.
- **Fix:** a present object counts as written: `(1, size)` for a container, `+ (1, len)` per
  plain or chunk object, the same numbers a directory resume reports for the same ids. The docs
  say so. No fingerprint, dataset id, or AST changes.
- **Evidence:** `object::a_resumed_object_dataset_counts_the_objects_it_finds` (MinIO) fails
  without the fix (left 2 files, right 10) and passes with it; `--test all object` 6 passed.
  Fixed in an isolated worktree and applied.

## 43. `datagen --resume` (and grow, shrink, settle) refused a config file that set `[datagen]` root

- **Where:** `runner/aeiou/src/main.rs` `datagen_cmd` (the root match's `Some(op)` arm),
  `runner/aeiou/src/options.rs` (new `Layers::unused`); runner/REFERENCE.md §14,
  man/aeiou.1.md `--resume`, man/aeiou-config.5.md (the `[datagen]` table).
- **Found by:** review of the coordinator/CLI: the modes that change a dataset root take its
  place from the manifest, so they never asked the layers for `root` or `endpoint`.
- **Defect:** keys a subcommand never reads are refused by `Layers::finish`, so a `[datagen]`
  table with `root`/`endpoint` (the natural file for a fresh datagen) made `aeiou datagen
  --resume root/train --config a.toml` exit 2 with "[datagen] endpoint: not an option of
  `aeiou datagen` (or the command line's alone)", while `AEIOU_ROOT` in the same mode was
  ignored with nothing said: the file and environment layers disagreed, and one config could
  not serve a datagen and its resume (§14).
- **Fix:** in `--resume`, `--grow`, `--shrink`, and `--settle`, `--root`/`--endpoint` on the
  command line stays a usage error; a value from the environment or the file is passed over
  with one WARNING line per option in the printed options block (and in the report's
  `warnings`), naming the layer that would have won, and no row for it. No fingerprint,
  dataset id, or AST changes.
- **Evidence:** `options::a_change_of_a_dataset_root_passes_over_the_place_from_the_lower_layers`
  (a fresh datagen from config root/endpoint, `--resume` with `--config`, `--grow` via
  `AEIOU_CONFIG`, `--shrink` with `AEIOU_ROOT`, `--resume --root` still refused) fails without
  the fix with the message above and passes with it; `--test all options man` 13 passed,
  `coord` 20 passed. Fixed in an isolated worktree and applied.

## 44. Two `consume`s of one dataset drew the same samples

- **Where:** `schema/check.py` and `runner/aeiou/src/validate.rs` (new `v6_one_consume`, the
  record in `handle()`); schema/README.md §4 V6, man/aeiou-abstract.7.md V6, the refusal table
  of builder/REFERENCE.md; `vm.rs` `consume` (unchanged).
- **Found by:** review of the runner's semantics: the consume position is
  `g + G·((b mod epoch_len)·B + j)` from the actor's number and its enclosing frames' ordinals
  (runner/REFERENCE.md §2), with no site and no node identity in the key.
- **Defect:** any two `consume` nodes of one dataset reach the same positions. In a loop
  `j < 4` with `let a = consume d; let b = consume d; stat a; stat b`, both checkers said ok and
  `dry-run --gpu 0` statted `d/f0006`, `d/f0000`, `d/f0002`, `d/f0004` each twice; sibling
  loops (a train phase then an eval phase) and two actor templates re-read the same ids alike.
  A "next sample without replacement" stream silently became one with reuse, inflating cache
  hits, against the positional-randomness rules.
- **Fix:** V6 now also says a dataset is consumed in one place: both checkers refuse the second
  `consume` (revisits of one node are de-duplicated by path), naming the first's path and the
  remedy (consume once and use the handle in each place, or put the one `consume` in an inner
  loop), with the same message and path in both. The builder refuses it at build time through
  `validate()`. No committed AST does this (all 14 checked by both checkers, the most any
  dataset has is one `consume`; `aeiou-build --hermetic --twice --check` exits 0). No
  fingerprint, dataset id, or AST shape changes.
- **Evidence:** `golden::a_dataset_is_consumed_in_one_place` and
  `test_builder.py::test_a_dataset_is_consumed_in_one_place` (same body, sibling loops, two
  actors; one `consume` used twice accepted) fail with the rule disabled and pass with it;
  golden 16 passed, test_builder + test_man 69 passed. Fixed in an isolated worktree and
  applied. Left open: one `consume` under an inner loop whose count depends on an outer index
  can still repeat a position (queued separately).

## 45. A multipart upload that failed at completion was left on the store, and big objects hit S3's 10,000-part limit late

- **Where:** `runner/aeiou/src/object.rs` (`Store::put_parts`, new `send_parts` and
  `part_for`, `Upload::send`/`fill`, `MAX_PARTS`, `MAX_OBJECT`); runner/REFERENCE.md §16
  (`--object-part-mib`, "`aeiou datagen`"); the `datagen.rs` module comment.
- **Found by:** review of the object engine against `object_store` 0.14.2, whose
  `S3MultiPartUpload` has no `Drop`: an upload is aborted only by an explicit `abort()`.
- **Defect:** (a) datagen's `put_parts` aborted on a failed part but returned a failed
  `complete()` without aborting, leaving an orphaned multipart upload: billed parts on real
  S3, invisible to the `LIST` a resume uses. (The run path's `Upload` already aborted.)
  (b) `put_parts` always used 8 MiB parts and the run's `Upload` never counted parts, so an
  object over 78.1 GiB failed at part 10,001 after uploading 78 GiB, with the store's cryptic
  error.
- **Fix:** `put_parts` sends through `send_parts`, which aborts on any error, a part's or the
  completion's, and returns the original error; its part size is `part_for(size, 8 MiB)`, the
  whole MiB that keeps the object within 10,000 parts (about 525 MiB at 5 TiB), and an object
  over 5 TiB is refused with `EFBIG` before any request. A run's `Upload`, whose size is not
  known up front, refuses part 10,001 ("a larger --object-part-mib uploads it") and a write
  past 5 TiB with `EFBIG` before sending, and its `Drop` aborts it. Part size never changes the
  bytes, the manifest, a fingerprint, or a dataset id.
- **Evidence:** lib tests `object::tests::the_part_size_keeps_an_upload_within_10000_parts`,
  `object::imp::tests::a_multipart_upload_that_fails_is_aborted_with_the_error_that_stopped_it`
  (a fake upload logging its calls: no "abort" after a failed completion before the fix) and
  `object::imp::tests::a_run_upload_refuses_part_10001_and_the_5_tib_object_before_sending`
  fail on the old behaviour and pass on the new; `--lib` 44 passed, MinIO `--test all object` 6
  passed. Fixed in an isolated worktree and applied.

## 46. A Parquet geometry the writer cannot frame was refused only after the manifest and files were written

- **Where:** `runner/aeiou/src/formats.rs` (new `Stream::skim`, `Writer::check`),
  `runner/aeiou/src/datagen.rs` `datagen` (the pre-flight `checked` loop).
- **Found by:** review of datagen: `parquet_page_header` (a page past i32) and
  `Stream::advance_parquet` (a page header outside the layout's varint band), and the
  footer/file-size drift checks, fire only when the writer reaches the file.
- **Defect:** a geometry that could never be written left a half-made dataset: `aeiou datagen`
  of a Parquet dataset whose second file's page header is 23 bytes where the layout predicts 26
  printed the space line, wrote `p/0000.parquet` (19 MB), an empty `p/0001.parquet`, and the
  manifest, then exited 1; one 2.56 GB page left an empty file and the manifest. A structural
  refusal must come before anything is written (bug 2), and on an object store the objects
  would have been uploaded.
- **Fix:** `Writer::check` drives the real writer's `advance()` over every file, skipping the
  payload instead of making it, and compares each file with `meta.file_size(file)`; it runs on
  threads over contiguous file ranges and returns the lowest failing file's error (an atomic
  minimum), with the writer's own message prefixed by the file. `datagen` calls it in the
  pre-flight, before the space check, the emptiness checks, the manifest, and any upload. O(samples), as
  the space projection (20 M samples in 104,167 files: 0.26 s); nothing per file is kept. HDF5
  is skipped (its header comes from `aeiou-header`). No fingerprint, dataset id, or AST changes.
- **Evidence:** `layout::parquet_geometry_refused_before_anything_is_written` (a valid control
  written; the band and over-2-GiB cases leave the root empty) fails without the pre-flight
  ("left behind: p, p/.aeiou-dataset.json, p/0000.parquet, p/0001.parquet") and passes with it;
  `--test all layout datagen` 11 passed, `--lib formats` 5 passed, `test_formats.py` 38 passed
  (Rust and Python writers still byte for byte). Fixed in an isolated worktree and applied.

## 47. A datagen host stopped by another host's failure said "stopped" and exited 130

- **Where:** `runner/aeiou/src/datagen.rs` (the end of a rank's datagen, before the
  `mode.halted()` branch); man/aeiou.1.md DATAGEN (the Ctrl-C paragraph), runner/REFERENCE.md
  the datagen *Ctrl-C* paragraph.
- **Found by:** review of the coordinator/CLI: the watch thread sets `mode.halt` both on a
  coordinator `Halt` (Ctrl-C) and on an abort (`tcp.abort_reason()`), and a rank other than 0
  then took the halted branch, `datagen::Stopped`, which `main.rs` exits with 130.
- **Defect:** with three ranks and rank 2's `train/` read-only, rank 0 exited 1 naming rank 2's
  error, rank 2 exited 1 with it, and rank 1 printed only "stopped" and exited 130, the Ctrl-C
  exit: a failure was reported as a user's interrupt and its reason was lost, against aeiou(1)
  (exit 1 is "a failure during the work") and REFERENCE §6 (a failure fails on every host
  naming the originating rank and reason). A lost coordinator was reported as "stopped" too.
- **Fix:** when the writers were halted and the coordinator holds an abort reason, the rank
  returns that reason as its error: "aeiou datagen: rank 2 (HOST): dataset `train`: creating
  …: Permission denied", exit 1. Ctrl-C keeps "stopped" and 130; the manifest's
  `progress.stopped` is written as before. No fingerprint, dataset id, or AST changes.
- **Evidence:** `coord::a_failing_datagen_host_fails_the_others_with_its_reason` (three
  processes; every rank exits 1 with the path and "Permission denied", ranks 0 and 1 name
  "rank 2 (", no "stopped" line, the manifest `writing` with `progress.stopped`) fails without
  the fix (rank 1 exit 130) and passes with it; `--test all coord` 21 passed, `man` 9 passed.
  Fixed in an isolated worktree and applied.

## 48. A negative dataset size was accepted and datagen wrapped it to 16384 PiB

- **Where:** `schema/check.py` and `runner/aeiou/src/validate.rs` (new rule V25, also applied
  by the runner's `check_given`), `runner/aeiou/src/eval.rs` `DsMeta::sample_size`;
  schema/README.md §4 V25, man/aeiou-abstract.7.md (V24 and V25 added to its rule list).
- **Found by:** review of the runner's semantics: every size of a non-`array` dataset goes
  through `sample_size`, which returned the draw unchecked, and its users (`file_size`,
  `sample_offset`, `unit_lens`, `chunk_size`, datagen's `as u64`, the space projection,
  `formats.rs`, `vm.rs`) took it as a byte count.
- **Defect:** `size: const(small − 100)` with `small` 10, or `normal(10, 100)` with no `min`:
  both checkers said ok, and `aeiou datagen` printed "16384.00 PiB to write" (with
  `--ignore-space` it would have tried).
- **Fix:** V25: both checkers refuse, with the same message and path, a non-`array` size the
  document shows can be below 0 (a known number after rounding, a `const` or parameter of one,
  a `uniform` lo, an `empirical` value, a `normal`/`lognormal` after the runner's clamp, one
  unbounded below, a `mixture` arm of positive weight); the runner applies it again to the
  values given with `--param`; and `sample_size` errors on a negative draw the document could
  not show, naming the dataset and sample. The builder refuses through `validate()`. No
  committed AST is affected; no AST, fingerprint, or dataset id changes.
- **Evidence:** `golden::a_dataset_size_is_never_below_zero` (fails without the checker change,
  and without the `eval.rs` change alone with "no sample of 4 drew below 0: [221, -69, -93,
  -100]") and `test_builder.py::test_a_dataset_size_is_never_below_zero` fail before and pass
  after; golden 17 passed, `--test all` 95 passed, test_builder + test_man 70 passed, both
  checkers over every committed AST unchanged. Fixed in an isolated worktree and applied.

## 49. A WebDataset member of 8 GiB or more panicked the tar header writer

- **Where:** `runner/aeiou/src/formats.rs` (`octal`, `tar_header`, `USTAR_MAX_SIZE`),
  `builder/aeiou/formats.py` (`WebDataset.__init__`, a fixed member).
- **Found by:** the fixer of bug 46, whose pre-flight drove the writer over a webdataset
  geometry with an 8 GiB sample.
- **Defect:** ustar's 12-byte size field holds 11 octal digits (at most 8 GiB − 1); `octal()`
  wrote the longer string past the field: "index out of bounds: the len is 12 but the index is
  12", a panic mid-dataset before bug 46 and "a format check thread panicked" after it. The
  Python format class accepted a fixed member of 8**11 bytes.
- **Fix:** `octal` returns an error when the value needs more digits than its field (mode,
  uid, gid, mtime, and the checksum now go through it too; their bytes are unchanged), and
  `tar_header` refuses "tar member `NAME` of N bytes: more than ustar's 8 GiB − 1
  (8589934591) size field", reached in datagen's pre-flight before anything is written. The
  Python class refuses a fixed member over `USTAR_MAX_SIZE` at build time (a sample member's
  size is the runner's to check). No GNU base-256 or pax extension; header bytes of every
  member that fits are unchanged.
- **Evidence:** `layout::tar_member_past_ustar_refused_before_anything_is_written` panics
  without the fix and passes with it, leaving the root empty (no 8 GiB file is written);
  `tar_header_is_cpythons` gains the 8 GiB − 1 and 8 GiB cases; a `test_formats.py` case for
  the Python refusal; `--test all layout` 7 passed, `--lib formats` 5 passed,
  `test_formats.py` 38 passed. Fixed in an isolated worktree and applied.

## 50. Bug 42's fix put the objects a resume found into the resume's own record (regression)

- **Where:** `runner/aeiou/src/datagen.rs` (`Track`, `Written`, `note`, `part_json`, the
  multi-host merge, the provenance), `runner/aeiou/src/main.rs` (the datagen summary lines),
  `runner/aeiou/tests/object.rs` (bug 42's test, amended); runner/REFERENCE.md §5 "Resume" and
  man/aeiou.1.md `--resume` (bug 42's sentences, corrected).
- **Found by:** CI after fa1787f (bug 42) and every later push:
  `builder/tests/test_object.py::test_aeiou_datagen_resumes_into_the_bucket[tfrecord|plain]`
  failed "the two missing, not the one there / assert 3 == 2". Bug 42 was verified with the
  Rust object tests only, and its own test asserted the wrong record.
- **Defect:** bug 42 counted a present object as written everywhere. Right for the dataset's
  totals (progress and `provenance.files_written`/`bytes_written`, every file once), wrong for
  the resume's own record: `provenance.resumes[i]`, its printed "N file(s)" summary, and its
  rate claimed objects it never sent.
- **Fix:** the found objects are their own count (`found_files`/`found_bytes` in `Track`,
  `found` in `Written`, summed across hosts): totals = before + sent + found; `resumes[i]`
  `files_written`/`bytes_written` = sent, with `files_found`/`bytes_found` when nonzero; the
  rate counts sent bytes; the summary adds ", N already there". A directory dataset never finds
  any, so its manifests and output are unchanged. No fingerprint, dataset id, or AST changes.
- **Evidence:** the amended `object::a_resumed_object_dataset_counts_the_objects_it_finds`
  (totals 10 files; `resumes[0]` 2 written, 8 found; "2 file(s), … 8 already there") fails on
  fa1787f's code (resumes[0] 10) and passes; `test_object.py` 15 passed against MinIO (both
  resume params, which fail on HEAD's binary); `--test all -- object coord datagen` 28 passed,
  on HEAD eddce97 plus this change. The fixers now also run the builder tests that exercise a
  runner change.

## 51. A failed run could report "run aborted: another actor failed" instead of its cause

- **Where:** `runner/aeiou/src/run.rs` (`Channel::fail`, new `echoes_abort` and
  `keep_cause`, `join_loaders`, the `parallel` fork's join, `run_with`'s actor join, the trace
  lanes' selection), `runner/aeiou/src/uring.rs` (`run`, `run_s3`), `runner/aeiou/src/aio.rs`
  (`run`).
- **Found by:** review of the execution engines, then confirmed by
  `run::an_op_whose_buffer_cannot_be_allocated_ends_the_run_with_its_error` (bug 31's test)
  failing about 1 run in 10: "take `batches`: producer failed: run aborted: another actor
  failed".
- **Defect:** two mechanisms. `Channel::fail` overwrote the recorded error: loader worker 0
  failed and raised the abort, worker 1's next op bailed "run aborted" and failed the channel
  again, replacing the cause the consumer then reported. And every join kept the first error in
  join order, so a thread that only saw the abort beat the thread that raised it (actors, loader
  workers, `parallel` sub-actors, event loops). The cause of a failed run was lost by timing.
- **Fix:** an error whose message says "run aborted" is an echo (`echoes_abort`, the test the
  trace-lane join already used); `keep_cause` keeps the first error that is not an echo, else
  the first; `Channel::fail` keeps its first message unless that is an echo and the new one is
  not; every join site uses `keep_cause`. No fingerprint, dataset id, or AST changes.
- **Evidence:** the flaky test failed 4/40 before and passed 40/40 after; the new
  `run::a_failed_run_reports_its_cause_not_the_abort_it_raised` (a loader, a `parallel` of
  two, and two actors, under `sync` and `io_uring`, one thread failing at 100 ms and another
  meeting the abort at 400 ms) fails 3/3 without the fix with four wrong reports and passes
  10/10 with it; `--test all run` 29 passed, `coord` 21, `--lib` 42.

## 52. `aeiou dry-run` silently ignored options given without the one they narrow

- **Where:** `runner/aeiou/src/main.rs` (`DryRunArgs`, `dry_run`); man/aeiou.1.md (dry-run
  output and metrics options), runner/REFERENCE.md (the `aeiou dry-run` row of the command
  table); `runner/aeiou/tests/usage.rs`.
- **Found by:** review of the coordinator/CLI against the help and the man page, which say
  "With `--gpu`:" for `--steps`/`--limit` and "With `--metrics`:" for
  `--metrics-block`/`--metrics-sample`.
- **Defect:** `dry-run … --gpus 2 --limit 2` exited 0 and printed instance 0's stream as if
  `--gpu 0` had been given (`a.gpu.unwrap_or(0)`), and `--metrics-block 8192 --metrics-sample
  4` without `--metrics` exited 0 having computed nothing, with no word either way. The block
  and sample were detected by comparing with their defaults, so an explicit default could not
  be told from an absent one, and the options block showed it as `[default]`.
- **Fix:** each is a usage error in the shared frame (§15, exit 2) when the option it narrows is
  absent: `--gpu` "(needed by --limit)", `--metrics` "(needed by --metrics-block)", listed with
  every other missing argument, as `run` does for `--report-takes`; `--metrics-json` implies
  `--metrics`, and a later `--no-metrics` brings the error back. `--metrics-block` and
  `--metrics-sample` are `Option`s with their defaults stated in the help. No fingerprint,
  dataset id, or AST changes.
- **Evidence:** `usage::an_option_narrowing_an_absent_one_is_a_usage_error` fails before the
  fix (exit 0 where 2 was expected) and passes after; `--test all -- usage:: man:: options::
  dry` 16 passed; test_trace.py, test_catalog.py, test_man.py 152 passed (their dry-run calls
  pair the metrics options with `--metrics-json`). Fixed in an isolated worktree and applied.

## 53. io_uring issued a read or write over 4 GiB at its length modulo 2^32

- **Where:** `runner/aeiou/src/uring.rs` `LoopIo::issue` (the Read and Write SQEs), new
  `rw_len`; runner/REFERENCE.md §8, the **Buffers** bullet; `runner/aeiou/tests/trace.rs`.
- **Found by:** review of the execution engines: `opcode::Read::new(f, ptr, len as u32)` and
  `opcode::Write::new(f, ptr, op.len as u32)`, the SQE's length field being 32 bits.
- **Defect:** an op of 4 GiB + 4096 bytes was issued as 4096: under io_uring a read of an 8 KiB
  file at that length returned 4096 bytes and failed "expected 8192 bytes, got 4096: the corpus
  does not match the dataset definition", where `sync` (one `pread` the kernel clamps to
  `MAX_RW_COUNT`) and `libaio` (a u64 `aio_nbytes`, clamped alike) moved 8192: the backend
  changed what an op moves, and a length of exactly 4 GiB was issued as 0.
- **Fix:** the SQE length is `rw_len(len)`, the length clamped to `MAX_RW_COUNT` (`INT_MAX`
  rounded down to a page, page-aligned so `O_DIRECT` lengths stay aligned), the most the kernel
  moves in one call under every backend; the buffer is still the op's whole length, as under
  `sync`, and a write fills the issued prefix (the payload depends only on position). An op
  over the limit moves what it moves under `sync`, judged by the same settle. No fingerprint,
  dataset id, or AST changes.
- **Evidence:** `trace::a_read_over_4_gib_moves_what_sync_moves_under_every_backend` (Sync,
  Uring, UringDirect, LibAio; the dry-run fingerprint and 8192 bytes read) fails under io_uring
  without the fix with the message above and passes with it; the unit test
  `uring::tests::a_read_or_write_over_max_rw_count_is_issued_at_it_never_at_its_low_32_bits`;
  `--test all -- uring aio run trace` 37 passed, `--lib` 43 passed.

## 54. The mmap backend issued `fstat` calls the application never makes, timed as reads

- **Where:** `runner/aeiou/src/backend.rs` (`OpenFile::size`, `Mmap::fstat`, `Mmap::mapping`,
  `Mmap::write`/`fallocate`/`ftruncate`, `MmapStats::sized`); runner/REFERENCE.md §9, the
  `mmap` bullet.
- **Found by:** review of the execution engines, then `strace` of `aeiou run` of
  `workloads/model-loading/safetensors-load` (`posix = "mmap"`, the synthetic params).
- **Defect:** `Mmap::mapping` asked `fstat` itself inside the timed `read`: at the first read of
  every open, even right after the abstract's own `fstat` of it, and at every read past the
  mapped length (an `until_eof` loop's last two reads). A config file showed the abstract's two
  `fstat`s plus two of the backend's, a shard its one plus one: syscalls safetensors never makes
  (it maps at the size its `fstat` gave), charged to the reads, against the abstract being the
  application's call stream.
- **Fix:** each open records the size it learned (the abstract's `fstat`, a write through it, a
  truncation); the mapping is made at that size, and the backend asks `fstat` itself only when
  the open has none yet, once per open (the technique's real cost, counted in
  `MmapStats::sized`); a read past the known size returns what `pread` would at that size
  without a syscall; a file grown under the open is mapped again after the abstract's next
  `fstat`, as an application must. The fingerprint (645003b9925d8ef4) and the mmap line (7
  mappings, 417.84 KiB, 284 pages) are unchanged; after the fix the strace shows exactly the
  abstract's calls.
- **Evidence:** `backend::tests::a_mapping_is_made_at_the_size_the_abstract_asked_and_the_backend_asks_only_when_it_did_not`
  and the amended `a_mapping_belongs_to_the_open_file_and_a_grown_file_is_mapped_again` fail on
  the old code and pass on the new; `--test all` 98 passed, `--lib` 43, `test_formats.py` 38
  (including the corpus run under `mmap`). Fixed in an isolated worktree and applied.
  `MmapStats::sized` is not yet printed in the report (queued).

## 55. A multi-host run's merged report counted rank 0's io_uring loops only

- **Where:** `runner/aeiou/src/run.rs` (new `UringReport::merge`, `Report::merge_all`, the doc
  of `Report::uring`); runner/REFERENCE.md §8, the "Knobs" paragraph;
  `runner/aeiou/tests/coord.rs`.
- **Found by:** review of the coordinator/CLI: `merge_all` merged only `in_flight_peak` of the
  `io_uring` section and passed rank 0's `loops` through, while `libaio`'s
  `AioReport::merge` sums its loops.
- **Defect:** `file_per_sample`, `--gpus 3` over two ranks under `--posix io_uring`: the hosts
  reported loops 2 and 1, the merged `result.io_uring.loops` said 2 (with `threads` 3, summed)
  and the merged text line "io_uring: loops 2", so the run-scope report misstated the run.
- **Fix:** `UringReport::merge` sums the loops over the hosts and keeps the largest in-flight
  peak, as `AioReport::merge` does; the knobs and io-wq caps stay rank 0's (a host's tuning and
  facts, each host's in its own report), as documented. No report field, fingerprint, dataset
  id, or AST changes.
- **Evidence:** `coord::the_merged_report_counts_every_hosts_event_loops` (io_uring and libaio,
  3 GPUs over 2 ranks: hosts 2 and 1, merged 3, the peak the hosts' largest, the text line
  "loops 3") fails without the fix ("the run's loops are every host's: … loops 2") and passes
  with it; `--test all -- coord report run` 53 passed, `--lib` 44, builder test_trace,
  test_formats, test_builder 172 passed.

## 56. The busy fraction set unscaled compute against wall-clock stall

- **Where:** `runner/aeiou/src/report.rs` (new `busy`, `take_summary`, `result`),
  `runner/aeiou/src/run.rs` (`Report::time_scale`, `assemble`, `merge_all`, `write_report`);
  runner/REFERENCE.md §3, the "Measurement" bullet; `runner/aeiou/tests/report.rs`.
- **Found by:** review of the coordinator/CLI: §3 defines busy as `compute / (compute +
  stall)`, and `compute` is recorded unscaled while it sleeps `ns × --time-scale`.
- **Defect:** the fraction mixed units, so it depended on `--time-scale` in the wrong way:
  `file_per_sample` (1000 samples, 10 steps) reported busy 0.988 at scale 0.5 where the compute
  slept gives 0.976, and 0.992 at `--time-scale 0`, where nothing was slept and busy is 0.
- **Fix:** `busy(compute, stall, ts) = compute·ts / (compute·ts + stall)` (0 when both are 0),
  used by the JSON summary and the text report (overall and per step bucket, labelled with the
  formula); the report carries `time_scale` (serde default 1.0; every rank has the same, as it
  is in the coordinator's identity). `compute_ns` stays unscaled. No report field removed; no
  fingerprint, dataset id, or AST changes.
- **Evidence:** `report::busy_is_the_compute_slept_against_the_stall` (scales 0, 0.5, 2, the
  JSON and the text) fails when `busy` ignores the scale ("scale 0: busy 0.992…, slept 0 ns
  against stall 3317507 ns gives 0") and passes with the fix; `--test all -- run report coord`
  54 passed, `--lib` 44.

## 57. Under io_uring a run of computes behind its deadlines never caught up (bug 41's test failed on CI)

- **Where:** `runner/aeiou/src/uring.rs` `LoopIo::wait`; runner/REFERENCE.md §8, the
  **Control** bullet; `runner/aeiou/tests/run.rs`
  `consecutive_computes_take_their_sum_on_every_backend`.
- **Found by:** CI after dd94d62 (bug 41): the test failed on GitHub's runner under io_uring
  ("5000 computes of 20000 ns took 584.59 ms, 484.59 ms over their sum"), Sync passing in the
  same run; it passed locally.
- **Defect:** the pacing was right (absolute deadlines, nothing clearing `pace` between
  computes), but the event loop always waited with `submit_with_args(1, timespec)`, even for a
  timer already due: an enter that waits 0 ns still arms an hrtimer and sleeps until its
  interrupt (about 5.7 µs here, about 100 µs on the CI VM, inferred from its log). The blocking
  path does not sleep at all past a deadline, so it catches up after a late wake; the loop paid
  a timer round trip per compute and never caught up, so emulated time again depended on the
  backend and the host.
- **Fix:** when the timeout is zero the loop calls `enter(queued, min_complete 0,
  GETEVENTS [| SQ_WAKEUP])` with no argument: it submits, runs deferred task work
  (`DEFER_TASKRUN` completions are still reaped), and returns without sleeping. Nonzero
  timeouts are unchanged; the aio and `WaitIo` engines already return at once on 0. No
  fingerprint, dataset id, or AST changes.
- **Evidence:** with every sleeping wait forced to at least 100 µs (a scratch simulation),
  Uring failed before ("405.69 ms over their sum") and passed after at 100 µs, 300 µs, and 1 ms.
  The test now also runs 40,000 computes of 2 µs, with a fixed 50 ms bound: it fails on this host
  without the fix ("113.46 ms over their sum") and passes 10/10 with it, and 3/3 under 40 `yes`
  loops; `--test all` 100 passed, `--test uring_knobs --test limits` passed (DEFER_TASKRUN,
  COOP, SQPOLL). Fixed in an isolated worktree and applied.

## 58. Integers outside 64 bits hashed and ran differently in Python and Rust

- **Where:** `runner/aeiou/src/canon.rs` (new `int_range_errors`), `runner/aeiou/src/lib.rs`
  `load_str`, `runner/aeiou/src/eval.rs` (`ParamSet::parse`, `Params::new`), `schema/check.py`
  (new `int_range`), `builder/aeiou/nodes.py` `lift`, `builder/aeiou/validate.py`,
  `builder/aeiou/params.py` `check`; schema/README.md §1 "Integers are 64-bit".
- **Found by:** review of the builder core: Python ints are unbounded and `lift` passed any
  through; the runner's literal is `Int(i64) | Float(f64)` and serde_json reads an integer
  past `u64` as a float.
- **Defect:** with `file_per_sample`'s `step_time` default set out of range, both checkers said
  ok every time, but at 2^63 the runner held `Float(9.223372036854776e18)` (it executes a
  different number than the AST says), and at 2^64, -2^63-1, and 10^30 the AST hash differed
  between Python and Rust (e.g. `c311bb87…` against `c07900cf…`), so a parameter file's
  `ast_sha256` pin or a recorded hash disagreed by language. `lift` also turned an integral
  float such as 1e30 into a 31-digit int. Parameter files and `--param` had the same gap.
- **Fix:** the contract's integers are i64, except a dataset's or namespace's `seed`, u64 (as
  the schema already said). A scanner over the raw JSON text (a parsed value has already made
  the integer a float) refuses every integer token out of range with its path, in the runner's
  every load, its parameter files and `--param`, and in `check.py`, the builder's validation,
  and `aeiou-params check`, word for word alike; `lift` refuses one where the author writes it,
  and an integral float out of range stays a float. A newly refused document; no hash,
  fingerprint, or dataset id changes (all 14 committed ASTs hash as before in both languages).
- **Evidence:** `golden::integers_outside_64_bits_are_refused_as_check_py_refuses_them` (both
  ends of the range, 2^63, 2^64, -2^63-1, a 43-digit number, seeds of 2^64 and -1, values in an
  array and an expression, a params file, `--param`) and
  `test_builder.py::test_integers_outside_64_bits_are_refused_by_builder_checker_and_runner`
  fail before the fix and pass after; golden 18 passed, `--lib` 44, test_builder, test_params,
  test_man, test_usage 114 passed; both checkers ok over every committed AST. Fixed in an
  isolated worktree and applied.

## 59. A libaio run counted the teardown of its AIO contexts as run time

- **Where:** `runner/aeiou/src/aio.rs` `run` (each loop thread), `runner/aeiou/src/run.rs`
  (`run_with`'s `elapsed`, `AioReport::teardown_ns`, `merge`, `describe`); runner/REFERENCE.md
  §9, the `libaio` and **Report.** bullets; `runner/aeiou/tests/run.rs`.
- **Found by:** the fixer of bug 41: libaio runs were 45–59 ms longer than every other backend,
  a fixed amount whatever the computes; `strace` showed `io_destroy` taking about 49 ms.
- **Defect:** `elapsed` was taken after `aio::run` returned, after every loop thread had dropped
  its `AioIo` (whose `Drop` calls `io_destroy`, 35–60 ms per context). A one-compute abstract of
  400 ms gave `elapsed 0.454 s` under libaio against 0.400 under `sync` and 0.403 under
  io_uring, so every libaio figure computed from elapsed was charged a cost that is not the
  workload's. (Per-actor end times are set inside the loop and were right; io_uring's ring
  close costs under 1 ms.)
- **Fix:** each loop thread takes its end instant after its loop runs and before it drops the
  loop, and times the drop as `teardown_ns`; `aio::run` returns the latest end, which
  `run_with` uses for `elapsed`; the teardown is reported on its own (`libaio: … teardown T
  (after the run's end)`, `teardown_ns` in the JSON, `#[serde(default)]`, summed across hosts).
  No fingerprint, dataset id, or AST changes.
- **Evidence:** `run::libaio_teardown_is_not_in_the_elapsed_time` (one 100 ms compute under
  LibAio and LibAioDirect, elapsed under compute + 20 ms) fails without the fix ("took 148.20
  ms, 48.20 ms over it") and passes with it; afterwards libaio and libaio-direct give elapsed
  0.400 s with teardown 0.040 and 0.036 s shown; `--test all -- run aio uring report` 36
  passed, `--lib` 44.

## 60. `x @ i` read a different binding than the checkers proved when a name was shadowed

- **Where:** `runner/aeiou/src/vm.rs` (`at`, the shift's scope `level` and definition `pos`,
  `Binding::def`, `ref` inside a shift), `runner/aeiou/src/validate.rs` and `schema/check.py`
  (a body that binds the `at`'s name twice is refused); schema/README.md V3 ("Which binding").
- **Found by:** review of the runner's semantics: the VM looked `x` up by name through the
  frames, while both checkers resolve it lexically (a `let` of the body the `at` is written
  in, before or after it, else the binding in scope from an enclosing body).
- **Defect:** with an outer loop binding `x = 1000·i` and an inner loop binding
  `let x = cond(j == 0, 0, x @ (j - 1) + 1)`, both checkers said ok, proving V3 against the
  inner `x`, but `aeiou dry-run` evaluated the outer one at the inner index: `o/1001/1000`,
  `o/2001/2000` where the inner chain gives `o/2/1`, `o/3/2` (the soundness gap of bug 39 by
  another road). A re-evaluated definition also read the wrong names: a `let` written before an
  inner `x` read that later `x`, and an outer `let` re-evaluated by `k @ (i - 1)` read an inner
  `c = 500` instead of its own `c = 7`.
- **Fix:** the VM resolves an `at`'s name from the body it is written in (a shift's body under
  a shift), else only from scopes enclosing it, taking the exact `let` that bound it; inside a
  shift a `ref` reads the nearest earlier sibling `let` re-evaluated at the shifted index
  (memoized by position), each sibling's names resolving where it is written. A body that binds
  the name twice is refused by both checkers alike ("`x @` names a binding this body defines
  more than once"). No committed AST has a shadowed or duplicated `let` (458 checked); every
  dry-run fingerprint of every committed AST and parameter set is unchanged.
- **Evidence:** `golden::at_names_the_binding_its_body_shadows` fails with the original
  `vm.rs` (`o/1001/1000` where `o/2/1` is expected) and passes with the fix;
  `test_builder.py::test_at_names_the_binding_its_body_shadows` fails with the original
  `check.py` and passes; on HEAD f018912 plus the change, `--test all` 106 passed, `--lib` 44,
  test_builder and test_params 91 passed. Fixed in an isolated worktree and applied.

## 61. One `consume` under a loop whose count varied drew the same samples

- **Where:** `schema/check.py` and `runner/aeiou/src/validate.rs` (a varying-count record per
  scope, `varies`, the `consume` record, `v6_one_consume`); schema/README.md §4 V6,
  man/aeiou-abstract.7.md V6, the V6 cell of builder/REFERENCE.md's refusal table.
- **Found by:** the fixer of bug 44: the consume position is the row-major ordinal of the
  enclosing frames (runner/REFERENCE.md §2), which assumes fixed counts.
- **Defect:** `loop i < 3 { loop j < 3 - i { stat consume d } }`: both checkers said ok and
  `dry-run --gpu 0` statted `d/f0002` at (0, 2), (1, 0), and (2, 0), the same sample three
  times, the silent reuse of bug 44 from a single `consume`.
- **Fix:** both checkers refuse, with the same message and path, a `consume` under a loop,
  `parallel`, or `loader` whose count (`from`, `to`, `step`, `width`, `batches`) names an
  enclosing index, an `at`, a binding marked varying, or a draw inside a frame; parameters,
  the actor number, `len`/`sum`/`count`/`dirs`/`inventory`, and draws at the actor's top stay
  allowed. The message names the loop and the remedy (a fixed count, the most it can be, with
  a `cond`/`when` inside). Deliberately conservative (`from: i, to: i + 4` is refused too). No
  committed AST is affected (both checkers ok over all 14; `aeiou-build --hermetic --twice
  --check` exits 0); no fingerprint, dataset id, or AST changes.
- **Evidence:** `golden::a_consume_is_under_loops_of_fixed_count` and
  `test_builder.py::test_a_consume_is_under_loops_of_fixed_count` (four refused shapes, three
  accepted; the Python test also holds `aeiou check` to the same message) fail without the fix
  and pass with it; merged with bug 60's checker changes and run on HEAD f9afd6c: golden 20
  passed, test_builder, test_man, test_params 110 passed, both checkers ok over every committed
  AST. Fixed in an isolated worktree and applied.

## 62. The runner's own buffer allocation and page faults were timed as part of ops

- **Where:** `runner/aeiou/src/run.rs` (new `prefault`, `reserve`, `issue_timed`;
  `Ring::ensure`; the timed sites of `Runner::op`, `trace_issue`, `trace_group`),
  `runner/aeiou/src/uring.rs` (`Pool::get`, the object loop's `buffer`, `issue`'s clock,
  `inline_op`), `runner/aeiou/src/aio.rs` (`issue`'s clock, `inline_op`); runner/REFERENCE.md,
  both **Buffers** bullets.
- **Found by:** review of the execution engines: `issue_blocking` sliced the thread's ring,
  allocating or growing it, after every caller had started the op's clock, and no buffer was
  touched when allocated.
- **Defect:** the first op of a size, and every op landing on fresh pages, was charged the
  runner's allocation and page faults: one actor writing 256 MiB in 32 MiB ops and reading it
  back four times saw its slowest read under `sync` at 3.74× the mean of the others (19.3 ms
  against 5.2 ms) and its slowest write at 2.16×; the event loops took their clock before
  `pool.get` (writes 1.6–1.9×). Runner overhead in op latency, against charging a backend only
  its API's costs; `trace_group`'s member threads paid it per member through their fresh rings.
- **Fix:** a ring is pre-faulted (one write per 4 KiB) when allocated or grown, and so is a new
  pool chunk or object buffer; `issue_timed` sizes the ring for the op (`reserve`: the issued
  length, the `O_DIRECT` round-out) and only then starts the clock, at every timed site; the
  loops start an op's clock once it has its chunk. A write's payload fill stays inside the
  timer, as before. No fingerprint, dataset id, or AST changes.
- **Evidence:** the same measurement after: `sync` 1.45× and 1.20× (reads), 1.18× and 1.07×
  (writes), io_uring writes 1.05×; `run::tests::runner_buffers_are_prefaulted_when_allocated`
  (`mincore`: a first ring, a grown ring, a pool chunk resident; 0 pages before) and
  `run::the_first_large_read_is_not_charged_the_rings_growth` (best of 3, slowest 16 MiB read
  under 2.5× the mean; 3.10× before) fail on the old code and pass on the new; `--test all --
  run trace uring aio` 39 passed in 4 of 4 runs, `--lib` 45, builder test_object and test_trace
  97 passed, MinIO object tests 6.

## 63. An io_uring loop could end with requests in the kernel that write into freed memory

- **Where:** `runner/aeiou/src/uring.rs` (new `Engine::drain`, `LoopIo::drain`,
  `LoopIo::efd_pending`, the `CANCEL` user_data, `Loop::run`, the `undrained` test hook);
  runner/REFERENCE.md §8, the **Buffers** bullet; `runner/aeiou/tests/run.rs`.
- **Found by:** review of the execution engines (suspected), then verified: `Loop::run` returned
  `serve()`'s result at once, and the `Loop`'s fields drop the ring first, then the tasks, then
  the eventfd's buffer; closing a ring fd only begins the kernel's teardown, which cancels and
  reaps its requests later, on a worker.
- **Defect:** a loop that ended with requests in the kernel (another actor failed, the run was
  aborted, or a success with the eventfd read still posted) freed what they name while the
  kernel could still write into it: each task's `statx` box (an `IORING_OP_STATX` in the
  io-wq), the loop's `efd_buf` (the read posted while a task waits at a barrier or an object op
  is in flight), and `cpath` under SQPOLL. A use-after-free by the kernel: silent heap
  corruption. (Read and write chunks were leaked, not freed; libaio is safe, its `io_destroy`
  waiting out every request before the tasks go.)
- **Fix:** before it lets go of anything, a loop drains its ring: it asks
  `IORING_OP_ASYNC_CANCEL` for every task in `Wait::Io` and for the eventfd read, and waits
  until nothing is in the kernel (a request that cannot be cancelled is waited out), on the
  error and the success paths alike; should the drain itself fail (no async cancel before Linux
  5.5), the tasks and the eventfd buffer are leaked rather than freed. No fingerprint, dataset
  id, or AST changes.
- **Evidence:** `run::a_failed_io_uring_loop_drains_its_ring_before_it_ends` (two GPUs on one
  loop: one waits at a barrier with the eventfd read posted, the other fails on an allocation;
  `undrained()` counts rings closed with requests in the kernel) fails with the drain disabled
  ("a ring was closed with a request still in the kernel, left: 1") and passes with it;
  `--test all` 113 passed, 1 ignored; `--lib` 60; builder test_trace 82 passed.

## 64. An HDF5 datagen refused for its header tool left an empty dataset root

- **Where:** `runner/aeiou/src/datagen.rs` `datagen` (the roots' `create_dir_all` moved from
  the plan loop to after the headers); `runner/aeiou/tests/layout.rs` (the test un-ignored).
- **Found by:** the coverage tests of `formats.rs` (8ca2bd3), which marked
  `layout::hdf5_without_its_header_tool_refused_before_anything_is_written` ignored.
- **Defect:** the plan loop created each directory root before rank 0 made the HDF5 headers,
  so a missing or failing `aeiou-header` ("not found: aeiou-header makes HDF5 headers and must
  sit beside aeiou", or a version it refuses) exited 1 with `out/h` left behind, empty: short of
  the standard of bugs 2 and 46 that a refused datagen leaves nothing.
- **Fix:** the roots are created after every refusal the plan and the headers decide (after
  `make_headers`, the resume check that the headers match the corpus's, and the share to the
  other ranks), still before any manifest is written; the empty-root checks are unmoved and the
  message is the same. No fingerprint, dataset id, or AST changes.
- **Evidence:** the un-ignored test fails on the old code ("left behind: [\"h\"]") and passes;
  on HEAD 455fa1d plus the change, `--test all -- layout datagen coord` 35 passed, and the
  builder files that run datagen (test_formats, test_payload, test_object, test_trace,
  test_coord, test_launch, test_options, test_usage, test_man) 176 passed. Fixed in an isolated
  worktree and applied.

## 65. An S3-only event loop busy-spun to every timer under 1 ms

- **Where:** `runner/aeiou/src/uring.rs` `WaitIo::wait` (the engine of a loop with only S3
  names under `--s3 async`).
- **Found by:** the fixers of bugs 41 and 57, noted and queued.
- **Defect:** the wait passed `timeout.as_millis()` to `poll`, so a timer under 1 ms (a
  `compute` of < 1 ms, a trace gap) became `poll(0)` and the loop spun to its deadline, and
  1.9 ms was cut to 1 ms: 300 waits of 200 µs used 60.19 ms of CPU in 60.21 ms, a whole CPU
  taken from the host under test and from the runtime's threads carrying the S3 requests.
- **Fix:** `ppoll` with a nanosecond `timespec` from the `Duration`; a zero timeout (a timer
  already due) still returns without sleeping, and with the 1 ns timer slack of bug 41 the
  wait wakes on time. No fingerprint, dataset id, or AST changes.
- **Evidence:** `uring::tests::an_s3_loop_sleeps_to_a_sub_millisecond_timer_never_spins_to_it`
  (500 µs and 1.9 ms waits last that long, 300 × 200 µs use under half their wall time in
  thread CPU, an eventfd write still wakes it) fails before ("woke after 4.641µs" on a 500 µs
  wait) and passes after, 5 of 5; `--lib` 63 passed, MinIO object tests 6 passed.

## 66. The stream abstracts failed when `batch` did not divide `samples_per_file`

- **Where:** `workloads/input-pipeline/stream/tfrecord/tfrecord_stream.py` and
  `workloads/input-pipeline/stream/parquet/parquet_stream.py` (the training loop's take), their
  regenerated ASTs, catalog pages, and shapes; `runner/aeiou/tests/golden.rs` (the two AST
  hashes); `runner/aeiou/tests/run.rs`.
- **Found by:** review of the runner's datagen/metrics: a cryptic error with
  `samples_per_file < batch`; the fixer found a second failure the review had not.
- **Defect:** the training loop took a shard `every(samples_per_file // batch)` steps. With
  `batch > samples_per_file` the period was 0: `aeiou dry-run` failed "mod: overflow or
  division by zero (0, 0)". With `samples_per_file` not a multiple of `batch` (1000 and 256),
  the floored period took more shards than the loader dispatches
  (`ceil_div(steps · batch, samples_per_file)`): the dry run passed and `aeiou run` failed
  "take `shards`: take after the last of 6 items". The parameter's doc said "a multiple of
  batch", but nothing stated or checked it, and the contract has no construct to refuse it.
- **Fix:** each step takes the shards whose first sample falls in its batch,
  `ceil_div((step + 1)·batch, spf) − ceil_div(step·batch, spf)` of them, which totals the
  loader's dispatch for any two positive values and, when `batch` divides `samples_per_file`,
  fires at exactly the old steps. **The two ASTs' hashes change** (tfrecord_stream
  `22929c8b95c0…`, parquet_stream `863e072da997…`); their dry-run fingerprints for every
  parameter set that ran before are unchanged (five sets each, the golden fingerprints
  included). The catalog's measured shapes and op counts are unchanged.
- **Evidence:** `run::stream_abstracts_take_every_shard_for_any_batch` (both abstracts at
  `samples_per_file` 128 with batch 32, 48, 200: datagen and a run whose takes equal
  `ceil(steps·batch/128)` and whose fingerprint, ops, and bytes equal the dry run's) fails with
  the old ASTs ("take after the last of 3 items") and passes; on HEAD: `aeiou-build --hermetic
  --twice --check` over every workload exits 0, `aeiou-catalog check` "13 mechanisms, 16
  cases: ok", golden 21 passed. Fixed in an isolated worktree and applied.

## 67. Integer arithmetic past 64 bits wrapped silently in the geometry, loops, and offsets

- **Where:** `runner/aeiou/src/eval.rs` (new `ceil_div`; `align_up`, `Framing::row_bytes`,
  `file_size`, `sample_offset`, `col_lens`, `col_offset_in_unit`, `unit_len`, `data_start`,
  `unit_offset`, `file_size_from_units`, the array slot offset, `build_model`,
  `Value::as_int`), `runner/aeiou/src/vm.rs` (loop trip counts, `frame_ordinal` and the
  `consume` position, `div`/`mod`/`ceil_div`, read/write offset bounds `past_end`,
  `add_written`, `lseek`, unit and column offsets, float namespace fields),
  `runner/aeiou/src/project.rs` (the projection's sums); schema/README.md §1 "Integers are
  64-bit".
- **Found by:** review of the runner's semantics (the expression arithmetic was found already
  checked; the wraps were in the geometry and the loops).
- **Defect:** release builds wrap: a compound file of 4 samples of 2^62 bytes came out 0 bytes
  (`read d/f0 … -> 0`, "total 0 B"); a loop from -(2^63-1) to 2^63-1 step 2^62 ran 0 times
  (fingerprint 0000000000000000) instead of 4; an lseek to 2^63-8 and two writes of 100 put
  the second at `off=-9223372036854775716`; an array of 4 slots of 2^62 had a negative size;
  `div`/`mod`/`ceil_div` of -2^63 by -1 panicked; a float past i64 saturated to 2^63-1.
- **Fix:** every computed sum, product, count, and offset is checked, refusing the run with the
  dataset or op named ("the size of file 0 overflows the 64-bit integers of the contract");
  trip counts are computed in i128 (the index steps exactly); `div`/`mod`/`ceil_div` use
  `checked_*_euclid`; a read or write at a negative offset or past 2^63-1 is refused, the
  kernel's `rw_verify_area` rule; an array or inventory whose size overflows is refused at
  model build; a float outside i64 where an integer is needed is refused. Nothing changes for
  values that do not overflow: 29 dry-run fingerprints over every committed AST and parameter
  set identical before and after; no measurable slowdown (hdf5_map 2.29 s / 2.28 s,
  lmcache_local_disk 68.81 s / 68.01 s).
- **Evidence:** `golden::integer_arithmetic_past_64_bits_is_refused_not_wrapped` (file size,
  array, inventory, loop, write, write repeat, lseek, div/mod/ceil_div, float, and an edge case
  that must still run) fails on the old code and passes; on HEAD plus the change `--test all`
  116 passed, `--lib` 61, the builder suite 358 passed, `aeiou-catalog check` ok. Fixed in an
  isolated worktree and applied.

## 68. The mmap backend's own `fstat` calls were counted but never reported

- **Where:** `runner/aeiou/src/run.rs` (`MmapReport::fstat_calls`, its merge, `describe`,
  `assemble`); runner/REFERENCE.md §9, the **Report.** bullet; `runner/aeiou/tests/report.rs`.
- **Found by:** the fixer of bug 54 (queued), and independently the regression review of the
  RC1 fixes.
- **Defect:** bug 54 counts in `MmapStats::sized` the `fstat` the backend issues itself (an
  open read before the abstract asked its size), a syscall the application did not make, timed
  inside a read; `Report::mmap` never read the counter, so no run could show it (the JSON had no
  field, the `mmap:` line no figure).
- **Fix:** the report carries it: `fstat_calls` in the JSON's `mmap` object
  (`#[serde(default)]`, summed across hosts) and `backend fstat N` on the `mmap:` line,
  documented in §9. No field removed; no fingerprint, dataset id, or AST changes.
- **Evidence:** `report::the_mmap_backend_s_own_fstat_calls_are_reported` (a `file_per_sample`
  variant opening and reading 30 files without `fstat` under `--posix mmap` reports 30, one per
  open, in the JSON and the text; `safetensors_load` reports 0) fails before (`left: Null`) and
  passes after; `--test all -- report run mmap` 41 passed on HEAD plus the change,
  `test_formats.py` 38 passed. Fixed in an isolated worktree and applied.

## 69. The trace's root check placed reads after a short write wrongly, and a failed `lseek` reset the position

- **Where:** `runner/aeiou/src/trace.rs` (new `moved`, used by `TraceFile::index` and
  `build`); runner/REFERENCE.md §13; `runner/aeiou/tests/trace.rs`.
- **Found by:** the regression review of the RC1 fixes (bug 27 had changed `build` only); the
  fixer's audit of `index` against `build` found the `lseek` rule.
- **Defect:** (1) `index`, which `check_root` uses, moved a no-offset write's position by its
  `len`, while `build` (since bug 27) moves it by the traced count: a trace writing 1000 bytes
  that returned 10 and then reading 100 dry-ran at `off=10` but `aeiou run --root` refused a
  valid 200-byte file, "the trace reads to 1100". (2) Both set the position to an `lseek`'s
  `ret.count().max(0)`, so a failed `lseek` (`EINVAL`) moved it to 0, where the kernel, the
  Python reader, and a run under a non-positional backend leave it: later sequential offsets
  in the dry run and the fingerprint were wrong.
- **Fix:** one position rule, `moved`, for both: a read or write without an offset moves by the
  traced count (checked for overflow), an `lseek` sets what it returned, a failed one leaves the
  position. **The fingerprint changes** for a trace file with a failed `lseek` followed by a
  sequential op on that open; no committed trace or fixture has an `lseek`, and every existing
  trace fingerprint is unchanged. No AST or dataset id changes.
- **Evidence:** `trace::the_root_check_places_a_read_where_the_walk_does_after_a_short_write_or_a_failed_lseek`
  (write 1000 → 10, read 100, failed lseek, read 50: reads at 10 and 110, a 159-byte file
  refused "reads to 160", a 160-byte one run under sync, io_uring, libaio to the dry run's
  fingerprint) fails with the old `trace.rs` (the read after the failed lseek at off=0); the CLI
  repro "reads to 1100" before, the dry run's fingerprint after; `--test all -- trace` 11
  passed, `--lib` 61, test_trace.py 82.

## 70. `aeiou-params check` judged some rules on the defaults, passing files the runner refuses

- **Where:** `schema/check.py` (new `Check.param_value`, used by `run`'s per-parameter check,
  `at_least`, `number`, and `least`), `builder/aeiou/params.py` (`Given` overrides
  `param_value` only); `builder/tests/test_params.py`.
- **Found by:** the regression review of the RC1 fixes (bugs 5 and 39): `Given` overrode only
  `at_least`, while `number()`, `least()`, and `exact()` read `ast.params[...].default`, and the
  runner's `check_given` reads the given value.
- **Defect:** a given scalar was never judged in Python by the normal/lognormal min > max rule,
  the V3 proof that rule feeds, or V25: with `max = param hi` given 0, `aeiou-params check`
  said ok while `aeiou dry-run --params-file` refused "normal: min 1 is above max 0" and the
  `x @` proof; with a dataset size `param sz` given -5, ok against the runner's V25 refusal. The
  tool's documented promise is to refuse every file `run --params-file` refuses.
- **Fix:** one hook for a parameter's value in effect (the default in `Check`, the given value
  in `Given`), read by every rule the runner's `check_given` applies; V23's divisor check keeps
  the default on purpose, as the runner does. No fingerprint, dataset id, or AST changes.
- **Evidence:** `test_params.py::test_given_scalars_are_judged_by_every_rule_that_reads_them`
  (hi 0, 9, 1; sz -5, 8192, 0: `params.check`, `aeiou-params check`'s exit, and the runner's
  words alike) fails 2 of 6 before and passes 6 after; test_params, test_builder, test_man 116
  passed; both checkers ok over every committed AST; all 22 committed parameter sets accepted by
  both tools. (The test resolves `AEIOU_RUNNER` before running in its temporary directory, since
  CI gives a path relative to builder/.) Fixed in an isolated worktree and applied.

## 71. V25 rounded and printed sizes differently in the two checkers, and check.py crashed on a non-finite one

- **Where:** `schema/check.py` (`round_away`, new `integral`, `size_not_negative`),
  `runner/aeiou/src/validate.rs` (`size_not_negative`); schema/README.md §4 V25.
- **Found by:** the regression review of the RC1 fixes (bug 48's V25).
- **Defect:** check.py rounded with `int(abs(x) + 0.5)`, the pattern bug 14 removed from
  `rng.rust_round`, and the runner printed `v.round() as i64`, saturating. A size of
  `-0.49999999999999994`: Python refused "it can be -1", Rust ok (the runner rounds it to 0);
  `-4503599627370497.0`: -…498 against -…497; `-1e20`: -100000000000000000000 against
  -9223372036854775808; `mul(-1e308, 10)` and `inf − inf`: a Python traceback ("cannot
  convert float infinity/NaN to integer", from the builder's `validate()` too) against Rust's
  i64::MIN and ok.
- **Fix:** check.py rounds with `rng.rust_round`'s algorithm (Rust's `f64::round`), and both
  print the rounded value unsaturated (`{:.0}` in both); a size whose least value is not
  finite is refused in both, "a size that is not a finite number (V25)". No committed AST is
  affected; no fingerprint, dataset id, or AST changes.
- **Evidence:** `golden::a_dataset_size_is_judged_and_printed_as_check_py_judges_and_prints_it`
  and `test_builder.py::test_a_dataset_size_is_judged_and_printed_as_the_runner_judges_and_prints_it`
  fail on the old code and pass; the five repros give identical lines in both checkers after.

## 72. The normal/lognormal min > max rule compared and printed differently in the two checkers

- **Where:** `schema/check.py` (`Check.dist`, `Check.at_least`), `runner/aeiou/src/validate.rs`
  (`bounds`).
- **Found by:** the regression review of the RC1 fixes (bug 39's rule).
- **Defect:** Rust printed floats with `Debug` and compared in f64; check.py printed `f"{x}"` and
  compared ints exactly: min 1e20 over max 1e19 read "1e20 … 1e19" in Rust and "1e+20 …
  1e+19" in Python, 1e-5/1e-7 "1e-5"/"1e-05", and min 9007199254740993 over max
  9007199254740992 was refused by Python and accepted by Rust.
- **Fix:** Rust prints with `canon::python_float_repr`, and both compare in f64, as the runner's
  clamp (`eval::clamp`) does at run time, where 2^53+1 and 2^53 are one value; the V3 belief
  that a `min` bounds a draw only under a `max` not below it compares the same way. No
  fingerprint, dataset id, or AST changes.
- **Evidence:** `golden::min_above_max_is_judged_and_printed_as_check_py_judges_and_prints_it`
  and `test_builder.py::test_min_above_max_is_judged_and_printed_as_the_runner_judges_and_prints_it`
  (also in a V3 `at` chain) fail on the old code and pass; both bugs on HEAD 0bee688 plus the
  change: golden 23 passed, test_builder, test_params, test_man, test_describe 119 passed, both
  checkers ok over every committed AST. Fixed in an isolated worktree and applied.

## 73. `aeiou datagen --root NAME` refused a bare relative root that did not exist yet

- **Where:** `runner/aeiou/src/datagen.rs` `free_space`; `runner/aeiou/tests/options.rs`.
- **Found by:** the regression review of the RC1 fixes (pre-existing at the tag).
- **Defect:** the space check walks up the root's parents to find a directory to `statvfs`;
  the parent of a bare relative name is the empty path, which does not exist, and the next is
  `None`, so `aeiou datagen --root dg1 …` failed "dg1/train: no existing directory above it",
  while `--root ./dg1` and an absolute path worked.
- **Fix:** an empty parent is the current directory. The other parent walks (the abstract's
  directory, `open_for_write`, `prune`) were checked and are right. No fingerprint, dataset
  id, or AST changes.
- **Evidence:** `options::datagen_takes_a_bare_relative_root_that_does_not_exist_yet` (the binary
  run in a temporary directory with `--root dg1`) fails before with the message above and passes
  after; `--test all -- usage options datagen` 20 passed, `--lib` 61.

## 74. The two checkers judged a size that is not a number by the order of its operands

- **Where:** `schema/check.py` and `runner/aeiou/src/validate.rs` (V25: `exact` `min`/`max`,
  the `normal`/`lognormal` clamp, `mixture` in `least`); `schema/README.md` §4 V25.
- **Found by:** the regression review of the RC1 fixes (bugs 71 and 72's rule).
- **Defect:** a NaN (an overflow such as `1e308*10 - 1e308*10`) reaching a `min`/`max` node, a
  clamp, or a mixture arm was dropped by Rust's `f64::min`/`max` and kept or dropped by
  Python's `min`/`max` according to argument order; a mixture also returned at its first
  unbounded arm. `mixture [NaN, 5]` was refused by check.py and accepted by `aeiou check`,
  `[5, NaN]` accepted by both, `max(NaN, -5)` refused as "not finite" by one and "-5" by the
  other; a `+inf` arm was accepted by both.
- **Fix:** `least_of`/`most_of` (NaN when either side is) in both checkers for the `min`/`max`
  nodes and the clamp; a mixture with a positive-weight arm whose least is not finite is not a
  finite number, and "unbounded" is decided after every arm, so arm order never matters. No
  fingerprint, dataset id, or AST changes; every committed AST is still accepted.
- **Evidence:** `golden::a_size_that_is_not_a_number_is_refused_whatever_the_order` and
  `test_builder.py::test_a_size_that_is_not_a_number_is_refused_whatever_the_order` (every
  mixture pair both ways, `min`/`max` both ways, three clamps, a weight-0 NaN arm still
  accepted) fail on the old code and pass; on HEAD df06a74 plus the change: golden 24 passed,
  test_builder and test_params 101 passed, check.py ok over every committed AST.

## 75. A channel declared inside a `parallel` body failed or hung by timing (regression of bug 37)

- **Where:** `schema/check.py` and `runner/aeiou/src/validate.rs` (new rule V26,
  `declared_outside_forks` in `loader`/`channel`); `runner/aeiou/src/run.rs` `Control::Channel`
  (comment); `schema/README.md` §3, §4; `man/aeiou-abstract.7.md`; `runner/REFERENCE.md` §4.
- **Found by:** the regression review of the RC1 fixes: bug 37 lets a channel be declared
  again in each iteration of a loop, and a declaration inside a fork body is made once by each
  sub-actor over the shared `Instance.channels`.
- **Defect:** `parallel width 2 { channel c; put c; take c }` passed both checkers, then
  `aeiou run` failed "channel c declared again with 1 item(s) put and not taken" under sync and
  io_uring, hung under sync at `--time-scale 0`, and failed "nothing will complete" under
  io_uring at time scale 0: which sub-actor's declaration replaced which depended on timing.
  A loader inside a `parallel` did the same.
- **Fix:** a channel is the actor's, shared by its sub-actors (as the docs already said); both
  checkers refuse a `channel` or `loader` inside a `parallel` or `loader` body at any depth,
  with one path and message (V26, contract 0.10, no committed AST affected). The per-iteration
  redeclaration bug 37 allowed is kept. No fingerprint, dataset id, or AST changes.
- **Evidence:** `golden::a_channel_or_loader_inside_a_fork_is_refused`,
  `run::a_channel_declared_inside_a_fork_is_refused_whatever_the_timing` (the intended shape
  runs with 6 puts and 6 takes under sync and io_uring at time scale 0 and 0.2), and
  `test_builder.py::test_a_channel_or_loader_inside_a_fork_is_refused` fail on the old code
  and pass; on HEAD 3f2c3cc plus the change: `--test all -- run golden loader parallel man` 68
  passed, `--lib` 61, test_builder, test_catalog, test_man, test_params 172 passed, both
  checkers ok over every committed AST. Fixed in an isolated worktree and applied.

## 76. The Python tools read `NaN`, `Infinity`, and `1e400` in JSON that the runner refuses

- **Where:** `schema/check.py` (`loads`, `main`, `int_range`); `builder/aeiou/emit.py`
  (`loads`, `load`); `builder/aeiou/params.py` `load`; `builder/aeiou/catalog.py` (four
  readers); `builder/aeiou/header.py` `--format`; `builder/aeiou/cli.py` (hermetic child).
- **Found by:** the RC1 fix of bug 74 (its out-of-scope note): Python's `json` takes the
  non-standard `NaN`, `Infinity`, `-Infinity` and turns `1e400` into an infinity, where the
  runner's serde_json refuses all four at parse time.
- **Defect:** `aeiou-params check` accepted `{"params": {"step_time": Infinity}}` (ok, exit 0)
  while `aeiou dry-run --params-file` refused it "not JSON"; `check.py` died with a traceback on
  such an AST (and on any invalid JSON) where `aeiou check` said "not an abstract". A
  non-finite float authored where `lift()` does not look (a framing column's `weight`) failed
  `aeiou-build` with a ValueError traceback in process, and the hermetic child wrote it with
  `allow_nan=True` and built outside its `BuildError` handler.
- **Fix:** a `loads` (parse hooks refusing the constants and an overflow to infinity) in
  check.py and in `emit`, used by every reader of an abstract or a parameter file in the
  builder's tools, each giving its existing invalid-JSON error; check.py reports unreadable
  JSON as `not an abstract (…)`, exit 1, as `aeiou check`; `int_range` refuses a non-finite
  float with its path in an AST the builder made; the hermetic child builds inside its handler
  and writes with `allow_nan=False`. No fingerprint, AST hash, or contract change: check.py's
  output over every committed AST is unchanged and `emit.loads` equals `json.loads` on them.
- **Evidence:** `test_builder.py::test_non_json_numbers_are_refused_as_the_runner_refuses_them`
  (5 literals, scalar/array/object, through check.py, `emit.load`, `aeiou check`),
  `test_builder.py::test_non_finite_floats_authored_are_refused_with_their_path` (in process and
  `--hermetic`), `test_params.py::test_non_json_numbers_in_a_parameter_file_are_refused` (4
  literals, `params.load`, `aeiou-params check`, `aeiou dry-run --params-file`): 10 failed on
  the old code, pass after; on HEAD 1d91efe plus the change: test_builder, test_params,
  test_man, test_catalog, test_trace, test_usage, test_formats 308 passed, CI's
  `aeiou-build --hermetic --twice --check` over every workload script rc 0, check.py ok over
  every committed AST.

## 77. A channel a loader's workers use, declared again under them, failed by timing

- **Where:** `schema/check.py` and `runner/aeiou/src/validate.rs` (V26's second clause:
  `channel_sites`, `used_by_workers`, called from `put`/`take`); `schema/README.md` §4 V26;
  `runner/REFERENCE.md` §4; `man/aeiou-abstract.7.md`.
- **Found by:** the RC1 fix of bug 75 (its adjacent note): bug 37 lets the actor declare a
  channel again in each iteration of a loop, and a loader's workers run beside the actor.
- **Defect:** `channel c; loader q {compute; put c}; loop e<4 {channel c; take q; take c}`
  passed both checkers and dry-run, then `aeiou run` failed "channel c declared again with 2
  item(s) put and not taken" in 4/4 runs at `--time-scale 0` under sync and io_uring and passed
  4/4 at time scale 1: which declaration a worker reached depended on timing. (`parallel` joins
  before the actor goes on and is not affected; a loader's workers touch channels only while
  its batches are untaken.)
- **Fix:** a `put` or `take` in a loader body (any depth) names a channel declared at one site
  (counting `channel` and `loader` nodes), that site outside every loop or in the body the
  loader is in with the loader declared at one site too; both checkers give one path and
  message. No committed AST puts or takes in a loader body. No fingerprint, dataset id, or AST
  changes.
- **Evidence:** `golden::a_channel_a_loaders_workers_use_is_declared_once_or_beside_it` (3
  accepted, 7 refused shapes), `run::a_channel_a_loaders_workers_use_is_not_declared_again_under_them`
  (the repro refused; both good shapes run with 12 puts and 24 takes under sync and io_uring at
  time scales 0 and 0.2), and `test_builder.py::test_a_channel_a_loaders_workers_use_is_declared_once_or_beside_it`
  fail with the rule disabled and pass; on HEAD ff4bd2b plus the change: `--test all` 126
  passed, `--lib` 61, test_builder, test_man, test_params 131 passed, `aeiou-build --hermetic
  --twice --check` rc 0, both checkers ok over every committed AST. Fixed in an isolated
  worktree and applied.

## 78. Every op on a container sample redrew the size of every sample in its file

- **Where:** `runner/aeiou/src/eval.rs` (`DsMeta::file_geom`, `FileGeom`); `runner/aeiou/src/vm.rs`
  (`Vm::geom`, `file_geom`, `file_size`, `sample_offset`; `target`, `size_of`, `offset_of`).
- **Found by:** the runner semantics review (S5, performance): `Vm::target` called
  `DsMeta::file_size(file)` and an `offset` called `DsMeta::sample_offset(id)` on every op, and
  both draw the size of every sample of the file.
- **Defect:** each op cost O(samples_per_file), so reading a file's samples one by one cost
  O(spf²): hdf5_map (open, fstat, nine reads, close per sample) dry-ran 20 steps in 10.0 s at
  spf 100000 against 0.05 s after; a one-file pass over 16000 TFRecord-framed samples took
  67.5 s, and 64000 more than 300 s (×16 per ×4 samples).
- **Fix:** `DsMeta::file_geom` computes a file's size and every sample's offset in one pass
  (two under a framing) with the same checked arithmetic; each VM holds the geometry of the
  files it addressed last, bounded at 16 files and 2^20 offsets, never a table over the
  dataset, never shared, cleared on `resume_from` (as `unit_lens`). One sample per file and
  array datasets bypass it; a refusal falls back to `DsMeta` for the same error. A shuffled
  `map` over more than 16 files still pays one O(spf) pass per visit, which the per-file-table
  invariant requires. No fingerprint, dataset id, AST, or datagen byte changes.
- **Evidence:** `eval::tests::a_pass_over_a_files_samples_draws_their_sizes_a_bounded_number_of_times`
  (at most 4n size draws per pass over n=400 samples, packed and framed, every read's offset and
  length equal to `DsMeta`'s, a 20-file shuffled variant past the cap) failed on the old vm.rs
  with 880200 draws and passes; `file_geom_is_file_size_and_sample_offset`; 67 of 76 committed
  (AST, params, gpus) fingerprints compared identical before and after (9 over 120 s with both
  binaries), datagen trees identical; on HEAD e06816e plus the change: `--lib` 63 passed,
  `--test all` 126, test_formats and test_builder 104, hdf5_map, parquet_stream,
  tfrecord_stream fingerprints at 1 and 4 GPUs unchanged. Fixed in an isolated worktree and
  applied.

## 79. Several producers on a small channel hung or finished as the threads happened to run

- **Where:** `runner/aeiou/src/run.rs` (channels: `Seqs`, `admits`, `Watch`, `Channel::park`/
  `unpark`/`wake`, `Instance.watch`, `Join.parked`, `spawn_sub`, the `parallel` and `loader`
  forks); `runner/aeiou/src/uring.rs` (`Chan::try_put`, `ChanWait::Put`); `schema/README.md`
  §3, `man/aeiou-abstract.7.md`, `runner/REFERENCE.md` §4 (the `put`/`take` rows).
- **Found by:** the RC1 fix of bug 77 (a repro that hung 3 in 32 runs).
- **Defect:** `put` waited while `capacity` items were held, so with several producers a later
  item could take the room an earlier one needed while the consumer waited for the earlier
  one. Loader workers putting on a channel of 2 with 2 workers × prefetch 2 hung 7 in 12 runs
  under sync at `--time-scale 0` and passed at 1; the loader written out in primitives
  (`GRAMMAR_OPTIONS.md` §3) and wider shapes hung 12/12 under sync and failed 12/12 under
  io_uring ("nothing will complete"). The thread runner hung where the event loop failed.
- **Fix (runtime semantics, recorded in the docs, 2026-10-15):** an item's `seq` is its position
  on the channel, and `put` waits for its window, `seq < taken + capacity`, as a loader's batch
  waits for its slot (`admits`, shared by both engines); at most `capacity` are still held.
  The thread runner now reports a wait every thread of the actor is in, each parked on one of
  its channels with nothing left to change one (`Watch`), as the event loop did; a thread in
  compute, I/O, or a barrier counts as live, so it never fires on a run that could finish. A
  loader worker's slot wait that fails for a reason other than a closed channel returns the
  error. No committed workload uses `channel`/`put`. No fingerprint, dataset id, or AST
  changes (channel operations are never hashed).
- **Evidence:** `run::a_put_waits_for_its_position_so_no_run_hangs_by_timing` (four shapes, both
  engines, time scales 0 ×3 and 0.0001) failed "time scale 0: hung" on the old run.rs/uring.rs
  and passes; the shapes passed 32/32 per engine and time scale after; on HEAD 2cc99f5 plus the
  change: `--lib` 63, `--test all` 128 passed three times in a row, builder suite 365 passed
  (14 object tests skipped without MinIO; run in the final check), `aeiou-build --hermetic
  --twice --check` rc 0. Fixed in an isolated worktree and applied.

## 80. A `seq` put twice on a channel waited forever instead of failing

- **Where:** as bug 79 (`Seqs::check`; the error's context in `Control::Put` and `ChanWait::Put`).
- **Found by:** the RC1 fix of bug 77.
- **Defect:** an ordered channel declared once outside a loop whose loader put `seq b` again
  each epoch waited forever under sync (4/4 runs) for an item numbered past the repeats, and
  failed with only "2 actor(s) wait on channels … nothing will complete" under io_uring; an
  unordered one kept the repeated item once, so it worked only when drained in between.
- **Fix:** a `seq` below 0, or one put before on the channel while it is declared, is a run-time
  error naming the put, the channel, and the seq, with the remedy (number the items across the
  loop, `e·n + b`, or declare the channel in it). The record is bounded by the channel's
  capacity. The bug-77 test's good shape now numbers its items `e*4+b`.
- **Evidence:** `run::a_repeated_seq_and_a_wait_nothing_ends_are_errors_not_hangs` (repeated seq
  and `seq -1` at two time scales, three stuck shapes, both engines) failed "hung" on the old
  code and passes; suites as bug 79.

## 81. An `x @ i` in a short loop nested in a long one sorted its whole cache on every insert

- **Where:** `runner/aeiou/src/vm.rs` (`AtCache`, `AtKey`; `Vm::at`, `Vm::at_cache`).
- **Found by:** the runner semantics review (S4, performance). Measured before fixing: for an
  ordinary `x @ (i-d)` chain the old eviction (sort the indices at 65,536 entries, keep those
  `>=` the median) was already amortized O(log n), as the review had not seen; the defect is
  the case where most keys share one index.
- **Defect:** the key is (definition, index, enclosing indices), so in
  `loop j<N { loop i<2 { x = cond(i>0, x @ (i-1)+1, j) } }` the index repeats and only `j`
  moves; every entry equal to the median was kept, nothing was shed, and from 65,536 entries
  every insert sorted the whole cache, which grew without bound: `aeiou dry-run` took 0.08 s
  at N=60k, 17.7 s at 100k, 84.6 s at 150k (hours at 1e6); `--metrics` at 100k 23.8 s.
- **Fix:** each entry carries its insert sequence; a full cache sheds exactly half, ordered by
  (index, sequence), the same smaller-indices-first policy with ties shed oldest first, the cut
  found by `select_nth_unstable` (O(n)), so an insert is O(1) amortized and the cache stays at
  `AT_CACHE_MAX`. The cache memoizes a pure function, so no value changes; N=100k 0.13 s,
  1e6 1.1 s; wide-`d` chains are no slower. No fingerprint, dataset id, or AST changes.
- **Evidence:** `golden::an_at_in_a_short_loop_nested_in_a_long_one_stays_linear` (200,000 × 2,
  values checked, under 20 s) failed on the old vm.rs in 159 s and passes in 0.59 s; unit tests
  `the_at_cache_is_bounded_when_indices_repeat` and `the_at_cache_keeps_the_larger_indices`;
  72 committed (AST, params, gpus) dry-runs identical before and after (7 over 120 s in both);
  on HEAD fdf31c8 plus the change: `--lib` 65 passed, `--test all` 129, the 11 committed ASTs
  under 120 s keep their 4-GPU fingerprints. Fixed in an isolated worktree and applied.

## 82. Bug 80's test expected the first repeated `seq` the workers' timing chose

- **Where:** `runner/aeiou/tests/run.rs` `a_repeated_seq_and_a_wait_nothing_ends_are_errors_not_hangs`.
- **Found by:** CI on fdf31c8 (bugs 79 and 80): under sync at time scale 0.0001 the run was
  refused "put 1 on `c`: seq 1 was put on this channel before", and the test asserted seq 0.
- **Defect:** in the second epoch two loader workers put seqs 0 and 1; which reaches the
  channel first is timing, so either is the first repeat. The runner's error is right; the
  test pinned one order (a timing dependence in a test, against the invariant it checks).
- **Fix:** the test accepts either first repeat.
- **Evidence:** the CI log of run 38763410446; the test passed 8 runs in a row locally after.

## 83. An event-loop instance stuck on its channels hung while another of its loop waited at a barrier

- **Where:** `runner/aeiou/src/uring.rs` (`Loop::serve`, new `Loop::stuck_instance`, `next_scan`);
  `runner/REFERENCE.md` §8.
- **Found by:** the RC1 timing review of the channel machinery (after bugs 79/80): the thread
  runner reports a channel deadlock per instance (`Watch`); the event loop's "nothing will
  complete" check looked at the loop as a whole and was skipped whenever the eventfd was
  posted, which any barrier waiter or object request in flight does.
- **Defect:** `channel c cap 1; cond(actor id == 1){ take c }; barrier global` at `--gpus 2
  --threads 1` (or `--gpus 40` with loops shared) hung under io_uring and libaio, gpu#0 at the
  barrier gpu#1 had to reach, gpu#1 on its channel; sync reported "every thread of gpu#1 waits
  on a channel that nothing will change".
- **Fix:** at most every `IDLE` (50 ms) after the runnable queue drains, an instance every task
  of which is unqueued and parked on its channels, sub-actors, loaders, or trace order fails
  with the existing wording listing its own tasks, whatever the loop's other instances do
  (channels and trace tables are per instance); a task in I/O, a timer, or a barrier keeps its
  instance live, so no run that could finish is failed. It takes the existing failure path, so
  the root cause is reported. No fingerprint, dataset id, or AST changes.
- **Evidence:** `run::an_instance_stuck_on_its_channels_is_an_error_while_a_sibling_waits_at_a_barrier`
  (2 GPUs, Sync, Uring and LibAio at 1 and 2 loops, hang-safe) hung 30 s with the call undone
  and passes in 0.26 s; the reviewer's shapes s1–s11 10/10 ok under io_uring and libaio at time
  scales 0 and 0.0001, 1 and 4 GPUs, no false report; on b108d7d plus the change `--test all`
  130 passed, `--lib` 65. Fixed in an isolated worktree and applied.

## 84. Under the thread runner a loader worker's failure was reported as "run aborted"

- **Where:** `runner/aeiou/src/run.rs` (`run_with`'s actor thread, new `Runner::abandon`,
  `Shared::abort_error`; `Channel::begin`/`put`/`take` take `&Shared`; `Sink::op`).
- **Found by:** the RC1 timing review of the channel machinery: `drive(...).and_then(|_|
  sink.finish())` skipped `finish` when the main line erred, so the instance's loader workers
  were never closed or joined and their errors were dropped; the channel waits returned a bare
  "run aborted".
- **Defect:** `channel c cap 4; loader q {put c seq (b==2 ? -1 : b)}; take c ×4; take q ×4` under
  `--posix sync` ended 20/20 with only "gpu#0 [1] take `c`: run aborted" where io_uring named
  "put -1 on `c`: seq -1 is below 0"; a reviewer's fuzzer found 112 of 4,800 sync runs of
  random loader shapes disagreeing with io_uring on the cause; loader threads could outlive
  the run, doing I/O.
- **Fix:** on the main line's error (or panic) the actor sets `aborted`, closes the instance's
  channels, joins its loader workers (stats merged), and ranks its error with theirs through
  `keep_cause`, so a real cause beats an echo; every abort echo reads "run aborted: <reason>"
  as `Sink::op`'s did; a put or take on a channel closed by an aborted run returns the echo,
  not "on a closed channel". No fingerprint, op stream, dataset id, or AST changes.
- **Evidence:** `run::a_loader_workers_failure_is_the_cause_whatever_the_main_line_waits_on`
  (three shapes, Sync and Uring, time scales 0 ×3 and 0.001, hang-safe) failed "take `c`: run
  aborted" on the old run.rs and passes; the fuzzer 0 disagreements in 4,800 runs after; on
  HEAD 40392a7 (bug 83) plus the change: `--test all` 131 passed twice, `--lib` 65, builder suite
  365 passed (object tests skipped, run in the final check). Fixed in an isolated worktree and
  applied.

## 85. The runner read some JSON float literals one ULP off, so its AST hash was not the builder's

- **Where:** `runner/aeiou/Cargo.toml` (serde_json's `float_roundtrip` feature);
  `runner/aeiou/tests/golden.rs`; `builder/tests/test_params.py`.
- **Found by:** the second regression review (pre-existing at the tag): serde_json without
  `float_roundtrip` parses floats "best effort", up to one ULP off.
- **Defect:** a dataset size `{"const": X}` hashed differently in `aeiou check` and check.py for
  X = 4503599627370495.5, 9007199254740993.0, 123456789.12345679; a normal's min > max message
  printed max `4503599627370495.0` for `…495.5`; a parameter file pinned to the builder's hash
  passed `aeiou-params check` and was refused by `aeiou dry-run --params-file` ("is for AST
  9dd21b2c…, not 93bfce21…"), and the runner ran a value one ULP from the document's. The
  committed ASTs happened to hash alike.
- **Fix:** serde_json with `float_roundtrip` (the locked 1.0.151 has it; Cargo.lock unchanged,
  no new crate), which every JSON reader of the runner uses (abstracts, parameter files,
  manifests, the coordinator's frames, trace files). Every committed AST's sha256 and every
  fingerprint compared (1 and 4 GPUs, those under 120 s with both binaries) unchanged.
- **Evidence:** `golden::float_literals_are_read_as_check_py_reads_them` (three sha256 pinned
  from check.py, the min > max message) and
  `test_params.py::test_a_file_pinned_to_the_builders_hash_is_taken_by_the_runner` fail on the
  old binary and pass; `--lib` 66 passed, `--test all` 132, test_builder, test_params,
  test_formats 152.

## 86. The geometry cache held every sample offset of its newest file, however many

- **Where:** `runner/aeiou/src/eval.rs` (`DsMeta::file_geom(file, max_offsets)`);
  `runner/aeiou/src/vm.rs` (`file_geom_within`, `geom_held`, the bound's doc).
- **Found by:** the second regression review of bug 78: the eviction loop never evicts the
  newest entry, so one file of N samples kept 8·N bytes of offsets in every VM, a per-sample
  table over a one-file dataset (CLAUDE.md: never per-file data structures; caches bounded).
- **Defect:** a one-file compound dataset of 8,000,000 samples dry-ran at maxrss 71 MB per VM.
- **Fix:** a file with more samples than `GEOM_OFFSETS` (2^20) is held by its size alone, its
  offsets never built; `sample_offset` then asks `DsMeta`, as before bug 78, so results are
  identical (the same run: 9.4 MB, same fingerprint). Such a file's offsets cost O(samples per
  file) again (0.64 s → 2.07 s for that run). No fingerprint, dataset id, or AST changes.
- **Evidence:** `vm::tests::the_geometry_cache_never_holds_more_offsets_than_its_bound` (cap 30,
  files of 100/100/20, packed and TFRecord, sizes and offsets equal `DsMeta`'s, all 220 ids at
  the production bound) failed "100 offsets held after file 0" with the gate undone and
  passes; suites as bug 85.

## 87. A value of a kind the runner refuses (`"count": "gpus"`) passed both checkers

- **Where:** `schema/check.py` and `runner/aeiou/src/validate.rs` (new rule V27: `kind`, `typed`,
  `drawn`, `operand`, a binding's kind in `Scope`; `check_given` judges given values too);
  `schema/README.md` §4 V27; `man/aeiou-abstract.7.md`; `runner/aeiou/tests/golden.rs`.
- **Found by:** the RC1 timing review of the channels (an aside): neither checker inferred a
  value's kind.
- **Defect:** `{"actors":{"gpu":{"count":"gpus",…}}}` was `ok` in check.py and `aeiou check`, and
  `dry-run`/`run` failed "actor `gpu` count: expected an integer, got str"; the same at every
  integer or number site (loop bounds, widths, loader counts, capacity, seq, ns, len, offset,
  ids, distribution arguments), condition, arithmetic operand, and namespace field; `--param
  n=abc` reached the run.
- **Fix:** a value's kind where the document shows it (a literal, a parameter by its value in
  effect, an element of a one-kind parameter array, a binding by its value, a handle, a
  distribution, an expression whose result kind is fixed, a `cond` with arms of one kind) is
  judged at each site against the kinds the runner refuses there, with one path and message in
  both checkers, e.g. `actors/gpu/count: expected an integer, got a string (V27)`; a float or
  bool where an integer is read stays accepted (the runner rounds it, reads a bool as 0/1); a
  literal `null` on an optional key is the key left out. Two parity gaps on the way: check.py
  read a `ref` where an expression or handle may stand as a handle (message differed), and
  passed a file handle `{dataset, id: null}` the runner refuses. A golden V3 case that used a
  distribution parameter as an `add` operand (which the runner refuses) now draws it. No
  fingerprint, dataset id, AST, or schema change.
- **Evidence:** `golden::a_value_of_a_kind_the_runner_refuses_is_refused_at_check` and
  `test_builder.py::test_a_value_of_a_kind_the_runner_refuses_is_refused_at_check` (6 accepted,
  14 refused, exact messages; `--param n=abc`) fail on the old checkers and pass; a 724-document
  battery (24 sites × 23 kinds) gives identical errors in both checkers, and every V27 refusal
  failed `dry-run` before; on HEAD 6a748fc plus the change: `--test all` 133 passed, `--lib` 66,
  builder suite with MinIO 381 passed, CI's checker diff empty, `aeiou-build --hermetic --twice
  --check` rc 0. Fixed in an isolated worktree and applied.

## 88. A `take` in a loader's body passed both checkers, then failed or passed by timing and engine

- **Where:** `schema/check.py` and `runner/aeiou/src/validate.rs` (`used_by_workers`, V26);
  `schema/README.md` §4 V26; `man/aeiou-abstract.7.md`; `runner/REFERENCE.md` §4.
- **Found by:** the second regression review: bug 77's rule kept a channel from being declared
  again under a loader's workers, but which worker took which item the actor put was still
  timing.
- **Defect:** `channel c; loader q workers 2 batches 8 {compute; take c}; loop j<8 {put c seq j;
  take q}` was `ok` in both checkers; `run` under sync failed 10/10 at time scale 0 and
  passed 1/10 at 1 ("every thread of gpu#0 waits …"), io_uring and libaio failed 10/10 at 0
  and passed 10/10 at 1.
- **Fix:** a `take` inside a `loader` body, at any depth, on a plain channel or another
  loader's, is refused by both checkers with one path and message ("… which of them took which
  item, and whether the run finished, would depend on timing; take in the actor, a loader's
  body only puts (V26)"); puts stay allowed. Bug 77's tests now use a put in their accepted
  shape. No committed AST takes in a loader body. No fingerprint, dataset id, or AST changes.
- **Evidence:** `golden::a_take_inside_a_loader_body_is_refused` and
  `test_builder.py::test_a_take_inside_a_loader_body_is_refused` (1 accepted, 4 refused) fail
  with the rule undone and pass; suites as bug 90.

## 89. Two sub-actors of one `parallel` taking on one channel finished or deadlocked by timing

- **Where:** `runner/aeiou/src/run.rs` (`ForkPath`, `one_taker`, `ActorState.fork`,
  `Instance::fork`, `Channel::take`, `Channel::park`); `runner/aeiou/src/uring.rs` (`Loop::forks`,
  the take path); `runner/REFERENCE.md` §4; `schema/README.md` §3; `man/aeiou-abstract.7.md`.
- **Found by:** the RC1 timing review of the channels: which taker got which item was timing,
  so work was distributed by timing (CLAUDE.md invariant) and the run's outcome varied.
- **Defect:** `parallel w<3` with w0 `put c 0; take d; put c 1`, w1 `take c; put d 0`, w2
  `take c` failed under sync 1/20 at time scale 0 and 7/20 at 1, passed 20/20 under io_uring
  and libaio; with w1/w2 swapped sync failed 10–14 of 20 and the event loop 20/20.
- **Fix:** each actor state carries its fork path (fork instance, index name, index per fork
  from the main line) and each channel its last taker's; a take by a sibling of the last taker
  in the same fork instance is an error in both engines, naming the channel and both indices;
  the main line before or after a fork, a sub-actor and the forks nested in it, and successive
  fork instances (a `parallel` in a loop) stay allowed. Every sub-actor's op stream is fixed,
  so two takers always both take and the error is every run's. A thread runner `park` that saw
  a just-failed sibling as not running reported "stuck" over its error; `park` now returns the
  abort echo when the run is aborted, so the cause wins. Bug 75's and 80's tests now take in one
  sub-actor. No fingerprint, dataset id, or AST changes (channel ops are never hashed).
- **Evidence:** `run::two_takers_in_one_fork_are_an_error_in_every_run` (four refused shapes, a
  good shape taking in turn across forks, Sync, Uring, LibAio at time scales 0 ×3 and 0.001)
  fails with `one_taker` disabled and passes; a fuzzer of 800 two-taker shapes × 8 runs (6 sync
  at three time scales, io_uring, libaio, under load) gave one outcome per shape; the
  reviewers' fuzzers 0 sync/io_uring disagreements; suites as bug 90.

## 90. An unordered channel missing a `seq` grew its record without bound and finished

- **Where:** `runner/aeiou/src/run.rs` (`Seqs::window`, `admits`, the stuck message);
  `runner/aeiou/src/uring.rs` (`Chan::try_put`, `Loop::parked_label`); `runner/REFERENCE.md`
  §4; `schema/README.md` §3.
- **Found by:** the RC1 timing review (bug 79's `Seqs` was bounded only for ordered channels).
- **Defect:** an unordered channel of 2 whose puts were `seq s+1` (seq 0 never put) succeeded
  at maxrss 81 MB against 40 MB without the gap (about 20 B per item held), the gap accepted.
- **Fix:** a put is admitted while `seq < min(taken, lowest seq not yet put) + capacity`, the
  same for ordered channels as before, so at most `capacity` seqs are held past the gap and a
  missing seq is the existing wait-nothing-ends error, which now names it ("waits to put 2
  (seq 0 is not put yet)") alike in both engines. An unordered producer can no longer run more
  than `capacity` seqs ahead of a seq not yet put (documented). No fingerprint, dataset id, or
  AST changes.
- **Evidence:** `run::an_unordered_channels_missing_seq_is_a_wait_nothing_ends` (1000 items, Sync,
  Uring, LibAio) succeeded with the old window and passes; the gap shape is refused at about 9 MB.
  For bugs 88–90 on HEAD c7764af plus the change: `--test all` 136 passed twice, `--lib` 66, builder
  suite with MinIO 382 passed, CI's checker diff empty, `aeiou-build --hermetic --twice
  --check` rc 0. Fixed in an isolated worktree and applied.

## 91. A file past the geometry cache's bound paid O(samples per file) for every sample offset

- **Where:** `runner/aeiou/src/eval.rs` (`FileGeom {file, size, every, anchors}`,
  `DsMeta::file_geom`, new `DsMeta::offset_from_geom`); `runner/aeiou/src/vm.rs`
  (`geom_held`, `sample_offset_within`).
- **Found by:** bug 86's trade-off: a file with more than 2^20 samples was held by its size
  alone, so each offset drew every earlier sample's size, and a pass over it was O(n²) again.
- **Defect:** dry-runs of a one-file dataset of 8,000,000 samples: 64 reads took 10.1 s packed,
  32.1 s TFRecord-framed, 13.3 s in units of 1000 (about 0.15–0.48 s per op).
- **Fix:** sparse anchors: a file of n samples holds the offset of every k-th sample, k the
  least with at most 2^20 anchors (k = 1 is the dense table), at the sample (packed), at the
  unit start (a unit of at most k samples), or at the row within a unit; an offset is its
  anchor plus at most k−1 size draws with `sample_offset`'s checked arithmetic, `None` (so
  `DsMeta`) for multi-column files or any failure. The cache still holds at most 2^20 entries
  in all. The same runs: 0.6–1.1 s whatever the reads, maxrss +7 MB (the bounded table);
  fingerprints identical. No fingerprint, dataset id, AST, or datagen byte changes.
- **Evidence:** `vm::tests::the_geometry_cache_never_holds_more_offsets_than_its_bound`
  (extended: cap 30, files of 100/100/20, packed, TFRecord, units of 1, 3, 7, 40; every offset
  equals `DsMeta`'s within k−1 draws; held anchors within the bound) failed "anchors None"
  with bug 86's gate and passes; on HEAD b721c06 plus the change `--lib` 66 passed, `--test
  all` 136, test_formats and test_builder 106, hdf5_map, tfrecord_stream, parquet_stream
  fingerprints at 1 and 4 GPUs unchanged. Fixed in an isolated worktree and applied.

## 92. The Python tools read duplicate keys and lone surrogates the runner refuses

- **Where:** `schema/check.py` and `builder/aeiou/emit.py` (`loads`); `builder/aeiou/trace.py`
  (`TraceReader.feed`); `builder/tests/test_builder.py`, `test_params.py`, `test_trace.py`.
- **Found by:** the second regression review (bug 76's fix was incomplete).
- **Defect:** `{"ast":"0.10","name":"x","name":"y",…}` was `ok` in check.py where `aeiou check`
  said "duplicate field `name`"; a phase name `"a\ud800b"` (a lone surrogate) was `ok` where
  the runner said "not an abstract (unexpected end of hex escape …)"; `aeiou-params check`
  took a parameter file with one; `aeiou-trace metrics` read a trace file the runner refuses
  and ended in a traceback on a line that was not JSON.
- **Fix:** `loads` refuses every repeated key (serde refuses repeated struct fields; map keys
  it takes last-wins, so Python is stricter there, and no committed file or writer repeats
  one) and any lone surrogate in a key or string after the parse (a valid pair is one
  character and kept); each tool keeps its invalid-JSON frame. The trace reader uses
  `emit.loads` and refuses a bad line as `<file>: line N: …`, rc 1. `emit.loads` equals
  `json.loads` on all 85 committed JSON files. No fingerprint, AST hash, or contract change.
- **Evidence:** `test_builder.py::test_duplicate_keys_and_lone_surrogates_are_refused_as_the_runner_refuses_them`,
  `test_params.py::test_duplicate_keys_and_lone_surrogates_in_a_parameter_file_are_refused`,
  `test_trace.py::test_an_exported_trace_is_read_as_the_runner_reads_it` fail on the old code
  and pass; suites as bug 93.

## 93. A parameter defined through itself overflowed both checkers' stacks

- **Where:** `runner/aeiou/src/validate.rs` and `schema/check.py` (new `param_cycles`, V1);
  `schema/README.md` §4 V1; `runner/aeiou/tests/golden.rs`; `builder/tests`.
- **Found by:** the second regression review (pre-existing at the tag).
- **Defect:** `params.pa.default = {"const": {"param": "pa"}}` with a dataset `size {"param":
  "pa"}`: `aeiou check` aborted "thread 'main' has overflowed its stack" (rc 134), check.py
  raised RecursionError (V25's least and V3's at_least follow a parameter to its value).
- **Fix:** before any walk that follows a `param`, each parameter whose value in effect (given,
  else default) reaches itself through `param` references is refused with its shortest cycle,
  `params/pa: parameter `pa` is defined through itself (pa -> pb -> pa) (V1)`, alike in both
  checkers and on given values (`check_given`, aeiou-params, `--param`, `--params-file`).
  References between parameters stay allowed; a cycle can only be distributions, which the run
  refuses anyway, so no runnable abstract is refused. No fingerprint, dataset id, or AST changes.
- **Evidence:** `golden::a_parameter_defined_through_itself_is_refused_with_its_cycle` aborted
  with a stack overflow on the old code and passes;
  `test_builder.py::test_a_parameter_defined_through_itself_is_refused_with_its_cycle` and
  `test_params.py::test_a_parameter_file_closing_a_cycle_is_refused_as_the_runner_refuses_it`
  fail before and pass. For bugs 92 and 93 on HEAD 3a5c36b plus the change: `--test all` 137
  passed, `--lib` 66, builder suite with MinIO 387 passed, CI's checker diff empty, every
  committed parameter file ok, `aeiou-build --hermetic --twice --check` rc 0. Fixed in an
  isolated worktree and applied.

## 94. A draw whose result the runner refuses (`const` of a distribution parameter) passed both checkers

- **Where:** `schema/check.py` and `runner/aeiou/src/validate.rs` (V27: new `drawkind`/`draw_kind`,
  `kind` of a `draw`, `drawn`); `schema/README.md` §4 V27; `man/aeiou-abstract.7.md`; tests.
- **Found by:** the RC1 fix of bug 93: V27 (bug 87) took every `draw` as of unknown kind.
- **Defect:** `params.pa = {const: {param: pb}}`, `pb = {const: 3}`, `compute ns {draw {param: pa}}`
  was `ok` in both checkers and failed `dry-run`/`run` "expected an integer, got dist"; so did a
  draw yielding a distribution, string, null, array, or (for a condition) float: `empirical`
  over strings, mixtures of such arms, a size `{const: {param: d}}`, a `let` bound to such a
  draw, `if: draw {const: 0.5}`.
- **Fix:** the kind a draw yields where the document shows it (`const` its argument's; the
  numeric distributions an integer; `empirical`, `mixture`, `cond` with one kind throughout;
  a parameter's distribution by its value in effect) feeds every V27 site, so the refusal lands
  where the runner blames, e.g. `actors/gpu/body/0/compute/ns: expected an integer, got a
  distribution (V27)`; `drawn` (a size, a pick's dist) refuses any non-number. A set of
  parameters being followed keeps the walk finite before V1's cycle check. Values given
  (`check_given`, aeiou-params, `--params-file`) are judged alike. Bug 93's tests named a
  parameter through this very shape and now name it through a mixture arm. No fingerprint,
  dataset id, AST, or schema change.
- **Evidence:** `golden::a_draw_of_a_kind_the_runner_refuses_is_refused_at_check` and
  `test_builder.py::test_a_draw_of_a_kind_the_runner_refuses_is_refused_at_check` (5 accepted,
  11 refused, a given value) fail on the old checkers and pass; a 455-document battery (11
  sites × 41 draw forms) gives identical errors in both checkers, 213 refused (10 before), each
  failing HEAD's dry-run; on HEAD 088919c plus the change: `--test all` 138 passed, `--lib` 66,
  builder suite with MinIO 388, CI's checker diff empty, every committed parameter file ok,
  `aeiou-build --hermetic --twice --check` rc 0. Fixed in an isolated worktree and applied.

## 95. A comparison of kinds the runner cannot compare, or an array read of a non-array, passed both checkers

- **Where:** `schema/check.py` (`COMPARISONS`, `compared`, `array`) and `runner/aeiou/src/validate.rs`
  (`Checker::compared`, `Checker::array`), V27; `schema/README.md` §4 V27;
  `man/aeiou-abstract.7.md`; tests.
- **Found by:** the RC1 fix of bug 94: V27 judged single operands but not a comparison's pair,
  nor what `elem`, `len`, `sum` read.
- **Defect:** `cond if {eq: [1, "a"]}` was `ok` in both checkers and failed `dry-run`/`run`
  "eq: cannot compare int with str"; `len`/`elem`/`sum` of a parameter that is no array failed
  "parameter `x` is not an array", `sum` of an array with a string "element of `x` is str".
- **Fix:** where both operands' kinds are known and the runner refuses the pair (it compares
  int with float, str with str, bool with bool, handle with handle, null with anything), both
  checkers refuse it at the comparison's path, `…/cond/if: eq: cannot compare an integer with a
  string (V27)`; `elem`, `len`, `sum` judge their parameter's value in effect (`len: parameter
  `n` is not an array (V27)`, `sum: an element of `sarr` is a string (V27)`). Unknown kinds stay
  accepted; given values are judged alike. No fingerprint, dataset id, AST, or schema change.
- **Evidence:** `golden::a_comparison_or_array_read_the_runner_refuses_is_refused_at_check` and
  `test_builder.py::test_a_comparison_or_array_read_the_runner_refuses_is_refused_at_check` (8
  accepted, 8 refused, a parameter file) fail on the old checkers and pass; a 689-document
  battery gives identical errors in both checkers, 317 refused (0 before), each failing HEAD's
  dry-run; on HEAD 0a71dd5 plus the change: `--test all` 139 passed, `--lib` 66, builder suite
  with MinIO 389, CI's checker diff empty, every committed parameter file ok, `aeiou-build
  --hermetic --twice --check` rc 0. Fixed in an isolated worktree and applied.

## 96. S3 object keys were percent-encoded, so bucket names differed from POSIX names and `--clean-namespaces` silently missed keys

- **Where:** `runner/aeiou/src/object.rs` (`Store::key` and its callers `read`, `upload`, `stat`,
  `list_dir`, `delete`, `rename`, `non_empty`, `get`, `put`, `put_parts`, `keys_below`,
  `delete_all`; `start` Open/Read arms); `runner/REFERENCE.md` §16 **Where.**; tests.
- **Found by:** RC1 third-round object-engine review (RV3a). `Store::key` built keys with
  object_store's `Path::from`, which percent-encodes `~ [ ] # % { } ^ | < > " \`` and controls,
  while listing returns stored keys as they are: `keys_below` handed back encoded strings and
  `delete_all` encoded them again (`%7E`→`%257E`), and S3 answers a DELETE of a missing key
  with success. Repro `o/ck~pt [{k}]#%.bin`: the bucket held `ck%7Ept %5B0%5D%23%25.bin` (POSIX
  writes `ck~pt [0]#%.bin`); after a foreign `foreign~key`, `--clean-namespaces` printed
  "namespace root o/ emptied" yet all 4 keys stayed, and the next run was refused "not empty
  (4 objects)". The same mismatch made datagen `--resume` (listed keys vs generated names)
  send such objects again.
- **Fix:** `key` joins prefix and name, drops empty components (as POSIX resolves `a//b`), and
  builds the key with `Path::parse`, so the key is the name byte for byte; a `.`/`..`
  component or a control character fails with EINVAL naming the raw name. `read`/`upload`
  return `io::Result`; listed keys are raw, so re-keying them is the identity. No fingerprint,
  dataset id, or AST change; object keys of names with those characters change (now the names).
- **Evidence:** `object::object_keys_are_the_names_as_written` (keys equal the POSIX run's names,
  manifest `o/ck~pt [0]#%.bin`, clean removes a foreign and a stale key, `..` gives EINVAL) and
  `a_resumed_object_dataset_counts_the_objects_it_finds` (pattern now `s~ample#{id:09}.npz`;
  before the fix it re-sent 10 objects where 2 were missing) fail with `Path::from` restored and
  pass; on HEAD 0dd3c25 plus the change: `--test all` with MinIO 146 passed (after freeing disk
  that had pushed MinIO into 507 storage-full), builder `test_object.py` + `test_formats.py` 53
  passed, `cargo check --release` without features builds. Fixed in an isolated worktree and applied.

## 97. `aeiou-header` took a non-entry `--format` and a row count below 1 as runtime failures, and its man EXAMPLE selected nothing

- **Where:** `builder/aeiou/header.py` (`main`); `man/aeiou-header.1.md` EXAMPLES and EXIT STATUS; tests.
- **Found by:** RC1 third-round UI review (RV3b): the man EXAMPLE's `jq -c .datasets.train.files.format`
  yields null (the AST has no `files` key), and `--format null`/`{}` then said "no header for
  format class `None`", `--format []` "class `[]`", exit 1; `--rows 0`/`-1` exited 1 though a
  meaningless value is a usage error (exit 2, runner/REFERENCE.md §15).
- **Fix:** a `--format` that is not an object with `class` is a usage error "--format: not a format
  entry (an object with `class`)", and `--rows` below 1 is a usage error, both through the
  shared usage frame; the EXAMPLE uses `.datasets.train.format`.
- **Evidence:** `test_usage.py::test_aeiou_header_refuses_what_is_no_format_entry_or_row_count` and
  `test_man.py::test_aeiou_header_example_selects_the_format_entry` fail on the old code and pass;
  on HEAD 2db51a3 plus the change the builder suite with MinIO gives 403 passed, `aeiou-build
  --hermetic --twice --check` rc 0, `aeiou-catalog check` 13 mechanisms, 16 cases ok.

## 98. The Python tools' `--help` broke long words and hyphenated words, so `aeiou-trace --help` printed a broken strace command

- **Where:** `builder/aeiou/usage.py` (`Formatter._split_lines`); tests.
- **Found by:** RV3b: at 78 columns `aeiou-trace --help` printed `-e trac` / `e=%file,…` and `aeiou dry-`
  / `run --metrics-json`, so a copied command fails.
- **Fix:** the shared formatter wraps with `break_long_words=False, break_on_hyphens=False`, as clap does.
- **Evidence:** `test_usage.py::test_the_help_wraps_at_spaces_only[80]` (all 16 Python tools and
  subcommands, no hyphen-broken line) fails on the old formatter and passes; `test_man.py` passes;
  same suite runs as bug 97.

## 99. `aeiou-trace metrics`/`export` exited 0 with nothing counted, and `export` wrote an empty trace file

- **Where:** `builder/aeiou/trace.py` (`Tracer.lines_read`/`calls_parsed`, `metrics_of`/`export_of`
  `seen`, the CLI); `man/aeiou-trace.1.md` EXIT STATUS; tests.
- **Found by:** RV3b: `metrics t.st --root /nonexist` printed `0 ops` and exited 0, a non-strace input
  gave a zero-filled document with exit 0, and `export --root /nonexist` wrote a 0-lane trace and
  its sha256, against aeiou-trace(1) "1 … a trace with no counted call".
- **Fix:** the CLI exits 1 when nothing was counted, saying how many lines were read and parsed
  ("no call under --root DIR (N line(s) read, M parsed as strace calls)", or "not an strace" when
  none parsed); `export` writes no file; an exported trace with no call fails too. The library
  functions keep returning zero counts.
- **Evidence:** `test_trace.py::test_a_trace_with_no_counted_call_fails_and_exports_nothing` fails on
  the old code and passes; same suite runs as bug 97.

## 100. `aeiou-trace compare` described an exported trace's metrics as "trace of None seed None gpus None"

- **Where:** `builder/aeiou/trace.py` (`_describe`); tests.
- **Found by:** RV3b: `_describe` knew only the strace and dry-run sources.
- **Fix:** a `source == "trace"` branch: "trace file X of ROOT (order: completion; N instance(s), M lane(s))".
- **Evidence:** `test_trace.py::test_compare_describes_the_metrics_of_an_exported_trace` fails on the
  old code and passes; same suite runs as bug 97.

## 101. `aeiou-build` of a missing script blamed its own source line, and its man promised a traceback it never prints

- **Where:** `builder/aeiou/cli.py` (script check, frame lookup); `man/aeiou-build.1.md` EXIT STATUS; tests.
- **Found by:** RV3b: `aeiou-build nonexist.py` → `FAIL nonexist.py: FileNotFoundError at line 1013`
  (runpy's line); a directory gave `IsADirectoryError at line 1013`.
- **Fix:** the script is checked first ("no such file (PATH); aeiou-build takes builder scripts
  (`.py`)", "not a file"); "at line N" only from a frame of the script itself; the man describes
  the one-line `FAIL <script>: <reason>`.
- **Evidence:** `test_usage.py::test_aeiou_build_says_a_script_does_not_exist` fails on the old code
  and passes; same suite runs as bug 97.

## 102. `aeiou-trace` accepted `--max-distance nan` (a gate that never fails) and negative bounds

- **Where:** `builder/aeiou/usage.py` (`non_negative`), `builder/aeiou/trace.py` (`--max-distance`,
  `--chain-gap-us`); `man/aeiou-trace.1.md`; tests.
- **Found by:** RV3b: `--max-distance nan`/`inf` exited 0 (every comparison with NaN is false),
  `-1` failed every row, `--chain-gap-us -1`/`nan` were taken.
- **Fix:** a finite, non-negative argument type, refused with argparse wording, exit 2.
- **Evidence:** `test_trace.py::test_a_bound_that_is_not_a_finite_non_negative_number_is_a_usage_error`
  (5 cases) fails on the old code and passes; same suite runs as bug 97.

## 103. `aeiou-trace` had no `-V/--version`, which every other tool has

- **Where:** `builder/aeiou/trace.py`; `man/aeiou-trace.1.md` SYNOPSIS and options; tests.
- **Found by:** RV3b: `aeiou-trace --version` → "unrecognized arguments", exit 2, against §15.
- **Fix:** `ap.version(f"aeiou-trace {__version__}")` as the other tools.
- **Evidence:** `test_trace.py::test_aeiou_trace_has_a_version_as_every_tool_does` fails on the old
  code and passes; `test_man.py` passes; same suite runs as bug 97.

## 104. QUICKSTART sent first-time users to the deprecated `aeiou-datagen` for Parquet and listed four of ten man pages

- **Where:** `QUICKSTART.md` "Other workloads".
- **Found by:** RV3b: it said Parquet is written "by the Python side" with `aeiou-datagen`, while
  `aeiou datagen parquet_stream.ast.json --param samples=2048` writes it (checked: two shards,
  "parquet, aeiou datagen parquet writer").
- **Fix:** the section uses `aeiou datagen` for all three container formats, keeps the wheel for
  authoring and trace comparison, and lists all ten man pages (aeiou-datagen(1) marked deprecated).
- **Evidence:** the command run as written by the fixer; no test reads QUICKSTART; same suite runs
  as bug 97.

## 105. `aeiou-params check` of a missing file printed Python's errno text instead of the suite's wording

- **Where:** `builder/aeiou/params.py` (`load`); tests.
- **Found by:** RV3b: `[Errno 2] No such file or directory: 'nonexist.json'`, where the runner says
  `nonexist.json: No such file or directory (os error 2)`.
- **Fix:** `load` words an OSError as the runner does.
- **Evidence:** `test_usage.py::test_aeiou_params_check_words_a_missing_file_as_the_runner_does`
  (compared byte for byte with the runner's line) fails on the old code and passes; same suite
  runs as bug 97.

## 106. Ctrl-C on `aeiou-launch` returned at once while every rank kept running

- **Where:** `runner/aeiou-launch` (remote command, rank start, trap, wait); `man/aeiou-launch.1.md`
  (DESCRIPTION, new SIGNALS, ENVIRONMENT, EXIT STATUS); `runner/REFERENCE.md` §6 aeiou-launch
  bullet; `builder/tests/test_launch.py`.
- **Found by:** RC1 third-round UI review (RV3b): each rank was `( $rsh … ) | sed … &`; a
  non-interactive sh starts async lists with SIGINT ignored, ssh keeps it ignored, and a remote
  command without a tty gets no signal. Simulated with `AEIOU_RSH` in its own session: the
  launcher exited 130 at once, both `aeiou run` ranks ran to their fingerprints, the mktemp dir
  leaked; aeiou(1) promises "Ctrl-C on any host stops every host", and datagen never got its
  `stopped` manifest.
- **Fix:** the signal travels over each ssh session's stdin (no tty needed, any `AEIOU_RSH` that
  passes stdin): a POSIX-sh watcher before the `exec`ed remote command kills it with INT/TERM on
  a line `INT`/`TERM`, and does nothing at EOF; the launcher traps INT/TERM, writes the line to
  each rank's FIFO holder (a second Ctrl-C sends a second INT), waits again for every rank,
  prints each one's status, removes its temp dir, and exits 130/143; a rank not yet started is
  not started. Limits documented: a POSIX remote shell, stdin passed (no `ssh -n`), SIGKILL of
  the launcher stops nothing.
- **Evidence:** `test_launch.py::test_ctrl_c_stops_every_rank[SIGINT-130]` and `[SIGTERM-143]`
  fail with HEAD's launcher (`-2 == 130`, `-15 == 143`) and pass; a real two-rank `aeiou run` of
  file_per_sample stopped in 0.07 s with no aeiou left, both ranks status 130; checked under dash
  and `bash --posix`; on HEAD 1cd5eb2 plus the change `test_launch.py test_man.py test_usage.py`
  37 passed, Rust `man` tests 9 passed, `sh -n` clean.

## 107. `aeiou-launch` spliced ranks' lines into each other when its output went to a file or pipe

- **Where:** `runner/aeiou-launch` (the per-rank prefixer); `builder/tests/test_launch.py`.
- **Found by:** RV3b: the per-rank `sed` is block-buffered when stdout is not a tty, so two ranks'
  buffers interleave mid-line (`[0 127.0.0.1]   lseek … [default]` carrying rank 1's tail, an
  unprefixed `n=32`).
- **Fix:** a `while IFS= read -r line` loop with one `printf` per line (read takes a pipe byte by
  byte, each printf is one write), the same on GNU and BSD; awk avoided since mawk buffers input.
- **Evidence:** `test_launch.py::test_lines_stay_whole_in_a_file` fails with HEAD's launcher
  (spliced `[0 a] 984 line[1 b] 0 line…`) and passes; 3000 lines from each of two ranks: 68 bad
  lines before, 0 after; same suite runs as bug 106.

## 108. `--clean-namespaces` deleted a read-only dataset (or an input namespace) placed by its own endpoint in the same bucket under the namespace's prefix

- **Where:** `runner/aeiou/src/run.rs` (`prepare_namespaces`); `runner/aeiou/src/object.rs` (`Store::place`,
  `same_bucket`, `full`); `runner/REFERENCE.md` §4 Startup and §16 **Namespaces**; tests.
- **Found by:** RC1 third-round object-engine review (RV3a). A dataset's keys were skipped only when
  `Arc::ptr_eq(&s, &store)`, and `Endpoints::parse` builds one `Store` per placed root, so a
  dataset with its own endpoint was never skipped. Repro: indexed s3 dataset `d`, s3 namespace
  `o`, `--endpoint d=s3://bench/ns/data --endpoint o=s3://bench/ns`: datagen, then `run
  --clean-namespaces` printed "namespace root o/ emptied" and failed `read d/f_0 … 404`; dataset
  and manifest gone. POSIX compared physical paths and was safe for datasets, but neither POSIX
  nor objects protected input namespaces (V14: a run never empties an input root).
- **Fix:** a `Store` knows its place (endpoint URL, bucket); the kept roots are every dataset root
  and every input-namespace root; on POSIX the physical comparison covers both, on objects
  every kept root in the same bucket becomes a skip prefix relative to the namespace's prefix,
  and a kept root equal to or containing the namespace prefix keeps everything below it. No
  fingerprint, dataset id, or AST change.
- **Evidence:** `object::endpoints_in_one_bucket_share_its_keys` (dataset and input namespace
  under an output prefix survive two cleans) and `run::an_input_namespace_inside_an_output_root_is_not_emptied`
  fail with the change reverse-applied and pass; after the fix the repro keeps `ns/data/f_0`
  (GET 200). Merged with bug 96's key change on HEAD 1ecdf00: `--test all` with MinIO 148
  passed, `--lib` 68, `cargo check --release` without features clean, builder `test_object.py`
  + `test_formats.py` 53 passed. Fixed in an isolated worktree and applied.

## 109. A rename between two endpoints' prefixes in one bucket was refused with EXDEV ("another object store")

- **Where:** `runner/aeiou/src/object.rs` (`Store::rename(from, into, to)`, the rename op); `runner/REFERENCE.md`
  §16 op table; tests.
- **Found by:** RV3a: the rename op used the same `Arc::ptr_eq` test as bug 108. Repro: namespaces
  `a` and `b`, `--endpoint a=s3://bench/x/tmp --endpoint b=s3://bench/x/fin`: `rename tmp/f0 to
  fin/f0: the target is in another object store (EXDEV)`, where POSIX with two directories on
  one file system renames.
- **Fix:** the op tests `same_bucket`; the rename copies to `into`'s key, so it moves across
  prefixes of one bucket; into another bucket it is still EXDEV, now "the target is in another
  bucket (s3://…)".
- **Evidence:** `object::endpoints_in_one_bucket_share_its_keys` (cross-prefix rename succeeds,
  `x/fin/f0` present and `x/tmp/f0` gone; another bucket EXDEV with the new message) fails with
  the change reverse-applied and passes; same suite runs as bug 108.

## 110. An s3 write handle never closed passed the checkers and the run, and the manifest listed objects the bucket did not hold

- **Where:** `schema/check.py` and `runner/aeiou/src/validate.rs` (V18: a later close in the same
  body); `builder/aeiou/builder.py`; `runner/aeiou/src/run.rs` (`ActorState::unclosed`, checked at
  `Runner::finish`, pool sub-actors, `finalize_main`), `runner/aeiou/src/uring.rs` (`end_child`
  returns `Result`), `runner/aeiou/src/object.rs` (`Handle::unfinished`, EBUSY on a reopen);
  `schema/README.md` V18, `man/aeiou-abstract.7.md`, `man/aeiou.1.md` OBJECT STORES,
  `runner/REFERENCE.md` §16 op table; tests.
- **Found by:** RC1 third-round object-engine review (RV3a): an `Upload` dropped without `close` is
  discarded (one part never sent, a multipart upload aborted), while `ActorState::opened` records
  the creation at open. Repro: two `open WRONLY|CREAT|TRUNC` + `write 4096`, no close: `aeiou
  check` ok, run exit 0, manifest `objects_created: 2, bytes_written: 8192, objects: [nc/f0,
  nc/f1]`, the bucket held only `.aeiou-namespace.json` (POSIX leaves both files).
- **Fix:** V18 refuses an `open` with `WRONLY` on an s3 name with no `close` of the same handle
  expression later in the same body, before it is opened again or a name it reads is bound again
  (a close in a nested body does not count: simple and sound, handles compared as V24 does), at
  `…/open/file`, identical in both checkers and the builder. At run time an actor that ends with
  an unsent upload fails ("opened for writing and never closed, so its upload was never sent …
  (V18)"), and an open of a path whose own upload is unsent is EBUSY. Stricter V18 only: no
  fingerprint, dataset id, or AST change; all committed ASTs pass.
- **Evidence:** golden `validator_rejects_what_check_py_rejects` cases (unclosed, reopened, cond-arm
  close, rebinding `let`, close of another object refused; unrelated `let` and writes in a loop
  accepted) and `test_builder.py::test_v18_an_object_is_written_once_in_order` fail with HEAD's
  checkers and builder and pass; the runtime check alone fails `nc` under `--s3 blocking` and
  `async`; 12 edge ASTs give the same path and message in both checkers. Merged on HEAD b52eba3:
  `--test all` with MinIO 148 passed, `--lib` 68, builder suite with MinIO 406, CI's checker diff
  empty, `aeiou-build --hermetic --twice --check` rc 0, 22 parameter files ok. Fixed in an
  isolated worktree and applied.

## 111. `open` with `RDONLY|CREAT` (or `TRUNC` without `WRONLY`) on an s3 name began an upload: reads failed EBADF and the close replaced the object with 0 bytes

- **Where:** `runner/aeiou/src/object.rs` (`writes`); `schema/check.py`, `runner/aeiou/src/validate.rs`,
  `builder/aeiou/builder.py` (V18); the same docs as bug 110; tests.
- **Found by:** RV3a: `writes()` counted CREAT or TRUNC alone as writing, and V18 refused only
  APPEND/RDWR/EXCL. Repro: write 4096, close, `open ["RDONLY","CREAT"]`, read 4096: `aeiou check`
  ok, the run failed `read rc/f0 …: Bad file descriptor (EBADF)` (POSIX passes); without the read
  the run passed with POSIX's fingerprint and `HEAD` showed `Content-Length: 0` where POSIX has
  4096: data silently destroyed.
- **Fix:** only `WRONLY` begins an upload; both checkers and the builder refuse CREAT or TRUNC
  without WRONLY on an s3 name at `…/open/flags` ("an object is made only by the upload an open
  with WRONLY begins, and a read-only handle writes none (V18)").
- **Evidence:** the golden R30 flag cases and the builder's V18 test fail on HEAD's code and pass;
  with the runtime change alone `rc`/`rc2` exit 0 and `f0` keeps 4096 bytes; same suite runs as
  bug 110.

## 112. `aeiou check`/`run` panicked under `| head`, and `dry-run` failed with "Broken pipe"

- **Where:** `runner/aeiou/src/main.rs` (writer `Out`, `STDOUT_GONE`, exit 141; every former `println!`),
  `datagen.rs`/`dryrun.rs` output through it; `man/aeiou.1.md` EXIT STATUS 141; `runner/REFERENCE.md`
  §15 "A reader that goes away"; `runner/aeiou/tests/usage.rs`.
- **Found by:** RC1 third-round UI review (RV3b): `aeiou check $(find workloads -name '*.ast.json') |
  head -1` → "failed printing to stdout: Broken pipe" panic, exit 101; `run … | head -1` panicked at
  its "report" line and the report recorded `"error": "Broken pipe"`; `dry-run | head -3` → "aeiou
  dry-run: Broken pipe (os error 32)", exit 1.
- **Fix:** all stdout goes through one fallible writer; the first EPIPE marks stdout gone and drops
  later writes, the work runs to its end (a run measures and writes `--report-json` with its true
  verdict), and the process exits 141 (128+SIGPIPE) quietly; a failing work's own exit 1 and
  message still win.
- **Evidence:** `usage::a_reader_that_goes_away_is_a_quiet_141` (check, dry-run, datagen, run, with a
  closed stdout) fails on HEAD (`(Some(1), "aeiou check: Broken pipe (os error 32)")` vs `(Some(141),
  "")`) and passes; on HEAD 3ca804d plus the change: `--tests --lib --bins` with MinIO all pass (all
  151, lib 68), builder suite with MinIO 406, CI's checker diff empty, `aeiou-catalog check` ok.
  Fixed in an isolated worktree and applied.

## 113. An unwritable `--report-json` was found only after the whole run

- **Where:** `runner/aeiou/src/main.rs` (`run_cmd` creates the report file first); `man/aeiou.1.md`
  `--report-json`; tests.
- **Found by:** RV3b: `--report-json /nonexist/r.json` ran to the end, printed its fingerprint, then
  failed exit 1; an hours-long run lost its JSON.
- **Fix:** the file is created (truncated) before anything else, replacing the old `remove_file`; a
  failure is a usage error, exit 2 ("cannot write the report there: …").
- **Evidence:** `usage::an_unwritable_report_is_refused_before_the_run` fails on HEAD and passes; same
  suite runs as bug 112.

## 114. The io_uring-knob refusal named knobs never given and blamed their defaults

- **Where:** `runner/aeiou/src/main.rs` (`run_checked`, the refusal built from the knobs set); tests.
- **Found by:** RV3b: `aeiou run --sqpoll 5` under sync → "--iowq-max-workers, --sqpoll,
  --defer-taskrun, --coop-taskrun are io_uring knobs; … (--iowq-max-workers by default, …)".
- **Fix:** the message names only the knobs set, with their layers ("--sqpoll is an io_uring knob;
  --posix sync has no ring").
- **Evidence:** `usage::messages_say_what_is_wrong` (R53 case) fails on HEAD and passes; same suite
  runs as bug 112.

## 115. `--subset` had no `--no-subset`, and repeating any `--[no-]x` boolean was refused

- **Where:** `runner/aeiou/src/main.rs` (arg structs: `--[no-]subset`; `overrides_with_all = [self,
  twin]` on every boolean pair); `man/aeiou.1.md`, `runner/REFERENCE.md` §2 synopsis; tests.
- **Found by:** RV3b: §14 says every boolean has `--[no-]x` and the last wins, yet `--no-subset` was
  "unexpected" and `--subset --subset` (and `--clean-namespaces --clean-namespaces`) "cannot be
  used multiple times".
- **Fix:** a hidden `--no-subset` twin as `--clean-namespaces` has; every boolean pair overrides
  itself and its twin, so repeats parse and the last wins.
- **Evidence:** `usage::messages_say_what_is_wrong` (all four forms) fails on HEAD and passes; same
  suite runs as bug 112.

## 116. Stale `aeiou --help` text: "a blocking I/O backend", `--threads` without `--s3 async`, no `--mmap-mode` default

- **Where:** `runner/aeiou/src/main.rs` (about line, module doc, `--threads`, `--mmap-mode` help);
  `man/aeiou.1.md`.
- **Found by:** RV3b: io_uring, libaio, and `--s3 async` are not blocking; the `--threads` refusal names
  `--s3 async` while its help did not; the `fault` default was stated nowhere.
- **Fix:** the run line says "through the I/O APIs it declares, or --posix and --s3"; `--threads`
  names `--s3 async`; `--mmap-mode` says "Default fault", in help and man alike.
- **Evidence:** the man tests (`runner/aeiou/tests/man.rs`, `builder/tests/test_man.py`) hold help and
  man together and pass; same suite runs as bug 112.

## 117. An unparsable `--coordinator` exited 1 at bind time instead of a usage error

- **Where:** `runner/aeiou/src/main.rs` (`check_coordinator` at resolve time, run and datagen); tests.
- **Found by:** RV3b: `--coordinator localhost:notaport` → "coordinator address … invalid port value",
  exit 1 (coord.rs), where §15 makes an unparsable value exit 2.
- **Fix:** HOST:PORT is checked when options resolve: "--coordinator localhost:notaport: expected
  HOST:PORT, a port number 0 to 65535", exit 2.
- **Evidence:** `usage::messages_say_what_is_wrong` (R61 case) fails on HEAD and passes; same suite
  runs as bug 112.

## 118. Runner messages that misled: "limit estimate:" on workload errors, negative counts, `--param =4`, a literal ROOT, the wrong usage line, clap's "-1", a stray space, an unchecked dataset id

- **Where:** `runner/aeiou/src/limits.rs` (prefix dropped), `eval.rs` (count and samples_per_file
  messages), `main.rs` (`--param` NAME=VALUE, per-form usage lines for `--resume/--grow/--shrink/
  --settle`, `allow_negative_numbers` plus range checks for `--time-scale`, `--max-gap`,
  `--progress-secs`, `--expect-dataset-id` as 64 hex), `datagen.rs` (Ctrl-C line with the real root),
  `dryrun.rs` (`(0.12×)`); tests.
- **Found by:** RV3b item 25 (R65): each message checked by hand against what it should say.
- **Fix:** the workload's own error without "limit estimate:"; "count -4: a count is 0 or more";
  "--param =4: expected NAME=VALUE" exit 2; "`aeiou datagen --resume <root>`" with the root; each
  datagen form's own usage line (the fresh form's bytes unchanged); "--time-scale -1: a factor of 0
  or more" etc.; no stray space; "not a dataset id (64 lowercase hex digits, …)" exit 2.
- **Evidence:** `usage::messages_say_what_is_wrong` fails on HEAD and passes; same suite runs as bug 112.

## 119. Any non-UTF-8 environment variable or argument made every `aeiou` subcommand panic

- **Where:** `runner/aeiou/src/main.rs` (`args_os`, clap `InvalidUtf8` naming the argument),
  `runner/aeiou/src/options.rs` (`vars_os` filtered on the `AEIOU_` bytes, environment held as
  `OsString`, `Layered::from_env(&OsStr)`); `runner/aeiou/tests/options.rs`, unit tests.
- **Found by:** RC1 third-round UI review (RV3b): `env "FOO=$(printf 'a\xff')" aeiou dry-run …` panicked
  in `std::env::vars()` (any variable on the host, not only `AEIOU_*`), exit 101; a non-UTF-8
  `--root`, `aeiou check` path, `AEIOU_ROOT`, or `--param` value panicked in `args()`.
- **Fix:** arguments and environment are read as OS strings; a path takes any bytes; a text option
  or variable that is not UTF-8 is a usage error naming it ("AEIOU_COORDINATOR: \"a\\xFF\" is not
  UTF-8", "\"steps=\\xFF\": not UTF-8, where the option takes text (a path may be any bytes)"), exit 2.
- **Evidence:** `options::odd_values_from_the_lower_layers_and_the_defaults_in_effect` panics on HEAD's
  options.rs/main.rs (`env.rs:162:83 … "a\xFF"`) and passes; unit test
  `empty_and_non_utf8_values_and_the_toml_types`; merged on HEAD 2ca77ed: `--tests --lib --bins`
  with MinIO all pass (all 152, lib 70), builder suite with MinIO 406, CI's checker diff empty.
  Fixed in an isolated worktree and applied.

## 120. The documented cluster config (`threads = 8` in `[run]`) was refused for every sync or mmap abstract, as were backend knobs from env or config

- **Where:** `runner/aeiou/src/main.rs` (`run_checked` pass-over block, `pass_over`/`pass_over_flag`,
  `object_engine`/`object_part` take `&mut Layers`, the `--defer-taskrun`/`--sqpoll` exclusion after
  the pass-over), `options.rs` (`Layers::pass_over`); `man/aeiou-config.5.md`, `man/aeiou.1.md`,
  `runner/REFERENCE.md` §14; tests.
- **Found by:** RV3b: the aeiou-config(5) EXAMPLE verbatim on file_per_sample → "--threads sets the
  event-loop threads of … this run has one thread per actor (--threads from config …)", exit 2;
  13 of 14 committed abstracts declare sync or mmap; `AEIOU_OBJECT_THREADS` refused POSIX datagen.
- **Fix:** a backend knob from env or config the run's API cannot use (io_uring knobs, `--aio-depth`,
  `--threads`, mmap knobs, POSIX knobs when every name is s3, `--object-threads`/`--object-part-mib`
  with no object store) is passed over with a `WARNING: … is not used: …` line in the options block
  and `layers.warnings` in the report, and dropped from `layers.options`; the command line's is
  still refused.
- **Evidence:** `options::fixed_options_are_refused_from_the_environment_and_the_file` (rewritten)
  fails on HEAD and passes; unit test `a_knob_from_a_lower_layer_is_passed_over_and_a_default_shows_its_value`;
  same suite runs as bug 119.

## 121. An empty `AEIOU_ROOT` (or `root = ""`) was taken as the current directory

- **Where:** `runner/aeiou/src/options.rs` (empty values from env or config); `man/aeiou-config.5.md`; tests.
- **Found by:** RV3b: `AEIOU_ROOT= aeiou datagen …` wrote `./train` with `root =  [env AEIOU_ROOT]`,
  exit 0, while `--root ""` is refused; an unset shell variable in a launcher would put a corpus, or
  `--clean-namespaces`, in the cwd. `AEIOU_COORDINATOR=`, `AEIOU_CONFIG=` were accepted too.
- **Fix:** an empty value from env or config where a value is required is a usage error ("AEIOU_ROOT:
  empty, and a value is required (leave it unset for none)"), for root, coordinator, report-json,
  endpoint, and config (an empty `AEIOU_CONFIG` no longer means no file); `endpoint = []` stays the
  empty list.
- **Evidence:** `options::odd_values_from_the_lower_layers_and_the_defaults_in_effect` (nothing written
  in the cwd) and the unit test; before-evidence by command, since the test stops at bug 119's panic
  on HEAD; same suite runs as bug 119.

## 122. `AEIOU_NO_X` for a command-line-only boolean suggested a spelling that is itself refused

- **Where:** `runner/aeiou/src/options.rs` (the no-x check moved from `ConfigFile::load` to `finish`);
  `man/aeiou-config.5.md`; tests.
- **Found by:** RV3b: `AEIOU_NO_METRICS=1` → "set in the environment as AEIOU_METRICS=true or false",
  which is then refused as the command line's alone; also `AEIOU_NO_CLEAN_NAMESPACES`, `[dry-run]
  no-metrics`.
- **Fix:** "AEIOU_NO_METRICS: --metrics is the command line's alone (`--no-metrics` there)".
- **Evidence:** the extended no-x unit test and the integration test; same suite runs as bug 119.

## 123. Config type errors showed Rust Debug output (`got String("3")`)

- **Where:** `runner/aeiou/src/options.rs` (`toml_described`); tests.
- **Found by:** RV3b: `[dry-run] threads = "3"` → "expected an integer, got String(\"3\")".
- **Fix:** "expected an integer, got the string \"3\"", "expected true or false, got the integer 1".
- **Evidence:** unit test `empty_and_non_utf8_values_and_the_toml_types`; same suite runs as bug 119.

## 124. The options block showed `none` where a default was in effect

- **Where:** `runner/aeiou/src/options.rs`, `main.rs` (`show_default` in `datagen_cmd`, `dry_run`,
  `run_checked`); tests.
- **Found by:** RV3b: datagen `writers = none [default]` then "writers 20"; `object-threads = none` while
  help says 2; dry-run `threads = none` then "on 20 thread(s)", against §14 "its value in effect".
- **Fix:** the block shows the value in effect ("20 (all cores)", "one per actor", "256", "fault", "2",
  "8"); the report JSON keeps `null` for a default.
- **Evidence:** the integration and unit tests above; same suite runs as bug 119.

## 125. The docs disagreed with the code, and each other, on which options are layered

- **Where:** `man/aeiou-config.5.md` (DESCRIPTION, FILE FORMAT, ENVIRONMENT, OPTIONS BY KIND),
  `runner/REFERENCE.md` §14 (lists, pass-over, Values, negation, refusals, Tests), `man/aeiou.1.md`.
- **Found by:** RV3b: aeiou-config(5) listed datagen layered as only root/endpoint/writers/object-threads
  while aeiou(1) marked `--rank`, `--ranks`, `--coordinator`, `--progress-secs` *Layered* and their
  variables work; §14 omitted `--object-threads`, `--object-part-mib`, `--progress-secs`.
- **Fix:** the lists follow the code, with datagen's fixed forms (`--resume`, `--grow`, `--shrink`,
  `--settle`) named.
- **Evidence:** man tests (`man.rs`, `test_man.py`) pass; same suite runs as bug 119.

## 126. An integral float for an integer parameter (`samples=3e3`) gave another dataset id, and a fractional one (`samples=20.7`) was taken and rounded

- **Where:** `runner/aeiou/src/eval.rs` (`integral`, `keep_int`, `keep_int_json`), applied in
  `Params::new` and `runner/aeiou/src/payload.rs` `param_values`; `builder/aeiou/params.py`
  (`keep_int` in `check`, normalized values to `given_rules`); `schema/README.md` §8 Rules; tests.
- **Found by:** RC1 third-round datagen review (RV3c): `--param samples=3e1` recorded `"count": 30.0`,
  id 51a94b9a… against 8b486790… for `samples=30`, byte-identical files; `run --param samples=30`
  was refused "count: 30 vs 30.0"; `samples=20.7` wrote 21 files with `count: 20.7`; yet
  `files_per_directory=10.0` was refused "not an integer".
- **Fix:** where a parameter's default (or an array default's elements) is an integer, an integral
  float within i64 becomes that integer and a fractional or out-of-range one is refused ("--param
  samples: the default is an integer, the value 20.7 is not"), in parameter files and `--param`
  alike, in the runner and the Python checker. No id changes for any value already an integer; no
  committed parameter file has such a float.
- **Evidence:** `golden::an_integral_float_is_the_integer_its_default_is` and
  `test_params.py::test_an_integral_float_is_the_integer_its_default_is` fail on HEAD and pass;
  `samples=3e1` now gives id 8b486790… and the run accepts it. Merged on HEAD 44969ca:
  `--tests --lib --bins` with MinIO all pass (all 158, lib 70), builder suite with MinIO 407, CI's
  checker diff empty, 22 parameter files ok, `aeiou-build --hermetic --twice --check` rc 0,
  `aeiou-catalog check` 13 mechanisms, 16 cases ok. Fixed in an isolated worktree and applied.

## 127. Progress on an array file read "0 B written, 0 B/s" with no ETA for its whole life

- **Where:** `runner/aeiou/src/datagen.rs` (`Track::pieces`/`pieces_done`, `Plan::note` `live`,
  `Ticker::tick`); tests.
- **Found by:** RV3c: faiss `--dataset lists --param lists=20000 --writers 1 --progress-secs 0.5` printed
  `0/1 files, 0 B written, 0 B/s` until the end; the default corpus is a 256 GiB file.
- **Fix:** the manifest's counts stay whole-file only; the printed line adds the live bytes of an
  unfinished array file, its rate, and an ETA from the pieces done.
- **Evidence:** `coord::an_array_files_progress_is_live` fails on HEAD and passes ("32.53 MiB written,
  644.55 MiB/s, about 0 s left"; id unchanged 2e3da472…); same suite runs as bug 126.

## 128. The O_DIRECT and unaligned-tail findings were made once per datagen and applied to every root

- **Where:** `runner/aeiou/src/datagen.rs` (`Mode` per dataset via `Mode::sharing(&halt)`, `part_json`
  `direct`/`pad_tails` per dataset per host, manifest `direct` the AND for that dataset); 
  `runner/REFERENCE.md` §5; tests.
- **Found by:** RV3c: lmcache with `--root` on ext4 (refuses unaligned direct tails) and an endpoint on
  tmpfs: strace showed an `ftruncate` after every tmpfs write; by the same path an O_DIRECT refusal
  on one root would send every later dataset through the page cache and record `direct: false` for all.
- **Fix:** each dataset keeps its own mode, sharing only the halt flag; notes and the recorded
  `direct`/`pad_tails` are per dataset.
- **Evidence:** `coord::the_direct_findings_are_each_roots_own` (probes each file system) fails on HEAD
  and passes; strace after: 0 `ftruncate` on tmpfs; an outright O_DIRECT refusal is covered by
  structure only (no local file system refuses it); same suite runs as bug 126.

## 129. A failed root creation left the roots made before it

- **Where:** `runner/aeiou/src/datagen.rs` (topmost new ancestor per plan, `unmake` on failure); tests.
- **Found by:** RV3c: `--endpoint sysp=/mnt/nfs/sys` (not writable) failed "Permission denied" and left
  `d15/kv/sub/`, against the comment that a refused datagen leaves nothing.
- **Fix:** on a `create_dir_all` failure the directories this datagen made are removed in reverse
  order with `remove_dir` (empty ones only, NotFound tolerated).
- **Evidence:** `coord::a_root_that_cannot_be_made_leaves_nothing_behind` fails on HEAD and passes; same
  suite runs as bug 126.

## 130. `--compress` ratios a 4 KiB piece cannot hold were recorded as asked, not as made

- **Where:** `runner/aeiou/src/payload.rs` (`compress_achieved`), `datagen.rs` (note and provenance),
  `main.rs` (`--compress`/`--write-compress` at most 4096); `man/aeiou.1.md`; tests.
- **Found by:** RV3c: `random_per_piece` gives `4096/ceil(4096/C)`: `--compress 5000` made 4096,
  2049–4095 made 2048, 1000 made 819.2, while the manifest recorded C.
- **Fix:** C above 4096 is a usage error (exit 2) for both flags; a C the pieces cannot hold exactly
  is accepted (3 makes 2.9985), a note says what is made, and the provenance records
  `compress_achieved`; the payload still records C, so no dataset id changes for any C.
- **Evidence:** `coord::a_compress_ratio_a_piece_cannot_hold_is_recorded_as_made` and
  `usage::a_meaningless_number_is_a_usage_error` fail on HEAD and pass; `--compress 1000` id
  c92e609f… before and after; same suite runs as bug 126.

## 131. A shrink's plan line read "3266 id(s) of 1234", and a killed corpus rerun was told to remove it

- **Where:** `runner/aeiou/src/datagen.rs` (the plan line; `unfinished_corpus` in the directory and
  object non-empty checks); tests.
- **Found by:** RV3c: `--shrink 1234` on 4,500 files printed "3266 id(s) of 1234 … from the top down";
  rerunning a killed datagen without `--resume` said "is not empty … remove it first" although the
  root held a `writing` manifest.
- **Fix:** "the ids from 12 to 44 of 45 (those from 12 up removed)" (with "N written" when the new
  short last file is rewritten); a root holding an unfinished corpus says "`aeiou datagen --resume
  ROOT` finishes it, `--settle ROOT` keeps what it reached, or remove it first".
- **Evidence:** `coord::grow_and_shrink_make_the_corpus_of_the_new_count` and
  `coord::an_unfinished_corpus_points_at_resume_and_settle` fail on HEAD and pass; same suite runs as
  bug 126.

## 132. An object request slower than 30 s failed the run after minutes of silent retries, each re-sending the whole part

- **Where:** `runner/aeiou/src/object.rs` (`ClientOptions::with_timeout_disabled`, `retry_config()`,
  `RangedService` counting retries by cause, `object::retries()`), `runner/aeiou/src/run.rs` and
  `report.rs` (`result.object {retries, retries_by_cause}`, merged across hosts; text line and
  WARNING); `runner/REFERENCE.md` §12 and §16 "Timeouts and retries"; `man/aeiou.1.md` OBJECT
  STORES; tests.
- **Found by:** RC1 third-round object-engine review (RV3a): `AmazonS3Builder::from_env()` kept
  object_store's 30 s per-request timeout and 10 retries within 180 s, while `--object-part-mib`
  allows 5120. With upstream throttled to 1 MiB/s, a 40 MiB part failed after 2m50s "… in 180.6s,
  after 5 retries … (EIO)", sent 6 times; the report showed no retries.
- **Fix:** no deadline per request (a POSIX op has none; the store's time is what is measured; the
  5 s connect timeout stays, `AWS_TIMEOUT` still applies); retries stay (every S3 SDK retries: part
  of the API), pinned in code (10 within 180 s, 100 ms doubling to 15 s, on 5xx/429/408/connection
  failures), and every retry is counted by cause and reported (`object retries N (503 ×n)` plus a
  WARNING; `result.object` in the JSON, `aeiou_report` stays 1), so a throttling store shows.
- **Evidence:** unit test `object::imp::tests::a_request_has_no_deadline_but_its_connect_and_its_retries_are_pinned`
  and `object::a_store_throttling_is_retried_and_every_retry_reported` (a proxy answering two `503
  SlowDown`: `object retries 2 (503 ×2)`, `result.object == {"retries":2,…}`) fail with the change
  undone and pass; the throttled 40 MiB repro now exits 0. Merged on HEAD 0a872a9: `--tests --lib
  --bins` with MinIO all pass (all 160, lib 73), `--features formats` usage 10, builder suite with
  MinIO 407, `aeiou-catalog check` ok. Fixed in an isolated worktree and applied.

## 133. Two datasets placed at one directory (or one inside the other) overwrote each other's manifest, exit 0

- **Where:** `runner/aeiou/src/endpoint.rs` (`Endpoints::datasets_apart`, called from `parse` and with
  `--root` from datagen and run in `main.rs`); `runner/REFERENCE.md` §4; `man/aeiou.1.md` ENDPOINTS; tests.
- **Found by:** RC1 third-round datagen review (RV3c): `Endpoints::parse` refused two names only with
  one abstract root. `aeiou datagen --endpoint lists=shared --endpoint index=shared …
  faiss_ivf_ondisk.ast.json` exited 0 with both datasets in `shared/` and its manifest naming only
  `lists`; a later run could not find `index`.
- **Fix:** every dataset's root is resolved to where it lands (its endpoint, a placed namespace it
  moves with, or `--root`; for s3 the server, bucket, and full prefix), directories made absolute
  with `.`/`..` dropped and symlinks followed; equal or nested places (whole components) are a
  usage error naming both, before anything is written. Namespaces holding datasets stay allowed.
- **Evidence:** unit tests `endpoint::tests::two_datasets_are_never_placed_at_one_directory_or_one_inside_the_other`,
  `two_object_datasets_are_never_placed_at_one_prefix_or_one_inside_the_other`, and
  `usage::two_datasets_in_one_place_are_a_usage_error` fail with `datasets_apart` short-circuited and
  pass; same suite runs as bug 132.
