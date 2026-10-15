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

