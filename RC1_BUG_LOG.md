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

