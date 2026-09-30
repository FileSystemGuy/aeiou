# CLAUDE.md

Abstract-driven I/O benchmark runner (Rust + `io-uring` crate + a pure-Rust TCP coordinator for multi-host; no MPI). Pre-implementation stage.

Built for the MLPerf Storage WG, meant to be general: name packages, modules, env vars, CLIs, files, and config keys for the mechanism or the repo (`aeiou`), never for MLPerf (convention in `PROJECT_BRIEF.md` §8: runner binary `aeiou` with subcommands, Python helpers `aeiou-build`, `aeiou-verify`, `aeiou-fit`); anything that is WG process (divisions, published hashes, reference parameters) is listed in `PROJECT_BRIEF.md` §8 and says so where it appears. Adding something WG-specific is a conscious, documented decision.

Read these before any design or coding work:
- `PROJECT_BRIEF.md`: original requirements, decisions made so far, open items
- `NAPKIN_MATH.md`: DRAM/IOPS estimates, risk register, spikes, round-2 decisions (§8)
- `GRAMMAR_OPTIONS.md`: abstract-language extension options
- `DESIGN_REVIEW.md`: the 2026-09-25 review and the reasoning behind the determinism rules below
- `ABSTRACTS.md`: paper abstracts for the eight target workloads, and the constructs they surfaced (§9, all accepted 2026-09-30)
- `schema/README.md` and `schema/abstract-ast.schema.json`: the AST contract (v0.1), canonical form, and validator rules
- `builder/README.md`: the Python builder (`aeiou`), its API, the build-time discipline, and the hermetic harness; `builder/abstracts/*.py` are the eight workloads as authoring scripts and `schema/examples/*.ast.json` their generated ASTs (never edit them by hand; JSON is the only format on the contract)

Invariants that must not be broken:
- Exact per-GPU reproducibility. Randomness is positional: keyed on (seed, gpu_id, site, enclosing loop indices). No per-actor draw counters, no shared stateful RNGs, no work distributed based on timing.
- Split work by global GPU id over positions in the Feistel shuffle, never by host rank or by file id. Position is a formula, `g + G·(b·B + j)`, never a counter shared by a GPU's workers. Epochs are `drop_last`.
- Producers are finite: a loader dispatches exactly `steps` batches. The op multiset is fixed before the run starts.
- The workload fingerprint is order-independent (sum of per-op hashes). Issue order within a GPU is timing-dependent and must never be hashed.
- The dataset seed is separate from `--seed` and lives in the dataset definition and the datagen manifest (`.aeiou-dataset.json` per dataset root; `aeiou run` compares the resolved dataset definition, never the whole parameter set or the abstract hash; datasets are read-only (V12); no generated name may begin with `.aeiou` (V13); namespaces must be empty at start).
- Never materialize per-file data structures. Filenames and sizes are computed from patterns.
- Data is reproducible, not self-describing (revised 2026-09-29): the bytes at `(dataset seed, file id, offset)` are a pure function of that tuple with controlled dedupe and compression ratios; no per-block headers, ever, since they force a dedupe ratio of one. The runner does structural checks only (byte counts, short reads, index bytes match the computed layout); content verification is a separate Python tool, never in a scored run. The runner never interprets sample data.
- Datasets are sample spaces: the Feistel shuffle runs over sample ids (`map` access) or shard ids (`stream` access), declared per dataset. Container layout (sample → file, offset, length) is a formula; format reader protocols come from format classes in the Python builder that emit plain POSIX nodes. The Rust runner is format-ignorant and refuses access modes the format class does not support.
- Random access is the null model, never the model, for data-dependent workloads (VDB, KV cache). Structure storage can exploit (reuse, runs, skew, dependency shape, write-then-read lag) is expressed with positional distributions and `x @ i` (a binding evaluated at an earlier index of its loop; self-reference only at a strictly smaller index), and checked against real traces with `--dry-run --metrics` (`GRAMMAR_OPTIONS.md` §5).
- The application/solution boundary is the interposition test (`PROJECT_BRIEF.md` §5): CLOSED runs the `sync` backend; anything an `LD_PRELOAD` shim could do under an unmodified app is solution. The run seed and file order are application-private.
- The abstract is POSIX-shaped and identical for every I/O backend. Backends (`sync`, `sync-direct`, `posix-aio`, `libaio`, `io_uring`, `mmap`, `gds`, `nixl-posix`, `libnfs`) map ops to APIs behind one trait with an issue half and a completion-source half; they never change the op stream or the fingerprint. List and axes in `PROJECT_BRIEF.md` §4.
- Authoring is the Python builder (Option D, decided 2026-09-30); the serde AST is the contract and the only thing the runner executes. The AST stays symbolic: loops are loop nodes, draws are nodes, parameters are references. Python never runs the workload and never reaches the client nodes. Object sizes in a `namespace` are computed from the writes' positional expressions, never observed with `fstat`.
- No MPI, no tokio/tonic in the runner. Cross-host coordination is blocking `std::net` on one thread behind the `Coordinator` trait (`NAPKIN_MATH.md` §8.A).
