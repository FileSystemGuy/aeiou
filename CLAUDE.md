# CLAUDE.md

Abstract-driven I/O benchmark runner (Rust + `io-uring` crate + MPI). Pre-implementation stage.

Read these before any design or coding work:
- `PROJECT_BRIEF.md`: original requirements, decisions made so far, open items
- `NAPKIN_MATH.md`: DRAM/IOPS estimates, risk register, spikes, round-2 decisions (§8)
- `GRAMMAR_OPTIONS.md`: abstract-language extension options

Invariants that must not be broken:
- Exact per-GPU reproducibility. Use a counter-based RNG keyed on (seed, gpu_id, epoch, step, site, draw). Never use shared stateful RNGs or distribute work based on timing.
- Split work by global GPU id over positions in the Feistel shuffle, never by MPI rank or by file id.
- Never materialize per-file data structures. Filenames and sizes are computed from patterns.
