// One test binary for every file but uring_knobs.rs and limits.rs, which count their own
// process's threads and open files and stay binaries of their own: three links of the crate,
// not twelve (CI, 2026-10-07).
mod coord;
mod golden;
mod layout;
mod man;
mod metrics;
#[cfg(feature = "object")]
mod object;
mod options;
mod report;
mod run;
mod trace;
mod usage;
