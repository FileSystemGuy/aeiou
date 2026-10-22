// One test binary for every file but uring_knobs.rs and limits.rs, which count their own
// process's threads and open files and stay binaries of their own: three links of the crate,
// not twelve (CI, 2026-10-07).
mod common;
mod coord;
mod cov_c2;
mod cov_c3;
mod cases;
mod cov_c1;
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
