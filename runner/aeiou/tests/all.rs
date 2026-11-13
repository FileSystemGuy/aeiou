// One test binary for every file but uring_knobs.rs and limits.rs, which count their own
// process's threads and open files and stay binaries of their own: three links of the crate,
// not twelve (CI, 2026-10-07).
mod common;
mod coord;
mod cov_c2;
mod cov_c3;
mod capture;
mod capture_requests;
mod capture_ebpf;
mod cases;
mod listing;
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
mod strace_corpus;
mod digest_corpus;
mod compare_corpus;
mod compare_tool;
mod emit;
mod applications;
mod trace_tool;
mod usage;
mod timeline;
mod seqcache;
mod tiers;
