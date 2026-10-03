//! Resource limits before the start gate (`NAPKIN_MATH.md` R8, `runner/README.md` §11): an
//! estimate of the files a host will hold open and the threads it will run, compared with
//! `RLIMIT_NOFILE`, `RLIMIT_NPROC`, `kernel.threads-max`, and `vm.max_map_count`, so that a
//! run that cannot fit is refused with the limit named instead of failing at hour two with
//! `EMFILE`. The soft limits are raised to the hard ones first.
//!
//! The need is an **estimate**: the first instance of each template on this host is walked
//! without I/O for at most `BUDGET` ops, its concurrent contexts are counted (the
//! sub-actors of a `parallel` are all live at once, a `loader`'s workers live beside the
//! line that forked them), and the result is multiplied by the instances this host runs.
//! Opens that come later than the budget, or that differ between instances, are not seen;
//! the measured peak is in the report (`HostCounters::open_files_peak`).

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::eval::Model;
use crate::run::{gpu_range, RunOpts};
use crate::vm::{actor_counts, Event, ForkKind, OpKind, Snapshot, Vm};

/// Ops walked per template for the estimate.
pub const BUDGET: u64 = 1 << 20;

/// What one context and everything it forks holds at its peak.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Peak {
    files: u64,
    threads: u64,
}

struct Walk {
    left: u64,
    truncated: bool,
}

impl Walk {
    /// Walk the body the VM has started; `Peak` of this context with its sub-actors.
    fn context<'m, 'a: 'm>(&mut self, vm: &mut Vm<'m, 'a>) -> Result<Peak> {
        let mut own: u64 = 0; // files this context has opened and not closed
        let mut beside = Peak::default(); // loader workers, live until the context ends
        let mut peak = Peak { files: 0, threads: 1 };
        loop {
            if self.left == 0 {
                self.truncated = true;
                break;
            }
            enum Step<'a> {
                End,
                Open,
                Close,
                Other,
                Fork(ForkKind<'a>),
                /// A `trace` node: its lanes are threads at once, its peak of open descriptors.
                Trace(u64, u64, u64),
            }
            let step = match vm.next()? {
                None => Step::End,
                Some(Event::Op(op, _)) => match op.kind {
                    OpKind::Open => Step::Open,
                    OpKind::Close => Step::Close,
                    _ => Step::Other,
                },
                Some(Event::Control(..)) => continue,
                Some(Event::Fork(k)) => Step::Fork(k),
                Some(Event::Trace(t, _)) => Step::Trace(t.lanes() as u64, t.peak_open, t.ops),
            };
            match step {
                Step::End => break,
                Step::Trace(lanes, files, ops) => {
                    self.left = self.left.saturating_sub(ops);
                    peak.files = peak.files.max(own + beside.files + files);
                    peak.threads = peak.threads.max(lanes + beside.threads);
                }
                Step::Open => {
                    self.left -= 1;
                    own += 1;
                    peak.files = peak.files.max(own + beside.files);
                }
                Step::Close => {
                    self.left -= 1;
                    // closing an inherited file hides it from this context; the parent holds it
                    own = own.saturating_sub(1);
                }
                Step::Other => self.left -= 1,
                Step::Fork(kind) => {
                    let snap = vm.snapshot();
                    vm.accept_fork();
                    match kind {
                        ForkKind::Parallel { width, .. } => {
                            let sum = self.subs(&snap, (0..width.max(0)).collect(), width.max(0) as u64)?;
                            peak.files = peak.files.max(own + beside.files + sum.files);
                            // sub-actor 0 runs on the forking thread
                            peak.threads = peak.threads.max(sum.threads + beside.threads);
                        }
                        ForkKind::Loader { workers, batches, .. } => {
                            // the first batch of every worker stands for its later ones
                            let n = workers.min(batches).max(0);
                            let sum = self.subs(&snap, (0..n).collect(), n as u64)?;
                            beside.files += sum.files;
                            beside.threads += sum.threads;
                            peak.files = peak.files.max(own + beside.files);
                            peak.threads = peak.threads.max(1 + beside.threads);
                        }
                    }
                }
            }
        }
        Ok(peak)
    }

    /// The sum over the sub-actors at `indices`, scaled to `width` when the budget ends first.
    fn subs<'m, 'a: 'm>(&mut self, snap: &Snapshot<'m, 'a>, indices: Vec<i64>, width: u64) -> Result<Peak> {
        let mut sum = Peak::default();
        let mut walked = 0u64;
        let mut vm = Vm::resume(snap.clone());
        for k in indices {
            if walked > 0 {
                if self.left == 0 {
                    self.truncated = true;
                    break;
                }
                vm.resume_from(snap);
            }
            vm.start_sub(k);
            let p = self.context(&mut vm)?;
            sum.files += p.files;
            sum.threads += p.threads;
            walked += 1;
        }
        if walked > 0 && walked < width {
            sum.files = (sum.files * width).div_ceil(walked);
            sum.threads = (sum.threads * width).div_ceil(walked);
        }
        Ok(sum)
    }
}

/// What this host is estimated to need.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Need {
    /// Files the actors hold open at once.
    pub open_files: u64,
    /// Sequential contexts live at once: the threads of a thread-per-actor backend.
    pub contexts: u64,
    /// The budget ended before a walked instance did.
    pub truncated: bool,
}

/// Walk the first instance of each template on this host and scale by the instances it runs.
pub fn estimate(model: &Model<'_>, opts: &RunOpts) -> Result<Need> {
    let mut need = Need::default();
    for (name, count) in actor_counts(model)? {
        let (lo, hi) = gpu_range(count, opts.ranks, opts.rank, opts.rank_rotate);
        if hi <= lo {
            continue;
        }
        let mut w = Walk { left: BUDGET, truncated: false };
        let mut vm = Vm::new(model, name, lo, count);
        vm.start(&model.ast.actors[name].body);
        let p = w.context(&mut vm).with_context(|| format!("limit estimate: actor `{name}` instance {lo}"))?;
        let n = (hi - lo) as u64;
        need.open_files += p.files * n;
        need.contexts += p.threads * n;
        need.truncated |= w.truncated;
    }
    Ok(need)
}

fn rlimit(res: libc::__rlimit_resource_t) -> (u64, u64) {
    let mut l = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    // SAFETY: `l` is a valid rlimit for the call to fill
    unsafe { libc::getrlimit(res, &mut l) };
    (l.rlim_cur, l.rlim_max)
}

/// Raise the soft limit to the hard one; returns `(soft before, soft now, hard)`.
fn raise(res: libc::__rlimit_resource_t) -> (u64, u64, u64) {
    let (soft, hard) = rlimit(res);
    if soft < hard {
        let l = libc::rlimit { rlim_cur: hard, rlim_max: hard };
        // SAFETY: a valid rlimit; raising the soft limit to the hard one needs no privilege
        unsafe { libc::setrlimit(res, &l) };
    }
    (soft, rlimit(res).0, hard)
}

fn sysctl(path: &str) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn show(v: u64) -> String {
    if v == libc::RLIM_INFINITY { "unlimited".into() } else { v.to_string() }
}

/// The limits of this host against the estimate, as checked before the gate.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Limits {
    pub need: Need,
    /// Descriptors the process needs: the estimate plus what was open at the check and the
    /// event loops' own.
    pub fds_needed: u64,
    pub nofile_soft_before: u64,
    pub nofile_soft: u64,
    pub nofile_hard: u64,
    pub threads_needed: u64,
    pub nproc_soft_before: u64,
    pub nproc_soft: u64,
    pub nproc_hard: u64,
    pub threads_max: Option<u64>,
    pub maps_needed: u64,
    pub max_map_count: Option<u64>,
}

impl Limits {
    pub fn lines(&self) -> Vec<String> {
        let raised = |before: u64, now: u64| if before != now { format!(" (raised from {})", show(before)) } else { String::new() };
        vec![
            format!(
                "limits: open files ~{} (+{} of the process){}; RLIMIT_NOFILE soft {}{} hard {}",
                self.need.open_files,
                self.fds_needed - self.need.open_files,
                if self.need.truncated { format!(", estimate from the first {BUDGET} ops") } else { String::new() },
                show(self.nofile_soft),
                raised(self.nofile_soft_before, self.nofile_soft),
                show(self.nofile_hard)
            ),
            format!(
                "limits: threads ~{}; RLIMIT_NPROC soft {}{} hard {}, kernel.threads-max {}; mappings ~{}, vm.max_map_count {}",
                self.threads_needed,
                show(self.nproc_soft),
                raised(self.nproc_soft_before, self.nproc_soft),
                show(self.nproc_hard),
                self.threads_max.map_or("?".into(), |v| v.to_string()),
                self.maps_needed,
                self.max_map_count.map_or("?".into(), |v| v.to_string())
            ),
        ]
    }

    /// What does not fit, each with the limit and how to raise it.
    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.fds_needed > self.nofile_soft {
            out.push(format!(
                "about {} open files needed, RLIMIT_NOFILE is {} (hard): raise it (`ulimit -Hn`, `nofile` in /etc/security/limits.conf, or `LimitNOFILE=` for a service; `fs.nr_open` bounds it)",
                self.fds_needed,
                show(self.nofile_hard)
            ));
        }
        // SAFETY: no arguments, no side effects
        let root = unsafe { libc::geteuid() } == 0;
        if !root && self.threads_needed > self.nproc_soft {
            out.push(format!(
                "about {} threads needed, RLIMIT_NPROC is {} (hard; it counts every task of this user): raise it (`ulimit -Hu`, `nproc` in limits.conf), or use an event-loop backend (`io_uring`, `libaio`)",
                self.threads_needed,
                show(self.nproc_hard)
            ));
        }
        if let Some(max) = self.threads_max {
            if self.threads_needed > max {
                out.push(format!("about {} threads needed, kernel.threads-max is {max}: raise the sysctl, or use an event-loop backend", self.threads_needed));
            }
        }
        if let Some(max) = self.max_map_count {
            if self.maps_needed > max {
                out.push(format!(
                    "about {} memory mappings needed (two per thread{}), vm.max_map_count is {max}: raise the sysctl",
                    self.maps_needed,
                    ", one per open file under `mmap`"
                ));
            }
        }
        out
    }
}

/// Estimate, raise the soft limits, and compare. `loops` is the event-loop thread count of
/// an event-loop backend (0 for the thread-per-actor backends); `mmap` adds a mapping per
/// open file.
pub fn measure(model: &Model<'_>, opts: &RunOpts, loops: u64) -> Result<Limits> {
    let need = estimate(model, opts)?;
    let open_now = std::fs::read_dir("/proc/self/fd").map(|d| d.count() as u64).unwrap_or(16);
    let tasks_now = std::fs::read_dir("/proc/self/task").map(|d| d.count() as u64).unwrap_or(1);
    let (nofile_soft_before, nofile_soft, nofile_hard) = raise(libc::RLIMIT_NOFILE);
    let (nproc_soft_before, nproc_soft, nproc_hard) = raise(libc::RLIMIT_NPROC);
    // a ring and an eventfd per loop; rank 0 holds a socket per host; slack for the report,
    // the manifests, and the sampler's reads of /proc
    let fds_needed = need.open_files + open_now + 4 * loops + opts.ranks.max(1) as u64 + 16;
    let threads_needed = tasks_now + if opts.backend.event_loop() { loops } else { need.contexts } + 2;
    let maps_now = std::fs::read_to_string("/proc/self/maps").map(|s| s.lines().count() as u64).unwrap_or(256);
    let maps_needed = maps_now + 2 * threads_needed + if opts.backend == crate::backend::BackendKind::Mmap { need.open_files } else { 0 };
    Ok(Limits {
        need,
        fds_needed,
        nofile_soft_before,
        nofile_soft,
        nofile_hard,
        threads_needed,
        nproc_soft_before,
        nproc_soft,
        nproc_hard,
        threads_max: sysctl("/proc/sys/kernel/threads-max"),
        maps_needed,
        max_map_count: sysctl("/proc/sys/vm/max_map_count"),
    })
}

/// `measure`, refusing the run when something does not fit unless `ignore` is set.
pub fn check(model: &Model<'_>, opts: &RunOpts, loops: u64, ignore: bool) -> Result<Limits> {
    let l = measure(model, opts, loops)?;
    let problems = l.problems();
    if !problems.is_empty() && !ignore {
        bail!("this host's limits do not fit the run (estimates; --ignore-limits starts anyway):\n  {}", problems.join("\n  "));
    }
    Ok(l)
}
