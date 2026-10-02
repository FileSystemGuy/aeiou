//! Host-side counters around a run: what the client machine did while the abstract ran, as
//! distinct from what the abstract did (`run::Stats`). They are measurements of the
//! solution, never part of the fingerprint, and they differ by backend where the op stream
//! does not: the kernel threads `io_uring` punts to, the CPU a backend burns, the RPCs the
//! NFS client puts on the wire for the same POSIX ops.
//!
//! Three sources, all Linux `/proc` or `getrusage`, each optional where the kernel lacks it:
//!
//! - `/proc/self/status` `Threads:` sampled every [`SAMPLE`] for the peak task count of the
//!   process; at every sample where the count changed, `/proc/self/task/*/comm` is scanned
//!   and the `iou-wrk-*` threads counted (io-wq workers; they linger idle for seconds, so a
//!   sampler at this rate sees the peak).
//! - `getrusage(RUSAGE_SELF)` before and after: user and system CPU time, and the peak RSS.
//! - `/proc/self/mountstats` before and after for the mount `--root` is on: the device,
//!   mount point, and filesystem type; for an NFS mount, the deltas of the `bytes:` line
//!   and of the per-op RPC statistics. **These are the mount's counters, not the process's**:
//!   every process on the host using that mount is in them.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// The task sampler's period.
pub const SAMPLE: Duration = Duration::from_millis(10);

/// What the host did over the run; one per host, summed by `Report::merge_all`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct HostCounters {
    /// Peak count of tasks (threads) in the process, sampled.
    pub tasks_peak: u64,
    /// Peak count of io-wq worker threads (`iou-wrk-*`), sampled; 0 under the `sync` backends.
    pub iowq_workers_peak: u64,
    /// The `SQPOLL` submission threads of the rings: the distinct threads the rings' `fdinfo`
    /// states (`SqThread:`) when its loop's work is done and the ring still open. They live
    /// exactly as long as their rings, so this is not a sample; 0 without `--sqpoll`.
    #[serde(default)]
    pub sqpoll_threads: u64,
    /// CPU time of the process over the run.
    pub cpu_user_ns: u64,
    pub cpu_sys_ns: u64,
    /// Peak resident set of the process, bytes (`ru_maxrss`; a process-lifetime peak, not a delta).
    pub maxrss_bytes: u64,
    /// Page faults of the process over the run (`getrusage`): minor (the page was in
    /// memory) and major (it took I/O). Under the `mmap` backend these carry the reads.
    #[serde(default)]
    pub minor_faults: u64,
    #[serde(default)]
    pub major_faults: u64,
    /// The most files the actors held open at once: counted at every open and close the
    /// runner makes (`backend::open_files_peak`), not sampled.
    #[serde(default)]
    pub open_files_peak: u64,
    /// The mount `--root` is on, when `/proc/self/mountstats` lists one.
    pub mount: Option<MountCounters>,
}

/// One mount's identity and, for NFS, its client counters over the run.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MountCounters {
    pub mount_point: String,
    pub device: String,
    pub fstype: String,
    /// The mount's options: the `opts:` line of `mountstats` on NFS (`vers`, `rsize`,
    /// `acregmin`…, `lookupcache`, `nconnect`: what decides the client's caching), the
    /// options field of `/proc/self/mounts` elsewhere. Distinct hosts' joined with ` | `.
    #[serde(default)]
    pub opts: Option<String>,
    /// `read_ahead_kb` of the mount's backing device info (`/sys/class/bdi/MAJOR:MINOR`): the
    /// readahead window, which bounds what one page fault reads under the `mmap` backend
    /// and shapes buffered sequential reads. A setting of the solution, recorded, never
    /// set. Empty where the filesystem has none (tmpfs); the distinct values once hosts merge.
    #[serde(default)]
    pub read_ahead_kb: Vec<u64>,
    pub nfs: Option<NfsCounters>,
}

/// `read_ahead_kb` of the backing device info of the filesystem `path` is on. `st_dev` names
/// it directly for NFS and other anonymous devices and for a whole disk; a partition's is
/// its disk's, reached through `/sys/dev/block`.
pub fn read_ahead_kb(path: &Path) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    let dev = std::fs::metadata(path).ok()?.dev();
    read_ahead_kb_at(Path::new("/sys"), libc::major(dev), libc::minor(dev))
}

pub fn read_ahead_kb_at(sys: &Path, major: u32, minor: u32) -> Option<u64> {
    let id = format!("{major}:{minor}");
    [sys.join("class/bdi").join(&id), sys.join("dev/block").join(&id).join("bdi"), sys.join("dev/block").join(&id).join("../bdi")]
        .iter()
        .find_map(|d| std::fs::read_to_string(d.join("read_ahead_kb")).ok())
        .and_then(|v| v.trim().parse().ok())
}

/// Deltas of the NFS client's `bytes:` line and per-op RPC statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct NfsCounters {
    /// Bytes read and written by the application through the page cache (`normalreadbytes`,
    /// `normalwritebytes`) and with `O_DIRECT` (`directreadbytes`, `directwritebytes`).
    pub normal_read_bytes: u64,
    pub normal_write_bytes: u64,
    pub direct_read_bytes: u64,
    pub direct_write_bytes: u64,
    /// Bytes moved to and from the server (`serverreadbytes`, `serverwritebytes`).
    pub server_read_bytes: u64,
    pub server_write_bytes: u64,
    /// RPCs by procedure, those with a non-zero count over the run.
    pub ops: BTreeMap<String, NfsOp>,
}

/// One procedure's deltas from the `per-op statistics` block.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct NfsOp {
    pub ops: u64,
    pub transmissions: u64,
    pub timeouts: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    /// Cumulative milliseconds queued, on the wire (`rtt`), and from issue to completion
    /// (`execute`); divide by `ops` for the means.
    pub queue_ms: u64,
    pub rtt_ms: u64,
    pub execute_ms: u64,
    /// Present from Linux 5.3 on.
    pub errors: u64,
}

impl HostCounters {
    /// Sum another host's counters in: peaks add (each host has its own process), as do CPU
    /// time, RSS, and NFS deltas by procedure; the mount identity becomes the set of distinct
    /// mount points, devices, and types joined with `,`.
    pub fn merge(&mut self, o: &HostCounters) {
        self.tasks_peak += o.tasks_peak;
        self.iowq_workers_peak += o.iowq_workers_peak;
        self.sqpoll_threads += o.sqpoll_threads;
        self.cpu_user_ns += o.cpu_user_ns;
        self.cpu_sys_ns += o.cpu_sys_ns;
        self.maxrss_bytes += o.maxrss_bytes;
        self.minor_faults += o.minor_faults;
        self.major_faults += o.major_faults;
        self.open_files_peak += o.open_files_peak;
        match (&mut self.mount, &o.mount) {
            (Some(m), Some(n)) => m.merge(n),
            (None, Some(n)) => self.mount = Some(n.clone()),
            _ => {}
        }
    }
}

fn join_distinct(a: &mut String, b: &str) {
    if !a.split(',').any(|x| x == b) {
        a.push(',');
        a.push_str(b);
    }
}

impl MountCounters {
    pub fn merge(&mut self, o: &MountCounters) {
        join_distinct(&mut self.mount_point, &o.mount_point);
        join_distinct(&mut self.device, &o.device);
        join_distinct(&mut self.fstype, &o.fstype);
        match (&mut self.opts, &o.opts) {
            (Some(a), Some(b)) if !a.split(" | ").any(|x| x == b) => {
                a.push_str(" | ");
                a.push_str(b);
            }
            (None, Some(b)) => self.opts = Some(b.clone()),
            _ => {}
        }
        for v in &o.read_ahead_kb {
            if !self.read_ahead_kb.contains(v) {
                self.read_ahead_kb.push(*v);
            }
        }
        self.read_ahead_kb.sort();
        match (&mut self.nfs, &o.nfs) {
            (Some(m), Some(n)) => m.merge(n),
            (None, Some(n)) => self.nfs = Some(n.clone()),
            _ => {}
        }
    }
}

impl NfsCounters {
    pub fn merge(&mut self, o: &NfsCounters) {
        self.normal_read_bytes += o.normal_read_bytes;
        self.normal_write_bytes += o.normal_write_bytes;
        self.direct_read_bytes += o.direct_read_bytes;
        self.direct_write_bytes += o.direct_write_bytes;
        self.server_read_bytes += o.server_read_bytes;
        self.server_write_bytes += o.server_write_bytes;
        for (k, v) in &o.ops {
            self.ops.entry(k.clone()).or_default().add(v);
        }
    }

    /// Subtract a snapshot taken earlier, keeping the procedures with a non-zero count.
    fn delta(&self, before: &NfsCounters) -> NfsCounters {
        let mut d = NfsCounters {
            normal_read_bytes: self
                .normal_read_bytes
                .wrapping_sub(before.normal_read_bytes),
            normal_write_bytes: self
                .normal_write_bytes
                .wrapping_sub(before.normal_write_bytes),
            direct_read_bytes: self
                .direct_read_bytes
                .wrapping_sub(before.direct_read_bytes),
            direct_write_bytes: self
                .direct_write_bytes
                .wrapping_sub(before.direct_write_bytes),
            server_read_bytes: self
                .server_read_bytes
                .wrapping_sub(before.server_read_bytes),
            server_write_bytes: self
                .server_write_bytes
                .wrapping_sub(before.server_write_bytes),
            ops: BTreeMap::new(),
        };
        for (k, v) in &self.ops {
            let z = NfsOp::default();
            let b = before.ops.get(k).unwrap_or(&z);
            let x = v.sub(b);
            if x.ops > 0 || x.transmissions > 0 || x.errors > 0 {
                d.ops.insert(k.clone(), x);
            }
        }
        d
    }
}

impl NfsOp {
    fn add(&mut self, o: &NfsOp) {
        self.ops += o.ops;
        self.transmissions += o.transmissions;
        self.timeouts += o.timeouts;
        self.bytes_sent += o.bytes_sent;
        self.bytes_received += o.bytes_received;
        self.queue_ms += o.queue_ms;
        self.rtt_ms += o.rtt_ms;
        self.execute_ms += o.execute_ms;
        self.errors += o.errors;
    }
    fn sub(&self, o: &NfsOp) -> NfsOp {
        NfsOp {
            ops: self.ops.wrapping_sub(o.ops),
            transmissions: self.transmissions.wrapping_sub(o.transmissions),
            timeouts: self.timeouts.wrapping_sub(o.timeouts),
            bytes_sent: self.bytes_sent.wrapping_sub(o.bytes_sent),
            bytes_received: self.bytes_received.wrapping_sub(o.bytes_received),
            queue_ms: self.queue_ms.wrapping_sub(o.queue_ms),
            rtt_ms: self.rtt_ms.wrapping_sub(o.rtt_ms),
            execute_ms: self.execute_ms.wrapping_sub(o.execute_ms),
            errors: self.errors.wrapping_sub(o.errors),
        }
    }
}

// ---------------------------------------------------------------- the sampler

struct Rusage {
    user_ns: u64,
    sys_ns: u64,
    maxrss_bytes: u64,
    minflt: u64,
    majflt: u64,
}

fn rusage() -> Rusage {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: a valid pointer to a zeroed rusage; RUSAGE_SELF never fails
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let ns = |tv: libc::timeval| (tv.tv_sec as u64) * 1_000_000_000 + (tv.tv_usec as u64) * 1_000;
    Rusage {
        user_ns: ns(ru.ru_utime),
        sys_ns: ns(ru.ru_stime),
        maxrss_bytes: (ru.ru_maxrss as u64) * 1024,
        minflt: ru.ru_minflt as u64,
        majflt: ru.ru_majflt as u64,
    }
}

/// The process's task count, from `/proc/self/status`.
fn task_count() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    s.lines()
        .find_map(|l| l.strip_prefix("Threads:"))
        .and_then(|v| v.trim().parse().ok())
}

/// The io-wq workers (`iou-wrk-*`) and `SQPOLL` threads (`iou-sqp-*`) among the process's
/// tasks, by thread name. The kernel starts and ends the workers on its own schedule, so
/// their peak can only be sampled (the sampler uses the first count; the `SQPOLL` threads
/// of the report come from the rings' `fdinfo`, `uring::run`).
fn io_threads() -> (u64, u64) {
    let Ok(rd) = std::fs::read_dir("/proc/self/task") else {
        return (0, 0);
    };
    let (mut wrk, mut sqp) = (0, 0);
    for e in rd.filter_map(|e| e.ok()) {
        let Ok(c) = std::fs::read_to_string(e.path().join("comm")) else { continue };
        if c.starts_with("iou-wrk") {
            wrk += 1;
        } else if c.starts_with("iou-sqp") {
            sqp += 1;
        }
    }
    (wrk, sqp)
}

/// Counters open around a run: `Sampler::start` before the actors are spawned, `finish`
/// after they are joined.
pub struct Sampler {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<(u64, u64)>>,
    ru0: Rusage,
    mount0: Option<MountSnapshot>,
    root: std::path::PathBuf,
}

impl Sampler {
    pub fn start(root: &Path) -> Sampler {
        let stop = Arc::new(AtomicBool::new(false));
        let s = stop.clone();
        let handle = std::thread::Builder::new()
            .name("counters".into())
            .spawn(move || {
                let mut tasks_peak = task_count().unwrap_or(0);
                let mut last = tasks_peak;
                let mut iowq_peak = io_threads().0;
                while !s.load(Ordering::Relaxed) {
                    std::thread::sleep(SAMPLE);
                    let Some(n) = task_count() else { continue };
                    tasks_peak = tasks_peak.max(n);
                    if n != last {
                        last = n;
                        iowq_peak = iowq_peak.max(io_threads().0);
                    }
                }
                // one last look, so a run shorter than the period is still seen
                if let Some(n) = task_count() {
                    tasks_peak = tasks_peak.max(n);
                }
                (tasks_peak, iowq_peak.max(io_threads().0))
            })
            .ok();
        Sampler {
            stop,
            handle,
            ru0: rusage(),
            mount0: MountSnapshot::for_path(root),
            root: root.to_path_buf(),
        }
    }

    pub fn finish(mut self) -> HostCounters {
        self.stop.store(true, Ordering::Relaxed);
        let (tasks_peak, iowq_workers_peak) = self
            .handle
            .take()
            .and_then(|h| h.join().ok())
            .unwrap_or((0, 0));
        let ru1 = rusage();
        let read_ahead_kb: Vec<u64> = read_ahead_kb(&self.root).into_iter().collect();
        let mount = match (self.mount0.take(), MountSnapshot::for_path(&self.root)) {
            (Some(a), Some(b)) if a.mount_point == b.mount_point => Some(MountCounters {
                mount_point: b.mount_point,
                device: b.device,
                fstype: b.fstype,
                opts: b.opts,
                read_ahead_kb,
                nfs: match (&a.nfs, &b.nfs) {
                    (Some(x), Some(y)) => Some(y.delta(x)),
                    _ => None,
                },
            }),
            (_, Some(b)) => Some(MountCounters {
                mount_point: b.mount_point,
                device: b.device,
                fstype: b.fstype,
                opts: b.opts,
                read_ahead_kb,
                nfs: None,
            }),
            _ => None,
        };
        HostCounters {
            tasks_peak,
            iowq_workers_peak,
            sqpoll_threads: 0,
            cpu_user_ns: ru1.user_ns.saturating_sub(self.ru0.user_ns),
            cpu_sys_ns: ru1.sys_ns.saturating_sub(self.ru0.sys_ns),
            maxrss_bytes: ru1.maxrss_bytes,
            minor_faults: ru1.minflt.saturating_sub(self.ru0.minflt),
            major_faults: ru1.majflt.saturating_sub(self.ru0.majflt),
            open_files_peak: crate::backend::open_files_peak(),
            mount,
        }
    }
}

// ---------------------------------------------------------------- mountstats

/// One `/proc/self/mountstats` entry: the header line's fields and, for NFS, the counters.
#[derive(Debug, Clone, PartialEq)]
pub struct MountSnapshot {
    pub mount_point: String,
    pub device: String,
    pub fstype: String,
    pub opts: Option<String>,
    pub nfs: Option<NfsCounters>,
}

impl MountSnapshot {
    /// The entry for the mount `path` is on: the longest mount point that is a prefix of the
    /// canonical path (the last such entry, as a later mount over the same point is on top).
    pub fn for_path(path: &Path) -> Option<MountSnapshot> {
        let canon = std::fs::canonicalize(path).ok()?;
        let text = std::fs::read_to_string("/proc/self/mountstats").ok()?;
        let mut m = Self::find(&text, &canon)?;
        if m.opts.is_none() {
            // only NFS prints `opts:` in mountstats; the others' are in /proc/self/mounts
            // (the last entry for the mount point is the one on top)
            let mounts = std::fs::read_to_string("/proc/self/mounts").unwrap_or_default();
            m.opts = mounts.lines().rev().find_map(|l| {
                let f: Vec<&str> = l.split(' ').collect();
                (f.len() >= 4 && unescape(f[1]) == m.mount_point).then(|| f[3].to_string())
            });
        }
        Some(m)
    }

    pub fn find(text: &str, canon: &Path) -> Option<MountSnapshot> {
        let mut best: Option<MountSnapshot> = None;
        for m in parse(text) {
            // a mount point from /proc has octal escapes for space, tab, newline, backslash
            let mp = std::path::PathBuf::from(unescape(&m.mount_point));
            if canon.starts_with(&mp)
                && best
                    .as_ref()
                    .map(|b| mp.as_os_str().len() >= b.mount_point.len())
                    .unwrap_or(true)
            {
                best = Some(MountSnapshot {
                    mount_point: mp.to_string_lossy().into_owned(),
                    ..m
                });
            }
        }
        best
    }
}

fn unescape(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\'
            && i + 3 < b.len()
            && b[i + 1..i + 4].iter().all(|c| (b'0'..=b'7').contains(c))
        {
            let v = (b[i + 1] - b'0') * 64 + (b[i + 2] - b'0') * 8 + (b[i + 3] - b'0');
            out.push(v);
            i += 4;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Every entry of a `mountstats` text, in file order.
pub fn parse(text: &str) -> Vec<MountSnapshot> {
    let mut out: Vec<MountSnapshot> = Vec::new();
    let mut in_ops = false;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("device ") {
            in_ops = false;
            // device D mounted on M with fstype T [statvers=V]
            let Some((device, rest)) = rest.split_once(" mounted on ") else {
                continue;
            };
            let Some((mount_point, rest)) = rest.rsplit_once(" with fstype ") else {
                continue;
            };
            let fstype = rest.split_whitespace().next().unwrap_or("").to_string();
            let nfs = fstype.starts_with("nfs").then(NfsCounters::default);
            out.push(MountSnapshot {
                mount_point: mount_point.to_string(),
                device: device.to_string(),
                fstype,
                opts: None,
                nfs,
            });
            continue;
        }
        let Some(cur) = out.last_mut() else { continue };
        let t = line.trim();
        if let Some(v) = t.strip_prefix("opts:") {
            cur.opts = Some(v.trim().to_string());
            continue;
        }
        let Some(nfs) = cur.nfs.as_mut() else {
            continue;
        };
        if let Some(v) = t.strip_prefix("bytes:") {
            let f: Vec<u64> = v
                .split_whitespace()
                .map(|x| x.parse().unwrap_or(0))
                .collect();
            if f.len() >= 6 {
                nfs.normal_read_bytes = f[0];
                nfs.normal_write_bytes = f[1];
                nfs.direct_read_bytes = f[2];
                nfs.direct_write_bytes = f[3];
                nfs.server_read_bytes = f[4];
                nfs.server_write_bytes = f[5];
            }
        } else if t.starts_with("per-op statistics") {
            in_ops = true;
        } else if in_ops {
            if let Some((name, v)) = t.split_once(':') {
                let f: Vec<u64> = v
                    .split_whitespace()
                    .map(|x| x.parse().unwrap_or(0))
                    .collect();
                if f.len() >= 8 {
                    nfs.ops.insert(
                        name.trim().to_string(),
                        NfsOp {
                            ops: f[0],
                            transmissions: f[1],
                            timeouts: f[2],
                            bytes_sent: f[3],
                            bytes_received: f[4],
                            queue_ms: f[5],
                            rtt_ms: f[6],
                            execute_ms: f[7],
                            errors: f.get(8).copied().unwrap_or(0),
                        },
                    );
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TEXT: &str = "\
device /dev/sdd mounted on / with fstype ext4
device none mounted on /mnt/wsl with fstype tmpfs
device localhost:/srv/x mounted on /mnt/aeiou-nfs with fstype nfs4 statvers=1.1
\topts:\trw,vers=4.2
\tage:\t26282
\tevents:\t2013 114488 170
\tbytes:\t100 200 300 400 500 600 7 8
\tRPC iostats version: 1.1  p/v: 100003/4 (nfs)
\txprt:\ttcp 0 0 2 0 11 64243 64223 0 2804870 0 534 134378 1958499
\tper-op statistics
\t        NULL: 1 1 0 44 24 0 0 0 0
\t        READ: 10 10 0 1000 2000 1 20 21 0
\t       WRITE: 0 0 0 0 0 0 0 0 0
device /dev/sde mounted on /mnt/aeiou-nfs/deeper\\040dir with fstype ext4
";

    #[test]
    fn parses_and_picks_the_longest_mount_point() {
        let all = parse(SAMPLE_TEXT);
        assert_eq!(all.len(), 4);
        let nfs = &all[2];
        assert_eq!(nfs.fstype, "nfs4");
        assert_eq!(nfs.opts.as_deref(), Some("rw,vers=4.2"));
        assert_eq!(all[0].opts, None, "only NFS prints its options in mountstats");
        let c = nfs.nfs.as_ref().unwrap();
        assert_eq!((c.normal_read_bytes, c.server_write_bytes), (100, 600));
        assert_eq!(c.ops["READ"].ops, 10);
        assert_eq!(c.ops["READ"].rtt_ms, 20);
        assert_eq!(c.ops.len(), 3);

        let m = MountSnapshot::find(SAMPLE_TEXT, Path::new("/mnt/aeiou-nfs/run1")).unwrap();
        assert_eq!(m.mount_point, "/mnt/aeiou-nfs");
        let m = MountSnapshot::find(SAMPLE_TEXT, Path::new("/mnt/aeiou-nfs/deeper dir/x")).unwrap();
        assert_eq!(
            (m.mount_point.as_str(), m.fstype.as_str()),
            ("/mnt/aeiou-nfs/deeper dir", "ext4")
        );
        assert!(m.nfs.is_none());
        let m = MountSnapshot::find(SAMPLE_TEXT, Path::new("/home/x")).unwrap();
        assert_eq!(m.mount_point, "/");
    }

    #[test]
    fn deltas_keep_only_the_procedures_that_ran() {
        let a = parse(SAMPLE_TEXT)[2].nfs.clone().unwrap();
        let mut b = a.clone();
        b.server_read_bytes += 4096;
        b.ops.get_mut("READ").unwrap().ops += 3;
        b.ops.get_mut("READ").unwrap().rtt_ms += 9;
        let d = b.delta(&a);
        assert_eq!(d.server_read_bytes, 4096);
        assert_eq!(d.normal_read_bytes, 0);
        assert_eq!(d.ops.len(), 1);
        assert_eq!((d.ops["READ"].ops, d.ops["READ"].rtt_ms), (3, 9));
    }

    #[test]
    fn merge_sums_and_joins_identities() {
        let mut x = HostCounters {
            tasks_peak: 3,
            mount: Some(MountCounters {
                mount_point: "/a".into(),
                device: "d".into(),
                fstype: "nfs4".into(),
                opts: Some("rw,vers=4.2".into()),
                read_ahead_kb: vec![128],
                nfs: Some(NfsCounters::default()),
            }),
            ..Default::default()
        };
        let mut nfs = NfsCounters::default();
        nfs.ops.insert(
            "READ".into(),
            NfsOp {
                ops: 2,
                ..Default::default()
            },
        );
        let y = HostCounters {
            tasks_peak: 4,
            mount: Some(MountCounters {
                mount_point: "/b".into(),
                device: "d".into(),
                fstype: "nfs4".into(),
                opts: Some("rw,vers=4.1,nconnect=4".into()),
                read_ahead_kb: vec![15360],
                nfs: Some(nfs),
            }),
            ..Default::default()
        };
        x.merge(&y);
        x.merge(&y);
        assert_eq!(x.tasks_peak, 11);
        let m = x.mount.unwrap();
        assert_eq!((m.mount_point.as_str(), m.device.as_str()), ("/a,/b", "d"));
        assert_eq!(m.opts.as_deref(), Some("rw,vers=4.2 | rw,vers=4.1,nconnect=4"), "distinct option sets once each");
        assert_eq!(m.read_ahead_kb, vec![128, 15360], "distinct readahead windows once each");
        assert_eq!(m.nfs.unwrap().ops["READ"].ops, 4);
    }

    #[test]
    fn read_ahead_kb_is_found_through_the_bdi_or_the_partition_s_disk() {
        let sys = std::env::temp_dir().join(format!("aeiou-sys-{}", std::process::id()));
        let put = |rel: &str, v: &str| {
            let d = sys.join(rel);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("read_ahead_kb"), v).unwrap();
        };
        // an NFS mount's anonymous device, a whole disk, and a partition under its disk
        put("class/bdi/0:78", "128\n");
        put("dev/block/8:48/bdi", "8192\n");
        // as in sysfs: the partition's entry is a link into its disk's directory
        put("devices/nvme0n1/bdi", "256\n");
        std::fs::create_dir_all(sys.join("devices/nvme0n1/nvme0n1p1")).unwrap();
        std::os::unix::fs::symlink("../../devices/nvme0n1/nvme0n1p1", sys.join("dev/block/259:1")).unwrap();
        assert_eq!(read_ahead_kb_at(&sys, 0, 78), Some(128));
        assert_eq!(read_ahead_kb_at(&sys, 8, 48), Some(8192));
        assert_eq!(read_ahead_kb_at(&sys, 259, 1), Some(256), "a partition's is its disk's");
        assert_eq!(read_ahead_kb_at(&sys, 0, 75), None, "tmpfs has no backing device info");
        std::fs::remove_dir_all(&sys).unwrap();
        // the live tree answers without panicking, whatever this filesystem is
        let _ = read_ahead_kb(Path::new("/"));
    }

    #[test]
    fn the_sampler_sees_this_process() {
        let s = Sampler::start(Path::new("."));
        let h = std::thread::spawn(|| std::thread::sleep(SAMPLE * 4));
        std::thread::sleep(SAMPLE * 2);
        h.join().unwrap();
        let c = s.finish();
        assert!(c.tasks_peak >= 3, "{c:?}"); // this thread, the sampler, the sleeper
        assert!(c.mount.is_some());
    }
}
