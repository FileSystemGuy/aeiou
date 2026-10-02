//! `aeiou dry-run`: walk every actor instance without doing I/O, count ops and bytes, sum the
//! fingerprint, and optionally print one instance's op stream. Instances run in parallel and
//! in any order; the fingerprint is a sum, so the result does not depend on either.

use std::collections::BTreeMap;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::Result;

use crate::eval::Model;
use crate::vm::{actor_counts, op_hash, run_actor, Control, Op, OpCtx, OpKind, Sink};

/// Which ops to print: one actor instance, optionally a range of its outermost loop index.
#[derive(Debug, Clone, Default)]
pub struct Filter {
    pub actor: Option<i64>,
    pub steps: Option<(i64, i64)>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub ops: u64,
    pub counts: BTreeMap<OpKind, u64>,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub expect_ops: u64,
}

impl Stats {
    fn add(&mut self, op: &Op) {
        self.ops += 1;
        *self.counts.entry(op.kind).or_insert(0) += 1;
        match op.kind {
            OpKind::Read => self.bytes_read += op.bytes as u64,
            OpKind::Write => self.bytes_written += op.bytes as u64,
            _ => {}
        }
        if !op.expect.is_empty() {
            self.expect_ops += 1;
        }
    }

    fn merge(&mut self, o: &Stats) {
        self.ops += o.ops;
        for (k, v) in &o.counts {
            *self.counts.entry(*k).or_insert(0) += v;
        }
        self.bytes_read += o.bytes_read;
        self.bytes_written += o.bytes_written;
        self.expect_ops += o.expect_ops;
    }
}

#[derive(Debug, Clone, Default)]
pub struct DryRun {
    pub total: Stats,
    pub phases: BTreeMap<String, Stats>,
    pub fingerprint: u64,
    pub compute_ns: i128,
    pub barriers: u64,
    pub takes: u64,
    pub puts: u64,
    pub instances: u64,
    /// Bytes read and written per instance, for the per-host estimate.
    pub per_instance: Vec<(i64, u64, u64)>,
    /// The locality metrics, under `--metrics` (`metrics.rs`).
    pub metrics: Option<Box<crate::metrics::Metrics>>,
    filter: Option<Filter>,
    printed: usize,
    lines: Vec<String>,
}

impl DryRun {
    pub fn new(filter: Option<Filter>) -> Self {
        DryRun { filter, ..Default::default() }
    }

    pub fn with_metrics(mut self, opts: Option<crate::metrics::Opts>) -> Self {
        self.metrics = opts.map(|o| Box::new(crate::metrics::Metrics::new(o)));
        self
    }

    pub fn merge(&mut self, o: &DryRun) {
        self.total.merge(&o.total);
        for (k, v) in &o.phases {
            self.phases.entry(k.clone()).or_default().merge(v);
        }
        self.fingerprint = self.fingerprint.wrapping_add(o.fingerprint);
        self.compute_ns += o.compute_ns;
        self.barriers += o.barriers;
        self.takes += o.takes;
        self.puts += o.puts;
        self.instances += o.instances;
        self.per_instance.extend_from_slice(&o.per_instance);
        self.printed += o.printed;
        self.lines.extend_from_slice(&o.lines);
        if let Some(om) = &o.metrics {
            match &mut self.metrics {
                Some(m) => m.merge(om),
                None => self.metrics = Some(om.clone()),
            }
        }
    }

    pub fn take_lines(&mut self) -> Vec<String> {
        std::mem::take(&mut self.lines)
    }

    fn wants(&self, ctx: &OpCtx) -> bool {
        let Some(f) = &self.filter else { return false };
        if f.actor.is_some_and(|a| a != ctx.actor) {
            return false;
        }
        if let Some((lo, hi)) = f.steps {
            match ctx.indices.first() {
                Some(i) if *i >= lo && *i < hi => {}
                _ => return false,
            }
        }
        f.limit.map_or(true, |n| self.printed < n)
    }
}

impl<'m, 'a> Sink<'m, 'a> for DryRun {
    fn op(&mut self, op: &Op, ctx: &OpCtx) -> Result<()> {
        self.total.add(op);
        if let Some(p) = ctx.phase {
            self.phases.entry(p.to_string()).or_default().add(op);
        }
        self.fingerprint = self.fingerprint.wrapping_add(op_hash(op, ctx));
        if self.wants(ctx) {
            self.printed += 1;
            let idx: Vec<String> = ctx.indices.iter().map(|i| i.to_string()).collect();
            let mut line = format!("{}#{} [{}]", ctx.template, ctx.actor, idx.join(","));
            if let Some(p) = ctx.phase {
                line.push_str(&format!(" {p}:"));
            }
            line.push_str(&format!(" {} {}", op.kind.name(), op.path));
            match op.kind {
                OpKind::Read => line.push_str(&format!(" off={} len={} -> {}", op.offset, op.len, op.bytes)),
                OpKind::Write => line.push_str(&format!(" off={} len={}", op.offset, op.len)),
                OpKind::Open => line.push_str(&format!(" flags={:#x}", op.aux)),
                OpKind::Lseek => line.push_str(&format!(" off={} whence={}", op.offset, op.aux)),
                OpKind::Fadvise => line.push_str(&format!(" off={} len={} advice={:?}", op.offset, op.len, crate::ast::Advice::from_code(op.aux))),
                OpKind::Ftruncate => line.push_str(&format!(" len={}", op.len)),
                OpKind::Fallocate => line.push_str(&format!(" off={} len={}", op.offset, op.len)),
                OpKind::Rename => line.push_str(&format!(" -> {}", op.path2.unwrap_or("?"))),
                _ => {}
            }
            if !op.expect.is_empty() {
                line.push_str(&format!(" (expect {})", op.expect.join("|")));
            }
            self.lines.push(line);
        }
        Ok(())
    }

    fn control(&mut self, c: Control<'a>, _ctx: &OpCtx) -> Result<()> {
        match c {
            Control::Compute { ns } => self.compute_ns += ns as i128,
            Control::Barrier { .. } => self.barriers += 1,
            Control::Take { .. } => self.takes += 1,
            Control::Put { .. } => self.puts += 1,
            Control::Channel { .. } => {}
        }
        Ok(())
    }
}

pub struct TemplateReport {
    pub name: String,
    pub count: i64,
    pub run: DryRun,
}

pub struct Report {
    pub templates: Vec<TemplateReport>,
    pub total: DryRun,
}

/// Run the dry run over every instance of every template with `threads` worker threads.
pub fn run(model: &Model<'_>, threads: usize, filter: Option<Filter>) -> Result<Report> {
    run_with(model, threads, filter, None)
}

/// `run`, with the locality metrics when `metrics` is given: each instance is then walked in
/// the round-robin order of its sub-actors (`metrics.rs`); counts and fingerprint are the same.
pub fn run_with(model: &Model<'_>, threads: usize, filter: Option<Filter>, metrics: Option<crate::metrics::Opts>) -> Result<Report> {
    let counts = actor_counts(model)?;
    let mut jobs: Vec<(usize, i64, i64)> = Vec::new();
    for (t, (_, count)) in counts.iter().enumerate() {
        for i in 0..*count {
            jobs.push((t, i, *count));
        }
    }
    let threads = if filter.as_ref().is_some_and(|f| f.actor.is_some()) { 1 } else { threads.max(1).min(jobs.len().max(1)) };
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, DryRun)>> = Mutex::new(Vec::new());
    let errors: Mutex<Vec<anyhow::Error>> = Mutex::new(Vec::new());

    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut per_template: BTreeMap<usize, DryRun> = BTreeMap::new();
                loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    if j >= jobs.len() {
                        break;
                    }
                    let (t, actor, count) = jobs[j];
                    let (name, _) = counts[t];
                    let sink = DryRun::new(filter.clone()).with_metrics(metrics);
                    let walked = match metrics {
                        Some(_) => crate::metrics::walk(model, name, actor, count, sink),
                        None => run_actor(model, name, actor, count, sink),
                    };
                    match walked {
                        Ok(mut r) => {
                            r.instances = 1;
                            r.per_instance = vec![(actor, r.total.bytes_read, r.total.bytes_written)];
                            per_template.entry(t).or_insert_with(|| DryRun::new(None)).merge(&r);
                        }
                        Err(e) => {
                            errors.lock().unwrap().push(e);
                            break;
                        }
                    }
                }
                let mut out = results.lock().unwrap();
                for (t, r) in per_template {
                    out.push((t, r));
                }
            });
        }
    });

    if let Some(e) = errors.into_inner().unwrap().into_iter().next() {
        return Err(e);
    }
    let mut templates: Vec<TemplateReport> = counts
        .iter()
        .map(|(name, count)| TemplateReport { name: name.to_string(), count: *count, run: DryRun::new(None) })
        .collect();
    for (t, r) in results.into_inner().unwrap() {
        templates[t].run.merge(&r);
    }
    let mut total = DryRun::new(None);
    for t in &templates {
        total.merge(&t.run);
    }
    for t in &mut templates {
        t.run.per_instance.sort();
    }
    Ok(Report { templates, total })
}

/// Total memory of this host in bytes, from `/proc/meminfo`.
pub fn host_dram() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

pub fn human_bytes(b: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut x = b as f64;
    let mut u = 0;
    while x >= 1024.0 && u < UNITS.len() - 1 {
        x /= 1024.0;
        u += 1;
    }
    if u == 0 { format!("{b} B") } else { format!("{x:.2} {}", UNITS[u]) }
}

pub fn human_ns(ns: i128) -> String {
    let s = ns as f64 / 1e9;
    if s >= 3600.0 {
        format!("{:.2} h", s / 3600.0)
    } else if s >= 60.0 {
        format!("{:.2} min", s / 60.0)
    } else {
        format!("{s:.3} s")
    }
}

pub fn write_report(out: &mut impl Write, r: &Report, ranks: Option<i64>, gpus: i64) -> std::io::Result<()> {
    for t in &r.templates {
        writeln!(out, "actor {}: {} instance(s)", t.name, t.count)?;
        write_stats(out, &t.run, "  ")?;
        crate::metrics::write_text(out, &t.run, "  ")?;
        let mut phases: Vec<_> = t.run.phases.iter().collect();
        phases.sort_by(|a, b| b.1.ops.cmp(&a.1.ops).then(a.0.cmp(b.0)));
        for (name, s) in phases {
            writeln!(
                out,
                "  phase {:<16} ops={:<12} read={:<12} written={}",
                name,
                s.ops,
                human_bytes(s.bytes_read),
                human_bytes(s.bytes_written)
            )?;
        }
    }
    if r.templates.len() > 1 {
        writeln!(out, "total")?;
        write_stats(out, &r.total, "  ")?;
        crate::metrics::write_text(out, &r.total, "  ")?;
    }
    writeln!(out, "fingerprint {:016x}", r.total.fingerprint)?;
    if let Some(ranks) = ranks {
        let per_host = (gpus + ranks - 1) / ranks;
        // instances are assigned to hosts by contiguous id range: host h owns [h·per_host, (h+1)·per_host)
        let mut worst: u64 = 0;
        for t in &r.templates {
            let mut by_host: BTreeMap<i64, u64> = BTreeMap::new();
            for (actor, rd, _) in &t.run.per_instance {
                *by_host.entry(actor / per_host.max(1)).or_insert(0) += rd;
            }
            worst = worst.max(by_host.values().copied().max().unwrap_or(0));
        }
        write!(out, "per host: {} actors, up to {} read", per_host, human_bytes(worst))?;
        if let Some(dram) = host_dram() {
            writeln!(out, "; this host has {} DRAM ({:.2}× )", human_bytes(dram), worst as f64 / dram as f64)?;
        } else {
            writeln!(out)?;
        }
    }
    Ok(())
}

fn write_stats(out: &mut impl Write, d: &DryRun, indent: &str) -> std::io::Result<()> {
    let s = &d.total;
    writeln!(out, "{indent}ops {}  read {}  written {}", s.ops, human_bytes(s.bytes_read), human_bytes(s.bytes_written))?;
    let counts: Vec<String> = OpKind::ALL
        .iter()
        .filter_map(|k| s.counts.get(k).map(|n| format!("{}={}", k.name(), n)))
        .collect();
    writeln!(out, "{indent}by kind: {}", counts.join(" "))?;
    writeln!(
        out,
        "{indent}compute {}  barriers {}  takes {}  puts {}  expect-ops {}",
        human_ns(d.compute_ns),
        d.barriers,
        d.takes,
        d.puts,
        s.expect_ops
    )?;
    Ok(())
}
