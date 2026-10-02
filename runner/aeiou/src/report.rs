//! The JSON report of `aeiou run --report-json FILE` (`runner/README.md` §12): what the text
//! report prints, for a tool. Format `aeiou_report: 1`.
//!
//! Built after the run and its verdict, outside `elapsed`. Times are integer nanoseconds,
//! sizes bytes; the fingerprint and the hashes are hex strings (a 64-bit fingerprint is
//! not exact as a JSON number). Nothing here is per file: created objects are a count.

use std::path::Path;

use serde_json::{json, Map, Value};

use crate::run::{ActorRecord, LatHist, Report, Stats};
use crate::vm::OpKind;

pub const FORMAT: u64 = 1;

/// The document as it accumulates: the header is known before the gate, the results after
/// the run, the verdict last. `write` is called on success and on failure alike, so a
/// harness never reads the file of an earlier run.
#[derive(Debug, Default)]
pub struct Doc {
    /// The identity and configuration of the run (`header`), once the checks have passed.
    pub header: Map<String, Value>,
    /// This host's report.
    pub host: Option<Value>,
    /// Every host's, merged (rank 0 of several).
    pub merged: Option<Value>,
    /// The fingerprint the verdict is on: this host's alone, or the run's from rank 0.
    pub fingerprint: Option<u64>,
    pub expected_fingerprint: Option<u64>,
    /// The fingerprint covers every host.
    pub whole_run: bool,
    pub started: Option<f64>,
    pub finished: Option<f64>,
    /// With `takes`: every instance's takes, not only their sums.
    pub full_takes: bool,
}

impl Doc {
    pub fn set(&mut self, key: &str, v: Value) {
        self.header.insert(key.into(), v);
    }

    /// The document, with the verdict `error` gives (`None`: the run passed).
    pub fn value(&self, error: Option<&str>) -> Value {
        let mut m = Map::new();
        m.insert("aeiou_report".into(), json!(FORMAT));
        m.insert("runner".into(), json!(env!("CARGO_PKG_VERSION")));
        for (k, v) in &self.header {
            m.insert(k.clone(), v.clone());
        }
        m.insert("started".into(), json!(self.started));
        m.insert("finished".into(), json!(self.finished));
        // `result` is what the verdict is on: the merged report where this host has it,
        // this host's otherwise; `scope` says which
        match (&self.merged, &self.host) {
            (Some(run), host) => {
                m.insert("scope".into(), json!("run"));
                m.insert("result".into(), run.clone());
                if let Some(h) = host {
                    m.insert("this_host".into(), h.clone());
                }
            }
            (None, Some(h)) => {
                m.insert("scope".into(), json!(if self.whole_run_alone() { "run" } else { "host" }));
                m.insert("result".into(), h.clone());
            }
            (None, None) => {
                m.insert("scope".into(), Value::Null);
                m.insert("result".into(), Value::Null);
            }
        }
        let hex = |v: Option<u64>| v.map(|f| json!(format!("{f:016x}"))).unwrap_or(Value::Null);
        m.insert(
            "verdict".into(),
            json!({
                "ok": error.is_none(),
                "error": error,
                "fingerprint": hex(self.fingerprint),
                "fingerprint_scope": self.fingerprint.map(|_| if self.whole_run { "run" } else { "host" }),
                "expected_fingerprint": hex(self.expected_fingerprint),
            }),
        );
        Value::Object(m)
    }

    /// One host is the whole run.
    fn whole_run_alone(&self) -> bool {
        self.header.get("ranks").and_then(|v| v.as_i64()) == Some(1)
    }

    pub fn write(&self, path: &Path, error: Option<&str>) -> anyhow::Result<()> {
        let text = serde_json::to_string_pretty(&self.value(error))? + "\n";
        std::fs::write(path, text).map_err(|e| anyhow::anyhow!("--report-json {}: {e}", path.display()))
    }
}

/// A latency histogram: the moments, the quantiles the text report prints and two more, and
/// the non-empty buckets as `[lower bound ns, count]` (four buckets per octave; a bucket
/// ends where the next bound begins). A quantile is the lower bound of its bucket.
pub fn hist(h: &LatHist) -> Value {
    let buckets: Vec<Value> = h.buckets.iter().enumerate().filter(|(_, n)| **n > 0).map(|(i, n)| json!([LatHist::lower(i), n])).collect();
    json!({
        "count": h.count,
        "sum_ns": h.sum,
        "mean_ns": h.mean(),
        "p50_ns": h.quantile(0.5),
        "p90_ns": h.quantile(0.9),
        "p99_ns": h.quantile(0.99),
        "p999_ns": h.quantile(0.999),
        "max_ns": h.max,
        "buckets": buckets,
    })
}

fn stats(s: &Stats, secs: f64) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("ops".into(), json!(s.ops));
    m.insert("bytes_read".into(), json!(s.bytes_read));
    m.insert("bytes_written".into(), json!(s.bytes_written));
    m.insert(
        "rates".into(),
        json!({
            "ops_per_s": s.ops as f64 / secs,
            "read_bytes_per_s": s.bytes_read as f64 / secs,
            "written_bytes_per_s": s.bytes_written as f64 / secs,
        }),
    );
    let by_kind: Map<String, Value> = OpKind::ALL.iter().filter_map(|k| s.counts.get(k).map(|n| (k.name().to_string(), json!(n)))).collect();
    m.insert("by_kind".into(), Value::Object(by_kind));
    let lat: Map<String, Value> = OpKind::ALL.iter().filter_map(|k| s.lat.get(k).map(|h| (k.name().to_string(), hist(h)))).collect();
    m.insert("latency".into(), Value::Object(lat));
    let phases: Map<String, Value> = s
        .phases
        .iter()
        .map(|(name, p)| (name.clone(), json!({"ops": p.ops, "bytes_read": p.bytes_read, "bytes_written": p.bytes_written, "io_ns": p.io_ns})))
        .collect();
    m.insert("phases".into(), Value::Object(phases));
    m.insert("compute_ns".into(), json!(s.compute_ns));
    m.insert("io_ns".into(), json!(s.io_ns));
    m.insert("barriers".into(), json!(s.barriers));
    m.insert("barrier_wait_ns".into(), json!(s.barrier_wait_ns));
    m.insert("takes".into(), json!(s.takes));
    m.insert("puts".into(), json!(s.puts));
    m.insert("expected_errors".into(), json!(s.expected_errors));
    m.insert("threads".into(), json!(s.threads));
    m.insert("input_opens".into(), json!(s.input_opens));
    m.insert("warm_opens".into(), json!(s.warm_opens));
    m
}

/// The take table of the text report: stall per take and the busy fraction
/// (compute / (compute + stall)), over all takes and by take ordinal in at most ten ranges.
pub fn take_summary(actors: &[ActorRecord]) -> Value {
    let with: Vec<&ActorRecord> = actors.iter().filter(|a| !a.takes.is_empty()).collect();
    if with.is_empty() {
        return Value::Null;
    }
    let busy = |compute: u64, stall: u64| if compute + stall > 0 { compute as f64 / (compute + stall) as f64 } else { 0.0 };
    let n = with.iter().map(|a| a.takes.len()).max().unwrap_or(0);
    let ranges = n.clamp(1, 10);
    let per = n.div_ceil(ranges);
    let mut all = LatHist::default();
    let mut all_compute = 0u64;
    let mut steps = Vec::new();
    for b in 0..ranges {
        let (lo, hi) = (b * per, ((b + 1) * per).min(n));
        if lo >= hi {
            break;
        }
        let mut h = LatHist::default();
        let mut compute = 0u64;
        for a in &with {
            for t in a.takes.iter().take(hi).skip(lo) {
                h.add(t.stall_ns);
                compute += t.compute_ns;
            }
        }
        steps.push(json!({
            "from": lo,
            "to": hi,
            "takes": h.count,
            "stall_mean_ns": h.mean(),
            "stall_p99_ns": h.quantile(0.99),
            "stall_max_ns": h.max,
            "busy": busy(compute, h.sum),
        }));
        all.merge(&h);
        all_compute += compute;
    }
    json!({
        "per_instance": n,
        "instances": with.len(),
        "takes": all.count,
        "stall_ns": all.sum,
        "compute_ns": all_compute,
        "stall_mean_ns": all.mean(),
        "stall_p99_ns": all.quantile(0.99),
        "stall_max_ns": all.max,
        "busy": busy(all_compute, all.sum),
        "steps": steps,
    })
}

/// One report (a host's or the merged one) as the document's `result`. `full_takes` adds
/// every take of every instance as `[stall_ns, compute_ns]`.
pub fn result(r: &Report, full_takes: bool) -> anyhow::Result<Value> {
    let secs = r.elapsed.as_secs_f64().max(1e-9);
    let mut m = Map::new();
    m.insert("host".into(), json!(r.host));
    let ranks: Vec<Value> = r.ranks.iter().map(serde_json::to_value).collect::<Result<_, _>>()?;
    m.insert("ranks".into(), Value::Array(ranks));
    let templates: Map<String, Value> = r.templates.iter().map(|(name, n)| (name.clone(), json!(n))).collect();
    m.insert("templates".into(), Value::Object(templates));
    m.insert("elapsed_ns".into(), json!(r.elapsed.as_nanos() as u64));
    m.insert("fingerprint".into(), json!(format!("{:016x}", r.stats.fingerprint)));
    m.extend(stats(&r.stats, secs));
    m.insert("threads_peak".into(), json!(r.threads_peak));
    m.insert("counters".into(), serde_json::to_value(&r.counters)?);
    m.insert("io_uring".into(), serde_json::to_value(&r.uring)?);
    m.insert("libaio".into(), serde_json::to_value(&r.aio)?);
    m.insert("mmap".into(), serde_json::to_value(&r.mmap)?);
    m.insert("cold".into(), serde_json::to_value(&r.cold)?);
    m.insert("objects_created".into(), json!(r.created.len()));
    let departures: Map<String, Value> = r.departure_releases.iter().map(|(scope, n)| (scope.clone(), json!(n))).collect();
    m.insert("departure_releases".into(), Value::Object(departures));
    m.insert("take_summary".into(), take_summary(&r.actors));
    let instances: Vec<Value> = r
        .actors
        .iter()
        .map(|a| {
            let mut i = json!({
                "template": a.template,
                "actor": a.actor,
                "elapsed_ns": a.elapsed.as_nanos() as u64,
                "takes": a.takes.len(),
                "stall_ns": a.takes.iter().map(|t| t.stall_ns).sum::<u64>(),
                "compute_ns": a.takes.iter().map(|t| t.compute_ns).sum::<u64>(),
            });
            if full_takes {
                i["take_records"] = a.takes.iter().map(|t| json!([t.stall_ns, t.compute_ns])).collect();
            }
            i
        })
        .collect();
    m.insert("instances".into(), Value::Array(instances));
    let mut warnings = Vec::new();
    if r.stats.warm_opens > 0 {
        warnings.push(format!("{} read(s) of input objects hit the host that wrote them (page cache, not storage)", r.stats.warm_opens));
    }
    for (scope, n) in &r.departure_releases {
        warnings.push(format!("barrier `{scope}` was released {n} time(s) by instances finishing: not every instance hit it the same number of times"));
    }
    m.insert("warnings".into(), json!(warnings));
    Ok(Value::Object(m))
}
