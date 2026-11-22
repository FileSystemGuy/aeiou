//! The executor (`DESIGN_REVIEW.md` §3.99, the note "The run side of timelines", step 3; brief
//! item 72 (7)) on a simulated clock and storage: each op issued at `max(plan + L, the lane's
//! previous completion, its edges)`, L never shrinking, forks and joins, submissions and their
//! staging, eviction batches.

use aeiou::eval::{build_model, Config, Model, Params};
use aeiou::exec::{simulate, Simulated, Stats};
use aeiou::plan::{Plan, Record};
use serde_json::{json, Value};

fn model(body: Value) -> &'static Model<'static> {
    let doc = json!({"ast": "0.23", "name": "t",
           "datasets": {"d": {"layout": "whole", "access": "map", "pattern": "d/{id}", "count": 1000, "size": {"const": 1}, "seed": 1}},
           "namespaces": {"kv": {"pattern": "kv/{key}", "fields": {"key": "int"}, "size": 4096, "seed": 1}},
           "actors": {"gpu": {"count": 1, "body": body}}}).to_string();
    let cfg = Box::leak(Box::new(Config { seed: 1, gpus: 1, overrides: vec![], sets: vec![], st_blksize: Default::default() }));
    let loaded = aeiou::load_str(&doc).unwrap_or_else(|e| panic!("{e:#}"));
    let ast = Box::leak(Box::new(loaded.ast));
    let params = Box::leak(Box::new(Params::new(ast, cfg).unwrap()));
    Box::leak(Box::new(build_model(ast, cfg, params).unwrap()))
}

/// Simulate the timeline `body` with `latency` per op; the horizon 1 s.
fn run(body: Value, latency: impl FnMut(&Record, i64) -> i64) -> (Vec<Simulated>, Stats) {
    let m = model(body);
    let mut p = Plan::new(m, "gpu", 0, 1);
    simulate(&mut p, 1_000_000_000, latency).unwrap_or_else(|e| panic!("{e:#}"))
}

fn tl(body: Value) -> Value {
    json!([{"timeline": {"name": "tl", "workers": [{"name": "w", "lanes": 1, "staging": 8192}], "body": body}}])
}

fn stat(id: i64) -> Value {
    json!({"stat": {"file": {"file": {"dataset": "d", "id": id}}}})
}

fn compute(ns: i64) -> Value {
    json!({"compute": {"ns": ns}})
}

fn obj(name: &str, key: i64) -> Value {
    json!({"let": {"name": name, "value": {"object": {"namespace": "kv", "fields": {"key": key}}}}})
}

fn r(name: &str) -> Value {
    json!({"ref": name})
}

/// (path, issued) of each op, in the order issued.
fn when(ops: &[Simulated]) -> Vec<(String, i64)> {
    ops.iter().map(|o| (o.record.path.to_string(), o.issued)).collect()
}

fn pairs(v: &[(&str, i64)]) -> Vec<(String, i64)> {
    v.iter().map(|(p, t)| (p.to_string(), *t)).collect()
}

/// On storage that answers at once every op issues at its planned time.
#[test]
fn ops_issue_at_their_planned_times_when_storage_keeps_up() {
    let (ops, st) = run(tl(json!([stat(1), compute(100), stat(2), compute(50), stat(3)])), |_, _| 0);
    assert_eq!(when(&ops), pairs(&[("d/1", 0), ("d/2", 100), ("d/3", 150)]));
    assert_eq!(st.lag, [0]);
    assert_eq!(st.late, 0);
}

/// Ops at one planned instant go back to back, each adding its time; a lane past its plan
/// stays behind by its lag L, which never shrinks, however long the plan's next gap.
#[test]
fn a_lane_keeps_its_lag() {
    let (ops, st) = run(tl(json!([stat(1), stat(2), compute(100), stat(3), compute(1000), stat(4)])), |r, _| if &*r.path == "d/2" { 150 } else { 10 });
    // d/1 at 0 done at 10; d/2 at 10 done at 160; d/3 planned at 100: L = 60; d/4 at 1100 + 60
    assert_eq!(when(&ops), pairs(&[("d/1", 0), ("d/2", 10), ("d/3", 160), ("d/4", 1160)]));
    assert_eq!(st.lag, [60]);
}

/// A fork starts its sub-lanes with the parent's L, once the parent reaches it; a joined parent
/// goes on at the latest of its plan plus L and its sub-lanes' last completions.
#[test]
fn forks_inherit_the_parents_lag_and_joins_wait_for_their_lanes() {
    let body = tl(json!([stat(1), compute(10),
        {"parallel": {"index": "k", "width": 2, "body": [compute(5), stat(100)]}},
        stat(2)]));
    // d/1 takes 50: the root is 40 behind at the fork (plan 10); each sub-lane's stat (plan 15) at
    // 55; one takes 100, done at 155; the root's stat (planned 15, the joined end) at 155
    let mut n = 0;
    let (ops, st) = run(body, |r, _| match &*r.path {
        "d/1" => 50,
        "d/100" => {
            n += 1;
            if n == 1 { 100 } else { 1 }
        }
        _ => 1,
    });
    assert_eq!(when(&ops), pairs(&[("d/1", 0), ("d/100", 55), ("d/100", 55), ("d/2", 155)]));
    assert_eq!(st.lag[0], 140, "the root: planned 15, went on at 155");
}

/// A parent's joined forks are joined one at a time: the first join waits for its own fork's
/// sub-lanes, never the next fork's, which start after it (found by `agentx-tiered`, whose
/// lanes fork sessions one after another: every join waited for all of them, and the run stopped).
#[test]
fn a_parent_forking_twice_joins_each_fork_in_turn() {
    let fork = |id: i64, ns: i64| json!({"parallel": {"index": "k", "width": 2, "body": [compute(ns), stat(id)]}});
    let (ops, st) = run(tl(json!([fork(1, 10), fork(2, 20), stat(3)])), |_, _| 1);
    assert_eq!(when(&ops), pairs(&[("d/1", 10), ("d/1", 10), ("d/2", 31), ("d/2", 31), ("d/3", 32)]));
    assert_eq!(st.lag[0], 2, "the root: each join one op's completion behind");
}

/// Path order: a session reading what another stores waits until the store has completed, and
/// the wait goes into its L; readers never wait for readers.
#[test]
fn a_store_finishing_after_its_readers_launch_delays_the_reader() {
    let write = json!([{"open": {"file": r("f"), "flags": ["WRONLY", "CREAT"]}}, {"write": {"file": r("f"), "len": 4096}}, {"close": {"file": r("f")}}]);
    let read = json!([{"open": {"file": r("f"), "flags": ["RDONLY"]}}, {"read": {"file": r("f"), "len": 4096}}, {"close": {"file": r("f")}}]);
    let body = tl(json!([obj("f", 1), {"parallel": {"index": "k", "width": 3, "body": [
        compute(0), {"cond": {"if": {"eq": [{"index": "k"}, 0]}, "then": write,
            "else": [compute(100), {"cond": {"if": {"eq": [{"index": "k"}, 1]}, "then": read.clone(), "else": read}}]}}]}}]));
    // the write takes 500: its close completes at 502; both readers planned at 100
    let (ops, st) = run(body, |r, _| if r.op().kind == aeiou::vm::OpKind::Write { 500 } else { 1 });
    let t: Vec<(String, i64)> = ops.iter().map(|o| (format!("{} {}", o.record.op().kind.name(), o.record.lane), o.issued)).collect();
    let opens: Vec<i64> = ops.iter().filter(|o| o.record.op().kind.name() == "open" && o.issued > 0).map(|o| o.issued).collect();
    assert_eq!(opens, [502, 502], "{t:?}");
    let reads: Vec<i64> = ops.iter().filter(|o| o.record.op().kind.name() == "read").map(|o| o.issued).collect();
    assert_eq!(reads, [503, 503], "both readers at once: {t:?}");
    assert_eq!(st.edge_waits, 2);
    assert_eq!(st.edge_wait_ns, 2 * 402);
}

/// A submission's body runs on its worker lane no earlier than the submitter reaches it, and
/// the worker lane's own backlog delays the next; the submitter goes on at once.
#[test]
fn submissions_queue_on_their_worker_lane() {
    let sub = |id: i64| json!({"submit": {"workers": "w", "body": [stat(id)]}});
    let (ops, _) = run(tl(json!([sub(1), sub(2), stat(9), compute(10), sub(3)])), |r, _| if &*r.path == "d/9" { 1 } else { 100 });
    assert_eq!(when(&ops), pairs(&[("d/9", 0), ("d/1", 0), ("d/2", 100), ("d/3", 200)]));
}

/// Bytes submitted and not yet written are staged; past the workers' staging, a would-be drop.
#[test]
fn staging_past_its_capacity_is_a_would_be_drop() {
    let store = |key: i64| json!({"submit": {"workers": "w", "body": [obj("f", key),
        {"open": {"file": r("f"), "flags": ["WRONLY", "CREAT"]}}, {"write": {"file": r("f"), "len": 4096}}, {"close": {"file": r("f")}}]}});
    // staging 8192: two stores held at once fit, the third at the same instant does not
    let (_, st) = run(tl(json!([store(1), store(2), compute(10_000), store(3)])), |_, _| 1000);
    assert_eq!(st.staging[0].peak, 8192);
    assert!(st.staging[0].drops.is_empty(), "{:?}", st.staging[0].drops);
    let (_, st) = run(tl(json!([store(1), store(2), store(3)])), |_, _| 1000);
    assert_eq!(st.staging[0].peak, 3 * 4096);
    assert_eq!(st.staging[0].drops, [(0, 3 * 4096)]);
    assert_eq!(st.staging[0].held, 0, "every write completed");
}

/// An eviction batch that overruns its tick delays the next by its overage: the eviction lane's L.
#[test]
fn an_eviction_batch_overrunning_its_tick_delays_the_next() {
    let tiers = json!([{"name": "t", "order": [{"cached": "touched"}, {"neg": {"cached": "pos"}}], "capacity": 100,
        "on_evict": {"lanes": 1, "body": [{"stat": {"file": {"file": {"dataset": "d", "id": {"mod": [{"victim": {}}, 1000]}}}}}]},
        "evict": {"every": 100, "when": true, "count": 2}}]);
    let body = json!([{"let": {"name": "a", "value": {"shared": {"id": 1, "n": 4}}}},
        {"store": {"cache": "lm", "tier": "t", "keys": crate::common::chunk_keys(r("a"), json!(1), json!(0), Some(json!(4)))}}, compute(250)]);
    let ast = json!([{"timeline": {"name": "tl", "caches": [{"name": "lm", "tiers": tiers}], "body": body}}]);
    // each unlink-like stat takes 80: the batch at 100 ends at 260, the one at 200 starts at 260
    let (ops, st) = run(ast, |_, _| 80);
    let t: Vec<i64> = ops.iter().map(|o| o.issued).collect();
    assert_eq!(t, [100, 180, 260, 340]);
    assert_eq!(*st.lag.iter().max().unwrap(), 140);
}

/// The planner's paths are retired as their ops complete: a long run holds about those in
/// flight, never all it has touched.
#[test]
fn the_planner_forgets_completed_paths() {
    let n = 20_000;
    let m = model(tl(json!([{"loop": {"index": "j", "to": n, "body": [
        {"let": {"name": "g", "value": {"object": {"namespace": "kv", "fields": {"key": {"index": "j"}}}}}},
        {"stat": {"file": r("g")}}, compute(10)]}}])));
    let mut p = Plan::new(m, "gpu", 0, 1);
    let (ops, st) = simulate(&mut p, 100, |_, _| 5).unwrap();
    assert_eq!(ops.len(), n as usize);
    assert!(st.paths_peak >= aeiou::exec::RETIRE_FLOOR && st.paths_peak < 2 * aeiou::exec::RETIRE_FLOOR, "{} paths held", st.paths_peak);
}

/// A joined parent goes on at its sub-lanes' latest planned end even when they completed sooner.
#[test]
fn a_join_waits_for_the_planned_end_of_its_lanes() {
    let (ops, _) = run(tl(json!([{"parallel": {"index": "k", "width": 1, "body": [stat(1), compute(100)]}}, stat(2)])), |_, _| 1);
    assert_eq!(when(&ops), pairs(&[("d/1", 0), ("d/2", 100)]));
}

/// An eviction a store started begins no earlier than the storing lane reaches the store.
#[test]
fn an_eviction_at_a_store_waits_for_the_storing_lane() {
    let tiers = json!([{"name": "t", "order": [{"cached": "touched"}, {"neg": {"cached": "pos"}}], "capacity": 2,
        "on_evict": {"lanes": 1, "body": [{"stat": {"file": {"file": {"dataset": "d", "id": {"mod": [{"victim": {}}, 1000]}}}}}]}}]);
    let body = json!([stat(1), {"let": {"name": "a", "value": {"shared": {"id": 1, "n": 3}}}},
        {"store": {"cache": "lm", "tier": "t", "keys": crate::common::chunk_keys(r("a"), json!(1), json!(0), Some(json!(3)))}}]);
    let ast = json!([{"timeline": {"name": "tl", "caches": [{"name": "lm", "tiers": tiers}], "body": body}}]);
    let (ops, st) = run(ast, |r, _| if &*r.path == "d/1" { 50 } else { 1 });
    assert_eq!(ops.len(), 2);
    assert_eq!(&*ops[0].record.path, "d/1");
    assert_eq!(ops[1].issued, 50, "the victim waits for the store, after d/1's 50 ns");
    assert_eq!(st.lag.iter().max(), Some(&50));
}

/// The tiered KV abstract's case through the executor on storage of a fixed latency, at
/// `AEIOU_TEST_SESSIONS` sessions (2 unless set; the case's 40 runs its 2,998,578 ops in about
/// 25 s in a release build): every op the plan holds is issued, and how long the plan and the
/// executor take and what they hold.
#[test]
fn agentx_tiered_through_the_executor() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../workloads/kvcache/lmcache-local-disk");
    let lock: Value = serde_json::from_str(&std::fs::read_to_string(root.join("cases/agentx-tiered/case.lock.json")).unwrap()).unwrap();
    let sessions = std::env::var("AEIOU_TEST_SESSIONS").unwrap_or_else(|_| "2".into());
    let mut overrides: Vec<(String, String)> = lock["runs"][0]["params"].as_object().unwrap().iter()
        .map(|(k, v)| (k.clone(), v.as_object().unwrap().values().next().unwrap().to_string())).collect();
    overrides.push(("sessions".into(), sessions));
    let cfg = Box::leak(Box::new(Config { seed: 0, gpus: 1, overrides, sets: vec![], st_blksize: Default::default() }));
    let loaded = aeiou::load(&root.join("lmcache_local_disk_tiered.ast.json")).unwrap();
    let ast = Box::leak(Box::new(loaded.ast));
    let params = Box::leak(Box::new(Params::new(ast, cfg).unwrap()));
    let m = Box::leak(Box::new(build_model(ast, cfg, params).unwrap()));
    let mut planned = 0;
    let mut p = Plan::new(m, "gpu", 0, 1);
    while let Some(x) = p.next().unwrap() {
        planned += matches!(x, aeiou::plan::Planned::Op(_)) as usize;
    }
    let t0 = std::time::Instant::now();
    let mut p = Plan::new(m, "gpu", 0, 1);
    let (ops, st) = simulate(&mut p, 10_000_000_000, |_, _| 1_000_000).unwrap();
    let mut lag = st.lag.clone();
    lag.sort();
    eprintln!("{} ops in {:.1} s; ends at {:.1} h; {} lanes, L max {:.3} s, median {:.3} s; edge waits {}; staging peak {} MiB, {} would-be drops; paths peak {}",
        ops.len(), t0.elapsed().as_secs_f64(), ops.iter().map(|o| o.completed).max().unwrap_or(0) as f64 / 3.6e12, lag.len(),
        lag.last().copied().unwrap_or(0) as f64 / 1e9, lag.get(lag.len() / 2).copied().unwrap_or(0) as f64 / 1e9,
        st.edge_waits, st.staging.first().map_or(0, |s| s.peak >> 20), st.staging.first().map_or(0, |s| s.drops.len()), st.paths_peak);
    assert!(planned > 0);
    assert_eq!(ops.len(), planned);
    assert!(ops.iter().all(|o| o.completed >= o.issued && o.issued >= o.record.plan));
}

fn span(name: &str, body: Value) -> Value {
    json!({"span": {"name": name, "body": body}})
}

/// A span's D runs from its launch (its plan at entry plus the lane's L, or the lane's previous
/// completion if later) to its last op's completion: the lag before it is not in D, the wait
/// behind its own ops is (brief item 72 (7) step 5, `DESIGN_REVIEW.md` §3.99 point 3).
#[test]
fn a_span_measures_from_its_launch_to_its_last_completion() {
    // d/1 at 0 done at 50, planned gap 10: the lane is 40 behind when the span starts at 10
    let body = tl(json!([stat(1), compute(10), span("load", json!([stat(2), stat(3)])), compute(1000), span("load", json!([stat(4)]))]));
    let (ops, st) = run(body, |r, _| if &*r.path == "d/1" { 50 } else { 20 });
    // d/2 at 50 (L 40), d/3 at 70 (L 60)
    assert_eq!(when(&ops), pairs(&[("d/1", 0), ("d/2", 50), ("d/3", 70), ("d/4", 1070)]));
    let load = &st.spans["load"];
    // (planned start, D): launch at 50, done at 90; launch at 1010 + 60, done at 1090
    assert_eq!(load, &[(10, 40), (1010, 20)]);
    assert_eq!(st.body_lag, [60]);
}

/// A span's op that waits on another lane's change (an edge) has the wait in its D: the reader
/// launched at its plan, its open issued once the writer's close completed.
#[test]
fn a_span_includes_its_edge_waits() {
    let wr = json!([obj("f", 1), {"open": {"file": r("f"), "flags": ["WRONLY", "CREAT"]}}, {"close": {"file": r("f")}}]);
    let rd = json!([compute(10), obj("f", 1), span("load", json!([{"open": {"file": r("f"), "flags": ["RDONLY"]}}, {"close": {"file": r("f")}}]))]);
    // lane 1 writes (its open 30, its close 5 more); lane 2 reads in a span planned at 10
    let body = tl(json!([{"parallel": {"index": "k", "width": 2, "body": [{"cond": {"if": {"eq": [{"index": "k"}, 0]}, "then": wr, "else": rd}}]}}]));
    let (ops, st) = run(body, |r, _| if r.lane == 1 && r.kind == aeiou::vm::OpKind::Open { 30 } else { 5 });
    assert_eq!(when(&ops), pairs(&[("kv/1", 0), ("kv/1", 30), ("kv/1", 35), ("kv/1", 40)]));
    assert_eq!(st.spans["load"], [(10, 35)]);
    assert_eq!(st.edge_waits, 1);
}

/// The summary of a span's D: count, sum, nearest-rank p50 and p99, maximum, and each
/// 5-minute window's count and p99 by planned start, the worst window's p99 the result.
#[test]
fn span_summaries_are_by_window_of_planned_start() {
    use aeiou::exec::{summarize, WINDOW_NS};
    let mut ds: Vec<(i64, i64)> = (0..100).map(|i| (i, i + 1)).collect();
    ds.push((WINDOW_NS, 500));
    ds.push((WINDOW_NS + 5, 7));
    let s = summarize(&ds);
    assert_eq!((s.count, s.sum, s.p50, s.p99, s.max), (102, 5050 + 507, 50, 100, 500));
    assert_eq!(s.windows, [(0, 100, 99), (1, 2, 500)]);
    assert_eq!(s.window_p99, 500);
    let e = summarize(&[]);
    assert_eq!((e.count, e.p99, e.window_p99), (0, 0, 0));
}

/// A span naming its staging (contract 0.21, the user's choice of 2026-11-14) draws on the
/// workers' staging with its reads while they are in flight, as a load into the pool the
/// pending stores hold: one lane's read fits, two lanes' at once are past it, a would-be drop;
/// a span naming none is not counted (brief item 72 (7) step 5).
#[test]
fn a_spans_loads_draw_on_its_staging() {
    let load = |staging: Option<&str>| {
        let mut s = json!({"span": {"name": "load", "body": [obj("f", 1), {"open": {"file": r("f"), "flags": ["RDONLY"]}},
            {"read": {"file": r("f"), "len": 6000}}, {"close": {"file": r("f")}}]}});
        if let Some(w) = staging {
            s["span"]["staging"] = json!(w);
        }
        s
    };
    // one lane: 6000 of 8192 held while the read is in flight
    let (_, st) = run(tl(json!([load(Some("w"))])), |_, _| 10);
    assert_eq!((st.staging[0].peak, st.staging[0].held), (6000, 0));
    assert!(st.staging[0].drops.is_empty());
    // two lanes reading at once, each after its open: 12,000 held at 10, past 8192
    let (_, st) = run(tl(json!([{"parallel": {"index": "k", "width": 2, "body": [load(Some("w"))]}}])), |_, _| 10);
    assert_eq!(st.staging[0].peak, 12000);
    assert_eq!(st.staging[0].drops, [(10, 12000)]);
    assert_eq!(st.staging[0].held, 0);
    // a span naming no staging: nothing held
    let (_, st) = run(tl(json!([{"parallel": {"index": "k", "width": 2, "body": [load(None)]}}])), |_, _| 10);
    assert!(st.staging.iter().all(|s| s.peak == 0 && s.drops.is_empty()), "{:?}", st.staging);
}

/// A submitted write not yet written and a span's load in flight share the staging: 4096 + 6000
/// past 8192 when the read issues while the write is still in flight.
#[test]
fn stores_and_loads_share_the_staging() {
    let body = tl(json!([obj("g", 2), {"submit": {"workers": "w", "body": [{"open": {"file": r("g"), "flags": ["WRONLY", "CREAT"]}},
        {"write": {"file": r("g"), "len": 4096}}, {"close": {"file": r("g")}}]}},
        {"span": {"name": "load", "staging": "w", "body": [obj("f", 1), {"open": {"file": r("f"), "flags": ["RDONLY"]}},
            {"read": {"file": r("f"), "len": 6000}}, {"close": {"file": r("f")}}]}}]));
    // the write takes 100; the read issues at 1 (after the span's open) while it is in flight
    let (_, st) = run(body, |r, _| if r.kind == aeiou::vm::OpKind::Write { 100 } else { 1 });
    assert_eq!(st.staging[0].peak, 10096);
    assert_eq!(st.staging[0].drops.len(), 1, "{:?}", st.staging[0]);
}

fn run_with(body: Value, opts: &aeiou::exec::TimelineOpts, latency: impl FnMut(&Record, i64) -> i64) -> (Vec<Simulated>, Stats) {
    let m = model(body);
    let mut p = Plan::new(m, "gpu", 0, 1);
    aeiou::exec::simulate_with(&mut p, 1_000_000_000, opts, latency).unwrap_or_else(|e| panic!("{e:#}"))
}

/// Every op's lateness is recorded (on a simulated clock, none); a body lane whose L passes the
/// lag ceiling is a certain failure: the run stops there, what is in flight completing, unless
/// told to keep going (brief item 72 (7) step 5).
#[test]
fn the_lag_ceiling_stops_a_run_unless_told_to_keep_going() {
    // each stat takes 100, planned 10 apart: L grows by 90 an op
    let body = tl(json!([stat(1), compute(10), stat(2), compute(10), stat(3), compute(10), stat(4)]));
    let (ops, st) = run(body.clone(), |_, _| 100);
    assert_eq!(st.lateness.count, 4);
    assert_eq!(st.lateness.max, 0);
    assert_eq!(ops.len(), 4);
    let stop = aeiou::exec::TimelineOpts { lag_ceiling_ns: Some(150), ..Default::default() };
    let (ops, st) = run_with(body.clone(), &stop, |_, _| 100);
    // d/3 at 200 (L 180) passes 150: it is issued, and nothing after it
    assert_eq!(when(&ops), pairs(&[("d/1", 0), ("d/2", 100), ("d/3", 200)]));
    assert!(ops.iter().all(|o| o.completed >= o.issued));
    assert_eq!(st.ceiling, Some((200, 0, 180)));
    assert!(st.stopped);
    assert_eq!(st.certain.as_deref(), Some("lane 0's lag reached 0.000 s, past the lag ceiling of 0.000 s, at 0.000 s into the timeline"));
    let go = aeiou::exec::TimelineOpts { keep_going: true, ..stop };
    let (ops, st) = run_with(body, &go, |_, _| 100);
    assert_eq!(ops.len(), 4);
    assert!(!st.stopped);
    assert!(st.certain.is_some());
}

/// A watched constraint (`result.span.S.window_p99_s <= C`) is lost once a window closes with a
/// p99 above C: the run stops at the window's close, not at the span; a window within C, or one
/// not closed, does not stop it; `.max_s` is judged on the window's largest D.
#[test]
fn a_watched_constraint_stops_the_run_when_a_window_closes_above_it() {
    let w = aeiou::exec::WINDOW_NS;
    // a span each window: the first's read takes 10, the second's 5000, a third window after
    let body = tl(json!([span("load", json!([stat(1)])), compute(w), span("load", json!([stat(2)])), compute(w), span("load", json!([stat(3)])), compute(w), stat(4)]));
    let lat = |r: &Record, _: i64| if &*r.path == "d/2" { 5000 } else { 10 };
    let watch = |limit_ns: i64, max: bool| aeiou::exec::Watch { span: "load".into(), max, limit_ns, strict: false, text: "result.span.load.window_p99_s <= C".into() };
    let opts = aeiou::exec::TimelineOpts { watches: vec![watch(1000, false)], ..Default::default() };
    let (ops, st) = run_with(body.clone(), &opts, lat);
    // window 1 closes when the planner passes 2w (d/3's span planned there): d/3 is not issued
    assert_eq!(when(&ops).iter().map(|x| x.0.as_str()).collect::<Vec<_>>(), ["d/1", "d/2"]);
    assert!(st.stopped);
    let c = st.certain.unwrap();
    assert!(c.starts_with("constraint `result.span.load.window_p99_s <= C` lost: span `load`'s window 1 (planned 300 s to 600 s) closed with a p99 of 0.000 s over 1 spans"), "{c}");
    // within its limit, the whole run
    let opts = aeiou::exec::TimelineOpts { watches: vec![watch(10_000, true)], ..Default::default() };
    let (ops, st) = run_with(body, &opts, lat);
    assert_eq!(ops.len(), 4);
    assert!(st.certain.is_none() && !st.stopped);
}

/// A would-be drop is a certain failure: the run stops at it.
#[test]
fn a_would_be_drop_stops_the_run() {
    let load = json!({"span": {"name": "load", "staging": "w", "body": [obj("f", 1), {"open": {"file": r("f"), "flags": ["RDONLY"]}},
        {"read": {"file": r("f"), "len": 6000}}, {"close": {"file": r("f")}}]}});
    let body = tl(json!([{"parallel": {"index": "k", "width": 2, "body": [load, compute(100), stat(9)]}}]));
    let (ops, st) = run_with(body, &Default::default(), |_, _| 10);
    assert!(st.stopped);
    assert!(st.certain.unwrap().starts_with("staging of workers `w` held 12000 bytes, past its 8192"));
    assert!(!ops.iter().any(|o| &*o.record.path == "d/9"), "nothing issues after the stop");
}

/// A run's series (brief item 72 (7) step 5, `--series`): a row a second of run time, by when its
/// ops completed, and a row a second of planned time, by when they were planned; every op in
/// one of each, writes with their latency, a lagging op in a later run second than its planned
/// one, live lanes and the planner's lead sampled on the run clock, eviction batches on the
/// planned one.
#[test]
fn a_series_puts_each_op_in_its_second_of_each_clock() {
    let s = 1_000_000_000i64;
    let wr = |k: i64| json!([obj("f", k), {"open": {"file": r("f"), "flags": ["WRONLY", "CREAT"]}}, {"write": {"file": r("f"), "len": 4096}}, {"close": {"file": r("f")}}]);
    // second 0: a stat; second 1: a file written (its write takes 1.5 s); second 2: a stat
    let mut body = vec![stat(1), compute(s)];
    body.extend(wr(1).as_array().unwrap().iter().cloned());
    body.extend([compute(s), stat(2)]);
    let opts = aeiou::exec::TimelineOpts { series: true, ..Default::default() };
    let (ops, st) = run_with(tl(json!(body)), &opts, |r, _| if r.kind == aeiou::vm::OpKind::Write { 1_500_000_000 } else { 10 });
    let rows = &st.rows;
    let total = |planned: bool| -> u64 { rows.iter().filter(|r| r.planned == planned).map(|r| r.ops.values().sum::<u64>()).sum() };
    assert_eq!(total(true), ops.len() as u64);
    assert_eq!(total(false), ops.len() as u64);
    let planned: Vec<(i64, u64)> = rows.iter().filter(|r| r.planned).map(|r| (r.s, r.ops.values().sum())).collect();
    assert_eq!(planned, [(0, 1), (1, 3), (2, 1)]);
    // the write planned in second 1 completes in second 2, and the close and stat after it later
    let w = rows.iter().find(|r| !r.planned && r.writes == 1).unwrap();
    assert_eq!((w.s, w.write_ns, w.write_bytes), (2, 1_500_000_000, 4096));
    let p1 = rows.iter().find(|r| r.planned && r.s == 1).unwrap();
    assert_eq!((p1.creates, p1.writes, p1.write_bytes), (1, 1, 4096));
    assert!(rows.iter().filter(|r| !r.planned).all(|r| r.live_lanes.is_some()));
    assert!(rows.iter().any(|r| !r.planned && r.lead_ns.is_some()));
    let sum = st.series.unwrap();
    assert_eq!(sum.peak_write_bytes.1, 4096);
    assert!(sum.seconds >= 3);
}

/// A series counts eviction batches on the planned clock: a tier evicting at every tick.
#[test]
fn a_series_counts_eviction_batches_by_planned_second() {
    let s = 1_000_000_000i64;
    let tiers = json!([{"name": "t", "order": [{"cached": "stored"}], "capacity": 1, "evict": {"every": s, "when": {"gt": [{"tier": "entries"}, 0]}, "count": 1},
        "on_evict": {"lanes": 1, "body": [stat(7)]}}]);
    let body = json!([{"let": {"name": "a", "value": {"shared": {"id": 1, "n": 2}}}},
        {"store": {"cache": "lm", "tier": "t", "keys": crate::common::chunk_keys(r("a"), json!(1), json!(0), Some(json!(1)))}}, compute(3 * s), stat(1)]);
    let doc = json!([{"timeline": {"name": "tl", "caches": [{"name": "lm", "tiers": tiers}], "body": body}}]);
    let opts = aeiou::exec::TimelineOpts { series: true, ..Default::default() };
    let (_, st) = run_with(doc, &opts, |_, _| 10);
    let ev: u64 = st.rows.iter().filter(|r| r.planned).map(|r| r.evictions).sum();
    assert!(ev >= 1, "{:?}", st.rows);
    assert_eq!(st.rows.iter().filter(|r| !r.planned).map(|r| r.evictions).sum::<u64>(), 0);
}

/// Worker lanes are live while they have work due, and each workers' backlog (writes submitted
/// and not yet written, and their bytes) is sampled into the run's seconds: a body lane that
/// issues nothing itself submits two stores to one worker lane, each write taking 1.5 s.
#[test]
fn a_series_counts_worker_lanes_live_and_their_backlog() {
    let s = 1_000_000_000i64;
    let store = |key: i64| json!({"submit": {"workers": "w", "body": [obj("f", key),
        {"open": {"file": r("f"), "flags": ["WRONLY", "CREAT"]}}, {"write": {"file": r("f"), "len": 4096}}, {"close": {"file": r("f")}}]}});
    let opts = aeiou::exec::TimelineOpts { series: true, ..Default::default() };
    let (_, st) = run_with(tl(json!([store(1), store(2), compute(4 * s)])), &opts, |r, _| if r.kind == aeiou::vm::OpKind::Write { 1_500_000_000 } else { 10 });
    let row = |sec: i64| st.rows.iter().find(|r| !r.planned && r.s == sec).unwrap_or_else(|| panic!("no run second {sec}: {:?}", st.rows));
    // second 0: both submitted, the worker busy; second 1: the first written at 1.5 s
    assert_eq!((row(0).live_lanes, row(0).backlog.get("w").copied()), (Some(1), Some((2, 8192))));
    assert_eq!((row(1).live_lanes, row(1).backlog.get("w").copied()), (Some(1), Some((1, 4096))));
    // second 3: the second written at 3.0 s; the planner's wake at 3.0 s saw it in flight
    assert_eq!(row(3).backlog.get("w").copied(), Some((1, 4096)));
    let sum = st.series.as_ref().unwrap();
    assert_eq!((sum.peak_live_lanes.1, sum.peak_backlog), (1, (0, 2)));
    assert_eq!((st.staging[0].queued, st.staging[0].queued_peak, st.staging[0].queued_bytes_peak), (0, 2, 8192));
}

/// An eviction batch's duration runs from its start (its tick, or the storing lane reaching it)
/// to its last victim's completion; past the next batch's planned start, its overage. The
/// eviction lanes are live and their backlog counted while a batch's ops are due.
#[test]
fn eviction_batches_report_their_duration_and_overage() {
    let s = 1_000_000_000i64;
    let tiers = json!([{"name": "t", "order": [{"cached": "touched"}, {"neg": {"cached": "pos"}}], "capacity": 100,
        "on_evict": {"lanes": 1, "body": [{"stat": {"file": {"file": {"dataset": "d", "id": {"mod": [{"victim": {}}, 1000]}}}}}]},
        "evict": {"every": s, "when": {"gt": [{"tier": "entries"}, 0]}, "count": 2}}]);
    let body = json!([{"let": {"name": "a", "value": {"shared": {"id": 1, "n": 4}}}},
        {"store": {"cache": "lm", "tier": "t", "keys": crate::common::chunk_keys(r("a"), json!(1), json!(0), Some(json!(4)))}}, compute(5 * s)]);
    let ast = json!([{"timeline": {"name": "tl", "caches": [{"name": "lm", "tiers": tiers}], "body": body}}]);
    let opts = aeiou::exec::TimelineOpts { series: true, ..Default::default() };
    // each victim's stat takes 0.8 s: the batch at 1 s ends at 2.6 s, 0.6 s past the next's
    // planned start at 2 s, which starts at 2.6 s and ends at 4.2 s
    let (ops, st) = run_with(ast, &opts, |_, _| 800_000_000);
    assert_eq!(ops.iter().map(|o| o.issued).collect::<Vec<_>>(), [s, 1_800_000_000, 2_600_000_000, 3_400_000_000]);
    let mut ev = st.evictions.clone();
    ev.sort_unstable();
    assert_eq!(ev, [(s, 1_600_000_000, 600_000_000), (2 * s, 2_200_000_000, 0)]);
    let run = |sec: i64| st.rows.iter().find(|r| !r.planned && r.s == sec).unwrap_or_else(|| panic!("no run second {sec}: {:?}", st.rows));
    assert_eq!((run(1).live_lanes, run(1).evict_queued), (Some(1), 2));
    assert_eq!(run(2).evict_queued, 2, "the first batch's last op completes at 2.6 s; the second's two are due");
    assert_eq!(run(3).evict_queued, 1, "the second batch's first op completes at 3.4 s");
    let planned = |sec: i64| st.rows.iter().find(|r| r.planned && r.s == sec).unwrap();
    assert_eq!((planned(1).eviction_max_ns, planned(2).eviction_max_ns), (1_600_000_000, 2_200_000_000));
    assert_eq!(st.series.as_ref().unwrap().peak_evict_queued, (1, 2));
}

/// Eviction ops planned ahead of their tick are not yet due: neither live nor backlog before it.
#[test]
fn eviction_ops_before_their_tick_are_not_due() {
    let s = 1_000_000_000i64;
    let tiers = json!([{"name": "t", "order": [{"cached": "touched"}, {"neg": {"cached": "pos"}}], "capacity": 100,
        "on_evict": {"lanes": 1, "body": [{"stat": {"file": {"file": {"dataset": "d", "id": {"mod": [{"victim": {}}, 1000]}}}}}]},
        "evict": {"every": 2 * s, "when": {"gt": [{"tier": "entries"}, 0]}, "count": 1}}]);
    let body = json!([{"let": {"name": "a", "value": {"shared": {"id": 1, "n": 1}}}},
        {"store": {"cache": "lm", "tier": "t", "keys": crate::common::chunk_keys(r("a"), json!(1), json!(0), Some(json!(1)))}}, compute(s + s / 2), stat(9), compute(s)]);
    let ast = json!([{"timeline": {"name": "tl", "caches": [{"name": "lm", "tiers": tiers}], "body": body}}]);
    let opts = aeiou::exec::TimelineOpts { series: true, ..Default::default() };
    // the batch at 2 s is pushed by 1.5 s (the horizon 1 s); at 1.5 s the body's stat is the one live lane
    let (_, st) = run_with(ast, &opts, |_, _| 10);
    let run = |sec: i64| st.rows.iter().find(|r| !r.planned && r.s == sec).unwrap_or_else(|| panic!("no run second {sec}: {:?}", st.rows));
    assert_eq!((run(1).live_lanes, run(1).evict_queued), (Some(1), 0));
    // at 2 s the eviction lane's op; the body lane ended with its stat (its trailing compute
    // holds nothing)
    assert_eq!((run(2).live_lanes, run(2).evict_queued), (Some(1), 1));
}
