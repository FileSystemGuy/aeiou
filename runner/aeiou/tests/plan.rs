//! The planner (`DESIGN_REVIEW.md` §3.99, the note "The run side of timelines", step 2; brief
//! item 72 (7)): the walk `dry-run` makes of a timeline, each op an owned record with its
//! executor lane, its planned time, and the edges it waits on, the `trace` node's path order
//! (`runner/REFERENCE.md` §13) across the timeline's lanes in planned order.

use aeiou::eval::{build_model, Config, Model, Params};
use aeiou::plan::{Plan, Planned, Record};
use aeiou::vm::{actor_counts, op_hash, run_actor, Control, Op, OpCtx, Sink};
use serde_json::{json, Value};

fn doc(body: Value) -> String {
    json!({"ast": "0.20", "name": "t",
           "datasets": {"d": {"layout": "whole", "access": "map", "pattern": "d/{id}", "count": 1000, "size": {"const": 1}, "seed": 1}},
           "namespaces": {"kv": {"pattern": "kv/{key}", "fields": {"key": "int"}, "size": 4096, "seed": 1}},
           "actors": {"gpu": {"count": 1, "body": body}}}).to_string()
}

fn model(body: Value) -> &'static Model<'static> {
    let cfg = Box::leak(Box::new(Config { seed: 1, gpus: 1, overrides: vec![], sets: vec![], st_blksize: Default::default() }));
    let loaded = aeiou::load_str(&doc(body)).unwrap_or_else(|e| panic!("{e:#}"));
    let ast = Box::leak(Box::new(loaded.ast));
    let params = Box::leak(Box::new(Params::new(ast, cfg).unwrap()));
    Box::leak(Box::new(build_model(ast, cfg, params).unwrap()))
}

/// Everything the planner gives for actor 0 of `gpu`.
fn planned(body: Value) -> Vec<Planned> {
    let m = model(body);
    let mut p = Plan::new(m, "gpu", 0, 1);
    let mut out = Vec::new();
    while let Some(x) = p.next().unwrap_or_else(|e| panic!("{e:#}")) {
        out.push(x);
    }
    out
}

fn records(body: Value) -> Vec<Record> {
    planned(body).into_iter().filter_map(|p| match p {
        Planned::Op(r) => Some(r),
        Planned::Compute { .. } => None,
    }).collect()
}

fn stat(id: Value) -> Value {
    json!({"stat": {"file": {"file": {"dataset": "d", "id": id}}}})
}

fn compute(ns: Value) -> Value {
    json!({"compute": {"ns": ns}})
}

fn tl(body: Value) -> Value {
    json!([{"timeline": {"name": "tl", "workers": [{"name": "w", "lanes": 2, "staging": 1048576}], "body": body}}])
}

fn obj(name: &str, key: i64) -> Value {
    json!({"let": {"name": name, "value": {"object": {"namespace": "kv", "fields": {"key": key}}}}})
}

fn r(name: &str) -> Value {
    json!({"ref": name})
}

/// A lane of `width` sessions, session k at planned time 10·k, each running `body`.
fn sessions(width: i64, body: Value) -> Value {
    let mut b = vec![compute(json!({"mul": [{"index": "k"}, 10]}))];
    b.extend(body.as_array().unwrap().iter().cloned());
    json!({"parallel": {"index": "k", "width": width, "body": b}})
}

fn kind(r: &Record) -> &'static str {
    r.op().kind.name()
}

/// Each record has its planned time, and a lane of its own executor lane: the lanes of
/// sessions forked at once differ, a lane's own ops share one.
#[test]
fn records_carry_their_lane_and_planned_time() {
    let recs = records(tl(json!([sessions(3, json!([stat(json!({"index": "k"})), compute(json!(4)), stat(json!({"add": [100, {"index": "k"}]}))]))])));
    let got: Vec<(String, i64)> = recs.iter().map(|r| (r.path.to_string(), r.plan)).collect();
    assert_eq!(got, [("d/0", 0), ("d/100", 4), ("d/1", 10), ("d/101", 14), ("d/2", 20), ("d/102", 24)].map(|(p, t)| (p.to_string(), t)));
    let lanes: Vec<usize> = recs.iter().map(|r| r.lane).collect();
    assert_eq!(lanes[0], lanes[1]);
    assert_eq!(lanes[2], lanes[3]);
    assert_eq!(lanes[4], lanes[5]);
    assert!(lanes[0] != lanes[2] && lanes[2] != lanes[4] && lanes[0] != lanes[4], "{lanes:?}");
    let seqs: Vec<u64> = recs.iter().map(|r| r.seq).collect();
    assert_eq!(seqs, [0, 1, 2, 3, 4, 5], "a record's number is its place in planned order");
    assert!(recs.iter().all(|r| r.edges.is_empty()), "nothing changes a path: no edges");
}

/// A `compute` is given with its lane and the planned time it starts at.
#[test]
fn computes_are_planned_with_their_lane() {
    let p = planned(tl(json!([sessions(2, json!([stat(json!(1))]))])));
    let comp: Vec<(i64, i64)> = p.iter().filter_map(|x| match x {
        Planned::Compute { plan, ns, .. } => Some((*plan, *ns)),
        _ => None,
    }).collect();
    assert_eq!(comp, [(0, 0), (0, 10)]);
    let lanes: Vec<usize> = p.iter().map(|x| match x {
        Planned::Compute { lane, .. } => *lane,
        Planned::Op(r) => r.lane,
    }).collect();
    assert_eq!(lanes[0], lanes[1], "session 0's compute and stat");
    assert_eq!(lanes[2], lanes[3], "session 1's");
    assert_ne!(lanes[0], lanes[2]);
}

/// Path order across lanes, in planned order: a changing op waits for every earlier op on its
/// path, a reading op for the earlier changing ops, reads never for reads; an op never waits
/// for its own lane's earlier ops, which its lane issues first anyway, and of another lane's
/// reads only for the latest.
#[test]
fn edges_are_path_order_in_planned_order() {
    // session 0 writes kv/1 at 0; sessions 1 and 2 read it at 10 and 20; session 3 unlinks it at 30
    let body = json!([obj("f", 1),
        {"cond": {"if": {"eq": [{"index": "k"}, 0]}, "then": [
            {"open": {"file": r("f"), "flags": ["WRONLY", "CREAT"]}}, {"write": {"file": r("f"), "len": 4096}}, {"close": {"file": r("f")}}],
         "else": [{"cond": {"if": {"lt": [{"index": "k"}, 3]}, "then": [
            {"open": {"file": r("f"), "flags": ["RDONLY"]}}, {"read": {"file": r("f"), "len": 4096}}, {"close": {"file": r("f")}}],
         "else": [{"unlink": {"file": r("f")}}]}}]}}]);
    let recs = records(tl(json!([sessions(4, body)])));
    let ks: Vec<&str> = recs.iter().map(kind).collect();
    assert_eq!(ks, ["open", "write", "close", "open", "read", "close", "open", "read", "close", "unlink"]);
    let e: Vec<Vec<u64>> = recs.iter().map(|r| r.edges.clone()).collect();
    // the writer's own ops wait for nothing
    assert!(e[0].is_empty() && e[1].is_empty() && e[2].is_empty(), "{e:?}");
    // a reader's ops wait for the writable close, the path's last change
    for i in 3..9 {
        assert_eq!(e[i], [2], "op {i} {}", ks[i]);
    }
    // the unlink waits for the last change and each reader's last op
    assert_eq!(e[9], [2, 5, 8]);
}

/// A rename changes both its paths; an op after an unlink waits for the unlink.
#[test]
fn a_rename_waits_on_both_paths_and_an_unlink_is_a_change() {
    let body = json!([obj("f", 1), obj("g", 2),
        {"cond": {"if": {"eq": [{"index": "k"}, 0]}, "then": [stat_of("f")],
         "else": [{"cond": {"if": {"eq": [{"index": "k"}, 1]}, "then": [stat_of("g")],
         "else": [{"cond": {"if": {"eq": [{"index": "k"}, 2]}, "then": [{"rename": {"from": r("f"), "to": r("g")}}],
         "else": [{"cond": {"if": {"eq": [{"index": "k"}, 3]}, "then": [{"unlink": {"file": r("g")}}],
         "else": [{"open": {"file": r("g"), "flags": ["WRONLY", "CREAT"]}}]}}]}}]}}]}}]);
    let recs = records(tl(json!([sessions(5, body)])));
    let ks: Vec<&str> = recs.iter().map(kind).collect();
    assert_eq!(ks, ["stat", "stat", "rename", "unlink", "open"]);
    assert_eq!(recs[2].edges, [0, 1], "the rename waits for the reads of either path");
    assert_eq!(recs[3].edges, [2]);
    assert_eq!(recs[4].edges, [3], "a create after an unlink waits for it");
}

fn stat_of(name: &str) -> Value {
    json!({"stat": {"file": r(name)}})
}

/// A close, fsync, or fdatasync changes its path when the open was writable; of a read-only
/// open, it reads.
#[test]
fn a_close_changes_its_path_only_when_the_open_was_writable() {
    let body = json!([obj("f", 1),
        {"cond": {"if": {"eq": [{"index": "k"}, 0]}, "then": [
            {"open": {"file": r("f"), "flags": ["RDONLY"]}}, {"close": {"file": r("f")}}],
         "else": [{"open": {"file": r("f"), "flags": ["RDWR"]}}, {"fsync": {"file": r("f")}}, {"close": {"file": r("f")}}]}}]);
    let recs = records(tl(json!([sessions(3, body)])));
    let ks: Vec<&str> = recs.iter().map(kind).collect();
    assert_eq!(ks, ["open", "close", "open", "fsync", "close", "open", "fsync", "close"]);
    assert!(recs[2].edges.is_empty(), "an open RDWR without CREAT or TRUNC reads: {:?}", recs[2].edges);
    assert_eq!(recs[3].edges, [1], "the fsync of a writable open changes: it waits for the reader");
    assert_eq!(recs[5].edges, [4], "session 2's open reads: it waits for session 1's close");
    assert_eq!(recs[6].edges, [4], "and its fsync for the last change, which waited for session 0");
}

/// Submitted bodies run on the workers' lanes, submission i on lane i mod `lanes`: submissions
/// i and i + `lanes` share an executor lane, which issues them in order.
#[test]
fn submitted_bodies_share_the_workers_lanes() {
    let sub = |id: i64| json!({"submit": {"workers": "w", "body": [stat(json!(id))]}});
    let recs = records(tl(json!([sub(1), sub(2), sub(3), stat(json!(0))])));
    let got: Vec<&str> = recs.iter().map(|r| &*r.path).collect();
    assert_eq!(got, ["d/0", "d/1", "d/2", "d/3"]);
    let l: Vec<usize> = recs.iter().map(|r| r.lane).collect();
    assert_eq!(l[1], l[3], "submissions 0 and 2 on worker lane 0");
    assert_ne!(l[1], l[2]);
    assert!(l[0] != l[1] && l[0] != l[2]);
}

/// Victim i of an eviction batch runs on eviction lane i mod `lanes`, the same executor lane
/// from batch to batch.
#[test]
fn eviction_lanes_persist_across_batches() {
    let tiers = json!([{"name": "t", "order": [{"cached": "touched"}, {"neg": {"cached": "pos"}}], "capacity": 100,
        "on_evict": {"lanes": 2, "body": [stat(json!({"mod": [{"victim": {}}, 1000]}))]},
        "evict": {"every": 100, "when": true, "count": 3}}]);
    let body = json!([{"let": {"name": "a", "value": {"shared": {"id": 1, "n": 6}}}},
        {"store": {"cache": "lm", "tier": "t", "seq": r("a"), "per": 1, "from": 0, "count": 6}}, compute(json!(250))]);
    let ast = json!([{"timeline": {"name": "tl", "caches": [{"name": "lm", "tiers": tiers}], "body": body}}]);
    let recs = records(ast);
    assert_eq!(recs.len(), 6, "two batches of three");
    assert_eq!(recs.iter().map(|r| r.plan).collect::<Vec<_>>(), [100, 100, 100, 200, 200, 200]);
    let l: Vec<usize> = recs.iter().map(|r| r.lane).collect();
    // in a batch, lane 0's victims come first: victims 0 and 2, then 1
    assert_eq!(l[0], l[1]);
    assert_ne!(l[0], l[2]);
    assert_eq!(&l[..3], &l[3..], "the next batch on the same lanes");
}

/// Once the ops an entry names have completed, `retire` forgets it: a later op on the path
/// waits for nothing.
#[test]
fn retired_ops_are_no_longer_waited_on() {
    let m = model(tl(json!([sessions(2, json!([obj("f", 1), {"unlink": {"file": r("f")}}]))])));
    let mut p = Plan::new(m, "gpu", 0, 1);
    let mut ops = Vec::new();
    while let Some(x) = p.next().unwrap() {
        if let Planned::Op(r) = x {
            if ops.is_empty() {
                assert_eq!(p.paths(), 1);
                p.retire(|s| s == r.seq);
                assert_eq!(p.paths(), 0, "the path's only op completed");
            }
            ops.push(r);
        }
    }
    assert_eq!(ops.len(), 2);
    assert!(ops[1].edges.is_empty(), "{:?}", ops[1].edges);
    // not retired: the second unlink waits for the first
    let mut p = Plan::new(m, "gpu", 0, 1);
    let mut ops = Vec::new();
    while let Some(x) = p.next().unwrap() {
        if let Planned::Op(r) = x {
            p.retire(|_| false);
            ops.push(r);
        }
    }
    assert_eq!(ops[1].edges, [ops[0].seq]);
}

/// Ops outside a timeline are planned on a lane of their own, at the instance's clock.
#[test]
fn ops_outside_a_timeline_have_their_own_lane() {
    let recs = records(json!([stat(json!(7)), tl(json!([stat(json!(8))]))[0], stat(json!(9))]));
    let got: Vec<&str> = recs.iter().map(|r| &*r.path).collect();
    assert_eq!(got, ["d/7", "d/8", "d/9"]);
    assert_eq!(recs[0].lane, recs[2].lane);
    assert_ne!(recs[0].lane, recs[1].lane);
}

/// What the planner gives is what `dry-run` walks: the same ops in the same order, so the
/// same fingerprint (dry-run is driven by it).
#[derive(Default)]
struct Rec(Vec<String>, u64);

impl<'m, 'a> Sink<'m, 'a> for Rec {
    fn op(&mut self, op: &Op, ctx: &OpCtx) -> anyhow::Result<()> {
        self.0.push(format!("{} {}", op.kind.name(), op.path));
        self.1 = self.1.wrapping_add(op_hash(op, ctx));
        Ok(())
    }
    fn control(&mut self, _c: Control<'a>, _ctx: &OpCtx) -> anyhow::Result<()> {
        Ok(())
    }
}

#[test]
fn the_plan_is_the_dry_runs_walk() {
    let body = json!([obj("g", 9), stat_of("g"), {"timeline": {"name": "tl", "workers": [{"name": "w", "lanes": 2, "staging": 1}], "body": [
        sessions(3, json!([obj("f", 1), {"loop": {"index": "j", "to": 2, "body": [
            {"submit": {"workers": "w", "body": [stat(json!({"add": [{"index": "j"}, {"index": "k"}]}))]}},
            {"open": {"file": r("f"), "flags": ["WRONLY", "CREAT"]}}, {"write": {"file": r("f"), "len": 4096}}, {"close": {"file": r("f")}},
            compute(json!(3))]}}]))]}}]);
    let m = model(body);
    let mut want = Rec::default();
    for (name, count) in actor_counts(m).unwrap() {
        for a in 0..count {
            want = run_actor(m, name, a, count, want).unwrap();
        }
    }
    let mut p = Plan::new(m, "gpu", 0, 1);
    let (mut got, mut fp) = (Vec::new(), 0u64);
    while let Some(x) = p.next().unwrap() {
        if let Planned::Op(r) = x {
            got.push(format!("{} {}", kind(&r), r.path));
            fp = fp.wrapping_add(op_hash(&r.op(), &r.ctx()));
        }
    }
    assert_eq!(got, want.0);
    assert_eq!(fp, want.1);
    assert_eq!(got.len(), 1 + 3 * 2 * 4);
}

/// A dry run completes each op as it walks it, so its planner forgets its paths from time to
/// time: a timeline over more paths than it holds at once (65,536) walks the ops of the same
/// body outside a timeline.
#[test]
fn a_dry_run_over_many_paths_walks_every_op() {
    let n = 70_000;
    let lp = json!({"loop": {"index": "j", "to": n, "body": [
        {"let": {"name": "f", "value": {"object": {"namespace": "kv", "fields": {"key": {"index": "j"}}}}}}, stat_of("f")]}});
    let fingerprint = |body: Value| {
        let m = model(body);
        let r = run_actor(m, "gpu", 0, 1, Rec::default()).unwrap();
        (r.0.len(), r.1)
    };
    let (ops, fp) = fingerprint(tl(json!([lp.clone()])));
    assert_eq!(ops, n);
    assert_eq!((ops, fp), fingerprint(json!([lp])));
}

/// Outside a timeline a `parallel` is walked inline, as the dry run walks it; what waits across
/// actors is not planned.
#[test]
fn outside_a_timeline_forks_are_inline_and_barriers_are_not_planned() {
    let recs = records(json!([{"parallel": {"index": "k", "width": 2, "body": [stat(json!({"index": "k"}))]}}]));
    assert_eq!(recs.iter().map(|r| &*r.path).collect::<Vec<_>>(), ["d/0", "d/1"]);
    let m = model(json!([{"barrier": {"scope": "global"}}]));
    let e = Plan::new(m, "gpu", 0, 1).next().unwrap_err();
    assert!(format!("{e:#}").contains("a `barrier` is not planned"), "{e:#}");
}
