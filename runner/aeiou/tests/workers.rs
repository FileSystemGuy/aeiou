//! Contract 0.20 (`DESIGN_REVIEW.md` §3.99, the note "The run side of timelines"; brief item 72
//! (3) and (7)): a timeline's `workers {name, lanes, staging}`, `submit` (a body run on a worker
//! lane at the submitting lane's planned instant), and `span` (a measured part of a lane), as the
//! plan walks them. What the run does with them (a worker lane's lag, staging, D) is the run
//! side's; the plan's op set does not move.

use aeiou::eval::{build_model, Config, Params};
use aeiou::vm::{actor_counts, op_hash, run_actor, Control, Op, OpCtx, Sink};
use serde_json::{json, Value};

/// The ops (by path) and the computes (`+ns`) in the order the walk gave them, and the sum of
/// the ops' hashes (the fingerprint).
#[derive(Default)]
struct Rec(Vec<String>, u64);

impl<'m, 'a> Sink<'m, 'a> for Rec {
    fn op(&mut self, op: &Op, ctx: &OpCtx) -> anyhow::Result<()> {
        self.0.push(op.path.to_string());
        self.1 = self.1.wrapping_add(op_hash(op, ctx));
        Ok(())
    }
    fn control(&mut self, c: Control<'a>, _ctx: &OpCtx) -> anyhow::Result<()> {
        if let Control::Compute { ns } = c {
            self.0.push(format!("+{ns}"));
        }
        Ok(())
    }
}

fn doc(body: Value) -> String {
    json!({"ast": "0.20", "name": "t", "params": {"n": {"default": 2}},
           "datasets": {"d": {"layout": "whole", "access": "map", "pattern": "d/{id}", "count": 1000, "size": {"const": 1}, "seed": 1}},
           "actors": {"gpu": {"body": body}}}).to_string()
}

fn walk_doc(text: &str, check: bool) -> Result<(Vec<String>, u64), String> {
    let cfg = Config { seed: 1, gpus: 1, overrides: vec![], sets: vec![], st_blksize: Default::default() };
    let ast = if check {
        aeiou::load_str(text).map_err(|e| format!("{e:#}"))?.ast
    } else {
        let ast = aeiou::ast::parse(text).map_err(|e| format!("{e:#}"))?;
        aeiou::sites::annotate(&ast);
        ast
    };
    let params = Params::new(&ast, &cfg).map_err(|e| format!("{e:#}"))?;
    let model = build_model(&ast, &cfg, &params).map_err(|e| format!("{e:#}"))?;
    let (mut out, mut fp) = (Vec::new(), 0u64);
    for (name, count) in actor_counts(&model).map_err(|e| format!("{e:#}"))? {
        for a in 0..count {
            let r = run_actor(&model, name, a, count, Rec::default()).map_err(|e| format!("{e:#}"))?;
            out.extend(r.0);
            fp = fp.wrapping_add(r.1);
        }
    }
    Ok((out, fp))
}

fn walk(body: Value) -> (Vec<String>, u64) {
    walk_doc(&doc(body), true).unwrap_or_else(|e| panic!("{e}"))
}

fn walk_fails(body: Value, needle: &str) {
    match walk_doc(&doc(body), false) {
        Ok(ops) => panic!("walked, wanted `{needle}`: {ops:?}"),
        Err(e) => assert!(e.contains(needle), "wanted `{needle}` in:\n{e}"),
    }
}

fn refused(body: Value) -> Vec<String> {
    let ast = aeiou::ast::parse(&doc(body)).expect("parses");
    aeiou::validate::check(&ast)
}

fn stat(id: Value) -> Value {
    json!({"stat": {"file": {"file": {"dataset": "d", "id": id}}}})
}

fn compute(ns: i64) -> Value {
    json!({"compute": {"ns": ns}})
}

fn now() -> Value {
    json!({"now": {}})
}

fn submit(body: Value) -> Value {
    json!({"submit": {"workers": "w", "body": body}})
}

fn span(name: &str, body: Value) -> Value {
    json!({"span": {"name": name, "body": body}})
}

/// A timeline with workers `w` (`lanes`, `staging`) around `body`.
fn tl_w(lanes: Value, staging: Value, body: Value) -> Value {
    json!([{"timeline": {"name": "tl", "workers": [{"name": "w", "lanes": lanes, "staging": staging}], "body": body}}])
}

fn tl(body: Value) -> Value {
    tl_w(json!(2), json!(1048576), body)
}

// ---------------------------------------------------------------- the plan

/// A submitted body issues the ops it would have issued where it stands, with the same
/// enclosing indices, so the op multiset and the fingerprint are those of the body written
/// inline: only which lane issues them, and so when, differs.
#[test]
fn a_submitted_body_issues_the_ops_written_inline() {
    let lp = |inner: Value| json!([{"loop": {"index": "j", "to": 3, "body": [inner]}}]);
    let (mut a, fa) = walk(tl(lp(submit(json!([stat(json!({"index": "j"}))])))));
    let (mut b, fb) = walk(tl(lp(stat(json!({"index": "j"})))));
    assert_eq!(fa, fb, "the fingerprint moved");
    a.sort();
    b.sort();
    assert_eq!(a, b);
    assert_eq!(a, ["d/0", "d/1", "d/2"]);
}

/// A submitted body reads `prev` as the lane would where it stands: the value the binding took in
/// the loop's previous iteration, not the `init` of a walk with no history.
#[test]
fn a_submitted_body_reads_prev_as_written_inline() {
    let body = |inner: Value| tl(json!([{"loop": {"index": "i", "to": 3, "body": [
        {"let": {"name": "x", "value": {"mul": [{"index": "i"}, 10]}}},
        {"let": {"name": "p", "value": {"prev": {"ref": "x", "init": 99}}}},
        inner,
    ]}}]));
    let (a, fa) = walk(body(submit(json!([stat(json!({"prev": {"ref": "x", "init": 99}}))]))));
    let (b, fb) = walk(body(stat(json!({"prev": {"ref": "x", "init": 99}}))));
    assert_eq!(b, ["d/99", "d/0", "d/10"]);
    assert_eq!(a, b);
    assert_eq!(fa, fb);
}

/// A submitted body runs at the submitting lane's planned instant, after the lane's own step
/// to its next `compute`, and sees the lane's bindings where it stood.
#[test]
fn a_submitted_body_runs_at_the_submitters_planned_instant() {
    let (ops, _) = walk(tl(json!([
        compute(5),
        {"let": {"name": "x", "value": 7}},
        submit(json!([stat(now()), stat(json!({"ref": "x"}))])),
        stat(json!(100)),
        compute(10),
        stat(json!(200)),
    ])));
    assert_eq!(ops, ["+5", "d/100", "+10", "d/5", "d/7", "d/200"]);
}

/// Submissions at one instant run in the order submitted, and a worker lane goes after the
/// timeline's other lanes planned for the same instant.
#[test]
fn workers_go_in_submission_order_after_the_other_lanes_at_an_instant() {
    let (ops, _) = walk(tl(json!([
        submit(json!([stat(json!(1))])),
        submit(json!([stat(json!(2))])),
        submit(json!([stat(json!(3))])),
        {"parallel": {"index": "k", "width": 2, "join": false, "body": [stat(json!({"add": [10, {"index": "k"}]}))]}},
        stat(json!(0)),
    ])));
    assert_eq!(ops, ["d/0", "d/10", "d/11", "d/1", "d/2", "d/3"]);
}

/// A span is a measured part of a lane: in the plan it changes nothing.
#[test]
fn a_span_changes_nothing_in_the_plan() {
    let reads = json!([stat(json!(1)), {"let": {"name": "y", "value": 2}}, stat(json!({"ref": "y"}))]);
    let (a, fa) = walk(tl(json!([compute(3), span("load", reads.clone())])));
    let (b, fb) = walk(tl(json!([compute(3), reads[0], reads[1], reads[2]])));
    assert_eq!(a, b);
    assert_eq!(fa, fb);
    assert_eq!(a, ["+3", "d/1", "d/2"]);
}

#[test]
fn workers_used_wrongly_are_refused_when_walked() {
    walk_fails(tl_w(json!(0), json!(1), json!([])), "workers `w`: lanes 0, not positive");
    walk_fails(tl_w(json!(2), json!(0), json!([])), "workers `w`: staging 0, not positive");
    walk_fails(json!([submit(json!([]))]), "submit: outside a timeline");
    walk_fails(tl(json!([{"submit": {"workers": "x", "body": []}}])), "submit: timeline `tl` declares no workers `x` (V36)");
}

// ---------------------------------------------------------------- rule V36

fn has(errs: &[String], needle: &str) {
    assert!(errs.iter().any(|e| e.contains(needle)), "wanted `{needle}` in:\n{}", errs.join("\n"));
}

#[test]
fn the_validator_holds_workers_submit_and_span_to_their_rules() {
    // clean
    let ok = refused(tl(json!([span("load", json!([stat(json!(1))])), submit(json!([stat(json!(2))]))])));
    assert!(ok.is_empty(), "{ok:?}");
    let ok = refused(tl_w(json!({"param": "n"}), json!({"mul": [{"param": "n"}, 1024]}), json!([submit(json!([stat(now())]))])));
    assert!(ok.is_empty(), "{ok:?}");

    // workers
    has(&refused(json!([{"timeline": {"name": "tl", "workers": [{"name": "w", "lanes": 1, "staging": 1}, {"name": "w", "lanes": 1, "staging": 1}], "body": []}}])),
        "workers `w` declared twice (V36)");
    has(&refused(json!([{"timeline": {"name": "tl", "workers": [{"name": "w-1", "lanes": 1, "staging": 1}], "body": []}}])), "not an identifier");
    has(&refused(json!([{"let": {"name": "z", "value": 3}}, tl_w(json!({"ref": "z"}), json!(1), json!([]))[0].clone()])),
        "workers `w`: its `lanes` reads `ref`; it reads only parameters and literals (V36)");
    has(&refused(tl_w(json!(1), json!({"now": {}}), json!([]))), "workers `w`: its `staging` reads `now`; it reads only parameters and literals (V36)");

    // submit
    has(&refused(json!([submit(json!([]))])), "submit: outside a timeline, where nothing is planned (V36)");
    has(&refused(tl(json!([{"submit": {"workers": "x", "body": []}}]))), "submit: timeline `tl` declares no workers `x` (V36)");
    for (n, what) in [
        (compute(1), "compute"),
        (submit(json!([])), "submit"),
        (span("s", json!([])), "span"),
        (json!({"parallel": {"index": "k", "width": 1, "body": []}}), "parallel"),
        (json!({"publish": {"timeline": "tl", "values": {"a": 1}}}), "publish"),
        (json!({"acquire": {"resource": "r", "n": 1, "hold": "h"}}), "acquire"),
    ] {
        has(&refused(tl(json!([submit(json!([{"loop": {"index": "i", "to": 1, "body": [n]}}]))]))),
            &format!("a `{what}` in a `submit` body, which only issues its ops on a worker lane (V36)"));
    }
    // an eviction body issues its victim's ops, and submits nothing
    let ev = json!([{"timeline": {"name": "tl", "workers": [{"name": "w", "lanes": 1, "staging": 1}],
        "caches": [{"name": "c", "tiers": [{"name": "t", "capacity": 1, "order": [{"cached": "stored"}],
            "on_evict": {"lanes": 1, "body": [submit(json!([]))]}}]}], "body": []}}]);
    has(&refused(ev), "a `submit` in an `on_evict` body, which only issues the victim's ops at its tick (V35)");

    // span
    has(&refused(json!([span("load", json!([]))])), "span: outside a timeline, where no request is measured (V36)");
    has(&refused(tl(json!([span("a b", json!([]))]))), "not an identifier");
    has(&refused(tl(json!([span("a", json!([span("b", json!([]))]))]))), "a span inside span `a`: a lane's wait is measured once (V36)");
    for (n, what) in [
        (compute(1), "compute"),
        (json!({"parallel": {"index": "k", "width": 1, "body": []}}), "parallel"),
        (json!({"acquire": {"resource": "r", "n": 1, "hold": "h"}}), "acquire"),
    ] {
        has(&refused(tl(json!([span("load", json!([{"cond": {"if": true, "then": [n]}}]))]))),
            &format!("a `{what}` in span `load`: a span measures storage waits, never planned time (V36)"));
    }
    has(&refused(tl(json!([submit(json!([span("s", json!([]))]))]))), "a `span` in a `submit` body");
}
