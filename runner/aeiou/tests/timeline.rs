//! Contract 0.15 (`DESIGN_REVIEW.md` §3.99, brief items 71 and 72): the `timeline` and its lanes
//! taken in planned order, `publish`, `published`, and `entry`, a `parallel` not joined, `prev`,
//! and `array`, as the dry run walks them.

use aeiou::eval::{build_model, Config, Params};
use aeiou::vm::{actor_counts, run_actor, Control, Op, OpCtx, Sink};
use serde_json::{json, Value};

/// The ops (by path) and the computes (`+ns`) in the order the walk gave them.
#[derive(Default)]
struct Rec(Vec<String>);

impl<'m, 'a> Sink<'m, 'a> for Rec {
    fn op(&mut self, op: &Op, _ctx: &OpCtx) -> anyhow::Result<()> {
        self.0.push(op.path.to_string());
        Ok(())
    }
    fn control(&mut self, c: Control<'a>, _ctx: &OpCtx) -> anyhow::Result<()> {
        if let Control::Compute { ns } = c {
            self.0.push(format!("+{ns}"));
        }
        Ok(())
    }
}

fn doc(params: Value, body: Value) -> String {
    json!({"ast": "0.18", "name": "t", "params": params,
           "datasets": {"d": {"layout": "whole", "access": "map", "pattern": "d/{id}", "count": 1000, "size": {"const": 1}, "seed": 1}},
           "actors": {"gpu": {"body": body}}}).to_string()
}

fn walk_with(text: &str, seed: u64) -> Result<Vec<String>, String> {
    let loaded = aeiou::load_str(text).map_err(|e| format!("{e:#}"))?;
    let cfg = Config { seed, gpus: 1, overrides: vec![], sets: vec![], st_blksize: Default::default() };
    let params = Params::new(&loaded.ast, &cfg).map_err(|e| format!("{e:#}"))?;
    let model = build_model(&loaded.ast, &cfg, &params).map_err(|e| format!("{e:#}"))?;
    let mut out = Vec::new();
    for (name, count) in actor_counts(&model).map_err(|e| format!("{e:#}"))? {
        for a in 0..count {
            out.extend(run_actor(&model, name, a, count, Rec::default()).map_err(|e| format!("{e:#}"))?.0);
        }
    }
    Ok(out)
}

fn walk(body: Value) -> Vec<String> {
    walk_with(&doc(json!({}), body), 1).unwrap_or_else(|e| panic!("{e}"))
}

fn fails(body: Value, needle: &str) {
    match walk_with(&doc(json!({}), body), 1) {
        Ok(ops) => panic!("walked, wanted `{needle}`: {ops:?}"),
        Err(e) => assert!(e.contains(needle), "wanted `{needle}` in:\n{e}"),
    }
}

/// The messages `aeiou check` gives a document.
fn refused(body: Value) -> Vec<String> {
    let ast = aeiou::ast::parse(&doc(json!({}), body)).expect("parses");
    aeiou::validate::check(&ast)
}

fn stat(id: Value) -> Value {
    json!({"stat": {"file": {"file": {"dataset": "d", "id": id}}}})
}

fn compute(ns: Value) -> Value {
    json!({"compute": {"ns": ns}})
}

fn tl(body: Value) -> Value {
    json!({"timeline": {"name": "tl", "body": body}})
}

// ---------------------------------------------------------------- planned order

/// A lane runs until its next `compute`, which moves its planned clock; the lane planned
/// soonest goes next. A `parallel` not joined lets its parent go on at once.
#[test]
fn lanes_take_their_steps_in_planned_order() {
    let body = json!([tl(json!([
        {"parallel": {"index": "k", "width": 2, "join": false, "body": [
            compute(json!({"add": [{"mul": [{"index": "k"}, 10]}, 5]})), stat(json!({"add": [50, {"index": "k"}]}))]}},
        {"loop": {"index": "i", "to": 3, "body": [stat(json!({"index": "i"})), compute(json!(10))]}}]))]);
    // main at 0, 10, 20; sub-agent 0 at 5, sub-agent 1 at 15
    assert_eq!(walk(body), ["d/0", "+10", "+5", "+15", "d/50", "d/1", "+10", "d/51", "d/2", "+10"]);
}

/// A joined `parallel` holds its parent until the last of its sub-actors has ended, and the
/// parent goes on at that sub-actor's planned end, beside a lane that was not joined.
#[test]
fn a_joined_parallel_holds_its_parent_until_its_last_sub_actor_ends() {
    let body = json!([tl(json!([
        {"parallel": {"index": "w", "width": 1, "join": false, "body": [compute(json!(12)), stat(json!(90)), compute(json!(10)), stat(json!(91))]}},
        {"parallel": {"index": "k", "width": 2, "body": [
            compute(json!({"add": [{"mul": [{"index": "k"}, 10]}, 5]})), stat(json!({"add": [50, {"index": "k"}]}))]}},
        stat(json!(0))]))]);
    assert_eq!(walk(body), ["+12", "+5", "+15", "d/50", "d/90", "+10", "d/51", "d/0", "d/91"]);
}

/// Two lanes planned for the same time go in the order of their lane keys (the parent before
/// its sub-actors, a fork's sub-actors by index, an earlier fork's before a later one's),
/// never in the order they were put back.
#[test]
fn a_tie_is_broken_by_the_lane_key_not_by_insertion() {
    let body = json!([tl(json!([
        {"parallel": {"index": "c", "width": 1, "join": false, "body": [compute(json!(5)), stat(json!(2))]}},
        compute(json!(2)), compute(json!(3)), stat(json!(1))]))]);
    // the sub-actor was put back for 5 before main was; main's key is the smaller
    assert_eq!(walk(body), ["+2", "+5", "+3", "d/1", "d/2"]);
}

/// A lane's draws are keyed on its position as anywhere else, so another lane beside it
/// changes none of them.
#[test]
fn a_lanes_draws_do_not_depend_on_the_lanes_beside_it() {
    let main = json!({"loop": {"index": "i", "to": 20, "body": [
        {"let": {"name": "x", "value": {"draw": {"uniform": {"lo": 0, "hi": 500}}}}},
        stat(json!({"ref": "x"})), compute(json!(7))]}});
    // a draw's site is its node's path, so the lane beside stays where it is, of width 0 or 3
    let beside = |w: i64| json!({"parallel": {"index": "o", "width": w, "join": false, "body": [
        {"loop": {"index": "j", "to": 5, "body": [
            {"let": {"name": "y", "value": {"draw": {"uniform": {"lo": 500, "hi": 1000}}}}},
            stat(json!({"ref": "y"})), compute(json!(11))]}}]}});
    let alone: Vec<String> = walk(json!([tl(json!([beside(0), main.clone()]))])).into_iter().filter(|s| s.starts_with("d/")).collect();
    let with: Vec<String> = walk(json!([tl(json!([beside(3), main]))])).into_iter().filter(|s| s.starts_with("d/") && s[2..].parse::<i64>().unwrap() < 500).collect();
    assert_eq!(alone.len(), 20);
    assert_eq!(alone, with);
}

// ---------------------------------------------------------------- publish, published, entry

/// `published` counts the entries planned before the reading lane's step; `entry` reads one
/// of them by its place in planned order.
#[test]
fn a_lane_reads_what_the_others_published_before_its_planned_time() {
    let body = json!([tl(json!([
        {"parallel": {"index": "k", "width": 1, "join": false, "body": [
            compute(json!(15)),
            stat(json!({"published": "tl"})),
            stat(json!({"entry": {"timeline": "tl", "index": 1, "key": "v"}}))]}},
        {"loop": {"index": "i", "to": 3, "body": [
            {"publish": {"timeline": "tl", "values": {"v": {"mul": [{"index": "i"}, 7]}}}},
            compute(json!(10))]}}]))]);
    assert_eq!(walk(body), ["+10", "+15", "+10", "d/2", "d/7", "+10"]);
}

#[test]
fn an_entry_not_yet_published_or_a_key_it_lacks_is_refused_when_walked() {
    fails(json!([tl(json!([stat(json!({"entry": {"timeline": "tl", "index": 0, "key": "v"}})), {"publish": {"timeline": "tl", "values": {"v": 1}}}]))]),
          "entry 0 of timeline `tl`: 0 published");
    fails(json!([tl(json!([{"publish": {"timeline": "tl", "values": {"v": 1}}},
                           stat(json!({"entry": {"timeline": "tl", "index": 0, "key": "w"}})), {"publish": {"timeline": "tl", "values": {"w": 2}}}]))]),
          "entry 0 of timeline `tl` has no `w`");
}

/// Entries in groups (contract 0.17): `publish` puts an entry in its `group` (0 when none is
/// given), and `published` and `entry` count and index within one group, so a session's pool
/// is its own in a timeline every session of an engine shares.
#[test]
fn entries_in_groups_are_counted_and_indexed_within_their_group() {
    let put = |g: Option<i64>, v: i64| match g {
        Some(g) => json!({"publish": {"timeline": "tl", "group": g, "values": {"v": v}}}),
        None => json!({"publish": {"timeline": "tl", "values": {"v": v}}}),
    };
    let body = json!([tl(json!([
        put(None, 1), put(Some(1), 2), put(Some(1), 3), put(Some(0), 4),
        stat(json!({"published": "tl"})),
        stat(json!({"published": {"timeline": "tl", "group": 1}})),
        stat(json!({"published": {"timeline": "tl", "group": {"add": [6, 1]}}})),
        stat(json!({"entry": {"timeline": "tl", "group": 1, "index": 1, "key": "v"}})),
        stat(json!({"entry": {"timeline": "tl", "index": 1, "key": "v"}})),
        stat(json!({"entry": {"timeline": "tl", "group": 0, "index": 0, "key": "v"}}))]))]);
    assert_eq!(walk(body), ["d/2", "d/2", "d/0", "d/3", "d/4", "d/1"]);
    fails(json!([tl(json!([put(Some(1), 2), stat(json!({"entry": {"timeline": "tl", "group": 1, "index": 1, "key": "v"}})), put(Some(1), 3)]))]),
          "entry 1 of group 1 of timeline `tl`: 1 published");
    // a group is an integer
    let e = refused(json!([tl(json!([{"publish": {"timeline": "tl", "group": "s", "values": {"v": 1}}}]))]));
    assert!(e.iter().any(|m| m.contains("expected an integer, got a string (V27)")), "{e:#?}");
}

/// The timeline ends when its last lane does; the actor goes on after it, and a second
/// timeline starts with nothing published.
#[test]
fn a_timeline_ends_with_its_last_lane_and_the_next_starts_empty() {
    let pub_one = json!({"publish": {"timeline": "tl", "values": {"v": 3}}});
    let body = json!([
        tl(json!([pub_one, {"parallel": {"index": "k", "width": 1, "join": false, "body": [compute(json!(9)), stat(json!(9))]}}])),
        stat(json!(100)),
        tl(json!([stat(json!({"published": "tl"}))]))]);
    assert_eq!(walk(body), ["+9", "d/9", "d/100", "d/0"]);
}

#[test]
fn what_a_timeline_may_not_hold_is_refused_when_walked() {
    fails(json!([tl(json!([{"barrier": {"scope": "global"}}]))]), "inside a timeline");
    fails(json!([tl(json!([{"loader": {"name": "q", "index": "b", "workers": 1, "prefetch": 1, "batches": 1, "body": []}}]))]), "inside a timeline");
}

// ---------------------------------------------------------------- prev

/// `prev(x, init)`: the value `x` took in the previous iteration of its loop, `init` in the
/// first; the same as `x @ (i - 1)` for a positional `x`, read rather than recomputed.
#[test]
fn prev_is_the_value_of_the_previous_iteration() {
    let prev = |r: &str, init: Value| json!({"prev": {"ref": r, "init": init}});
    let body = json!([{"loop": {"index": "o", "to": 2, "body": [{"loop": {"index": "i", "to": 4, "body": [
        {"let": {"name": "y", "value": {"mul": [{"index": "i"}, 2]}}},
        {"let": {"name": "z", "value": prev("y", json!(50))}},
        {"let": {"name": "x", "value": {"add": [prev("x", json!({"mul": [{"index": "o"}, 100]})), {"index": "i"}]}}},
        {"let": {"name": "a", "value": {"cond": {"if": {"gt": [{"index": "i"}, 0]}, "then": {"at": {"ref": "y", "index": {"sub": [{"index": "i"}, 1]}}}, "else": 50}}}},
        stat(json!({"ref": "x"})), stat(json!({"ref": "z"})), stat(json!({"ref": "a"}))]}}]}}]);
    let ops = walk(body);
    let (x, z, a): (Vec<_>, Vec<_>, Vec<_>) = (ops.iter().step_by(3).cloned().collect(), ops.iter().skip(1).step_by(3).cloned().collect(), ops.iter().skip(2).step_by(3).cloned().collect());
    assert_eq!(x, ["d/0", "d/1", "d/3", "d/6", "d/100", "d/101", "d/103", "d/106"]);
    assert_eq!(z, a, "prev(y) is y @ (i - 1) for a positional y");
    assert_eq!(&z[..4], ["d/50", "d/0", "d/2", "d/4"]);
}

/// `prev` carries a lane's state that depends on what others published, which `@` could
/// not: `@` reaching a timeline read or a `prev` is refused when walked.
#[test]
fn prev_carries_state_from_the_timeline_and_at_may_not_reach_it() {
    let body = json!([tl(json!([
        {"parallel": {"index": "k", "width": 1, "join": false, "body": [
            {"loop": {"index": "j", "to": 3, "body": [{"publish": {"timeline": "tl", "values": {"v": 1}}}, compute(json!(10))]}}]}},
        {"loop": {"index": "i", "to": 3, "body": [
            compute(json!(5)),
            {"let": {"name": "n", "value": {"published": "tl"}}},
            {"let": {"name": "s", "value": {"add": [{"prev": {"ref": "s", "init": 0}}, {"ref": "n"}]}}},
            stat(json!({"ref": "s"}))]}}]))]);
    // the sub-agent publishes at 0, 10, 20; main reads at 5, 10, 15: 1, 1 (a tie: main's key first), 2
    assert_eq!(walk(body).into_iter().filter(|s| s.starts_with("d/")).collect::<Vec<_>>(), ["d/1", "d/2", "d/4"]);
    fails(json!([{"loop": {"index": "i", "to": 3, "body": [
        {"let": {"name": "a", "value": {"add": [{"prev": {"ref": "a", "init": 0}}, 1]}}},
        {"let": {"name": "b", "value": {"add": [{"ref": "a"}, 1]}}},
        {"let": {"name": "c", "value": {"cond": {"if": {"gt": [{"index": "i"}, 0]}, "then": {"at": {"ref": "b", "index": {"sub": [{"index": "i"}, 1]}}}, "else": 0}}}},
        stat(json!({"ref": "c"}))]}}]), "`prev` under `@`");
}

// ---------------------------------------------------------------- array

/// An `array`'s values are computed once, each with its index bound as a loop's, so its
/// draws are positional; `elem` and `len` read it as they read a parameter array.
#[test]
fn an_array_is_drawn_once_by_position_and_read_by_elem_and_len() {
    let body = json!([{"array": {"name": "bs", "count": 4, "index": "b", "value": {"draw": {"uniform": {"lo": 0, "hi": 1000}}}}},
        {"loop": {"index": "k", "to": {"len": "bs"}, "body": [stat(json!({"elem": {"array": "bs", "index": {"index": "k"}}}))]}},
        stat(json!({"elem": {"array": "bs", "index": 2}}))]);
    let text = doc(json!({}), body);
    let a = walk_with(&text, 1).unwrap();
    assert_eq!(a.len(), 5);
    assert_eq!(a[4], a[2], "computed once");
    assert_eq!(walk_with(&text, 1).unwrap(), a, "the same seed");
    assert_ne!(walk_with(&text, 2).unwrap(), a, "another seed");
    let distinct: std::collections::HashSet<_> = a[..4].iter().collect();
    assert_eq!(distinct.len(), 4, "each element at its own position: {a:?}");
    fails(json!([{"array": {"name": "bs", "count": 4, "index": "b", "value": 1}}, stat(json!({"elem": {"array": "bs", "index": 4}}))]),
          "elem: index 4 outside array `bs` of 4");
}

// ---------------------------------------------------------------- the checkers (V30 to V32)

#[test]
fn the_validator_holds_timelines_prev_and_arrays_to_their_rules() {
    let has = |body: Value, needle: &str| {
        let e = refused(body.clone());
        assert!(e.iter().any(|m| m.contains(needle)), "wanted `{needle}` for {body}:\n{e:#?}");
    };
    has(json!([{"publish": {"timeline": "tl", "values": {"v": 1}}}]), "(V30)");
    has(json!([stat(json!({"published": "tl"}))]), "(V30)");
    has(json!([tl(json!([tl(json!([]))]))]), "(V30)");
    has(json!([{"parallel": {"index": "k", "width": 1, "join": false, "body": []}}]), "(V30)");
    has(json!([tl(json!([{"channel": {"name": "q", "capacity": 1}}]))]), "(V30)");
    has(json!([tl(json!([{"publish": {"timeline": "tl", "values": {"v": 1}}}, stat(json!({"entry": {"timeline": "tl", "index": 0, "key": "w"}}))]))]), "(V30)");
    // `@` naming a definition that reads the timeline or `prev`
    has(json!([tl(json!([{"loop": {"index": "i", "to": 2, "body": [
        {"let": {"name": "n", "value": {"published": "tl"}}},
        {"let": {"name": "m", "value": {"cond": {"if": {"gt": [{"index": "i"}, 0]}, "then": {"at": {"ref": "n", "index": {"sub": [{"index": "i"}, 1]}}}, "else": 0}}}}]}}]))]), "(V30)");
    // prev outside a loop, of an unknown name, of a let not directly in its loop's body
    has(json!([{"let": {"name": "x", "value": {"prev": {"ref": "x", "init": 0}}}}]), "(V31)");
    has(json!([{"loop": {"index": "i", "to": 2, "body": [{"let": {"name": "x", "value": {"prev": {"ref": "q", "init": 0}}}}]}}]), "(V31)");
    has(json!([{"loop": {"index": "i", "to": 2, "body": [{"cond": {"if": true, "then": [{"let": {"name": "x", "value": {"prev": {"ref": "x", "init": 0}}}}]}}]}}]), "(V31)");
    // an array inside a loop, named as a parameter, or reading an unknown array
    has(json!([{"loop": {"index": "i", "to": 2, "body": [{"array": {"name": "bs", "count": 1, "index": "b", "value": 1}}]}}]), "(V32)");
    has(json!([stat(json!({"elem": {"array": "nope", "index": 0}}))]), "nope");
    let ok = json!([tl(json!([
        {"array": {"name": "bs", "count": 2, "index": "b", "value": {"index": "b"}}},
        {"parallel": {"index": "k", "width": 2, "join": false, "body": [{"publish": {"timeline": "tl", "values": {"v": {"elem": {"array": "bs", "index": {"index": "k"}}}}}}]}},
        {"loop": {"index": "i", "to": 2, "body": [
            {"let": {"name": "x", "value": {"add": [{"prev": {"ref": "x", "init": 0}}, {"published": "tl"}]}}},
            stat(json!({"entry": {"timeline": "tl", "index": 0, "key": "v"}}))]}}]))]);
    assert_eq!(refused(ok), Vec::<String>::new());
}

// ---------------------------------------------------------------- the run side, not built

/// `aeiou run` refuses a timeline until issuing its lanes' ops at their planned times is built
/// (brief item 72 (7)); `check` and `dry-run` walk it.
#[test]
fn run_refuses_a_timeline_that_check_and_dry_run_walk() {
    let dir = crate::common::tmpdir("timeline", "run");
    let ast = dir.join("t.ast.json");
    std::fs::write(&ast, doc(json!({}), json!([tl(json!([stat(json!(1))]))]))).unwrap();
    let aeiou = env!("CARGO_BIN_EXE_aeiou");
    let out = |args: &[&str]| std::process::Command::new(aeiou).args(args).output().unwrap();
    let a = ast.to_str().unwrap();
    assert!(out(&["check", a]).status.success());
    assert!(out(&["dry-run", a, "--gpus", "1"]).status.success());
    let r = out(&["run", a, "--gpus", "1", "--root", dir.to_str().unwrap()]);
    let err = String::from_utf8_lossy(&r.stderr);
    assert!(!r.status.success() && err.contains("a `timeline` is not run yet"), "{err}");
}
