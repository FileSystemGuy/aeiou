//! Contract 0.19 (`DESIGN_REVIEW.md` §3.99, the note "Contract 0.19 proposed" and the user's
//! answers of 2026-11-13): a tier's timed eviction (`evict {every, when, count}`, `tier`), the
//! eviction body (`on_evict {lanes, body}`, `victim`), and `probe`, `touch`, and `store` over a
//! range of a `seq`'s chunks.

use aeiou::eval::{build_model, Config, Params};
use aeiou::vm::{actor_counts, run_actor, Control, Op, OpCtx, Sink};
use serde_json::{json, Value};

#[derive(Default)]
struct Rec(Vec<String>);

impl<'m, 'a> Sink<'m, 'a> for Rec {
    fn op(&mut self, op: &Op, _ctx: &OpCtx) -> anyhow::Result<()> {
        self.0.push(op.path.to_string());
        Ok(())
    }
    fn control(&mut self, _c: Control<'a>, _ctx: &OpCtx) -> anyhow::Result<()> {
        Ok(())
    }
}

fn doc(body: Value) -> String {
    json!({"ast": "0.23", "name": "t", "params": {"cap": {"default": 10}},
           "datasets": {"d": {"layout": "whole", "access": "map", "pattern": "d/{id}", "count": 1000000, "size": {"const": 1}, "seed": 1}},
           "actors": {"gpu": {"body": body}}}).to_string()
}

fn walk_doc(text: &str, check: bool) -> Result<Vec<String>, String> {
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
    let mut out = Vec::new();
    for (name, count) in actor_counts(&model).map_err(|e| format!("{e:#}"))? {
        for a in 0..count {
            out.extend(run_actor(&model, name, a, count, Rec::default()).map_err(|e| format!("{e:#}"))?.0);
        }
    }
    Ok(out)
}

fn walk(body: Value) -> Vec<String> {
    walk_doc(&doc(body), true).unwrap_or_else(|e| panic!("{e}"))
}

fn fails(body: Value, needle: &str) {
    match walk_doc(&doc(body), true) {
        Ok(ops) => panic!("walked, wanted `{needle}`: {ops:?}"),
        Err(e) => assert!(e.contains(needle), "wanted `{needle}` in:\n{e}"),
    }
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

fn r(name: &str) -> Value {
    json!({"ref": name})
}

fn let_(name: &str, value: Value) -> Value {
    json!({"let": {"name": name, "value": value}})
}

fn compute(ns: i64) -> Value {
    json!({"compute": {"ns": ns}})
}

fn cached(a: &str) -> Value {
    json!({"cached": a})
}

fn shared(id: i64, n: i64) -> Value {
    json!({"shared": {"id": id, "n": n}})
}

fn probe(s: Value, from: i64, bind: &str) -> Value {
    json!({"probe": {"cache": "c", "tier": "t", "keys": crate::common::chunk_keys(s, json!(1), json!(from), None), "bind": bind}})
}

fn touch(s: Value, from: i64, count: i64) -> Value {
    json!({"touch": {"cache": "c", "tier": "t", "keys": crate::common::chunk_keys(s, json!(1), json!(from), Some(json!(count)))}})
}

fn store(s: Value, from: i64, count: i64) -> Value {
    json!({"store": {"cache": "c", "tier": "t", "keys": crate::common::chunk_keys(s, json!(1), json!(from), Some(json!(count)))}})
}

fn store_bind(s: Value, from: i64, count: i64, bind: &str) -> Value {
    json!({"store": {"cache": "c", "tier": "t", "keys": crate::common::chunk_keys(s, json!(1), json!(from), Some(json!(count))), "bind": bind}})
}

/// The key of chunk `j` of `s`, one block a chunk, folded into the dataset's ids.
fn key_id(s: Value, j: i64) -> Value {
    json!({"mod": [{"chunk_key": {"seq": s, "chunk": j, "per": 1}}, 1000000]})
}

fn victim_id() -> Value {
    json!({"mod": [{"victim": {}}, 1000000]})
}

/// LMCache 0.5.5's rule: every `every` ns, at 0.8 of the capacity, 20 % of the keys, at least 1.
fn lmcache_evict(every: i64) -> Value {
    json!({"every": every,
           "when": {"ge": [{"mul": [{"tier": "used"}, 10]}, {"mul": [{"param": "cap"}, 8]}]},
           "count": {"max": [1, {"div": [{"mul": [{"tier": "entries"}, 2]}, 10]}]}})
}

/// A tier in LMCache's order (least recently touched, a batch's tail first) with timed eviction.
fn timed(extra: Value) -> Value {
    let mut t = json!({"name": "t", "order": [cached("touched"), {"neg": cached("pos")}], "capacity": {"param": "cap"},
                       "evict": lmcache_evict(100)});
    for (k, v) in extra.as_object().unwrap() {
        t[k] = v.clone();
    }
    json!([t])
}

fn tl(tiers: Value, body: Value) -> Value {
    json!({"timeline": {"name": "tl", "caches": [{"name": "c", "tiers": tiers}], "body": body}})
}

// ---------------------------------------------------------------- probe, touch, store

/// `store` stores the chunks of a range the tier lacks, one event for the whole statement (so
/// `pos` orders within it); `probe` reads the leading run from `from` and changes nothing;
/// `touch` makes the chunks it finds recent.
#[test]
fn probe_touch_and_store_work_over_a_range() {
    let lru = json!([{"name": "t", "order": [cached("touched"), {"neg": cached("pos")}], "capacity": 4}]);
    // A's four chunks fill the tier; B's one evicts A's tail, a batch's last chunk least
    let body = json!([tl(lru.clone(), json!([
        let_("a", shared(1, 4)), store(r("a"), 0, 4), store(shared(2, 1), 0, 1),
        probe(r("a"), 0, "h"), stat(r("h")),
        probe(r("a"), 2, "h2"), stat(r("h2"))]))]);
    assert_eq!(walk(body), ["d/3", "d/1"]);
    // touched chunks outlive the rest: A's first two touched after B's store; C's two evict A's
    // third (stored with A) and then B
    let body = json!([tl(lru.clone(), json!([
        let_("a", shared(1, 3)), let_("b", shared(2, 1)),
        store(r("a"), 0, 3), store(r("b"), 0, 1), touch(r("a"), 0, 2), store(shared(3, 2), 0, 2),
        probe(r("a"), 0, "ha"), probe(r("b"), 0, "hb"), stat(r("ha")), stat(r("hb"))]))]);
    assert_eq!(walk(body), ["d/2", "d/0"]);
    // a probe changes nothing: B, stored first, is not saved by being probed
    let body = json!([tl(lru.clone(), json!([
        let_("b", shared(2, 1)), store(r("b"), 0, 1), store(shared(1, 3), 0, 3), probe(r("b"), 0, "x"),
        store(shared(3, 1), 0, 1), probe(r("b"), 0, "hb"), stat(r("hb")), stat(r("x"))]))]);
    assert_eq!(walk(body), ["d/0", "d/1"]);
    // a range past the sequence's end is cut at it; a store leaves what is there alone
    let body = json!([tl(lru, json!([
        let_("a", shared(1, 3)), store(r("a"), 1, 9), probe(r("a"), 0, "h0"), probe(r("a"), 1, "h1"),
        store(r("a"), 0, 1), probe(r("a"), 0, "h2"), stat(r("h0")), stat(r("h1")), stat(r("h2"))]))]);
    assert_eq!(walk(body), ["d/0", "d/2", "d/3"]);
}

/// Entries equal in a tier's order are still each a victim (found 2026-11-22): a store's chunks
/// share its event, so under an order of `touched` alone they tied, and the victim order, keyed
/// by the order's values and the event stored at, kept one of them; the others were never
/// evicted and a touch of one took another's place. Ties go by `pos`, least first.
#[test]
fn entries_equal_in_the_order_are_each_a_victim() {
    let lru = json!([{"name": "t", "order": [cached("touched")], "capacity": 2}]);
    // A's two chunks fill the tier in one event; C evicts A's first, D its second
    let body = json!([tl(lru.clone(), json!([
        let_("a", shared(1, 2)), let_("c", shared(2, 1)),
        store(r("a"), 0, 2), store(r("c"), 0, 1),
        probe(r("a"), 0, "a0"), probe(r("a"), 1, "a1"), stat(r("a0")), stat(r("a1")),
        store(shared(3, 1), 0, 1),
        probe(r("a"), 1, "b1"), probe(r("c"), 0, "c0"), stat(r("b1")), stat(r("c0"))]))]);
    assert_eq!(walk(body), ["d/0", "d/1", "d/0", "d/1"]);
    // a touch of both, one event again, leaves both victims: two stores evict both
    let body = json!([tl(lru, json!([
        let_("a", shared(1, 2)), store(r("a"), 0, 2), touch(r("a"), 0, 2),
        store(shared(2, 2), 0, 2), probe(r("a"), 0, "a0"), probe(r("a"), 1, "a1"), stat(r("a0")), stat(r("a1"))]))]);
    assert_eq!(walk(body), ["d/0", "d/0"]);
}

/// A key is an integer and the cache knows nothing of where it came from (contract 0.23): `key`
/// is computed with the range's index at `from`, `from + 1`, and so on, and an entry's `pos` is
/// that index.
#[test]
fn keys_are_integers_computed_at_each_index_from_from() {
    let keys = |index: &str, from: i64, count: i64, key: Value| json!({"index": index, "from": from, "count": count, "key": key});
    let j = || json!({"index": "j"});
    let lmc = json!([{"name": "t", "order": [cached("touched"), {"neg": cached("pos")}], "capacity": 3}]);
    // keys 105, 106, 107 at indices 5, 6, 7; a probe of 105 to 108, the index itself the key, finds 3
    let body = json!([tl(lmc.clone(), json!([
        {"store": {"cache": "c", "tier": "t", "keys": keys("j", 5, 3, json!({"add": [j(), 100]}))}},
        {"probe": {"cache": "c", "tier": "t", "keys": keys("j", 105, 4, j()), "bind": "h"}}, stat(r("h")),
        // one more evicts the entry of the greatest `pos`, key 107 at index 7
        {"store": {"cache": "c", "tier": "t", "keys": keys("j", 0, 1, json!(999))}},
        {"probe": {"cache": "c", "tier": "t", "keys": keys("j", 105, 4, j()), "bind": "h2"}}, stat(r("h2")),
        {"probe": {"cache": "c", "tier": "t", "keys": keys("j", 107, 1, j()), "bind": "h3"}}, stat(r("h3"))]))]);
    assert_eq!(walk(body), ["d/3", "d/2", "d/0"]);
    // `from` left out is 0; a key that comes twice is acted on once, at its first index
    let body = json!([tl(lmc, json!([
        {"store": {"cache": "c", "tier": "t", "keys": {"index": "j", "count": 4, "key": {"div": [j(), 2]}}, "taken": "tk"}},
        stat(json!({"len": "tk"})), stat(elem("tk", 0)), stat(elem("tk", 1))]))]);
    assert_eq!(walk(body), ["d/2", "d/0", "d/2"]);
    // a draw in a key is one at each index, as an array's value is: four keys, not one
    let body = json!([tl(json!([{"name": "t"}]), json!([
        {"store": {"cache": "c", "tier": "t", "keys": {"index": "j", "count": 4, "key": {"draw": {"uniform64": {}}}}, "taken": "tk"}},
        stat(json!({"len": "tk"}))]))]);
    assert_eq!(walk(body), ["d/4"]);
    walk_fails(json!([tl(json!([{"name": "t"}]), json!([{"touch": {"cache": "c", "tier": "t", "keys": {"index": "j", "count": -1, "key": 0}}}]))]), "touch: count -1");
}

/// A store of `present: touch` (contract 0.23) is a lookup that stores what it missed: the keys
/// the tier holds touched, the others stored, one event; `found` the leading run held before
/// it, `bind` the one after, `taken` the indices stored.
#[test]
fn a_store_that_touches_is_a_lookup_that_stores_what_it_missed() {
    let lru = |cap: i64| json!([{"name": "t", "order": [cached("touched"), cached("pos")], "capacity": cap}]);
    let all = |s: Value, extra: Value| {
        let mut n = json!({"cache": "c", "tier": "t", "keys": crate::common::chunk_keys(s, json!(1), json!(0), None), "present": "touch"});
        for (k, v) in extra.as_object().unwrap() {
            n[k] = v.clone();
        }
        json!({"store": n})
    };
    let body = json!([tl(lru(9), json!([
        let_("a", shared(1, 4)), store(r("a"), 0, 2),
        all(r("a"), json!({"found": "f", "bind": "b", "taken": "tk"})),
        stat(r("f")), stat(r("b")), stat(json!({"len": "tk"})), stat(elem("tk", 0)), stat(elem("tk", 1))]))]);
    assert_eq!(walk(body), ["d/2", "d/4", "d/2", "d/2", "d/3"]);
    // what it found is touched: B, stored after A, goes before A's chunks when C needs room
    let body = json!([tl(lru(3), json!([
        let_("a", shared(1, 2)), let_("b", shared(2, 1)),
        store(r("a"), 0, 2), store(r("b"), 0, 1), all(r("a"), json!({})), store(shared(3, 1), 0, 1),
        probe(r("a"), 0, "ha"), probe(r("b"), 0, "hb"), stat(r("ha")), stat(r("hb"))]))]);
    assert_eq!(walk(body), ["d/2", "d/0"]);
    // held, what it touched and stored is no victim until the release: C finds no room and is dropped
    let body = json!([tl(lru(2), json!([
        let_("a", shared(1, 2)), store(r("a"), 0, 1), all(r("a"), json!({"hold": "p"})),
        store_bind(shared(3, 1), 0, 1, "c1"), {"release": {"hold": "p"}}, store_bind(shared(3, 1), 0, 1, "c2"),
        stat(r("c1")), stat(r("c2"))]))]);
    assert_eq!(walk(body), ["d/0", "d/1"]);
}

/// `store`'s `bind` is the leading run of the range held after it: a store a full tier with
/// timed eviction skips is the planned drop (Q5), and `fail` refuses the walk instead.
#[test]
fn a_store_past_a_timed_tiers_capacity_is_a_planned_drop() {
    let body = json!([tl(timed(json!({"capacity": 3})), json!([
        let_("a", shared(1, 5)), store_bind(r("a"), 0, 5, "n"), stat(r("n")), probe(r("a"), 0, "h"), stat(r("h"))]))]);
    assert_eq!(walk(body), ["d/3", "d/3"]);
    fails(json!([tl(timed(json!({"capacity": 3, "full": "fail"})), json!([store(shared(1, 5), 0, 5)]))]),
        "store: tier `t` of cache `c` is full: past its capacity 3 (full: fail)");
}

// ---------------------------------------------------------------- timed eviction

/// At each planned tick, when `when` holds, `count` victims go, least in `order` first; a tick
/// at the same planned instant as a lane's step comes first; a tier with timed eviction
/// never evicts at store.
#[test]
fn timed_eviction_runs_on_the_planned_grid() {
    // nine of ten (0.9 >= 0.8): the tick at 100 evicts one, the tail
    let run = |n: i64, wait: i64| {
        walk(json!([tl(timed(json!({})), json!([
            let_("a", shared(1, n)), store(r("a"), 0, n), compute(wait), probe(r("a"), 0, "h"), stat(r("h"))]))]))
    };
    assert_eq!(run(9, 99), ["d/9"]);
    assert_eq!(run(9, 100), ["d/8"]);
    // 8 (0.8): one at 100, then 7 is below the mark and nothing more goes
    assert_eq!(run(8, 250), ["d/7"]);
    // 7 is below the mark: nothing
    assert_eq!(run(7, 1000), ["d/7"]);
    // ten: 2 at 100 (20 % of 10), then 8 at 0.8: 1 at 200, then 7
    assert_eq!(run(10, 150), ["d/8"]);
    assert_eq!(run(10, 200), ["d/7"]);
    // touched chunks outlive the rest: B's touch at 50 keeps it past the tick at 100
    let body = json!([tl(timed(json!({})), json!([
        let_("b", shared(2, 1)), store(r("b"), 0, 1), store(shared(1, 8), 0, 8), compute(50), touch(r("b"), 0, 1),
        compute(60), probe(r("b"), 0, "hb"), stat(r("hb"))]))]);
    assert_eq!(walk(body), ["d/1"]);
    // no ticks after the last lane ends, and none in a timeline with no timed tier
    let body = json!([tl(timed(json!({})), json!([store(shared(1, 9), 0, 9)])), stat(json!(5))]);
    assert_eq!(walk(body), ["d/5"]);
    // a held entry is not a victim: a lookup holding A's chunks spares them
    let body = json!([tl(timed(json!({})), json!([
        let_("a", shared(1, 2)), {"store": {"cache": "c", "tier": "t", "keys": crate::common::chunk_keys(r("a"), json!(1), json!(0), None), "present": "touch", "found": "x", "hold": "p"}},
        store(shared(2, 7), 0, 7), compute(100), {"release": {"hold": "p"}},
        probe(r("a"), 0, "h"), probe(shared(2, 7), 0, "h2"), stat(r("h")), stat(r("h2"))]))]);
    assert_eq!(walk(body), ["d/2", "d/6"]);
}

// ---------------------------------------------------------------- the eviction body

/// Each victim runs the body with `victim` its key, on eviction lanes at the tick's planned
/// instant, victim i on lane i mod `lanes`: the unlinks are ops of the plan.
#[test]
fn each_victim_runs_the_eviction_body() {
    let body_ = json!([stat(victim_id())]);
    // ten chunks: two victims at the tick, then one at the next: the tail first
    let body = json!([tl(timed(json!({"on_evict": {"lanes": 4, "body": body_}})), json!([
        let_("a", shared(1, 10)), store(r("a"), 0, 10), compute(200),
        stat(key_id(r("a"), 9)), stat(key_id(r("a"), 8)), stat(key_id(r("a"), 7))]))]);
    let ops = walk(body);
    assert_eq!(ops.len(), 6, "{ops:?}");
    assert_eq!(&ops[..3], &ops[3..], "the body's stats are the three evicted chunks' keys, tail first");
    // victims i and i + lanes share a lane: all are issued, worker 0's first at one instant
    let body = json!([tl(timed(json!({"on_evict": {"lanes": 2, "body": body_}, "evict": {
        "every": 100, "when": true, "count": 3}})), json!([
        let_("a", shared(1, 3)), store(r("a"), 0, 3), compute(100),
        stat(key_id(r("a"), 2)), stat(key_id(r("a"), 0)), stat(key_id(r("a"), 1))]))]);
    let ops = walk(body);
    assert_eq!(&ops[..3], &ops[3..], "{ops:?}");
    // an eviction at store runs the body at the storing lane's planned instant
    let body = json!([tl(json!([{"name": "t", "order": [cached("touched"), {"neg": cached("pos")}], "capacity": 2,
                                 "on_evict": {"lanes": 1, "body": [stat(victim_id())]}}]), json!([
        let_("a", shared(1, 2)), store(r("a"), 0, 2), store(shared(2, 1), 0, 1), stat(key_id(r("a"), 1))]))]);
    let ops = walk(body);
    assert_eq!(ops.len(), 2);
    assert_eq!(ops[0], ops[1]);
    // the body sees the bindings around the timeline, and parameters
    let body = json!([let_("base", json!(7)), tl(timed(json!({"on_evict": {"lanes": 1, "body": [stat(json!({"add": [r("base"), {"param": "cap"}]}))]}})), json!([
        store(shared(1, 9), 0, 9), compute(100)]))]);
    assert_eq!(walk(body), ["d/17"]);
}

/// `now` in an eviction body is the eviction's planned instant, a tick's or the evicting
/// store's (contract 0.22, V35): the body plans no time, so it is that for the whole body.
#[test]
fn now_in_an_eviction_body_is_the_evictions_instant() {
    let now_body = json!({"lanes": 1, "body": [stat(json!({"now": {}})), stat(json!({"now": {}}))]});
    // a tick at 100 ns evicts one of nine
    let body = json!([tl(timed(json!({"on_evict": now_body})), json!([store(shared(1, 9), 0, 9), compute(150)]))]);
    assert_eq!(walk(body), ["d/100", "d/100"]);
    // a store at 5 ns evicts one; the storing lane's later `compute` does not move it
    let body = json!([tl(json!([{"name": "t", "order": [cached("stored")], "capacity": 2, "on_evict": now_body}]), json!([
        compute(5), store(shared(1, 3), 0, 3), compute(7)]))]);
    assert_eq!(walk(body), ["d/5", "d/5"]);
}

/// An eviction body reads no state of the timeline (contract 0.22, V35): no `published` and no
/// `entry`, anywhere in it; its step comes after the evicting lane's, so what it would read is
/// the scheduler's, not the abstract's.
#[test]
fn an_eviction_body_reads_no_timeline_state() {
    let publish = json!({"publish": {"timeline": "tl", "values": {"a": 1}}});
    for (e, what) in [(json!({"published": "tl"}), "published"), (json!({"published": {"timeline": "tl", "group": 0}}), "published"),
                      (json!({"entry": {"timeline": "tl", "index": 0, "key": "a"}}), "entry")] {
        let ev = json!({"lanes": 1, "body": [{"loop": {"index": "i", "to": 1, "body": [stat(json!({"add": [e, 1]}))]}}]});
        let errs = refused(json!([tl(timed(json!({"on_evict": ev})), json!([publish]))]));
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].ends_with(&format!("{what}: in an `on_evict` body, which reads no state of its timeline (V35)")), "{errs:?}");
    }
}

#[test]
fn eviction_used_wrongly_is_refused_when_walked() {
    walk_fails(json!([stat(json!({"victim": {}}))]), "victim: outside a tier's `on_evict` body (V35)");
    walk_fails(json!([stat(json!({"tier": "used"}))]), "tier `used`: outside a tier's `evict` (V33)");
    walk_fails(json!([probe(shared(1, 1), 0, "h")]), "probe: no timeline declares cache `c` around it (V33)");
    walk_fails(json!([tl(timed(json!({})), json!([{"touch": {"cache": "c", "tier": "u", "keys": crate::common::chunk_keys(shared(1, 1), json!(1), json!(0), Some(json!(1)))}}]))]),
        "touch: cache `c` has no tier `u` (V33)");
    fails(json!([tl(timed(json!({"evict": {"every": 0, "when": true, "count": 1}})), json!([]))]),
        "tier `t` of cache `c`: evict every 0 ns");
    fails(json!([tl(timed(json!({"on_evict": {"lanes": 0, "body": []}})), json!([]))]), "tier `t` of cache `c`: on_evict lanes 0");
    fails(json!([tl(timed(json!({})), json!([store(shared(1, 1), -1, 1)]))]), "chunk_key: chunk -1 of a sequence of 1 whole chunks of 1");
}

// ---------------------------------------------------------------- the checkers (V33, V35)

#[test]
fn the_validator_holds_eviction_and_ranges_to_their_rules() {
    let has = |body: Value, needle: &str| {
        let e = refused(body.clone());
        assert!(e.iter().any(|m| m.contains(needle)), "wanted `{needle}` for {body}:\n{e:#?}");
    };
    let ev = lmcache_evict(100);
    has(json!([tl(json!([{"name": "t", "evict": ev}]), json!([]))]), "tier `t`: `evict` with no `capacity` (V33)");
    has(json!([tl(json!([{"name": "t", "capacity": 4, "evict": ev}]), json!([]))]), "tier `t`: `evict` with no `order`, so it has no victims (V33)");
    has(json!([tl(timed(json!({"full": "wait"})), json!([]))]), "tier `t`: `evict` with `full` `wait`; a timed tier's room comes at its ticks, not at a release (V33)");
    has(json!([tl(timed(json!({"evict": {"every": 100, "when": {"gt": [r("x"), 1]}, "count": 1}})), json!([]))]),
        "tier `t`: its `evict` `when` reads `ref`; it reads only parameters, literals, and `tier` (V33)");
    has(json!([tl(timed(json!({"evict": {"every": {"tier": "used"}, "when": true, "count": 1}})), json!([]))]),
        "tier `t`: its `evict` `every` reads `tier`; it reads only parameters and literals (V33)");
    has(json!([tl(timed(json!({"evict": {"every": 100, "when": true, "count": cached("pos")}})), json!([]))]), "reads `cached`");
    has(json!([tl(timed(json!({"evict": {"every": 100, "when": 1, "count": "x"}})), json!([]))]), "expected an integer");
    has(json!([stat(json!({"tier": "used"}))]), "tier `used`: outside a tier's `evict` (V33)");
    has(json!([stat(json!({"victim": {}}))]), "victim: outside a tier's `on_evict` body (V35)");
    has(json!([tl(json!([{"name": "t", "on_evict": {"lanes": 1, "body": []}}]), json!([]))]), "tier `t`: `on_evict` with no `order`, so nothing is evicted (V35)");
    has(json!([tl(timed(json!({"on_evict": {"lanes": {"tier": "used"}, "body": []}})), json!([]))]), "its `on_evict` `lanes` reads `tier`");
    for (stmt, what) in [
        (compute(1), "compute"),
        (store(shared(1, 1), 0, 1), "store"),
        (json!({"publish": {"timeline": "tl", "values": {"x": 1}}}), "publish"),
        (json!({"parallel": {"index": "w", "width": 2, "body": []}}), "parallel"),
    ] {
        has(json!([tl(timed(json!({"on_evict": {"lanes": 1, "body": [stmt]}})), json!([]))]),
            &format!("a `{what}` in an `on_evict` body, which only issues the victim's ops at its tick (V35)"));
    }
    has(json!([tl(timed(json!({"full": "skip"})), json!([{"loop": {"index": "i", "to": 2, "body": [probe(shared(1, 1), 0, "h"),
        stat(json!({"cond": {"if": {"gt": [{"index": "i"}, 0]}, "then": {"at": {"ref": "h", "index": {"sub": [{"index": "i"}, 1]}}}, "else": 0}}))]}}]))]), "V33");
    has(json!([tl(json!([{"name": "t", "order": [cached("freed")], "capacity": 4, "full": "wait"}]), json!([store(shared(1, 1), 0, 1)]))]),
        "store: tier `t` waits when full, which only a store of `present: touch` decides before it changes anything (V33)");
    has(json!([tl(json!([{"name": "t", "unit": "bytes", "capacity": 4}]), json!([store(shared(1, 1), 0, 1)]))]),
        "store: tier `t` counts bytes, so the store needs a `size` (V33)");
    has(json!([tl(timed(json!({})), json!([{"store": {"cache": "c", "tier": "t", "keys": crate::common::chunk_keys(shared(1, 1), json!(1), json!(0), Some(json!(1))), "size": 3}}]))]),
        "store: tier `t` counts entries, so the store takes no `size` (V33)");
    has(json!([tl(timed(json!({})), json!([{"probe": {"cache": "d", "tier": "t", "keys": crate::common::chunk_keys(shared(1, 1), json!(1), json!(0), None), "bind": "h"}}]))]),
        "probe: no timeline declares cache `d` around it (V33)");
    has(json!([tl(timed(json!({})), json!([{"probe": {"cache": "c", "tier": "t", "keys": crate::common::chunk_keys(json!(3), json!(1), json!(0), None), "bind": "h"}}]))]), "probe");
    let ok = json!([let_("base", json!(7)), tl(timed(json!({"capacity": {"param": "cap"}, "on_evict": {"lanes": 4, "body": [
        let_("f", victim_id()), {"cond": {"if": {"gt": [r("f"), 0]}, "then": [stat(r("f"))], "else": [stat(r("base"))]}}]}})), json!([
        let_("a", shared(1, 10)), probe(r("a"), 0, "h2"), probe(r("a"), 0, "h3"),
        touch(r("a"), 0, 10), store_bind(r("a"), 0, 10, "n"), stat(r("n")), compute(100)]))]);
    assert_eq!(refused(ok.clone()), Vec::<String>::new());
    assert_eq!(walk(ok).len(), 3);
}

// ---------------------------------------------------------------- taken, chunks, present

fn tiers_tu(cap_u: i64) -> Value {
    let lru = |name: &str, cap: i64| json!({"name": name, "order": [cached("touched"), {"neg": cached("pos")}], "capacity": cap});
    json!([lru("t", 100), lru("u", cap_u)])
}

fn store_in(tier: &str, s: Value, extra: Value) -> Value {
    // `from` and `count`, or `chunks`, of `extra` are the store's keys over `s`
    let e = extra.as_object().unwrap();
    let keys = match e.get("chunks") {
        Some(a) => crate::common::chunk_keys_of(s, json!(1), a.as_str().unwrap()),
        None => crate::common::chunk_keys(s, json!(1), e.get("from").cloned().unwrap_or(json!(0)), e.get("count").cloned()),
    };
    let mut n = json!({"cache": "c", "tier": tier, "keys": keys});
    for (k, v) in e {
        if !["chunks", "from", "count"].contains(&k.as_str()) {
            n[k] = v.clone();
        }
    }
    json!({"store": n})
}

fn probe_in(tier: &str, s: Value, from: i64, bind: &str) -> Value {
    json!({"probe": {"cache": "c", "tier": tier, "keys": crate::common::chunk_keys(s, json!(1), json!(from), None), "bind": bind}})
}

fn elem(a: &str, i: i64) -> Value {
    json!({"elem": {"array": a, "index": i}})
}

/// `taken` binds the indices of the chunks a store newly took, in order; `chunks` names an
/// array of chunk indices a `store` or `touch` acts on in place of `from` and `count`: LMCache's
/// G2 in front of G3, where only what G2 newly took is written to G3.
#[test]
fn a_store_binds_the_chunks_it_took_and_a_store_of_chunks_acts_on_those() {
    // A's chunks 0 and 1 are in t; a store of all five takes 2, 3, 4
    let body = json!([tl(tiers_tu(100), json!([
        let_("a", shared(1, 5)), store(r("a"), 0, 2), store_in("t", r("a"), json!({"from": 0, "count": 5, "taken": "tk"})),
        stat(json!({"len": "tk"})), stat(elem("tk", 0)), stat(elem("tk", 2)),
        store_in("u", r("a"), json!({"chunks": "tk", "taken": "tk2"})), stat(json!({"len": "tk2"})),
        probe_in("u", r("a"), 0, "h0"), probe_in("u", r("a"), 2, "h2"), stat(r("h0")), stat(r("h2"))]))]);
    assert_eq!(walk(body), ["d/3", "d/2", "d/4", "d/3", "d/0", "d/3"]);
    // a store that takes nothing binds an empty array; a chunk a full tier skips is not taken
    let body = json!([tl(json!([{"name": "t", "order": [cached("touched"), {"neg": cached("pos")}], "capacity": 3,
                                 "evict": lmcache_evict(100)}]), json!([
        let_("a", shared(1, 5)), store_in("t", r("a"), json!({"from": 0, "count": 5, "taken": "x"})),
        store_in("t", r("a"), json!({"from": 0, "count": 2, "taken": "y"})),
        stat(json!({"len": "x"})), stat(json!({"len": "y"}))]))]);
    assert_eq!(walk(body), ["d/3", "d/0"]);
    // a touch of chunks makes those recent: A's 0 and 2 outlive its 1 and 3 when B comes
    let body = json!([tl(tiers_tu(4), json!([
        let_("a", shared(1, 4)), {"array": {"name": "pick", "count": 2, "index": "k", "value": {"mul": [{"index": "k"}, 2]}}},
        store_in("u", r("a"), json!({"from": 0, "count": 4})),
        {"touch": {"cache": "c", "tier": "u", "keys": crate::common::chunk_keys_of(r("a"), json!(1), "pick")}},
        store_in("u", shared(2, 2), json!({"from": 0, "count": 2})),
        probe_in("u", r("a"), 0, "h0"), probe_in("u", r("a"), 2, "h2"), stat(r("h0")), stat(r("h2"))]))]);
    assert_eq!(walk(body), ["d/1", "d/1"]);
    // a parameter array names chunks too
    let doc_ = json!({"ast": "0.23", "name": "t", "params": {"some": {"default": [1, 3]}},
        "datasets": {"d": {"layout": "whole", "access": "map", "pattern": "d/{id}", "count": 1000000, "size": {"const": 1}, "seed": 1}},
        "actors": {"gpu": {"body": [tl(tiers_tu(9), json!([
            let_("a", shared(1, 4)), store_in("u", r("a"), json!({"chunks": "some", "taken": "tk"})),
            probe_in("u", r("a"), 1, "h"), stat(json!({"len": "tk"})), stat(r("h"))]))]}}}).to_string();
    assert_eq!(walk_doc(&doc_, true).unwrap(), ["d/2", "d/1"]);
}

/// `present: renew` stores again a chunk the tier holds, as if it were absent: a new entry, and
/// taken; `keep`, the default, leaves it alone. LMCache rewrites a chunk's file in G3 whenever
/// G2 newly takes it, and the key becomes the most recent.
#[test]
fn a_renewing_store_makes_a_held_chunk_new_and_takes_it() {
    let run = |present: &str| {
        walk(json!([tl(tiers_tu(4), json!([
            let_("a", shared(1, 3)), store_in("u", r("a"), json!({"from": 0, "count": 3})),
            store_in("u", shared(2, 1), json!({"from": 0, "count": 1})),
            {"array": {"name": "first", "count": 1, "index": "k", "value": 0}},
            store_in("u", r("a"), json!({"chunks": "first", "present": present, "taken": "tk"})),
            store_in("u", shared(3, 3), json!({"from": 0, "count": 3})),
            probe_in("u", r("a"), 0, "h"), probe_in("u", shared(2, 1), 0, "hb"),
            stat(json!({"len": "tk"})), stat(r("h")), stat(r("hb"))]))]))
    };
    assert_eq!(run("renew"), ["d/1", "d/1", "d/0"]);
    assert_eq!(run("keep"), ["d/0", "d/0", "d/1"]);
    // a chunk a lane holds is left as it is, renewed or not
    let body = json!([tl(tiers_tu(9), json!([
        let_("a", shared(1, 2)), {"store": {"cache": "c", "tier": "u", "keys": crate::common::chunk_keys(r("a"), json!(1), json!(0), None), "present": "touch", "found": "x", "hold": "p"}},
        {"array": {"name": "both", "count": 2, "index": "k", "value": {"index": "k"}}},
        store_in("u", r("a"), json!({"chunks": "both", "present": "renew", "taken": "tk"})), {"release": {"hold": "p"}},
        store_in("u", r("a"), json!({"chunks": "both", "present": "renew", "taken": "tk2"})),
        stat(json!({"len": "tk"})), stat(json!({"len": "tk2"}))]))]);
    assert_eq!(walk(body), ["d/0", "d/2"]);
}

#[test]
fn chunks_used_wrongly_are_refused_when_walked() {
    let arr = |vals: Value| json!({"array": {"name": "x", "count": 1, "index": "k", "value": vals}});
    fails(json!([tl(tiers_tu(9), json!([arr(json!(5)), store_in("u", shared(1, 3), json!({"chunks": "x"}))]))]),
        "chunk_key: chunk 5 of a sequence of 3 whole chunks of 1");
    fails(json!([tl(tiers_tu(9), json!([arr(json!(-1)), store_in("u", shared(1, 3), json!({"chunks": "x"}))]))]),
        "chunk_key: chunk -1 of a sequence of 3 whole chunks of 1");
    walk_fails(json!([tl(tiers_tu(9), json!([store_in("u", shared(1, 3), json!({"chunks": "nope"}))]))]),
        "unknown parameter array `nope`");
}

#[test]
fn the_validator_holds_taken_chunks_and_present_to_their_rules() {
    let has = |body: Value, needle: &str| {
        let e = refused(body.clone());
        assert!(e.iter().any(|m| m.contains(needle)), "wanted `{needle}` for {body}:\n{e:#?}");
    };
    let arr = json!({"array": {"name": "x", "count": 1, "index": "k", "value": 0}});
    // a hold is a store's that touches; a key is an integer, read with the range's index in scope
    has(json!([tl(tiers_tu(9), json!([arr, {"store": {"cache": "c", "tier": "u", "keys": {"index": "j", "count": 1, "key": {"index": "j"}}, "hold": "p"}}, {"release": {"hold": "p"}}]))]),
        "store: `hold` `p` holds what a store of `present: touch` touches and stores, and this one is not (V34)");
    has(json!([tl(tiers_tu(9), json!([{"touch": {"cache": "c", "tier": "u", "keys": {"index": "j", "count": 1, "key": shared(1, 3)}}}]))]),
        "expected an integer, got a seq (V27)");
    has(json!([tl(tiers_tu(9), json!([{"touch": {"cache": "c", "tier": "u", "keys": {"index": "j", "count": 1, "key": {"index": "j"}}}}, json!({"compute": {"ns": {"index": "j"}}})]))]),
        "index `j` not in scope");
    has(json!([tl(tiers_tu(9), json!([{"touch": {"cache": "c", "tier": "u", "keys": {"index": "1j", "count": 1, "key": 0}}}]))]), "not an identifier");
    has(json!([tl(tiers_tu(9), json!([store_in("u", shared(1, 3), json!({"chunks": "nope"}))]))]), "unknown parameter array `nope`");
    has(json!([tl(tiers_tu(9), json!([store_in("u", shared(1, 3), json!({"from": 0, "count": 1, "taken": "cap"}))]))]),
        "store: `taken` `cap` has the name of a parameter, which `elem` and `len` also read (V32)");
    has(json!([tl(tiers_tu(9), json!([store_in("u", shared(1, 3), json!({"from": 0, "count": 1, "taken": "1x"}))]))]), "not an identifier");
    // what a store took varies from one iteration to the next: a loop's count may not read it
    // where a fixed one is needed, and a later statement reads it after the store
    has(json!([stat(json!({"len": "tk"})), tl(tiers_tu(9), json!([store_in("u", shared(1, 3), json!({"from": 0, "count": 1, "taken": "tk"}))]))]),
        "unknown parameter array `tk`");
    let ok = json!([tl(tiers_tu(9), json!([{"loop": {"index": "i", "to": 3, "body": [
        let_("a", shared(1, 4)),
        store_in("t", r("a"), json!({"from": 0, "count": 4, "taken": "tk"})),
        store_in("u", r("a"), json!({"chunks": "tk", "present": "renew", "taken": "tk3"})),
        {"touch": {"cache": "c", "tier": "u", "keys": crate::common::chunk_keys_of(r("a"), json!(1), "tk3")}},
        {"loop": {"index": "j", "to": {"len": "tk3"}, "body": [stat(elem("tk3", 0))]}}]}}]))]);
    assert_eq!(refused(ok.clone()), Vec::<String>::new());
    assert_eq!(walk(ok), ["d/0", "d/0", "d/0", "d/0"]);
}

// ---------------------------------------------------------------- a probe over several tiers (contract 0.25)

fn probe_tiers(tiers: Value, s: Value, from: i64, bind: &str, held: Option<Value>) -> Value {
    let mut n = json!({"cache": "c", "tiers": tiers, "keys": crate::common::chunk_keys(s, json!(1), json!(from), None), "bind": bind});
    if let Some(h) = held {
        n["held"] = h;
    }
    json!({"probe": n})
}

fn walk_25(body: Value) -> Vec<String> {
    walk_doc(&doc(body).replacen("\"0.23\"", "\"0.25\"", 1), true).unwrap_or_else(|e| panic!("{e}"))
}

/// Contract 0.25 (brief item 78): a `probe` over `tiers` binds the leading run of the keys some
/// tier of the list holds, and `held` binds, a tier, the indices of that run it serves, a key
/// going to the first tier listed that holds it: LMCache's lookup over two L2 adapters, where
/// the union of what they hold is cut at its first gap and the lowest adapter loads each key.
#[test]
fn a_probe_over_several_tiers_reads_the_unions_leading_run() {
    // t holds A's chunks 0 and 2; u holds 1, 2, and 4; no tier holds 3
    let setup = || vec![let_("a", shared(1, 6)), store_in("t", r("a"), json!({"from": 0, "count": 1})), store_in("t", r("a"), json!({"from": 2, "count": 1})),
        store_in("u", r("a"), json!({"from": 1, "count": 2})), store_in("u", r("a"), json!({"from": 4, "count": 1}))];
    let with = |more: Vec<Value>| json!([tl(tiers_tu(100), Value::Array(setup().into_iter().chain(more).collect()))]);
    // neither tier alone has a run past one chunk; together, three; t serves 0 and 2, u the 1 between
    let body = with(vec![probe_in("t", r("a"), 0, "ht"), probe_in("u", r("a"), 0, "hu"),
        probe_tiers(json!(["t", "u"]), r("a"), 0, "h", Some(json!(["at", "au"]))),
        stat(r("ht")), stat(r("hu")), stat(r("h")), stat(json!({"len": "at"})), stat(json!({"len": "au"})),
        stat(elem("at", 0)), stat(elem("at", 1)), stat(elem("au", 0))]);
    assert_eq!(walk_25(body), ["d/1", "d/0", "d/3", "d/2", "d/1", "d/0", "d/2", "d/1"]);
    // the order of the list decides who serves a key both hold: u first, chunk 2 is u's
    let body = with(vec![probe_tiers(json!(["u", "t"]), r("a"), 0, "h", Some(json!(["au", "at"]))),
        stat(r("h")), stat(json!({"len": "at"})), stat(json!({"len": "au"})), stat(elem("at", 0)), stat(elem("au", 0)), stat(elem("au", 1))]);
    assert_eq!(walk_25(body), ["d/3", "d/1", "d/2", "d/0", "d/1", "d/2"]);
    // from chunk 1: the run is 1 and 2, the arrays hold the range's indices, not counts from its start
    let body = with(vec![probe_tiers(json!(["t", "u"]), r("a"), 1, "h", Some(json!(["at", "au"]))),
        stat(r("h")), stat(elem("at", 0)), stat(elem("au", 0))]);
    assert_eq!(walk_25(body), ["d/2", "d/2", "d/1"]);
    // from a chunk no tier holds: no run, and empty arrays; without `held`, the run alone
    let body = with(vec![probe_tiers(json!(["t", "u"]), r("a"), 3, "h", Some(json!(["at", "au"]))), probe_tiers(json!(["t", "u"]), r("a"), 4, "g", None),
        stat(r("h")), stat(json!({"len": "at"})), stat(json!({"len": "au"})), stat(r("g"))]);
    assert_eq!(walk_25(body), ["d/0", "d/0", "d/0", "d/1"]);
    // nothing is changed: with room for four in u, A's chunks stay in the order they were stored
    // in, so B's two evict A's 1 and 2, the probe between having touched none
    let lru = |name: &str, cap: i64| json!({"name": name, "order": [cached("touched"), cached("pos")], "capacity": cap});
    let body = json!([tl(json!([lru("t", 100), lru("u", 4)]), json!([
        let_("a", shared(1, 6)), store_in("u", r("a"), json!({"from": 1, "count": 2})), store_in("u", r("a"), json!({"from": 4, "count": 2})),
        probe_tiers(json!(["t", "u"]), r("a"), 1, "h", Some(json!(["at", "au"]))),
        store_in("u", shared(2, 2), json!({"from": 0, "count": 2})),
        probe_in("u", r("a"), 1, "h1"), probe_in("u", r("a"), 4, "h4"), stat(r("h")), stat(r("h1")), stat(r("h4"))]))]);
    assert_eq!(walk_25(body), ["d/2", "d/0", "d/2"]);
}

#[test]
fn the_validator_holds_a_probe_over_several_tiers_to_its_rules() {
    let has = |node: Value, needle: &str| {
        let text = doc(json!([tl(tiers_tu(9), json!([{"probe": node.clone()}]))])).replacen("\"0.23\"", "\"0.25\"", 1);
        let e = aeiou::validate::check(&aeiou::ast::parse(&text).expect("parses"));
        assert!(e.iter().any(|m| m.contains(needle)), "wanted `{needle}` for {node}:\n{e:#?}");
    };
    let ks = || json!({"index": "j", "count": 1, "key": {"index": "j"}});
    has(json!({"cache": "c", "tier": "t", "tiers": ["t", "u"], "keys": ks(), "bind": "h"}), "probe: `tier` and `tiers` both; it names one tier or a list of them (V33)");
    has(json!({"cache": "c", "keys": ks(), "bind": "h"}), "probe: neither `tier` nor `tiers` (V33)");
    has(json!({"cache": "c", "tiers": ["t"], "keys": ks(), "bind": "h"}), "probe: `tiers` names 1 tier; a probe of one tier names it by `tier` (V33)");
    has(json!({"cache": "c", "tiers": ["t", "u", "t"], "keys": ks(), "bind": "h"}), "probe: tier `t` named twice in `tiers` (V33)");
    has(json!({"cache": "c", "tiers": ["t", "v"], "keys": ks(), "bind": "h"}), "probe: cache `c` has no tier `v` (V33)");
    has(json!({"cache": "c", "tier": "t", "keys": ks(), "bind": "h", "held": ["x"]}), "probe: `held` without `tiers`; a probe of one tier binds its run alone (V33)");
    has(json!({"cache": "c", "tiers": ["t", "u"], "keys": ks(), "bind": "h", "held": ["x"]}), "probe: `held` names 1 arrays for 2 tiers (V33)");
    has(json!({"cache": "c", "tiers": ["t", "u"], "keys": ks(), "bind": "h", "held": ["x", "x"]}), "probe: `held` `x` named twice (V33)");
    has(json!({"cache": "c", "tiers": ["t", "u"], "keys": ks(), "bind": "h", "held": ["x", "cap"]}), "probe: `held` `cap` has the name of a parameter, which `elem` and `len` also read (V32)");
    has(json!({"cache": "c", "tiers": ["t", "u"], "keys": ks(), "bind": "h", "held": ["x", "1y"]}), "not an identifier");
    // the version: a document with such a probe carries 0.25, and one without it does not
    let with = doc(json!([tl(tiers_tu(9), json!([probe_tiers(json!(["t", "u"]), shared(1, 2), 0, "h", None)]))]));
    let e = aeiou::validate::check(&aeiou::ast::parse(&with).expect("parses"));
    assert_eq!(e, ["t: ast: version `0.23`, but the document holds a `probe` over several tiers, which contract `0.25` added: it carries `0.25`"]);
    let without = doc(json!([tl(tiers_tu(9), json!([probe_in("t", shared(1, 2), 0, "h")]))])).replacen("\"0.23\"", "\"0.25\"", 1);
    let e = aeiou::validate::check(&aeiou::ast::parse(&without).expect("parses"));
    assert!(e[0].contains("holds nothing past contract `0.23`"), "{e:?}");
    // what `held` binds varies with what the tiers hold, as a store's `taken` does: read after the probe
    let ok = json!([tl(tiers_tu(9), json!([let_("a", shared(1, 4)), probe_tiers(json!(["t", "u"]), r("a"), 0, "h", Some(json!(["at", "au"]))),
        {"touch": {"cache": "c", "tier": "u", "keys": crate::common::chunk_keys_of(r("a"), json!(1), "au")}},
        {"loop": {"index": "k", "to": {"len": "at"}, "body": [stat(elem("at", 0))]}}, stat(r("h"))]))]);
    let text = doc(ok).replacen("\"0.23\"", "\"0.25\"", 1);
    assert_eq!(aeiou::validate::check(&aeiou::ast::parse(&text).expect("parses")), Vec::<String>::new());
    assert_eq!(walk_doc(&text, true).unwrap(), ["d/0"]);
}

// ---------------------------------------------------------------- a tier's fill read by a lane (contract 0.27)

fn used(tier: &str, bind: &str) -> Value {
    json!({"used": {"cache": "c", "tier": tier, "bind": bind}})
}

fn walk_27(body: Value) -> Vec<String> {
    walk_doc(&doc(body).replacen("\"0.23\"", "\"0.27\"", 1), true).unwrap_or_else(|e| panic!("{e}"))
}

/// Contract 0.27 (brief item 74): `used` binds what a tier holds, in the tier's unit, at the
/// lane's planned instant, and changes nothing: what a lane needs to decide as LMCache's server
/// does whether a load fits in G2, all of it or none.
#[test]
fn used_binds_what_a_tier_holds_and_changes_nothing() {
    // empty, then three of A's chunks in t and one in u; each tier's own count
    let body = json!([tl(tiers_tu(100), json!([
        used("t", "u0"), stat(r("u0")),
        let_("a", shared(1, 6)), store_in("t", r("a"), json!({"from": 0, "count": 3})), store_in("u", r("a"), json!({"from": 0, "count": 1})),
        used("t", "ut"), used("u", "uu"), stat(r("ut")), stat(r("uu")),
        // read again after a store: the value is the one at the statement, not a later one
        store_in("t", r("a"), json!({"from": 3, "count": 2})), stat(r("ut")), used("t", "ut2"), stat(r("ut2"))]))]);
    assert_eq!(walk_27(body), ["d/0", "d/3", "d/1", "d/3", "d/5"]);
    // a tier that counts bytes: `used` is in bytes
    let bytes = json!([{"name": "t", "order": [cached("touched")], "capacity": 1000, "unit": "bytes"}]);
    let body = json!([tl(bytes, json!([
        {"store": {"cache": "c", "tier": "t", "keys": crate::common::chunk_keys(shared(1, 2), json!(1), json!(0), Some(json!(2))), "size": 300}},
        used("t", "b"), stat(r("b"))]))]);
    assert_eq!(walk_27(body), ["d/600"]);
    // the decision it is for: a load of `n` fits in what a tier of 4 has left, or nothing is done
    let lru = json!([{"name": "t", "order": [cached("touched"), {"neg": cached("pos")}], "capacity": 4}]);
    let fits = |n: i64| json!({"le": [n, {"sub": [4, r("f")]}]});
    let body = json!([tl(lru, json!([
        store(shared(1, 3), 0, 3), used("t", "f"),
        {"cond": {"if": fits(1), "then": [stat(json!(1))], "else": [stat(json!(10))]}},
        {"cond": {"if": fits(2), "then": [stat(json!(2))], "else": [stat(json!(20))]}}]))]);
    assert_eq!(walk_27(body), ["d/1", "d/20"]);
    // nothing is changed: A's chunks keep their order, so B's one evicts A's first-stored tail
    let lru = json!([{"name": "t", "order": [cached("touched"), cached("pos")], "capacity": 3}]);
    let body = json!([tl(lru, json!([
        let_("a", shared(1, 3)), store(r("a"), 0, 3), used("t", "f"), store(shared(2, 1), 0, 1),
        probe(r("a"), 0, "h0"), probe(r("a"), 1, "h1"), used("t", "g"), stat(r("h0")), stat(r("h1")), stat(r("g"))]))]);
    assert_eq!(walk_27(body), ["d/0", "d/2", "d/3"]);
}

#[test]
fn the_validator_holds_used_to_its_rules() {
    let check = |body: Value, version: &str| {
        let text = doc(body).replacen("\"0.23\"", &format!("\"{version}\""), 1);
        aeiou::validate::check(&aeiou::ast::parse(&text).expect("parses"))
    };
    let has = |body: Value, needle: &str| {
        let e = check(body.clone(), "0.27");
        assert!(e.iter().any(|m| m.contains(needle)), "wanted `{needle}` for {body}:\n{e:#?}");
    };
    has(json!([used("t", "f")]), "used: no timeline declares cache `c` around it (V33)");
    has(json!([tl(tiers_tu(9), json!([used("v", "f")]))]), "used: cache `c` has no tier `v` (V33)");
    has(json!([tl(tiers_tu(9), json!([used("t", "1f")]))]), "not an identifier");
    // its value is the one the step being walked sees: never under `@`, never a `prev`
    has(json!([tl(tiers_tu(9), json!([{"loop": {"index": "i", "to": 2, "body": [used("t", "f"), stat(json!({"at": {"ref": "f", "index": 0}}))]}}]))]),
        "`f @`: it is bound by a `probe`, a `used`, or a `store`");
    has(json!([tl(tiers_tu(9), json!([{"loop": {"index": "i", "to": 2, "body": [used("t", "f"), stat(json!({"prev": {"ref": "f", "init": 0}}))]}}]))]),
        "prev: `f` is bound by a `probe`, a `used`, or a `store`, not a `let` (V31)");
    // a body that only issues ops reads no state of its timeline
    let workers = |body: Value| json!([{"timeline": {"name": "tl", "caches": [{"name": "c", "tiers": tiers_tu(9)}],
        "workers": [{"name": "w", "lanes": 1, "staging": 4096}], "body": body}}]);
    has(workers(json!([{"submit": {"workers": "w", "body": [used("t", "f")]}}])), "used");
    let evicting = json!([{"name": "t", "order": [cached("touched")], "capacity": 4, "on_evict": {"lanes": 1, "body": [used("t", "f")]}}]);
    has(json!([tl(evicting, json!([]))]), "used");
    // the version: a document with it carries 0.27, and one without it does not
    let e = check(json!([tl(tiers_tu(9), json!([used("t", "f")]))]), "0.23");
    assert_eq!(e, ["t: ast: version `0.23`, but the document holds a `used`, a tier's fill read by a lane, which contract `0.27` added: it carries `0.27`"]);
    let e = check(json!([tl(tiers_tu(9), json!([probe_in("t", shared(1, 2), 0, "h")]))]), "0.27");
    assert!(e[0].contains("holds nothing past contract `0.23`"), "{e:?}");
    assert_eq!(check(json!([tl(tiers_tu(9), json!([used("t", "f"), stat(r("f"))]))]), "0.27"), Vec::<String>::new());
}

// ---------------------------------------------------------------- what a tier holds of a key range, as a set (contract 0.28)

fn holds(tier: &str, s: Value, from: i64, bind: &str) -> Value {
    json!({"holds": {"cache": "c", "tier": tier, "keys": crate::common::chunk_keys(s, json!(1), json!(from), None), "bind": bind}})
}

fn set(bind: &str, op: &str, of: &[&str]) -> Value {
    json!({"set": {"bind": bind, "op": op, "of": of}})
}

fn run_of(a: &str, from: Value) -> Value {
    json!({"run": {"array": a, "from": from}})
}

fn below(a: &str, bound: Value) -> Value {
    json!({"below": {"array": a, "bound": bound}})
}

/// A stat of each element of the first `n` of array `a`.
fn each(a: &str, n: Value) -> Value {
    json!({"loop": {"index": "k", "to": n, "body": [stat(json!({"elem": {"array": a, "index": {"index": "k"}}}))]}})
}

fn walk_28(body: Value) -> Vec<String> {
    walk_doc(&doc(body).replacen("\"0.23\"", "\"0.28\"", 1), true).unwrap_or_else(|e| panic!("{e}"))
}

/// A's chunks stored one by one: those of `mask`'s bits, in tier `tier`.
fn stored(tier: &str, mask: u32) -> Vec<Value> {
    (0..8).filter(|i| mask >> i & 1 == 1).map(|i| store_in(tier, r("a"), json!({"from": i, "count": 1}))).collect()
}

/// Contract 0.28 (the user's proposal, 2026-11-23): `holds` binds the indices of a key range
/// whose keys a tier holds, every one, not stopping at the first it lacks, and changes nothing;
/// `set` combines such arrays; `run` and `below` measure one. What a lookup is made of, so that
/// its rules (which run is the hit, which tier serves a key) are the abstract's statements.
#[test]
fn holds_binds_every_index_a_tier_holds_and_changes_nothing() {
    // t holds A's chunks 0, 2, 3, and 6
    let with = |more: Vec<Value>| json!([tl(tiers_tu(100), Value::Array(std::iter::once(let_("a", shared(1, 8))).chain(stored("t", 0b100_1101)).chain(more).collect()))]);
    let body = with(vec![holds("t", r("a"), 0, "s"), stat(json!({"len": "s"})), each("s", json!({"len": "s"}))]);
    assert_eq!(walk_28(body), ["d/4", "d/0", "d/2", "d/3", "d/6"]);
    // from chunk 1: the range's indices, not counts from its start; a tier holding none, empty
    let body = with(vec![holds("t", r("a"), 1, "s"), holds("u", r("a"), 0, "e"), each("s", json!({"len": "s"})), stat(json!({"len": "e"}))]);
    assert_eq!(walk_28(body), ["d/2", "d/3", "d/6", "d/0"]);
    // the value is the one at the statement, not a later one
    let body = with(vec![holds("t", r("a"), 0, "s"), store_in("t", r("a"), json!({"from": 1, "count": 1})), holds("t", r("a"), 0, "s2"),
        stat(json!({"len": "s"})), stat(json!({"len": "s2"}))]);
    assert_eq!(walk_28(body), ["d/4", "d/5"]);
    // nothing is changed: A's chunks keep their order, so B's one evicts A's first-stored
    let lru = json!([{"name": "t", "order": [cached("touched"), cached("pos")], "capacity": 3}]);
    let body = json!([tl(lru, json!([
        let_("a", shared(1, 3)), store(r("a"), 0, 3), holds("t", r("a"), 0, "s"), store(shared(2, 1), 0, 1),
        holds("t", r("a"), 0, "s2"), stat(json!({"len": "s"})), each("s2", json!({"len": "s2"}))]))]);
    assert_eq!(walk_28(body), ["d/3", "d/1", "d/2"]);
}

#[test]
fn sets_combine_and_run_and_below_measure_what_holds_bound() {
    // t holds A's chunks 0, 2, 3, 6; u holds 1, 2, 4, 6; v nothing
    let tiers = || { let mut t = tiers_tu(100); t.as_array_mut().unwrap().push(json!({"name": "v", "order": [cached("touched")], "capacity": 100})); t };
    let with = |more: Vec<Value>| json!([tl(tiers(), Value::Array(std::iter::once(let_("a", shared(1, 8)))
        .chain(stored("t", 0b100_1101)).chain(stored("u", 0b101_0110))
        .chain([holds("t", r("a"), 0, "st"), holds("u", r("a"), 0, "su"), holds("v", r("a"), 0, "sv")]).chain(more).collect()))]);
    let all = |a: &str| each(a, json!({"len": a}));
    let body = with(vec![set("x", "union", &["st", "su"]), all("x")]);
    assert_eq!(walk_28(body), ["d/0", "d/1", "d/2", "d/3", "d/4", "d/6"]);
    let body = with(vec![set("x", "intersect", &["st", "su"]), all("x")]);
    assert_eq!(walk_28(body), ["d/2", "d/6"]);
    let body = with(vec![set("x", "minus", &["st", "su"]), all("x"), set("y", "minus", &["su", "st"]), all("y")]);
    assert_eq!(walk_28(body), ["d/0", "d/3", "d/1", "d/4"]);
    // more than two: the union of all, what all hold, the first less all the rest; an empty one
    let body = with(vec![set("x", "union", &["sv", "st", "su"]), stat(json!({"len": "x"})), set("y", "intersect", &["st", "su", "sv"]), stat(json!({"len": "y"})),
        set("m", "union", &["st", "su"]), set("z", "minus", &["m", "st", "sv"]), all("z"), set("w", "minus", &["sv", "st"]), stat(json!({"len": "w"}))]);
    assert_eq!(walk_28(body), ["d/6", "d/0", "d/1", "d/4", "d/0"]);
    // a set of a set, and an array with itself
    let body = with(vec![set("x", "union", &["st", "su"]), set("y", "minus", &["x", "su"]), all("y"), set("z", "intersect", &["st", "st"]), stat(json!({"len": "z"})),
        set("w", "minus", &["st", "st"]), stat(json!({"len": "w"}))]);
    assert_eq!(walk_28(body), ["d/0", "d/3", "d/4", "d/0"]);
    // run: the consecutive indices held from `from`; none when `from` is not held, or past the end
    let body = with(vec![set("x", "union", &["st", "su"]), stat(run_of("x", json!(0))), stat(run_of("x", json!(2))), stat(run_of("x", json!(5))),
        stat(run_of("x", json!(6))), stat(run_of("x", json!(9))), stat(run_of("st", json!(2))), stat(run_of("sv", json!(0))), stat(run_of("x", json!(-1)))]);
    assert_eq!(walk_28(body), ["d/5", "d/3", "d/0", "d/1", "d/0", "d/2", "d/0", "d/0"]);
    // below: how many elements are less than the bound
    let body = with(vec![stat(below("st", json!(0))), stat(below("st", json!(1))), stat(below("st", json!(3))), stat(below("st", json!(6))),
        stat(below("st", json!(100))), stat(below("sv", json!(5))), stat(below("st", json!(-3)))]);
    assert_eq!(walk_28(body), ["d/0", "d/1", "d/2", "d/3", "d/4", "d/0", "d/0"]);
}

/// The proof that the functions suffice: over cache states and starts, a probe over two tiers
/// (contract 0.25) and its composition from `holds`, `set`, `run`, and `below` give the same
/// run and the same indices to each tier.
#[test]
fn a_probe_over_tiers_is_its_composition_from_sets() {
    let masks = [(0u32, 0u32), (0b1, 0), (0, 0b1), (0b101, 0b10), (0b1111_1111, 0), (0, 0b1111_1111), (0b1010_1010, 0b0101_0101), (0b0110_0111, 0b0011_1100),
        (0b1000_0001, 0b0111_1110), (0b0001_0111, 0b0001_1011), (0b1110_0000, 0b0001_1111), (0b0100_1101, 0b0101_0110)];
    let mut runs = std::collections::BTreeSet::new();
    for (mt, mu) in masks {
        for from in [0i64, 1, 3, 7] {
            for order in [["t", "u"], ["u", "t"]] {
                let with = |more: Vec<Value>| json!([tl(tiers_tu(100), Value::Array(std::iter::once(let_("a", shared(1, 8)))
                    .chain(stored("t", mt)).chain(stored("u", mu)).chain(more).collect()))]);
                let probed = walk_25(with(vec![probe_tiers(json!(order), r("a"), from, "h", Some(json!(["a0", "a1"]))),
                    stat(r("h")), each("a0", json!({"len": "a0"})), stat(json!(777)), each("a1", json!({"len": "a1"}))]));
                let end = json!({"add": [from, r("h")]});
                let composed = walk_28(with(vec![holds(order[0], r("a"), from, "s0"), holds(order[1], r("a"), from, "s1"),
                    set("any", "union", &["s0", "s1"]), let_("h", run_of("any", json!(from))), set("r1", "minus", &["s1", "s0"]),
                    stat(r("h")), each("s0", below("s0", end.clone())), stat(json!(777)), each("r1", below("r1", end))]));
                assert_eq!(probed, composed, "t {mt:#b} u {mu:#b} from {from} order {order:?}");
                runs.insert(probed[0].clone());
            }
        }
    }
    assert!(runs.len() >= 5, "{runs:?}");                // the states reach runs of several lengths
}

#[test]
fn the_validator_holds_sets_to_their_rules() {
    let check = |body: Value, version: &str| {
        let text = doc(body).replacen("\"0.23\"", &format!("\"{version}\""), 1);
        aeiou::validate::check(&aeiou::ast::parse(&text).expect("parses"))
    };
    let has = |body: Value, needle: &str| {
        let e = check(body.clone(), "0.28");
        assert!(e.iter().any(|m| m.contains(needle)), "wanted `{needle}` for {body}:\n{e:#?}");
    };
    let a = || shared(1, 4);
    let in_tl = |body: Value| json!([tl(tiers_tu(9), body)]);
    // holds: as a probe of one tier
    has(json!([holds("t", a(), 0, "s")]), "holds: no timeline declares cache `c` around it (V33)");
    has(in_tl(json!([holds("v", a(), 0, "s")])), "holds: cache `c` has no tier `v` (V33)");
    has(in_tl(json!([holds("t", a(), 0, "1s")])), "not an identifier");
    has(in_tl(json!([holds("t", a(), 0, "cap")])), "holds: `bind` `cap` has the name of a parameter, which `elem` and `len` also read (V32)");
    // set: an operation of the three, over two or more arrays a `holds` or a `set` bound
    has(in_tl(json!([holds("t", a(), 0, "s"), {"set": {"bind": "x", "op": "xor", "of": ["s", "s"]}}])), "set: `op` `xor` is not `union`, `intersect`, or `minus` (V33)");
    has(in_tl(json!([holds("t", a(), 0, "s"), set("x", "union", &["s"])])), "set: `of` names 1 array; an operation is over two or more (V33)");
    has(in_tl(json!([holds("t", a(), 0, "s"), set("x", "union", &["s", "q"])])), "set: `q` is no array a `holds` or a `set` bound before it (V33)");
    has(in_tl(json!([store_in("t", a(), json!({"taken": "k"})), holds("t", a(), 0, "s"), set("x", "union", &["s", "k"])])),
        "set: `k` is no array a `holds` or a `set` bound before it (V33)");
    has(in_tl(json!([holds("t", a(), 0, "s"), set("1x", "union", &["s", "s"])])), "not an identifier");
    has(in_tl(json!([holds("t", a(), 0, "s"), set("cap", "union", &["s", "s"])])), "set: `bind` `cap` has the name of a parameter, which `elem` and `len` also read (V32)");
    // run and below: of such an array alone
    has(in_tl(json!([store_in("t", a(), json!({"taken": "k"})), stat(run_of("k", json!(0)))])), "run: `k` is no array a `holds` or a `set` bound before it (V33)");
    has(in_tl(json!([stat(below("q", json!(0)))])), "below: `q` is no array a `holds` or a `set` bound before it (V33)");
    // a body that only issues ops reads no state of its timeline
    let workers = |body: Value| json!([{"timeline": {"name": "tl", "caches": [{"name": "c", "tiers": tiers_tu(9)}],
        "workers": [{"name": "w", "lanes": 1, "staging": 4096}], "body": body}}]);
    has(workers(json!([{"submit": {"workers": "w", "body": [holds("t", a(), 0, "s")]}}])), "holds");
    let evicting = json!([{"name": "t", "order": [cached("touched")], "capacity": 4, "on_evict": {"lanes": 1, "body": [holds("t", a(), 0, "s")]}}]);
    has(json!([tl(evicting, json!([]))]), "holds");
    // the version: a document with one carries 0.28, and one without does not
    let e = check(in_tl(json!([holds("t", a(), 0, "s")])), "0.27");
    assert_eq!(e, ["t: ast: version `0.27`, but the document holds a `holds` or a `set`, what a tier holds of a key range as a set, which contract `0.28` added: it carries `0.28`"]);
    let e = check(in_tl(json!([probe_in("t", a(), 0, "h")])), "0.28");
    assert!(e[0].contains("holds nothing past contract `0.23`"), "{e:?}");
    // what they bind varies with what the tiers hold, as a store's `taken` does
    let ok = in_tl(json!([let_("a", a()), holds("t", r("a"), 0, "s"), holds("u", r("a"), 0, "s2"), set("x", "union", &["s", "s2"]),
        let_("n", below("x", json!({"add": [0, run_of("x", json!(0))]}))), each("x", r("n")),
        {"touch": {"cache": "c", "tier": "u", "keys": crate::common::chunk_keys_of(r("a"), json!(1), "x")}}, stat(r("n"))]));
    assert_eq!(check(ok, "0.28"), Vec::<String>::new());
}
