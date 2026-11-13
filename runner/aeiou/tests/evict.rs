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
    json!({"ast": "0.19", "name": "t", "params": {"cap": {"default": 10}},
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
    json!({"probe": {"cache": "c", "tier": "t", "seq": s, "per": 1, "from": from, "bind": bind}})
}

fn touch(s: Value, from: i64, count: i64) -> Value {
    json!({"touch": {"cache": "c", "tier": "t", "seq": s, "per": 1, "from": from, "count": count}})
}

fn store(s: Value, from: i64, count: i64) -> Value {
    json!({"store": {"cache": "c", "tier": "t", "seq": s, "per": 1, "from": from, "count": count}})
}

fn store_bind(s: Value, from: i64, count: i64, bind: &str) -> Value {
    json!({"store": {"cache": "c", "tier": "t", "seq": s, "per": 1, "from": from, "count": count, "bind": bind}})
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
        let_("a", shared(1, 2)), {"lookup": {"cache": "c", "tier": "t", "seq": r("a"), "per": 1, "bind": "x", "hold": "p"}},
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

#[test]
fn eviction_used_wrongly_is_refused_when_walked() {
    walk_fails(json!([stat(json!({"victim": {}}))]), "victim: outside a tier's `on_evict` body (V35)");
    walk_fails(json!([stat(json!({"tier": "used"}))]), "tier `used`: outside a tier's `evict` (V33)");
    walk_fails(json!([probe(shared(1, 1), 0, "h")]), "probe: no timeline declares cache `c` around it (V33)");
    walk_fails(json!([tl(timed(json!({})), json!([{"touch": {"cache": "c", "tier": "u", "seq": shared(1, 1), "per": 1, "from": 0, "count": 1}}]))]),
        "touch: cache `c` has no tier `u` (V33)");
    fails(json!([tl(timed(json!({"evict": {"every": 0, "when": true, "count": 1}})), json!([]))]),
        "tier `t` of cache `c`: evict every 0 ns");
    fails(json!([tl(timed(json!({"on_evict": {"lanes": 0, "body": []}})), json!([]))]), "tier `t` of cache `c`: on_evict lanes 0");
    fails(json!([tl(timed(json!({})), json!([store(shared(1, 1), -1, 1)]))]), "store: from -1");
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
        "store: tier `t` waits when full, which only a `lookup` decides before it changes anything (V33)");
    has(json!([tl(json!([{"name": "t", "unit": "bytes", "capacity": 4}]), json!([store(shared(1, 1), 0, 1)]))]),
        "store: tier `t` counts bytes, so the store needs a `size` (V33)");
    has(json!([tl(timed(json!({})), json!([{"store": {"cache": "c", "tier": "t", "seq": shared(1, 1), "per": 1, "from": 0, "count": 1, "size": 3}}]))]),
        "store: tier `t` counts entries, so the store takes no `size` (V33)");
    has(json!([tl(timed(json!({})), json!([{"probe": {"cache": "d", "tier": "t", "seq": shared(1, 1), "per": 1, "from": 0, "bind": "h"}}]))]),
        "probe: no timeline declares cache `d` around it (V33)");
    has(json!([tl(timed(json!({})), json!([{"probe": {"cache": "c", "tier": "t", "seq": 3, "per": 1, "from": 0, "bind": "h"}}]))]), "probe");
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
    let mut n = json!({"cache": "c", "tier": tier, "seq": s, "per": 1});
    for (k, v) in extra.as_object().unwrap() {
        n[k] = v.clone();
    }
    json!({"store": n})
}

fn probe_in(tier: &str, s: Value, from: i64, bind: &str) -> Value {
    json!({"probe": {"cache": "c", "tier": tier, "seq": s, "per": 1, "from": from, "bind": bind}})
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
        {"touch": {"cache": "c", "tier": "u", "seq": r("a"), "per": 1, "chunks": "pick"}},
        store_in("u", shared(2, 2), json!({"from": 0, "count": 2})),
        probe_in("u", r("a"), 0, "h0"), probe_in("u", r("a"), 2, "h2"), stat(r("h0")), stat(r("h2"))]))]);
    assert_eq!(walk(body), ["d/1", "d/1"]);
    // a parameter array names chunks too
    let doc_ = json!({"ast": "0.19", "name": "t", "params": {"some": {"default": [1, 3]}},
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
        let_("a", shared(1, 2)), {"lookup": {"cache": "c", "tier": "u", "seq": r("a"), "per": 1, "bind": "x", "hold": "p"}},
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
        "store: chunk 5, outside the sequence's 3 chunks");
    fails(json!([tl(tiers_tu(9), json!([arr(json!(-1)), store_in("u", shared(1, 3), json!({"chunks": "x"}))]))]),
        "store: chunk -1, outside the sequence's 3 chunks");
    walk_fails(json!([tl(tiers_tu(9), json!([store_in("u", shared(1, 3), json!({"chunks": "nope"}))]))]),
        "unknown parameter array `nope`");
    walk_fails(json!([tl(tiers_tu(9), json!([store_in("u", shared(1, 3), json!({}))]))]),
        "store: a range needs `from` and `count`, or `chunks` (V33)");
}

#[test]
fn the_validator_holds_taken_chunks_and_present_to_their_rules() {
    let has = |body: Value, needle: &str| {
        let e = refused(body.clone());
        assert!(e.iter().any(|m| m.contains(needle)), "wanted `{needle}` for {body}:\n{e:#?}");
    };
    let arr = json!({"array": {"name": "x", "count": 1, "index": "k", "value": 0}});
    has(json!([tl(tiers_tu(9), json!([arr, store_in("u", shared(1, 3), json!({"chunks": "x", "from": 0, "count": 1}))]))]),
        "store: `chunks` in place of `from` and `count`, not beside them (V33)");
    has(json!([tl(tiers_tu(9), json!([{"touch": {"cache": "c", "tier": "u", "seq": shared(1, 3), "per": 1, "from": 0}}]))]),
        "touch: a range needs `from` and `count`, or `chunks` (V33)");
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
        {"touch": {"cache": "c", "tier": "u", "seq": r("a"), "per": 1, "chunks": "tk3"}},
        {"loop": {"index": "j", "to": {"len": "tk3"}, "body": [stat(elem("tk3", 0))]}}]}}]))]);
    assert_eq!(refused(ok.clone()), Vec::<String>::new());
    assert_eq!(walk(ok), ["d/0", "d/0", "d/0", "d/0"]);
}
