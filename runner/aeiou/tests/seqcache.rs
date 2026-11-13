//! Contract 0.16 (`DESIGN_REVIEW.md` §3.99, brief items 71 and 75): the `seq` value (`fresh`,
//! `shared`, `prefix`, `cat`, `seq_len`, `chunk_key`), a timeline's `caches`, and `lookup`, as the
//! dry run walks them.

use aeiou::eval::{build_model, Config, Params};
use aeiou::vm::{actor_counts, run_actor, Control, Op, OpCtx, Sink};
use serde_json::{json, Value};

/// The ops (by path) in the order the walk gave them; computes are left out.
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
    json!({"ast": "0.17", "name": "t", "params": {},
           "datasets": {"d": {"layout": "whole", "access": "map", "pattern": "d/{id}", "count": 1000, "size": {"const": 1}, "seed": 1}},
           "actors": {"gpu": {"body": body}}}).to_string()
}

fn walk_with(body: Value, seed: u64) -> Result<Vec<String>, String> {
    let text = doc(body);
    let loaded = aeiou::load_str(&text).map_err(|e| format!("{e:#}"))?;
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
    walk_with(body, 1).unwrap_or_else(|e| panic!("{e}"))
}

fn fails(body: Value, needle: &str) {
    match walk_with(body, 1) {
        Ok(ops) => panic!("walked, wanted `{needle}`: {ops:?}"),
        Err(e) => assert!(e.contains(needle), "wanted `{needle}` in:\n{e}"),
    }
}

/// The messages `aeiou check` gives a document.
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

fn r(name: &str) -> Value {
    json!({"ref": name})
}

fn let_(name: &str, value: Value) -> Value {
    json!({"let": {"name": name, "value": value}})
}

fn fresh(n: i64) -> Value {
    json!({"fresh": n})
}

fn shared(id: i64, n: i64) -> Value {
    json!({"shared": {"id": id, "n": n}})
}

fn prefix(s: Value, k: i64) -> Value {
    json!({"prefix": [s, k]})
}

fn cat(a: Value, b: Value) -> Value {
    json!({"cat": [a, b]})
}

fn key(s: Value, c: i64) -> Value {
    json!({"chunk_key": {"seq": s, "chunk": c, "per": 4}})
}

/// 1 when two expressions are equal, else 0, as an op's id.
fn same(a: Value, b: Value) -> Value {
    stat(json!({"cond": {"if": {"eq": [a, b]}, "then": 1, "else": 0}}))
}

/// A key as an op's id, small enough for the test dataset.
fn small(k: Value) -> Value {
    stat(json!({"mod": [k, 997]}))
}

fn lookup(seq: Value, bind: &str) -> Value {
    json!({"lookup": {"cache": "lm", "tier": "disk", "seq": seq, "per": 4, "bind": bind}})
}

fn tl(tiers: Value, body: Value) -> Value {
    json!({"timeline": {"name": "tl", "caches": [{"name": "lm", "tiers": tiers}], "body": body}})
}

fn unbounded() -> Value {
    json!([{"name": "disk", "evict": "none"}])
}

// ---------------------------------------------------------------- the seq value

#[test]
fn a_seq_is_built_by_fresh_prefix_and_cat() {
    let body = json!([
        let_("a", fresh(10)),
        let_("b", cat(prefix(r("a"), 6), fresh(3))),
        stat(json!({"seq_len": r("a")})), stat(json!({"seq_len": r("b")})),
        stat(json!({"seq_len": cat(r("a"), r("b"))})), stat(json!({"seq_len": prefix(r("a"), 0)})),
        stat(json!({"seq_len": shared(3, 7)}))]);
    assert_eq!(walk(body), ["d/10", "d/9", "d/19", "d/0", "d/7"]);
}

/// A chunk's key is the chained hash of every block up to its end, so two sequences give one
/// key exactly when they hold the same blocks up to it, however they were put together.
#[test]
fn chunk_keys_are_equal_exactly_when_the_prefixes_are() {
    let body = json!([
        let_("a", fresh(10)),
        let_("b", cat(prefix(r("a"), 6), fresh(3))),
        let_("c", prefix(cat(prefix(r("a"), 8), fresh(5)), 8)),
        same(key(r("a"), 0), key(r("b"), 0)),
        same(key(r("a"), 1), key(r("b"), 1)),
        same(key(r("a"), 1), key(r("c"), 1)),
        same(key(r("a"), 0), key(r("a"), 1)),
        same(key(shared(7, 4), 0), key(shared(7, 8), 0)),
        same(key(shared(7, 8), 1), key(cat(prefix(shared(7, 6), 6), prefix(cat(prefix(shared(7, 6), 0), shared(7, 8)), 2)), 1)),
        same(key(shared(7, 4), 0), key(shared(8, 4), 0)),
        same(key(shared(7, 4), 0), key(r("a"), 0)),
        stat(json!({"cond": {"if": {"ge": [key(r("a"), 0), 0]}, "then": 1, "else": 0}}))]);
    assert_eq!(walk(body), ["d/1", "d/0", "d/1", "d/0", "d/1", "d/0", "d/0", "d/0", "d/1"]);
}

/// `fresh` is positional, as a draw is: its blocks are new at every position and the same when
/// `@` recomputes them; another seed gives other blocks, and `shared` the same in every run.
#[test]
fn fresh_blocks_are_positional_and_shared_ones_are_common() {
    let prev = |n: &str, init: Value| json!({"prev": {"ref": n, "init": init}});
    let body = json!([{"loop": {"index": "i", "to": 4, "body": [
        let_("x", fresh(4)),
        let_("k", key(r("x"), 0)),
        same(r("k"), prev("k", json!(-1))),
        stat(json!({"cond": {"if": {"gt": [{"index": "i"}, 0]}, "then": {"cond": {"if": {"eq": [key(json!({"at": {"ref": "x", "index": {"sub": [{"index": "i"}, 1]}}}), 0), prev("k", json!(-1))]}, "then": 1, "else": 0}}, "else": 1}}))]}}]);
    assert_eq!(walk(body), ["d/0", "d/1", "d/0", "d/1", "d/0", "d/1", "d/0", "d/1"]);

    let keys = json!([{"loop": {"index": "i", "to": 6, "body": [small(key(fresh(4), 0)), small(key(shared(1, 4), 0))]}}]);
    let (one, two) = (walk_with(keys.clone(), 1).unwrap(), walk_with(keys, 2).unwrap());
    let col = |w: &[String], k: usize| w.iter().skip(k).step_by(2).cloned().collect::<Vec<_>>();
    assert_ne!(col(&one, 0), col(&two, 0), "fresh blocks follow the seed");
    assert_eq!(col(&one, 1), col(&two, 1), "shared blocks do not");

    // a lane's fresh blocks are its own; shared ones are every lane's
    let lanes = json!([tl(unbounded(), json!([{"parallel": {"index": "w", "width": 2, "body": [
        let_("x", fresh(4)), {"publish": {"timeline": "tl", "values": {"x": r("x"), "s": shared(2, 4)}}}]}},
        same(key(json!({"entry": {"timeline": "tl", "index": 0, "key": "x"}}), 0), key(json!({"entry": {"timeline": "tl", "index": 1, "key": "x"}}), 0)),
        same(key(json!({"entry": {"timeline": "tl", "index": 0, "key": "s"}}), 0), key(json!({"entry": {"timeline": "tl", "index": 1, "key": "s"}}), 0))]))]);
    assert_eq!(walk(lanes), ["d/0", "d/1"]);
}

#[test]
fn a_seq_used_wrongly_is_refused_when_walked() {
    fails(json!([stat(json!({"seq_len": prefix(fresh(10), 11)}))]), "prefix: 11 blocks of a sequence of 10");
    fails(json!([stat(json!({"seq_len": prefix(fresh(10), -1)}))]), "prefix: -1 blocks of a sequence of 10");
    fails(json!([stat(key(fresh(10), 2))]), "chunk_key: chunk 2 of a sequence of 2 whole chunks of 4");
    fails(json!([stat(json!({"chunk_key": {"seq": fresh(10), "chunk": 0, "per": 0}}))]), "chunk_key: `per` 0, not positive");
    fails(json!([stat(json!({"seq_len": fresh(-1)}))]), "fresh: -1 blocks");
    fails(json!([stat(json!({"seq_len": shared(1, -2)}))]), "shared: -2 blocks");
    fails(json!([stat(json!({"add": [fresh(1), 1]}))]), "got a seq (V27)");
    fails(json!([stat(json!({"seq_len": 3}))]), "seq_len: expected a seq, got int");
}

// ---------------------------------------------------------------- caches and lookup

/// A `lookup` binds the leading chunks found, touches them, and stores the rest.
#[test]
fn a_lookup_finds_the_longest_stored_prefix_and_stores_the_rest() {
    let body = json!([tl(unbounded(), json!([
        let_("s0", fresh(10)), lookup(r("s0"), "h0"),
        let_("s1", cat(r("s0"), fresh(6))), lookup(r("s1"), "h1"),
        let_("s2", prefix(r("s1"), 7)), lookup(r("s2"), "h2"),
        lookup(fresh(8), "h3"),
        lookup(prefix(r("s0"), 3), "h4"),
        stat(r("h0")), stat(r("h1")), stat(r("h2")), stat(r("h3")), stat(r("h4"))]))]);
    assert_eq!(walk(body), ["d/0", "d/2", "d/1", "d/0", "d/0"]);
}

/// An `lru` tier evicts the entry used longest ago when an insert would pass its capacity;
/// a hit is a use.
#[test]
fn an_lru_tier_evicts_the_entry_used_longest_ago() {
    let lru = json!([{"name": "disk", "evict": "lru", "capacity": 3}]);
    // A, B two chunks each in three entries: each store evicts the other's first chunk, so
    // alternating they always miss; A twice running hits whole the second time
    let body = json!([tl(lru.clone(), json!([
        let_("a", fresh(8)), let_("b", fresh(8)),
        lookup(r("a"), "h0"), lookup(r("b"), "h1"), lookup(r("a"), "h2"), lookup(r("b"), "h3"), lookup(r("a"), "h4"), lookup(r("a"), "h5"),
        stat(r("h0")), stat(r("h1")), stat(r("h2")), stat(r("h3")), stat(r("h4")), stat(r("h5"))]))]);
    assert_eq!(walk(body), ["d/0", "d/0", "d/0", "d/0", "d/0", "d/2"]);
    // a hit touches: A hit again after C makes C the oldest, so D evicts C, not A
    let body = json!([tl(lru, json!([
        let_("a", fresh(8)), let_("c", fresh(4)), let_("d", fresh(4)),
        lookup(r("a"), "h0"), lookup(r("c"), "h1"), lookup(r("a"), "h2"), lookup(r("d"), "h3"), lookup(r("a"), "h4"), lookup(r("c"), "h5"),
        stat(r("h2")), stat(r("h4")), stat(r("h5"))]))]);
    assert_eq!(walk(body), ["d/2", "d/2", "d/0"]);
}

/// A tier counted in bytes takes each lookup's `size`, the bytes of one chunk.
#[test]
fn a_tier_in_bytes_evicts_by_the_sizes_stored() {
    let tiers = json!([{"name": "disk", "evict": "lru", "capacity": 10, "unit": "bytes"}]);
    let lk = |s: Value, b: &str| json!({"lookup": {"cache": "lm", "tier": "disk", "seq": s, "per": 4, "bind": b, "size": 4}});
    let body = json!([tl(tiers, json!([
        let_("a", fresh(8)), let_("b", fresh(4)),
        lk(r("a"), "h0"), lk(r("a"), "h1"), lk(r("b"), "h2"), lk(r("a"), "h3"),
        stat(r("h1")), stat(r("h3"))]))]);
    assert_eq!(walk(body), ["d/2", "d/0"]);
    // an entry larger than the tier is not stored
    let tiers = json!([{"name": "disk", "evict": "lru", "capacity": 3, "unit": "bytes"}]);
    let body = json!([tl(tiers, json!([lk(shared(1, 4), "h0"), lk(shared(1, 4), "h1"), stat(r("h1"))]))]);
    assert_eq!(walk(body), ["d/0"]);
}

/// The cache is the timeline's: every lane's lookups meet it in planned order, and the next
/// timeline starts with it empty.
#[test]
fn the_cache_is_shared_by_the_lanes_in_planned_order() {
    let body = json!([
        tl(unbounded(), json!([
            {"parallel": {"index": "w", "width": 1, "join": false, "body": [
                compute(2), lookup(shared(1, 8), "a0"), compute(6), lookup(shared(2, 8), "a1")]}},
            compute(5), lookup(shared(1, 8), "m0"), lookup(shared(2, 8), "m1"), stat(r("m0")), stat(r("m1"))])),
        tl(unbounded(), json!([lookup(shared(1, 8), "n0"), stat(r("n0"))]))]);
    // the lane stores shared(1) at 2 and shared(2) at 8; main looks at 5
    assert_eq!(walk(body), ["d/2", "d/0", "d/0"]);
}

/// A lookup's value is the walk's, as a timeline read is: `@` cannot recompute it.
#[test]
fn at_may_not_reach_a_lookup() {
    fails(json!([tl(unbounded(), json!([{"loop": {"index": "i", "to": 3, "body": [
        lookup(fresh(4), "h"),
        let_("x", json!({"add": [r("h"), 1]})),
        let_("y", json!({"cond": {"if": {"gt": [{"index": "i"}, 0]}, "then": {"at": {"ref": "x", "index": {"sub": [{"index": "i"}, 1]}}}, "else": 0}})),
        stat(r("y"))]}}]))]), "under `@`");
}

// ---------------------------------------------------------------- the checkers (V33)

#[test]
fn the_validator_holds_seqs_caches_and_lookups_to_their_rules() {
    let has = |body: Value, needle: &str| {
        let e = refused(body.clone());
        assert!(e.iter().any(|m| m.contains(needle)), "wanted `{needle}` for {body}:\n{e:#?}");
    };
    has(json!([lookup(fresh(4), "h")]), "lookup: no timeline declares cache `lm` around it (V33)");
    has(json!([{"timeline": {"name": "tl", "body": [lookup(fresh(4), "h")]}}]), "lookup: no timeline declares cache `lm` around it (V33)");
    has(json!([tl(json!([{"name": "mem", "evict": "none"}]), json!([lookup(fresh(4), "h")]))]), "lookup: cache `lm` has no tier `disk` (V33)");
    has(json!([tl(json!([{"name": "disk", "evict": "lru"}]), json!([]))]), "tier `disk`: `lru` needs a `capacity` (V33)");
    has(json!([tl(json!([{"name": "disk", "evict": "none", "capacity": 4}]), json!([]))]), "tier `disk`: `none` keeps every entry, so it takes no `capacity` (V33)");
    has(json!([tl(json!([{"name": "disk", "evict": "none"}, {"name": "disk", "evict": "none"}]), json!([]))]), "tier `disk` declared twice in cache `lm` (V33)");
    has(json!([tl(json!([]), json!([]))]), "cache `lm` has no tiers (V33)");
    has(json!([{"timeline": {"name": "tl", "caches": [{"name": "lm", "tiers": unbounded()}, {"name": "lm", "tiers": unbounded()}], "body": []}}]),
        "cache `lm` declared twice (V33)");
    has(json!([tl(json!([{"name": "disk", "evict": "lru", "capacity": 8, "unit": "bytes"}]), json!([lookup(fresh(4), "h")]))]),
        "lookup: tier `disk` counts bytes, so the lookup needs a `size` (V33)");
    has(json!([tl(unbounded(), json!([{"lookup": {"cache": "lm", "tier": "disk", "seq": fresh(4), "per": 4, "bind": "h", "size": 4}}]))]),
        "lookup: tier `disk` counts entries, so the lookup takes no `size` (V33)");
    // `@` naming a lookup's binding, `prev` of one
    has(json!([tl(unbounded(), json!([{"loop": {"index": "i", "to": 2, "body": [
        lookup(fresh(4), "h"),
        let_("y", json!({"cond": {"if": {"gt": [{"index": "i"}, 0]}, "then": {"at": {"ref": "h", "index": {"sub": [{"index": "i"}, 1]}}}, "else": 0}}))]}}]))]), "(V33)");
    has(json!([tl(unbounded(), json!([{"loop": {"index": "i", "to": 2, "body": [
        lookup(fresh(4), "h"), let_("y", json!({"prev": {"ref": "h", "init": 0}}))]}}]))]), "prev: `h` is bound by a `lookup`, not a `let` (V31)");
    // a seq where a number goes, a number where a seq goes
    has(json!([stat(json!({"add": [fresh(1), 1]}))]), "seq");
    has(json!([stat(json!({"seq_len": 3}))]), "seq_len: expected a seq, got int");
    has(json!([stat(json!({"seq_len": prefix(json!(3), 1)}))]), "prefix: expected a seq, got int");
    has(json!([stat(json!({"seq_len": cat(fresh(1), json!(2))}))]), "cat: expected a seq, got int");
    has(json!([tl(unbounded(), json!([lookup(json!(5), "h")]))]), "lookup: expected a seq, got int");
    let ok = json!([tl(unbounded(), json!([
        {"loop": {"index": "i", "to": 2, "body": [
            let_("s", cat(prefix(json!({"prev": {"ref": "s", "init": shared(1, 4)}}), 4), fresh(4))),
            lookup(r("s"), "h"),
            {"publish": {"timeline": "tl", "values": {"p": r("s")}}},
            {"loop": {"index": "c", "from": r("h"), "to": {"div": [{"seq_len": r("s")}, 4]}, "body": [small(key(r("s"), 0))]}}]}}]))]);
    assert_eq!(refused(ok), Vec::<String>::new());
}

// ---------------------------------------------------------------- the walk's own refusals

/// A document not checked at load is checked again with its parameters (`Params::new`), and the
/// walk refuses the rest: either way the same words.
fn walk_unchecked(body: Value) -> Result<Vec<String>, String> {
    let ast = aeiou::ast::parse(&doc(body)).map_err(|e| format!("{e:#}"))?;
    aeiou::sites::annotate(&ast);
    let cfg = Config { seed: 1, gpus: 1, overrides: vec![], sets: vec![], st_blksize: Default::default() };
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

#[test]
fn the_walk_refuses_what_the_validator_would() {
    let no = |body: Value, needle: &str| match walk_unchecked(body.clone()) {
        Ok(ops) => panic!("walked {body}, wanted `{needle}`: {ops:?}"),
        Err(e) => assert!(e.contains(needle), "wanted `{needle}` in:\n{e}"),
    };
    no(json!([lookup(fresh(4), "h")]), "lookup: no timeline declares cache `lm` around it (V33)");
    no(json!([tl(json!([{"name": "mem", "evict": "none"}]), json!([lookup(fresh(4), "h")]))]), "lookup: cache `lm` has no tier `disk` (V33)");
    no(json!([tl(json!([{"name": "disk", "evict": "lru", "capacity": -1}]), json!([]))]), "tier `disk` of cache `lm`: capacity -1");
    no(json!([tl(unbounded(), json!([{"lookup": {"cache": "lm", "tier": "disk", "seq": fresh(4), "per": 0, "bind": "h"}}]))]), "lookup: `per` 0, not positive");
    no(json!([tl(unbounded(), json!([lookup(json!(3), "h")]))]), "lookup: expected a seq, got int");
    no(json!([stat(json!({"seq_len": cat(fresh(1), json!(2))}))]), "cat: expected a seq, got int");
    no(json!([stat(json!({"seq_len": prefix(json!(3), 1)}))]), "prefix: expected a seq, got int");
    no(json!([stat(json!({"chunk_key": {"seq": 3, "chunk": 0, "per": 4}}))]), "chunk_key: expected a seq, got int");
}
