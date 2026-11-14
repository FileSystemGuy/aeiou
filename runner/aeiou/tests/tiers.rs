//! Contract 0.18 (`DESIGN_REVIEW.md` §3.99, brief items 72 (1) and 75): a tier's victim `order`
//! as expressions over `cached`, holds (`lookup`'s `hold`, `acquire`, `release`), what a full
//! tier does (`full`), a timeline's counted `resources`, lanes that wait in planned time, and `now`.

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
    json!({"ast": "0.20", "name": "t", "params": {"cap": {"default": 4}},
           "datasets": {"d": {"layout": "whole", "access": "map", "pattern": "d/{id}", "count": 1000, "size": {"const": 1}, "seed": 1}},
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

/// The walk's own refusal, for what the validator lets through or would refuse first.
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

fn compute(ns: Value) -> Value {
    json!({"compute": {"ns": ns}})
}

fn now() -> Value {
    json!({"now": {}})
}

fn cached(a: &str) -> Value {
    json!({"cached": a})
}

/// A lookup of `s`, one key a block, in tier `t` of cache `c`.
fn lk(s: Value, bind: &str) -> Value {
    json!({"lookup": {"cache": "c", "tier": "t", "seq": s, "per": 1, "bind": bind}})
}

fn lk_hold(s: Value, bind: &str, hold: &str) -> Value {
    json!({"lookup": {"cache": "c", "tier": "t", "seq": s, "per": 1, "bind": bind, "hold": hold}})
}

fn release(hold: &str) -> Value {
    json!({"release": {"hold": hold}})
}

fn acquire(resource: &str, n: i64, hold: &str) -> Value {
    json!({"acquire": {"resource": resource, "n": n, "hold": hold}})
}

fn tier(order: Value, capacity: i64, full: &str) -> Value {
    json!([{"name": "t", "order": order, "capacity": capacity, "full": full}])
}

fn tl(tiers: Value, body: Value) -> Value {
    json!({"timeline": {"name": "tl", "caches": [{"name": "c", "tiers": tiers}], "body": body}})
}

fn tl_res(resources: Value, body: Value) -> Value {
    json!({"timeline": {"name": "tl", "resources": resources, "body": body}})
}

/// `width` lanes, each starting at its index's planned time.
fn lanes(width: i64, body: Value) -> Value {
    let mut b = vec![compute(json!({"index": "w"}))];
    b.extend(body.as_array().unwrap().iter().cloned());
    json!({"parallel": {"index": "w", "width": width, "join": false, "body": b}})
}

fn shared(id: i64, n: i64) -> Value {
    json!({"shared": {"id": id, "n": n}})
}

// ---------------------------------------------------------------- the victim order

/// FIFO, LRU, and LFU are three orders over one mechanism: the entry whose values are least
/// is evicted first.
#[test]
fn an_order_chooses_the_victim() {
    // two entries; A, B, A again, then C: FIFO evicts A (stored first), LRU B (touched longest ago)
    let run = |order: Value| {
        walk(json!([tl(tier(order, 2, "skip"), json!([
            lk(shared(1, 1), "a0"), lk(shared(2, 1), "b0"), lk(shared(1, 1), "a1"), lk(shared(3, 1), "c0"),
            lk(shared(1, 1), "a2"), stat(r("a1")), stat(r("a2"))]))]))
    };
    assert_eq!(run(json!([cached("stored")])), ["d/1", "d/0"]);
    assert_eq!(run(json!([cached("touched")])), ["d/1", "d/1"]);
    // LFU, ties to the one touched longest ago: A twice, B once; C evicts B, then B evicts C
    let body = json!([tl(tier(json!([cached("hits"), cached("touched")]), 2, "skip"), json!([
        lk(shared(1, 1), "a0"), lk(shared(1, 1), "a1"), lk(shared(2, 1), "b0"), lk(shared(3, 1), "c0"),
        lk(shared(2, 1), "b1"), lk(shared(1, 1), "a2"), lk(shared(3, 1), "c1"),
        stat(r("b1")), stat(r("a2")), stat(r("c1"))]))]);
    assert_eq!(walk(body), ["d/0", "d/1", "d/0"]);
    // an order may read parameters: the sign of `size` from one
    let body = json!([tl(tier(json!([{"mul": [cached("size"), {"param": "cap"}]}, cached("stored")]), 2, "skip"), json!([
        lk(shared(1, 1), "a0"), lk(shared(2, 1), "b0"), lk(shared(3, 1), "c0"), lk(shared(1, 1), "a1"), stat(r("a1"))]))]);
    assert_eq!(walk(body), ["d/0"]);
}

/// vLLM's free queue as an order: released least recently first, and within one release its
/// tail first (`freed`, then minus `pos`), so a prompt's head outlives its tail.
#[test]
fn a_release_frees_the_tail_first_under_vllms_order() {
    let run = |order: Value| {
        walk(json!([tl(tier(order, 4, "skip"), json!([
            let_("s", shared(1, 4)),
            lk_hold(r("s"), "h0", "p"), release("p"),
            lk(shared(2, 2), "x"),
            lk(shared(1, 4), "h1"),
            stat(r("h1"))]))]))
    };
    // two new blocks evict the released prompt's last two: its first two are still found
    assert_eq!(run(json!([cached("freed"), {"neg": cached("pos")}])), ["d/2"]);
    // least recently touched first evicts its first two instead, and nothing leads
    assert_eq!(run(json!([cached("touched")])), ["d/0"]);
}

// ---------------------------------------------------------------- holds and a full tier

/// A held entry is never a victim; `skip` leaves out what finds no room, `fail` refuses.
#[test]
fn held_entries_are_not_victims() {
    let body = json!([tl(tier(json!([cached("touched")]), 4, "skip"), json!([
        lk_hold(shared(1, 2), "a0", "p"),
        lk(shared(2, 3), "b0"),
        lk(shared(1, 2), "a1"), release("p"),
        lk(shared(2, 3), "b1"),
        stat(r("a1")), stat(r("b1"))]))]);
    // B's third block evicts its own first, A being held, so A is found whole and B's start is gone
    assert_eq!(walk(body), ["d/2", "d/0"]);
    fails(json!([tl(tier(json!([cached("touched")]), 2, "fail"), json!([
        lk_hold(shared(1, 2), "a0", "p"), lk(shared(2, 1), "b0"), release("p")]))]),
        "lookup: tier `t` of cache `c` is full: 1 more past its capacity 2 (full: fail)");
    // a lookup larger than the tier can never fit, whether it would wait or fail
    fails(json!([tl(tier(json!([cached("touched")]), 2, "wait"), json!([lk(shared(1, 3), "a")]))]),
        "lookup: 3 of new entries in tier `t` of cache `c`, past its capacity 2, can never fit");
    // a tier with no order evicts nothing: full past its capacity
    fails(json!([tl(json!([{"name": "t", "capacity": 2, "full": "fail"}]), json!([lk(shared(1, 2), "a"), lk(shared(2, 1), "b")]))]),
        "is full: 1 more past its capacity 2 (full: fail)");
    let body = json!([tl(json!([{"name": "t", "capacity": 2}]), json!([lk(shared(1, 2), "a"), lk(shared(2, 1), "b"), lk(shared(2, 1), "b1"), stat(r("b1"))]))]);
    assert_eq!(walk(body), ["d/0"]);
}

/// A lane whose lookup does not fit waits in planned time for a release, first come first.
#[test]
fn a_full_tier_makes_a_lane_wait_for_a_release() {
    let body = json!([tl(tier(json!([cached("freed")]), 4, "wait"), json!([lanes(3, json!([
        {"lookup": {"cache": "c", "tier": "t", "seq": {"fresh": 3}, "per": 1, "bind": "h", "hold": "p"}},
        stat(now()), compute(json!(10)), release("p")]))]))]);
    // three lanes at 0, 1, 2 each holding 3 of 4 entries for 10: admitted at 0, 10, 20
    assert_eq!(walk(body), ["d/0", "d/10", "d/20"]);
}

// ---------------------------------------------------------------- resources

/// A resource of C units is a queue with C servers: lanes wait in planned time, first come
/// first, and a lane behind the first waits even when its own units are free.
#[test]
fn a_resource_queues_lanes_first_come_first() {
    let body = json!([tl_res(json!([{"name": "slots", "capacity": 1}]), json!([lanes(3, json!([
        acquire("slots", 1, "s"), stat(now()), compute(json!(5)), release("s")]))]))]);
    assert_eq!(walk(body), ["d/0", "d/5", "d/10"]);
    // A holds 2 of 3 from 0 to 10; B (at 1) wants 2 and waits; C (at 2) wants 1, which is free,
    // and waits behind B; at 10 both go
    let body = json!([tl_res(json!([{"name": "u", "capacity": {"sub": [{"param": "cap"}, 1]}}]), json!([
        {"parallel": {"index": "w", "width": 3, "join": false, "body": [
            compute(json!({"index": "w"})),
            {"cond": {"if": {"eq": [{"index": "w"}, 2]},
                      "then": [acquire("u", 1, "s")],
                      "else": [acquire("u", 2, "s")]}},
            stat(now()), compute(json!(10)), release("s")]}}]))]);
    assert_eq!(walk(body), ["d/0", "d/10", "d/10"]);
    // a lane may hold several; joined lanes and releases in any order
    let body = json!([tl_res(json!([{"name": "u", "capacity": 2}, {"name": "v", "capacity": 1}]), json!([
        acquire("u", 1, "a"), acquire("v", 1, "b"), acquire("u", 1, "c"), release("a"), release("c"), release("b"),
        {"parallel": {"index": "w", "width": 2, "body": [acquire("v", 1, "b"), compute(json!(3)), release("b")]}},
        stat(now())]))]);
    assert_eq!(walk(body), ["d/6"]);
}

#[test]
fn holds_used_wrongly_are_refused_when_walked() {
    let slots = json!([{"name": "s", "capacity": 1}]);
    fails(json!([tl_res(slots.clone(), json!([acquire("s", 1, "a")]))]), "timeline `tl`: a lane ended holding `a`, which only its own `release` gives back (V34)");
    fails(json!([tl_res(slots.clone(), json!([acquire("s", 1, "a"), acquire("s", 1, "b"), release("a"), release("b")]))]),
        "timeline `tl`: lanes wait on resource `s`, and no lane that could release it is left (V34)");
    fails(json!([tl_res(slots.clone(), json!([acquire("s", 2, "a"), release("a")]))]), "acquire: 2 units of resource `s`, past its capacity 1, can never be held");
    fails(json!([tl_res(slots.clone(), json!([acquire("s", -1, "a"), release("a")]))]), "acquire: -1 units of resource `s`");
    fails(json!([tl_res(json!([{"name": "s", "capacity": 2}]), json!([acquire("s", 1, "a"), acquire("s", 1, "a"), release("a")]))]),
        "acquire: this lane holds `a` already; `release` it first (V34)");
    fails(json!([tl_res(slots.clone(), json!([acquire("s", 1, "a"), release("a"), release("a")]))]), "release: this lane holds no `a` (V34)");
    fails(json!([tl(tier(json!([cached("touched")]), 4, "skip"), json!([lk_hold(shared(1, 1), "x", "a"), lk_hold(shared(1, 1), "y", "a"), release("a")]))]),
        "lookup: this lane holds `a` already; `release` it first (V34)");
    fails(json!([tl_res(json!([{"name": "s", "capacity": -1}]), json!([]))]), "resource `s`: capacity -1");
    // what the validator refuses, refused by the walk too
    walk_fails(json!([acquire("s", 1, "a")]), "acquire: outside a timeline, where nothing is planned (V34)");
    walk_fails(json!([tl_res(slots.clone(), json!([acquire("t", 1, "a")]))]), "acquire: no timeline declares resource `t` around it (V34)");
    walk_fails(json!([release("a")]), "release: outside a timeline, where nothing is planned (V34)");
    walk_fails(json!([stat(now())]), "now: outside a timeline, where nothing is planned (V30)");
    walk_fails(json!([stat(cached("hits"))]), "cached `hits`: outside a tier's `order` (V33)");
}

/// `now` is the lane's planned time, a value of the walk as a timeline read is: `@` cannot
/// recompute it.
#[test]
fn now_is_the_planned_time_and_not_positional() {
    let body = json!([tl_res(json!([]), json!([compute(json!(7)), stat(now()), lanes(2, json!([compute(json!(3)), stat(now())]))]))]);
    assert_eq!(walk(body), ["d/7", "d/10", "d/11"]);
    walk_fails(json!([tl_res(json!([]), json!([{"loop": {"index": "i", "to": 3, "body": [
        let_("x", now()),
        stat(json!({"cond": {"if": {"gt": [{"index": "i"}, 0]}, "then": {"at": {"ref": "x", "index": {"sub": [{"index": "i"}, 1]}}}, "else": 0}}))]}}]))]),
        "(V30)");
}

// ---------------------------------------------------------------- the checkers (V33, V34, V30)

#[test]
fn the_validator_holds_orders_holds_and_resources_to_their_rules() {
    let has = |body: Value, needle: &str| {
        let e = refused(body.clone());
        assert!(e.iter().any(|m| m.contains(needle)), "wanted `{needle}` for {body}:\n{e:#?}");
    };
    has(json!([tl(tier(json!([r("x")]), 4, "skip"), json!([]))]), "tier `t`: its `order` reads `ref`; an order reads only parameters, literals, and `cached` (V33)");
    has(json!([tl(tier(json!([{"add": [cached("pos"), {"draw": {"const": 1}}]}]), 4, "skip"), json!([]))]), "its `order` reads `draw`");
    has(json!([tl(tier(json!([{"cond": {"if": true, "then": now(), "else": 1}}]), 4, "skip"), json!([]))]), "its `order` reads `now`");
    has(json!([tl(tier(json!([{"cond": {"if": {"gt": [cached("pos"), 1]}, "then": 1, "else": {"neg": cached("hits")}}}, "x"]), 4, "skip"), json!([]))]), "expected an integer");
    has(json!([stat(cached("pos"))]), "cached `pos`: outside a tier's `order` (V33)");
    has(json!([stat(now())]), "now: outside a timeline, where nothing is planned (V30)");
    has(json!([acquire("s", 1, "a")]), "acquire: outside a timeline, where nothing is planned (V34)");
    has(json!([release("a")]), "release: outside a timeline, where nothing is planned (V34)");
    has(json!([tl_res(json!([]), json!([acquire("s", 1, "a"), release("a")]))]), "acquire: no timeline declares resource `s` around it (V34)");
    has(json!([tl_res(json!([{"name": "s", "capacity": 1}]), json!([release("b")]))]), "release: no `acquire` or `lookup` of timeline `tl` holds `b` (V34)");
    has(json!([tl_res(json!([{"name": "s", "capacity": 1}, {"name": "s", "capacity": 2}]), json!([]))]), "resource `s` declared twice (V34)");
    has(json!([tl_res(json!([{"name": "s", "capacity": "x"}]), json!([]))]), "expected an integer");
    has(json!([tl_res(json!([{"name": "s", "capacity": 1}]), json!([acquire("s", 1, "a b"), release("a b")]))]), "not an identifier");
    has(json!([tl(tier(json!([cached("touched")]), 4, "skip"), json!([lk_hold(shared(1, 1), "x", "a-b"), release("a-b")]))]), "not an identifier");
    let ok = json!([{"timeline": {"name": "tl", "resources": [{"name": "slots", "capacity": {"param": "cap"}}],
        "caches": [{"name": "c", "tiers": tier(json!([cached("freed"), {"neg": cached("pos")}]), 4, "wait")}],
        "body": [lanes(2, json!([
            acquire("slots", 1, "s"), lk_hold(shared(1, 2), "h", "p"), stat(r("h")), stat(now()),
            {"cond": {"if": {"gt": [r("h"), 0]}, "then": [release("p")], "else": [release("p")]}}, release("s")]))]}}]);
    assert_eq!(refused(ok.clone()), Vec::<String>::new());
    assert_eq!(walk(ok), ["d/0", "d/0", "d/2", "d/1"]);
}
