//! The AST types, the validator, the model, and the VM where the other suites do not reach
//! them (RC1 review, coverage lane C1): the shared refusal cases both checkers are held to, and
//! the op streams and refusals of the statements, expressions, and handles the committed
//! abstracts do not use.

use aeiou::ast::{Advice, IoctlRequest, Whence};
use aeiou::eval::{build_model, Config, Model, ParamSet, Params};
use aeiou::vm::{actor_counts, run_actor, Control, Op, OpCtx, Sink};
use serde_json::{json, Value};

// ---------------------------------------------------------------- the shared refusal cases

/// `tests/refusals.json`, held to check.py by `builder/tests/test_builder.py`: each case's
/// errors as the runner gives them, path and message (a case the JSON Schema refuses in
/// check.py carries the runner's own words for the rule, and the schema's paths, unused here).
#[test]
fn the_shared_refusal_cases_are_refused_as_check_py_refuses_them() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/refusals.json")).unwrap();
    let cases: Vec<Value> = serde_json::from_str(&text).unwrap();
    assert!(cases.len() > 100);
    let mut wrong = Vec::new();
    for c in &cases {
        let why = c["why"].as_str().unwrap();
        let name = c["ast"]["name"].as_str().unwrap();
        let want: Vec<String> = c["errors"].as_array().unwrap().iter().map(|e| format!("{name}: {}", e.as_str().unwrap())).collect();
        let got: Vec<String> = match aeiou::load_str(&c["ast"].to_string()) {
            Ok(_) => Vec::new(),
            Err(e) => {
                let msg = format!("{e:#}");
                let body = msg.strip_prefix("invalid abstract:\n").unwrap_or_else(|| panic!("{why}: not a validator refusal:\n{msg}"));
                body.lines().map(|l| l.trim().to_string()).collect()
            }
        };
        if got != want {
            wrong.push(format!("{why}:\n  got  {got:?}\n  want {want:?}"));
        }
    }
    assert!(wrong.is_empty(), "{} of {} cases:\n{}", wrong.len(), cases.len(), wrong.join("\n"));
}

// ---------------------------------------------------------------- the AST types

#[test]
fn op_codes_come_back_as_the_values_they_were_made_from() {
    for w in [Whence::SET, Whence::CUR, Whence::END] {
        assert_eq!(Whence::from_code(w as u64), w);
    }
    for r in [IoctlRequest::TCGETS, IoctlRequest::FIONREAD, IoctlRequest::BLKGETSIZE64] {
        assert_eq!(IoctlRequest::from_code(r as u64), r);
    }
    for a in [Advice::NORMAL, Advice::RANDOM, Advice::SEQUENTIAL, Advice::WILLNEED, Advice::DONTNEED, Advice::NOREUSE] {
        assert_eq!(Advice::from_code(a as u64), a);
    }
    // a code no variant has is the default (what `aux` of another kind of op would carry)
    assert_eq!((Whence::from_code(9), IoctlRequest::from_code(9), Advice::from_code(9)), (Whence::SET, IoctlRequest::TCGETS, Advice::NORMAL));
}

#[test]
fn an_ast_of_another_contract_version_is_refused() {
    let e = aeiou::ast::parse(r#"{"ast": "0.9", "name": "t", "actors": {}}"#).unwrap_err();
    assert_eq!(e.to_string(), "ast version `0.9` is not `0.12`");
}

/// A cloned draw keeps the site `sites::annotate` gave it, so a clone draws what the original does.
#[test]
fn a_cloned_draw_keeps_its_site() {
    let e: aeiou::ast::Expr = serde_json::from_str(r#"{"draw": {"uniform": {"lo": 0, "hi": 9}}}"#).unwrap();
    let aeiou::ast::Expr::Node(n) = &e else { panic!() };
    let aeiou::ast::ExprNode::Draw(d) = n.as_ref() else { panic!() };
    d.site.set(0xfeed);
    let c = e.clone();
    let aeiou::ast::Expr::Node(n) = &c else { panic!() };
    let aeiou::ast::ExprNode::Draw(d) = n.as_ref() else { panic!() };
    assert_eq!(d.site.get(), 0xfeed);
}

// ---------------------------------------------------------------- walking a document

/// Every op and control event of every actor instance, in walk order, one line each.
#[derive(Default)]
struct Rec(Vec<String>);

impl<'m, 'a> Sink<'m, 'a> for Rec {
    fn op(&mut self, op: &Op, ctx: &OpCtx) -> anyhow::Result<()> {
        let phase = ctx.phase.map(|p| format!("[{p}] ")).unwrap_or_default();
        let to = op.path2.map(|p| format!(" -> {p}")).unwrap_or_default();
        self.0.push(format!("{phase}{} {}{to} off={} len={} bytes={} aux={}", op.kind.name(), op.path, op.offset, op.len, op.bytes, op.aux));
        Ok(())
    }
    fn control(&mut self, c: Control<'a>, _ctx: &OpCtx) -> anyhow::Result<()> {
        self.0.push(match c {
            Control::Compute { ns } => format!("compute {ns}"),
            other => format!("{other:?}"),
        });
        Ok(())
    }
}

fn doc(params: Value, datasets: Value, namespaces: Value, body: Value) -> String {
    json!({"ast": "0.12", "name": "t", "params": params, "datasets": datasets, "namespaces": namespaces, "actors": {"gpu": {"body": body}}}).to_string()
}

/// One plain dataset `d` of ten one-byte files.
fn d() -> Value {
    json!({"d": {"layout": "whole", "access": "map", "pattern": "d/{id}", "count": 10, "size": {"const": 1}, "seed": 1}})
}

fn with_model<T>(text: &str, gpus: i64, seed: u64, overrides: &[(&str, &str)], f: impl FnOnce(&Model) -> Result<T, String>) -> Result<T, String> {
    let loaded = aeiou::load_str(text).map_err(|e| format!("{e:#}"))?;
    let cfg = Config { seed, gpus, overrides: overrides.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), sets: vec![], st_blksize: Default::default() };
    let params = Params::new(&loaded.ast, &cfg).map_err(|e| format!("{e:#}"))?;
    let model = build_model(&loaded.ast, &cfg, &params).map_err(|e| format!("{e:#}"))?;
    f(&model)
}

fn walk_with(text: &str, gpus: i64, seed: u64) -> Result<Vec<String>, String> {
    with_model(text, gpus, seed, &[], |model| {
        let mut out = Vec::new();
        for (name, count) in actor_counts(model).map_err(|e| format!("{e:#}"))? {
            for a in 0..count {
                out.extend(run_actor(model, name, a, count, Rec::default()).map_err(|e| format!("{e:#}"))?.0);
            }
        }
        Ok(out)
    })
}

fn walk(text: &str) -> Vec<String> {
    walk_with(text, 1, 1).unwrap_or_else(|e| panic!("{e}"))
}

/// The walk's error, which must name `needle`.
fn fails(text: &str, needle: &str) {
    match walk_with(text, 1, 1) {
        Ok(ops) => panic!("walked, wanted `{needle}`: {ops:?}"),
        Err(e) => assert!(e.contains(needle), "wanted `{needle}` in:\n{e}"),
    }
}

fn computes(body: Value, params: Value) -> Vec<String> {
    walk(&doc(params, d(), json!({}), body))
}

fn stat(ds: &str, id: i64) -> Value {
    json!({"stat": {"file": {"file": {"dataset": ds, "id": id}}}})
}

// ---------------------------------------------------------------- statements

/// `choose` takes an arm by a positional draw at its site: in proportion to the weights, never
/// an arm of weight 0, the same arms again for the same seed, others for another.
#[test]
fn choose_takes_its_arms_by_weight_and_position() {
    let body = json!([{"loop": {"index": "i", "to": 400, "body": [{"choose": {"arms": [
        {"weight": 3, "body": [stat("d", 0)]}, {"weight": 1, "body": [stat("d", 1)]}, {"weight": 0, "body": [stat("d", 2)]}]}}]}}]);
    let text = doc(json!({}), d(), json!({}), body);
    let ops = walk(&text);
    let n = |p: &str| ops.iter().filter(|o| o.starts_with(&format!("stat {p} "))).count();
    assert_eq!(ops.len(), 400);
    assert_eq!(n("d/2"), 0, "an arm of weight 0");
    assert!((260..340).contains(&n("d/0")), "{} of 400 at weight 3 of 4", n("d/0"));
    assert_eq!(walk_with(&text, 1, 1).unwrap(), ops, "the same seed");
    assert_ne!(walk_with(&text, 1, 2).unwrap(), ops, "another seed");
    let zero = json!([{"choose": {"arms": [{"weight": 0, "body": [stat("d", 0)]}]}}]);
    fails(&doc(json!({}), d(), json!({}), zero), "choose: total weight is zero");
}

/// The ops on a handle the committed abstracts do not issue: `ftruncate`, `fallocate` (with
/// and without an offset), `fdatasync`, `rmdir`, and `rename`, each with the fields it carries.
#[test]
fn ftruncate_fallocate_fdatasync_and_rmdir_carry_their_arguments() {
    let ns = json!({"o": {"pattern": "o/{k}", "fields": {"k": "int"}, "size": 100, "seed": 2}});
    let o = |k: i64| json!({"object": {"namespace": "o", "fields": {"k": k}}});
    let body = json!([
        {"open": {"file": o(1), "flags": ["WRONLY", "CREAT"], "mode": 0o640}},
        {"ftruncate": {"file": o(1), "len": 10}},
        {"fallocate": {"file": o(1), "offset": 4, "len": 8}},
        {"fallocate": {"file": o(1), "len": 3}},
        {"fdatasync": {"file": o(1)}},
        {"close": {"file": o(1)}},
        {"rename": {"from": o(1), "to": o(2)}},
        {"mkdir": {"dir": o(3)}},
        {"rmdir": {"dir": o(3)}}
    ]);
    let ops = walk(&doc(json!({}), json!({}), ns.clone(), body));
    assert_eq!(&ops[1..], [
        "ftruncate o/1 off=0 len=10 bytes=0 aux=0",
        "fallocate o/1 off=4 len=8 bytes=0 aux=0",
        "fallocate o/1 off=0 len=3 bytes=0 aux=0",
        "fdatasync o/1 off=0 len=0 bytes=0 aux=0",
        "close o/1 off=0 len=0 bytes=0 aux=0",
        "rename o/1 -> o/2 off=0 len=0 bytes=0 aux=0",
        &format!("mkdir o/3 off=0 len=0 bytes=0 aux={}", 0o777),
        "rmdir o/3 off=0 len=0 bytes=0 aux=0",
    ]);
    assert!(ops[0].starts_with("open o/1 ") && ops[0].ends_with(&format!("aux={}", aeiou::vm::flag_bits(&[aeiou::ast::OpenFlag::WRONLY, aeiou::ast::OpenFlag::CREAT]) | (0o640 << 32))), "{}", ops[0]);
    for (op, what) in [(json!({"ftruncate": {"file": o(1), "len": 1}}), "ftruncate"), (json!({"fallocate": {"file": o(1), "len": 1}}), "fallocate")] {
        fails(&doc(json!({}), json!({}), ns.clone(), json!([op])), &format!("{what} on `o/1`, which this actor has not opened"));
    }
}

#[test]
fn a_loop_a_fork_or_a_phase_out_of_range_is_refused_when_walked() {
    let walked = |body: Value| doc(json!({}), d(), json!({}), body);
    fails(&walked(json!([{"loop": {"index": "i", "to": 4, "step": 0, "body": []}}])), "loop `i`: step 0 must be positive");
    fails(&walked(json!([{"loop": {"index": "i", "from": i64::MIN, "to": i64::MAX, "body": []}}])), "is more than 2^63-1 iterations");
    fails(&walked(json!([{"parallel": {"index": "p", "width": {"neg": 1}, "body": []}}])), "parallel `p`: width -1");
    fails(&walked(json!([{"loader": {"name": "q", "index": "b", "workers": 0, "prefetch": 1, "batches": 1, "body": []}}])), "loader: workers 0, prefetch 1, batches 1");
    fails(&walked(json!([{"phase": {"name": true, "body": []}}])), "phase name is bool");
    // a phase named by an integer is labelled with it
    assert_eq!(walk(&walked(json!([{"phase": {"name": 7, "body": [stat("d", 0)]}}]))), ["[7] stat d/0 off=0 len=0 bytes=0 aux=0"]);
    // a document loaded from text has no directory to find a trace in
    fails(&walked(json!([{"trace": {"file": "x.trace", "sha256": "0".repeat(64)}}])), "trace `x.trace`: not loaded");
}

// ---------------------------------------------------------------- expressions

#[test]
fn sum_len_elem_and_neg_of_parameters() {
    let params = json!({"arr": {"default": [1, 2, 3]}, "farr": {"default": [1.5, 2]}, "big": {"default": [i64::MAX, 1]}, "low": {"default": i64::MIN}});
    let c = |ns: Value| json!({"compute": {"ns": ns}});
    let got = computes(json!([c(json!({"sum": "arr"})), c(json!({"sum": "farr"})), c(json!({"len": "arr"})), c(json!({"elem": {"array": "arr", "index": 2}})),
        c(json!({"neg": {"neg": 5}})), c(json!({"neg": -2.5})), c(json!({"actor": "count"}))]), params.clone());
    // 3.5 and 2.5 are rounded half away from zero, as every float read as an integer is
    assert_eq!(got, ["compute 6", "compute 4", "compute 3", "compute 3", "compute 5", "compute 3", "compute 1"]);
    let refused = |ns: Value, needle: &str| fails(&doc(params.clone(), d(), json!({}), json!([c(ns)])), needle);
    refused(json!({"sum": "big"}), "sum: the integers of `big` pass 2^63-1");
    refused(json!({"neg": {"param": "low"}}), "neg overflow");
    refused(json!({"elem": {"array": "arr", "index": 3}}), "elem: index 3 outside parameter array `arr` of 3");
    // `{actor: count}` is the instance count of the actor template
    assert_eq!(walk_with(&doc(json!({}), d(), json!({}), json!([c(json!({"actor": "count"}))])), 3, 1).unwrap(), ["compute 3"; 3]);
}

/// Values whose kind the document does not show (an element of a mixed array) are judged by
/// the run as V27 judges the ones it does show.
#[test]
fn a_value_of_the_wrong_kind_the_document_does_not_show_is_refused_when_walked() {
    let params = json!({"mix": {"default": [1, "a"]}});
    let ns = json!({"o": {"pattern": "o/{k}", "fields": {"k": "int"}, "size": 1, "seed": 2}});
    let mix = json!({"elem": {"array": "mix", "index": 1}});
    let refused = |node: Value, needle: &str| fails(&doc(params.clone(), d(), ns.clone(), json!([node])), needle);
    refused(json!({"compute": {"ns": {"add": [mix, 1]}}}), "add: operands are str and int");
    refused(json!({"compute": {"ns": {"neg": mix}}}), "neg of str");
    refused(json!({"cond": {"if": {"lt": [mix, 2]}, "then": []}}), "lt: cannot compare str with int");
    refused(json!({"compute": {"ns": mix}}), "expected an integer, got str");
    refused(json!({"cond": {"if": mix, "then": []}}), "expected a condition, got str");
    refused(json!({"stat": {"file": {"object": {"namespace": "o", "fields": {"k": mix}}}}}), "object field `k` of `o` is Int, got str");
    refused(json!({"compute": {"ns": {"draw": {"normal": {"mean": mix, "sd": 1, "min": 0}}}}}), "expected a number, got str");
}

#[test]
fn comparisons_of_numbers_and_handles_and_the_short_circuits() {
    let f = |id: i64| json!({"file": {"dataset": "d", "id": id}});
    let when = |test: Value, id: i64| json!({"cond": {"if": test, "then": [stat("d", id)]}});
    let body = json!([
        {"let": {"name": "a", "value": f(0)}}, {"let": {"name": "b", "value": f(0)}}, {"let": {"name": "c", "value": f(1)}},
        when(json!({"eq": [{"ref": "a"}, {"ref": "b"}]}), 0),
        when(json!({"ne": [{"ref": "a"}, {"ref": "c"}]}), 1),
        when(json!({"eq": [{"ref": "a"}, {"ref": "c"}]}), 2),
        when(json!({"lt": [1.5, 2]}), 3),
        when(json!({"ge": [2, 2.5]}), 4),
        when(json!({"or": [false, 0]}), 5),
        when(json!({"or": [0, true]}), 6),
        when(json!({"and": [true, 0]}), 7)
    ]);
    let ops = computes(body, json!({}));
    let ids: Vec<&str> = ops.iter().map(|o| o.split(' ').nth(1).unwrap()).collect();
    assert_eq!(ids, ["d/0", "d/1", "d/3", "d/6"]);
}

/// A `let` of a distribution is drawn where a `draw` names it; `x @ i` below the loop's start
/// takes the `cond`'s other arm, whichever arm it is in, and with no `cond` around it the walk
/// is refused.
#[test]
fn a_distribution_binding_is_drawn_and_an_at_below_the_start_takes_the_other_arm() {
    let body = json!([
        {"let": {"name": "w", "value": {"uniform": {"lo": 5, "hi": 6}}}},
        {"compute": {"ns": {"draw": {"ref": "w"}}}},
        {"loop": {"index": "i", "from": 0, "to": 3, "body": [
            {"let": {"name": "x", "value": {"cond": {"if": true, "then": {"add": [{"at": {"ref": "x", "index": {"sub": [{"index": "i"}, 1]}}}, 1]}, "else": 0}}}},
            {"let": {"name": "y", "value": {"cond": {"if": false, "then": 100, "else": {"add": [{"at": {"ref": "y", "index": {"sub": [{"index": "i"}, 1]}}}, 1]}}}}},
            {"compute": {"ns": {"ref": "x"}}}, {"compute": {"ns": {"ref": "y"}}}
        ]}}
    ]);
    assert_eq!(computes(body, json!({})), ["compute 5", "compute 0", "compute 100", "compute 1", "compute 101", "compute 2", "compute 102"]);
    let outer = json!([{"loop": {"index": "j", "from": 2, "to": 4, "body": [
        {"let": {"name": "s", "value": {"mul": [{"index": "j"}, 10]}}},
        {"loop": {"index": "k", "to": 1, "body": [{"compute": {"ns": {"at": {"ref": "s", "index": {"sub": [{"index": "j"}, 1]}}}}}]}}]}}]);
    fails(&doc(json!({}), d(), json!({}), outer), "`at` below the loop's start with no alternative arm");
}

/// `x @ i` far past a loop that starts below zero: the ordinal of the frame is past 2^63-1, and
/// the `consume` it re-evaluates is refused rather than wrapped.
#[test]
fn a_position_past_the_64_bit_integers_is_refused() {
    let body = json!([{"loop": {"index": "j", "from": -5, "to": 1, "body": [
        {"let": {"name": "s", "value": {"consume": "d"}}},
        {"loop": {"index": "k", "to": 1, "body": [
            {"let": {"name": "t", "value": {"at": {"ref": "s", "index": i64::MAX}}}},
            {"stat": {"file": {"ref": "t"}}}]}}]}}]);
    fails(&doc(json!({}), d(), json!({}), body), "position overflow");
}

// ---------------------------------------------------------------- handles

/// A container dataset `c` (10 samples of 100 bytes, 4 to a file, units of 2), a chunked one `k`
/// (two files of 150 bytes in chunks of 64), an array `a` (3 slots of 16, 10 bytes each), and a
/// dataset `i` with an inventory.
fn containers() -> Value {
    json!({
        "c": {"layout": "compound", "access": "map", "pattern": "c/{id}", "count": 10, "samples_per_file": 4, "size": {"const": 100}, "seed": 1,
              "format": {"class": "x", "framing": {"unit": 2, "unit_header": 8, "columns": [{"header": 4, "weight": 0.5}, {"weight": 0.5}]}}},
        "k": {"layout": "chunked", "access": "map", "pattern": "k/{id}_{k}", "count": 2, "chunk": 64, "size": {"const": 150}, "seed": 2},
        "a": {"layout": "array", "access": "indexed", "pattern": "a/one.bin", "count": 3, "slot": 16, "size": {"const": 10}, "seed": 3},
        "i": {"layout": "whole", "access": "map", "pattern": "i/{id}", "count": 5, "size": {"const": 1}, "seed": 4, "inventory": {"name": "list.txt", "entry": 8}}
    })
}

#[test]
fn units_columns_chunks_and_slots_have_the_sizes_and_offsets_of_their_framing() {
    let c = |ns: Value| json!({"compute": {"ns": ns}});
    let file = |ds: &str, id: i64| json!({"file": {"dataset": ds, "id": id}});
    let sample = |id: i64| json!({"pick": {"dataset": "c", "dist": {"const": id}}});
    let unit = |id: i64, u: i64| json!({"unit": {"of": file("c", id), "index": u}});
    let body = json!([
        c(json!({"unit_index": sample(7)})),                                  // (7 - 4) / 2
        c(json!({"unit_index": unit(0, 1)})),
        c(json!({"unit_index": {"column": {"of": unit(0, 1), "index": 0}}})),
        c(json!({"units": sample(9)})),                                       // file 2: 2 samples
        c(json!({"units": unit(0, 0)})),
        c(json!({"chunks": file("k", 0)})),                                   // 150 in 64s
        c(json!({"chunks": {"pick": {"dataset": "k"}}})),
        c(json!({"size": {"file": {"dataset": "k", "id": 0, "chunk": 2}}})),  // 150 - 128
        c(json!({"size": {"file": {"of": {"pick": {"dataset": "k"}}, "chunk": 1}}})),
        c(json!({"size": unit(0, 0)})),                                       // 8 + (4 + 50 + 50) + (50 + 50)
        c(json!({"size": {"column": {"of": unit(0, 0), "index": 1}}})),
        c(json!({"size": {"column": {"of": sample(1), "index": 0}}})),        // 4 + 50 + 50
        c(json!({"offset": unit(0, 1)})),
        c(json!({"offset": {"column": {"of": unit(0, 1), "index": 1}}})),     // 212 + 8 + 104
        c(json!({"offset": {"unit": {"of": sample(3)}}})),
        c(json!({"offset": file("k", 1)})),
        c(json!({"size": file("a", 0)})),                                     // 2 × 16 + 10
        c(json!({"size": {"file": {"of": {"pick": {"dataset": "a"}}}}})),
        c(json!({"size": {"pick": {"dataset": "a"}}})),
        c(json!({"offset": {"pick": {"dataset": "a", "dist": {"const": 2}}}})),
        c(json!({"size": {"inventory": "i"}})),                               // 5 × 8
        c(json!({"offset": {"inventory": "i"}})),
        c(json!({"count": "c"}))
    ]);
    let got = walk(&doc(json!({}), containers(), json!({}), body));
    let want = [1, 1, 1, 1, 2, 3, 3, 22, 64, 212, 100, 104, 212, 324, 212, 0, 42, 42, 10, 32, 40, 0, 10];
    assert_eq!(got, want.iter().map(|n| format!("compute {n}")).collect::<Vec<_>>());
}

#[test]
fn a_handle_of_a_kind_its_place_does_not_take_is_refused_when_walked() {
    let c = |ns: Value| json!({"compute": {"ns": ns}});
    let refused = |node: Value, needle: &str| fails(&doc(json!({}), containers(), json!({"o": {"pattern": "o/{k}", "fields": {"k": "int"}, "size": "as_written", "seed": 9}}), json!([node])), needle);
    let file = |ds: &str, id: i64| json!({"file": {"dataset": ds, "id": id}});
    let dir = json!({"dir": {"dataset": "c", "id": 0}});
    refused(c(json!({"unit_index": file("c", 0)})), "unit_index: not a sample, unit, or column handle");
    refused(c(json!({"units": dir.clone()})), "units: not a file, sample, or unit handle");
    refused(c(json!({"chunks": {"unit": {"of": file("c", 0), "index": 0}}})), "chunks: not a file or sample handle");
    refused(c(json!({"chunks": file("c", 0)})), "dataset `c` has no `chunk`");
    refused(c(json!({"size": {"file": {"dataset": "k", "id": 0, "chunk": 3}}})), "chunk 3 outside file 0 of `k` (150 bytes)");
    refused(c(json!({"size": {"unit": {"of": file("c", 0), "index": 5}}})), "unit 5 outside file 0 (2 units)");
    refused(c(json!({"offset": {"unit": {"of": file("c", 0), "index": 5}}})), "unit 5 outside file 0 (2 units)");
    refused(c(json!({"size": {"column": {"of": {"unit": {"of": file("c", 0), "index": 0}}, "index": 2}}})), "column 2 outside the 2 columns of `c`");
    refused(c(json!({"offset": {"column": {"of": {"unit": {"of": file("c", 0), "index": 0}}, "index": 2}}})), "column 2 outside the 2 columns of `c`");
    refused(c(json!({"size": dir.clone()})), "size of a directory");
    refused(c(json!({"offset": dir.clone()})), "offset of Dir");
    refused(c(json!({"size": {"file": {"of": {"inventory": "i"}}}})), "file {of}: Inventory");
    refused(c(json!({"size": {"unit": {"of": file("c", 0)}}})), "unit {of: file} needs an index");
    refused(c(json!({"size": {"unit": {"of": dir.clone()}}})), "unit {of}: Dir");
    refused(c(json!({"size": {"column": {"of": file("c", 0), "index": 0}}})), "column {of}: File");
    refused(c(json!({"size": file("a", 1)})), "an array dataset has one file, id 0");
    refused(c(json!({"size": {"pick": {"dataset": "c", "dist": {"const": 50}}}})), "pick: id 50 outside dataset `c` of 10");
    refused(c(json!({"size": {"object": {"namespace": "o", "fields": {"k": 1}}}})), "size of as_written object `o/1`: no writes recorded at this position");
    refused(c(json!({"size": {"pick": {"dataset": "c", "dist": {"uniform": {"lo": 4, "hi": 4}}}}})), "uniform: empty range [4, 4)");
    refused(c(json!({"size": {"pick": {"dataset": "c", "dist": {"empirical": {"values": [1], "weights": [0]}}}}})), "empirical: total weight is zero");
    refused(c(json!({"size": {"pick": {"dataset": "c", "dist": {"mixture": [{"weight": 0, "dist": {"const": 1}}]}}}})), "mixture: total weight is zero");
    // a sample of an array is a slot; an offset or size of a container file is its file's
    let empty = json!({"e": {"layout": "whole", "access": "map", "pattern": "e/{id}", "count": 0, "size": {"const": 1}, "seed": 5}});
    fails(&doc(json!({}), empty.clone(), json!({}), json!([c(json!({"size": {"pick": {"dataset": "e"}}}))])), "pick from empty dataset `e`");
    fails(&doc(json!({}), empty, json!({}), json!([{"loop": {"index": "i", "to": 1, "body": [{"stat": {"file": {"consume": "e"}}}]}}])), "has 0 samples, fewer than one batch across 1 actors (1 per batch)");
}

/// `zipf` with `s` = 1 (its own inverse CDF), `hotset`, and a pick over a `stream` dataset (a
/// shard) stay inside the dataset.
#[test]
fn skewed_picks_stay_inside_their_dataset() {
    let ds = json!({"d": {"layout": "whole", "access": "map", "pattern": "d/{id}", "count": 10, "size": {"const": 1}, "seed": 1},
                    "s": {"layout": "compound", "access": "stream", "pattern": "s/{id}", "count": 12, "samples_per_file": 4, "size": {"const": 1}, "seed": 2}});
    for (set, dist, n) in [("d", json!({"zipf": {"s": 1.0}}), 10), ("d", json!({"hotset": {"fraction": 0.2, "weight": 0.9}}), 10), ("s", json!({"zipf": {"s": 1.5}}), 3)] {
        let body = json!([{"loop": {"index": "i", "to": 200, "body": [{"stat": {"file": {"pick": {"dataset": set, "dist": dist}}}}]}}]);
        let ops = walk(&doc(json!({}), ds.clone(), json!({}), body));
        let ids: std::collections::BTreeSet<i64> = ops.iter().map(|o| o.split(' ').nth(1).unwrap().split('/').nth(1).unwrap().parse().unwrap()).collect();
        assert!(ids.len() > 1 && ids.iter().all(|i| (0..n).contains(i)), "{set} {dist}: {ids:?}");
    }
}

// ---------------------------------------------------------------- reads, writes, positions

#[test]
fn positions_and_lengths_outside_what_a_file_holds_are_refused_when_walked() {
    let ns = json!({"o": {"pattern": "o/{k}", "fields": {"k": "int"}, "size": "as_written", "seed": 9}});
    let o = json!({"object": {"namespace": "o", "fields": {"k": 1}}});
    let f = json!({"file": {"dataset": "d", "id": 0}});
    let opened = |h: &Value, flags: Value, then: Vec<Value>| {
        let mut b = vec![json!({"let": {"name": "h", "value": h}}), json!({"open": {"file": {"ref": "h"}, "flags": flags}})];
        b.extend(then);
        json!([{"loop": {"index": "i", "to": 1, "body": b}}])
    };
    let h = json!({"ref": "h"});
    let refused = |body: Value, needle: &str| fails(&doc(json!({}), d(), ns.clone(), body), needle);
    refused(opened(&f, json!(["RDONLY"]), vec![json!({"read": {"file": h, "len": {"neg": 1}}})]), "read: negative length -1");
    refused(opened(&o, json!(["WRONLY", "CREAT"]), vec![json!({"write": {"file": h, "len": {"neg": 1}}})]), "write: negative length -1");
    refused(opened(&f, json!(["RDONLY"]), vec![json!({"read": {"file": h, "len": 0, "repeat": "until_eof"}})]), "until_eof with a zero-length read");
    refused(opened(&o, json!(["RDWR", "CREAT"]), vec![json!({"write": {"file": h, "len": 4, "repeat": 0}}), json!({"read": {"file": h, "len": 4, "repeat": "until_eof"}})]),
        "until_eof on `o/1`: size unknown");
    refused(opened(&o, json!(["RDWR", "CREAT"]), vec![json!({"lseek": {"file": h, "offset": 0, "whence": "END"}})]), "lseek END on `o/1`: size unknown");
    refused(opened(&f, json!(["RDONLY"]), vec![json!({"lseek": {"file": h, "offset": -5, "whence": "CUR"}})]), "lseek to -5 on `d/0`");
    refused(opened(&f, json!(["RDONLY"]), vec![json!({"fadvise": {"file": h, "advice": "WILLNEED", "len": {"neg": 1}}})]), "fadvise: offset 0, len -1");
    refused(opened(&o, json!(["WRONLY", "CREAT"]), vec![json!({"write": {"file": h, "len": i64::MAX, "offset": 0}}), json!({"write": {"file": h, "len": i64::MAX, "offset": 0}})]),
        "write: the size written, 9223372036854775807 + 9223372036854775807 bytes, passes 2^63-1");
    // a known size: an `END` seek from it, and an ioctl and fadvise with their codes
    let ops = walk(&doc(json!({}), d(), ns.clone(), opened(&f, json!(["RDONLY"]), vec![
        json!({"lseek": {"file": h, "offset": -1, "whence": "END"}}), json!({"read": {"file": h, "len": 4}}),
        json!({"ioctl": {"file": h, "request": "FIONREAD"}}), json!({"fadvise": {"file": h, "advice": "DONTNEED", "offset": 0, "len": 1}})])));
    assert_eq!(&ops[1..], ["lseek d/0 off=-1 len=0 bytes=0 aux=2", "read d/0 off=0 len=4 bytes=1 aux=0", "ioctl d/0 off=0 len=0 bytes=0 aux=1", "fadvise d/0 off=0 len=1 bytes=0 aux=4"]);
}

/// The entries a `readdir` expects, by the directory scheme of the pattern.
#[test]
fn a_readdir_expects_the_entries_its_directory_scheme_puts_there() {
    let ds = json!({
        "m": {"layout": "whole", "access": "indexed", "pattern": "m/{id mod 3}/f_{id}", "count": 10, "size": {"const": 1}, "seed": 1},
        "c": {"layout": "whole", "access": "map", "pattern": "c/x/f_{id}", "count": 10, "size": {"const": 1}, "seed": 2},
        "p": {"layout": "whole", "access": "map", "pattern": "p/{id}/f", "count": 10, "size": {"const": 1}, "seed": 3},
        "s": {"layout": "compound", "access": "map", "pattern": "s/{id div 2}/f_{id}", "count": 10, "samples_per_file": 2, "size": {"const": 1}, "seed": 4}
    });
    let rd = |set: &str, d: i64| {
        let h = json!({"dir": {"dataset": set, "id": d}});
        vec![json!({"open": {"file": h, "flags": ["RDONLY", "DIRECTORY"]}}), json!({"readdir": {"dir": h, "repeat": "until_end"}}), json!({"close": {"file": h}})]
    };
    let body: Vec<Value> = [("m", 0), ("m", 2), ("c", 0), ("p", 7), ("s", 1)].iter().flat_map(|(s, d)| rd(s, *d)).collect();
    let ops = walk(&doc(json!({}), ds.clone(), json!({}), json!(body)));
    let readdirs: Vec<&str> = ops.iter().filter(|o| o.starts_with("readdir")).map(|s| s.as_str()).collect();
    // `{id mod 3}` of 10: 4, 4 (not asked), 3; one constant directory; one per file; and
    // directory 1 of `s/{id div 2}`, files 2 and 3 of two samples each, whose entries are not
    // computed (-1)
    assert_eq!(readdirs, [
        "readdir m/0 off=0 len=0 bytes=4 aux=0",
        "readdir m/2 off=0 len=0 bytes=3 aux=0",
        "readdir c/x off=0 len=0 bytes=10 aux=0",
        "readdir p/7 off=0 len=0 bytes=1 aux=0",
        "readdir s/1 off=0 len=0 bytes=-1 aux=0",
    ]);
    fails(&doc(json!({}), ds.clone(), json!({}), json!([{"stat": {"file": {"dir": {"dataset": "m", "id": 3}}}}])), "directory 3 outside the 3 directories of `m`");
    let several = json!({"u": {"layout": "whole", "access": "indexed", "pattern": "u/{id div 4}/{id mod 2}/f_{id}", "count": 8, "size": {"const": 1}, "seed": 1}});
    fails(&doc(json!({}), several.clone(), json!({}), json!([{"compute": {"ns": {"dirs": "u"}}}])), "dirs: dataset `u` has several `id` fields in its directory part");
    fails(&doc(json!({}), several, json!({}), json!([{"stat": {"file": {"dir": {"dataset": "u", "id": 0}}}}])), "directory 0 outside the 0 directories of `u`");
}

// ---------------------------------------------------------------- the model

#[test]
fn a_dataset_whose_values_the_model_cannot_hold_is_refused_when_the_model_is_built() {
    let built = |ds: Value| with_model(&doc(json!({}), json!({"d": ds}), json!({}), json!([])), 1, 1, &[], |_| Ok(())).unwrap_err();
    let whole = |more: Value| {
        let mut d = json!({"layout": "compound", "access": "map", "pattern": "d/{id}", "count": 10, "samples_per_file": 2, "size": {"const": 1}, "seed": 1});
        d.as_object_mut().unwrap().extend(more.as_object().unwrap().clone());
        d
    };
    assert!(built(whole(json!({"samples_per_file": {"neg": 0}}))).contains("samples_per_file 0: a file holds at least 1 sample"));
    assert!(built(whole(json!({"format": {"class": "x", "framing": {"file_header": {"neg": 1}}}}))).contains("framing file_header is -1"));
    assert!(built(json!({"layout": "chunked", "access": "map", "pattern": "d/{id}_{k}", "count": 1, "chunk": {"neg": 0}, "size": {"const": 1}, "seed": 1})).contains("chunk must be positive"));
    assert!(built(json!({"layout": "whole", "access": "map", "pattern": "d/{id}", "count": 1, "size": {"const": 1}, "seed": 1, "inventory": {"name": "l", "entry": 0}})).contains("inventory entry 0 must be positive"));
    assert!(built(json!({"layout": "array", "access": "indexed", "pattern": "d/a.bin", "count": {"neg": 1}, "slot": 4, "size": {"const": 1}, "seed": 1})).contains("count -1, slot 4"));
    let ns = with_model(&doc(json!({}), json!({}), json!({"o": {"pattern": "o/{k}_{z}", "fields": {"k": "int"}, "size": 1, "seed": 1}}), json!([])), 1, 1, &[], |_| Ok(())).unwrap_err();
    assert!(ns.contains("namespace `o`: pattern field `{z}` is not a declared field"), "{ns}");
}

/// A draw of `normal` or `lognormal` past the 64-bit integers is refused, as `as_int` refuses
/// such a float; it became 2^63-1 silently (a file of 8 EiB), and a NaN draw 0 (RC1 review).
#[test]
fn a_normal_draw_past_the_64_bit_integers_is_refused() {
    let sized = |size: Value| doc(json!({}), json!({"d": {"layout": "whole", "access": "indexed", "pattern": "d/{id}", "count": 2, "size": size, "seed": 1}}), json!({}),
        json!([{"compute": {"ns": {"size": {"file": {"dataset": "d", "id": 0}}}}}]));
    fails(&sized(json!({"normal": {"mean": 1e19, "sd": 1, "min": 0}})), "normal: float 10000000000000000000 where an integer is needed is outside the 64-bit integers of the contract");
    fails(&sized(json!({"lognormal": {"median": 1e300, "sigma": 50, "min": 0}})), "lognormal: ");
    fails(&sized(json!({"normal": {"mean": {"div": [0.0, 0.0]}, "sd": 1, "min": 0}})), "normal: non-finite float where an integer is needed");
    assert_eq!(walk(&sized(json!({"normal": {"mean": 4.6e18, "sd": 0, "min": 0}}))), ["compute 4600000000000000000"]);
}

#[test]
fn a_dataset_framed_in_two_weighted_columns_splits_each_sample_between_them() {
    let ds = json!({"c": {"layout": "compound", "access": "map", "pattern": "c/{id}", "count": 3, "samples_per_file": 3, "size": {"const": 101}, "seed": 1,
        "format": {"class": "x", "framing": {"columns": [{"weight": 0.25}, {"weight": 0.0, "fixed": 8}, {"weight": 0.75}]}}}});
    with_model(&doc(json!({}), ds, json!({}), json!([])), 1, 1, &[], |model| {
        let m = &model.datasets[0];
        // floor(101 / 4) = 25; the last weighted column takes the other 76; the unweighted one only its fixed bytes
        assert_eq!(m.col_lens(0, 0).unwrap(), [75, 24, 228]);
        assert_eq!(m.framing().unwrap().part(1, 101), 0);
        assert!(m.sample_offset(0).unwrap_err().to_string().contains("its bytes are split over 3 columns"));
        Ok(())
    })
    .unwrap();
}

/// What `DsMeta` refuses of a dataset of the other kind, or of an id past its end.
#[test]
fn dataset_geometry_outside_a_dataset_is_refused() {
    let ds = json!({"c": {"layout": "compound", "access": "map", "pattern": "c/{id}", "count": 10, "samples_per_file": 4, "size": {"const": 1}, "seed": 1},
                    "a": {"layout": "array", "access": "indexed", "pattern": "a/one.bin", "count": 3, "slot": 16, "size": {"const": 30}, "seed": 2},
                    "z": {"layout": "array", "access": "indexed", "pattern": "z/one.bin", "count": 0, "slot": 16, "size": {"const": 1}, "seed": 3}});
    with_model(&doc(json!({}), ds, json!({}), json!([])), 1, 1, &[], |model| {
        let [a, c, z] = [&model.datasets[0], &model.datasets[1], &model.datasets[2]];
        let e = |r: anyhow::Result<i64>| r.unwrap_err().to_string();
        assert_eq!(e(c.sample_size(10)), "sample 10 outside dataset `c` of 10");
        assert_eq!(e(a.sample_size(3)), "region 3 outside dataset `a` of 3");
        assert_eq!(a.sample_size(1).unwrap(), 16, "a size is capped at the slot");
        assert_eq!(e(c.unit_offset(0, 1)), "unit 1 outside file 0 of `c`");
        assert_eq!(e(a.size_draw(0).map(|n| n as i64)), "`a` is an array dataset");
        assert_eq!(e(a.samples_in_file(0)), "`a` is an array dataset");
        assert_eq!(e(a.file_size(1)), "an array dataset has one file, id 0");
        assert_eq!(e(a.file_path(1, None).map(|_| 0)), "an array dataset has one file, id 0");
        assert_eq!(e(a.dir_path(0).map(|_| 0)), "`dir` on array dataset `a`");
        assert_eq!(e(a.consume_perm(0).map(|_| 0)), "`consume` on array dataset `a`: use `pick`");
        assert_eq!(e(a.framing().map(|_| 0)), "`a` is an array dataset: no container framing");
        assert_eq!(e(a.chunks(0)), "dataset `a` has no `chunk`");
        assert_eq!((a.files(), a.unit_of_sample(2).unwrap(), z.file_size(0).unwrap(), a.dir_entries(0)), (1, 0, 0, None));
        assert_eq!(c.dir_entries(0), None, "several samples to a file: no computed entries");
        Ok(())
    })
    .unwrap();
}

// ---------------------------------------------------------------- parameters

#[test]
fn parameters_given_are_held_to_their_declarations() {
    let text = doc(json!({"n": {"default": 3}, "fixed": {"default": 1, "cli": false}, "arr": {"default": [1, 2]}}), d(), json!({}), json!([]));
    let given = |overrides: &[(&str, &str)], sets: Vec<ParamSet>| {
        let loaded = aeiou::load_str(&text).unwrap();
        let cfg = Config { seed: 1, gpus: 1, overrides: overrides.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), sets, st_blksize: Default::default() };
        Params::new(&loaded.ast, &cfg).map(|_| ()).map_err(|e| format!("{e:#}"))
    };
    assert_eq!(given(&[("gpus", "2")], vec![]).unwrap_err(), "`gpus` is set with --gpus, not --param");
    assert_eq!(given(&[("fixed", "2")], vec![]).unwrap_err(), "--param fixed: not overridable (cli: false)");
    assert_eq!(given(&[("arr", r#"[1, {"const": 2}]"#)], vec![]).unwrap_err(), "--param arr[1]: the default's elements are a scalar, this one is a distribution");
    assert_eq!(given(&[("arr", "[1, 2.5]")], vec![]).unwrap_err(), "--param arr[1]: the default is an integer, the value 2.5 is not");
    let set = |text: &str| ParamSet::parse(text, "p.json").map_err(|e| format!("{e:#}"));
    assert_eq!(set(r#"{"params_version": 2, "abstract": "t", "ast_sha256": "xyz", "params": {}}"#).unwrap_err(), r#"`ast_sha256` is not 64 hex digits: "xyz""#);
    assert_eq!(set(r#"{"params_version": 2, "abstract": "t", "doc": 4, "params": {}}"#).unwrap_err(), "`doc` is not a string: 4");
    let other = set(r#"{"params_version": 2, "abstract": "u", "params": {}}"#).unwrap();
    assert_eq!(other.check_identity("t", &"0".repeat(64)).unwrap_err().to_string(), "parameter file p.json is for abstract `u`, not `t`");
    let pinned = set(&format!(r#"{{"params_version": 2, "abstract": "t", "ast_sha256": "{}", "params": {{"n": {{"default": 4}}}}}}"#, "A".repeat(64))).unwrap();
    assert_eq!(pinned.check_identity("t", &"a".repeat(64)).map_err(|e| e.to_string()), Ok(()), "hex digits of either case");
    assert_eq!(given(&[], vec![pinned]), Ok(()));
}
