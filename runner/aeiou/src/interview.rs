//! The questions `aeiou-trace digest` asks about a trace and the answers file (`PROJECT_BRIEF.md`
//! item 47; `DESIGN_REVIEW.md` §3.86 to §3.89, in Rust since §3.89 stage 3). The interview
//! identifies the application from the records (`applications.rs`), asks the record's options,
//! the facts of every site with an open question, and then the open items in rank order, and
//! checks every answer against the trace (each site's facts carry the least value the trace
//! allows); every document it shows is remade from the one analysis, the trace read once.
use std::io::{BufRead, Write};

use anyhow::{anyhow, Result};
use serde_json::{json, Map, Value};

use crate::applications::Records;
use crate::digest::{pct, py, site_line, site_name, Analysis, Opts, UNKNOWN};

pub const ANSWERS_FORMAT: u64 = 1;

/// What a candidate answer means, as the interview lists it.
pub fn label(candidate: &str) -> &'static str {
    match candidate {
        "consume" => "each file exactly once per pass (a permutation over the set)",
        "choose" => "a draw with replacement: a file may come up again within a pass",
        "per-path" => "the bytes are a property of the file (its size)",
        "drawn" => "the bytes are chosen per access",
        "sequential" => "in their sorted order",
        "permuted" => "in a shuffled order",
        "earlier_run" => "an earlier run of the application wrote them (a warm cache)",
        "other_writer" => "another process the trace does not cover writes them",
        "once" => "each key is read once",
        "repeated" => "a key is read again after its first read",
        "uniform" => "spread evenly over the file",
        "skewed" => "a few parts of the file take most of the reads",
        "fits" => "the file fits in memory at scale",
        "exceeds" => "the file exceeds memory at scale",
        "dependent" => "each offset depends on what the previous read returned",
        "independent" => "the offsets are independent of one another",
        "checkpoint" => "a checkpoint: the application's state, read back only after a failure",
        "output" => "output: an index build, a dataset, results",
        _ => "",
    }
}

/// Where the questions go and the answers come from.
pub trait Console {
    fn say(&mut self, text: &str);
    /// A line typed after `prompt`, without its newline; None at end of input.
    fn ask(&mut self, prompt: &str) -> Option<String>;
}

/// The terminal: questions on stderr (stdout may be the document), answers read from stdin.
pub struct Terminal;

impl Console for Terminal {
    fn say(&mut self, text: &str) {
        eprintln!("{text}");
    }

    fn ask(&mut self, prompt: &str) -> Option<String> {
        eprint!("{prompt}");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        match std::io::stdin().lock().read_line(&mut line) {
            Ok(0) | Err(_) => None,
            Ok(_) => Some(line.strip_suffix('\n').unwrap_or(&line).to_string()),
        }
    }
}

/// Why the trace contradicts a value of a site's fact (`f`, one of the site's `facts`), or None.
pub fn fact_check(f: &Value, value: i64) -> Option<String> {
    let least = f.get("least")?.as_i64()?;
    (value < least).then(|| format!("{}; {value} is {}", py(&f["why"]), if f["name"] == "size" { "smaller" } else { "fewer" }))
}

/// A whole number typed, as Python's `str.isdigit` and `int` read one.
fn number(line: &str) -> Option<i64> {
    (!line.is_empty() && line.bytes().all(|b| b.is_ascii_digit())).then(|| line.parse().ok()).flatten()
}

/// A number from 1 to n, or None at end of input.
fn choose(io: &mut dyn Console, prompt: &str, n: usize) -> Option<usize> {
    loop {
        let line = io.ask(prompt)?;
        match number(line.trim()) {
            Some(k) if k >= 1 && k as usize <= n => return Some(k as usize),
            _ => io.say(&format!("  a number from 1 to {n}")),
        }
    }
}

fn with_phase(mut a: Map<String, Value>, phase: &Value) -> Value {
    if !phase.is_null() {
        a.insert("phase".into(), phase.clone());
    }
    Value::Object(a)
}

/// Ask what the trace cannot decide, in this order: with `apps` (the application records), which
/// application the trace is of, confirmed from its path signatures or chosen from the list, its
/// version, and the record's options, whose assertions then answer what they can (as suggestions
/// the person confirms when no record claims the version); then the facts of every site with an
/// open question; then the open items in rank order. Every answer is checked against the trace
/// and re-asked when it contradicts it; end of input stops the asking. `application` given skips
/// the identification (`--application NAME@VERSION`). Returns the document and the answers
/// given, the list an answers file keeps and a replay applies, with the record's assertions among
/// them carrying their own `by`.
pub fn interview(an: &Analysis, io: &mut dyn Console, o: &Opts, by: &str, at: &str, apps: Option<&Records>, application: Option<Value>) -> (Value, Vec<Value>) {
    let mut answers: Vec<Value> = vec![];
    let mut state = application.clone().unwrap_or(Value::Null);
    let document = |answers: &[Value], app: &Value| an.document(o, answers, Some(by), Some(at), app);
    macro_rules! finish {
        () => {
            return (document(&answers, &state), answers)
        };
    }

    if let Some(apps) = apps {
        let all = an.document(&Opts { epochs: 1, candidates: None, paths: true }, &[], None, None, &Value::Null);
        let templates: Vec<(String, Vec<String>)> = all["templates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| (py(&t["template"]), t["raw_paths"].as_array().map(|a| a.iter().map(py).collect()).unwrap_or_default()))
            .collect();
        let (app, version) = if let Some(given) = &application {
            (given["name"].as_str().and_then(|n| apps.apps.get(n)), given["version"].as_str().map(str::to_string))
        } else {
            let found = apps.identify(templates.iter().flat_map(|(_, ps)| ps.iter().map(String::as_str)));
            io.say("");
            let mut app = None;
            let listed = if let Some((a0, sig)) = found.first() {
                io.say(&format!("This trace looks like {} (a path matched {sig}).", a0.display));
                io.say("    1. yes");
                io.say("    2. another application");
                io.say("    3. none of these, or don't know");
                let Some(n) = choose(io, "answer [1-3]: ", 3) else { finish!() };
                if n == 1 {
                    app = Some(*a0);
                }
                n == 2
            } else {
                io.say("No path in the trace identifies the application.");
                true
            };
            if listed {
                let names: Vec<&String> = apps.apps.keys().collect();
                io.say("Which application produced this trace?");
                for (i, name) in names.iter().enumerate() {
                    io.say(&format!("    {}. {}: {}", i + 1, apps.apps[*name].display, apps.apps[*name].summary));
                }
                io.say(&format!("    {}. none of these, or don't know", names.len() + 1));
                let Some(n) = choose(io, &format!("answer [1-{}]: ", names.len() + 1), names.len() + 1) else { finish!() };
                app = names.get(n - 1).map(|name| &apps.apps[*name]);
            }
            let mut version = None;
            if let Some(a) = app {
                let Some(v) = io.ask(&format!("which version of {}? (Enter for don't know): ", a.display)) else { finish!() };
                version = Some(v.trim().to_string()).filter(|v| !v.is_empty());
            }
            (app, version)
        };
        if let Some(app) = app {
            let (record, exact) = app.resolve(version.as_deref());
            state = json!({"name": app.name, "display": app.display, "version": version, "record": null, "exact": false, "options": {}, "blind_spots": []});
            match record {
                None => io.say(&format!("  no record of {} at {} or below; nothing is assumed about it", app.display, version.as_deref().unwrap_or("None"))),
                Some(record) => {
                    state["record"] = json!(record.version);
                    state["exact"] = json!(exact);
                    if !exact {
                        io.say(&format!(
                            "  no record claims {} {}; the nearest is {}, read {}: its answers are suggestions you confirm",
                            app.display,
                            version.as_deref().unwrap_or("of unknown version"),
                            record.version,
                            record.read
                        ));
                    }
                    let mut options = Map::new();
                    if let Some(given) = application.as_ref().and_then(|a| a["options"].as_object()) {
                        options.extend(given.clone());
                    }
                    for opt in &record.options {
                        if options.contains_key(&opt.name) {
                            continue;
                        }
                        io.say("");
                        io.say(&format!("  {}", opt.question));
                        for (i, c) in opt.choices.iter().enumerate() {
                            io.say(&format!("    {}. {c}{}", i + 1, if opt.default.as_ref() == Some(c) { " (the default)" } else { "" }));
                        }
                        io.say(&format!("    {}. don't know", opt.choices.len() + 1));
                        state["options"] = json!(options);
                        let Some(n) = choose(io, &format!("answer [1-{}]: ", opt.choices.len() + 1), opt.choices.len() + 1) else { finish!() };
                        if n <= opt.choices.len() {
                            options.insert(opt.name.clone(), json!(opt.choices[n - 1]));
                        }
                    }
                    state["options"] = json!(options);
                    let spots = apps.blind_spots(record, &options);
                    for b in &spots {
                        io.say(&format!("  blind spot ({}): {}", py(&b["kind"]), py(&b["note"])));
                    }
                    state["blind_spots"] = json!(spots);
                    for a in record.assertions(&options, &templates) {
                        if exact {
                            answers.push(a);
                            continue;
                        }
                        let what = match a.get("fact") {
                            Some(f) => format!("{} {}", py(f), py(&a["value"])),
                            None => format!("{} {}", py(&a["question"]), py(&a["answer"])),
                        };
                        let note = a.get("note").and_then(Value::as_str).filter(|n| !n.is_empty());
                        io.say("");
                        io.say(&format!("  the {} {} record says, for {}: {what}{}", record.display, record.version, py(&a["template"]), note.map_or(String::new(), |n| format!(" ({n})"))));
                        io.say("    1. yes, that holds here");
                        io.say("    2. no");
                        io.say("    3. don't know");
                        let Some(n) = choose(io, "answer [1-3]: ", 3) else { finish!() };
                        if n == 1 {
                            let mut mine: Map<String, Value> = a.as_object().unwrap().iter().filter(|(k, _)| *k != "by" && *k != "source").map(|(k, v)| (k.clone(), v.clone())).collect();
                            mine.insert(
                                "note".into(),
                                json!(format!("confirmed from the {} {} record{}", record.display, record.version, note.map_or(String::new(), |n| format!(": {n}")))),
                            );
                            answers.push(Value::Object(mine));
                        }
                    }
                }
            }
        }
    }

    let doc = document(&answers, &state);
    let open: Vec<(Value, Value)> = doc["open"].as_array().unwrap().iter().map(|it| (it["site"].clone(), it.get("phase").cloned().unwrap_or(Value::Null))).collect();
    for s in doc["sites"].as_array().unwrap() {
        if !open.contains(&(s["template"].clone(), s["phase"].clone())) {
            continue;
        }
        let name = py(&doc["templates"][s["template"].as_u64().unwrap() as usize]["template"]);
        io.say("");
        io.say(&format!("site {}  {name}: {}", site_name(&s["template"], &s["phase"]), site_line(s)));
        for f in s["facts"].as_array().unwrap() {
            io.say(&format!("  {}", py(&f["ask"])));
            loop {
                let Some(line) = io.ask("number (Enter for don't know): ") else { finish!() };
                let line = line.trim();
                if line.is_empty() {
                    break;
                }
                let Some(v) = number(line).filter(|v| *v >= 1) else {
                    io.say("  a whole number of at least 1, or Enter for don't know");
                    continue;
                };
                if let Some(why) = fact_check(f, v) {
                    io.say(&format!("  {why}"));
                    continue;
                }
                let a: Map<String, Value> = [("template".to_string(), json!(name)), ("fact".to_string(), f["name"].clone()), ("value".to_string(), json!(v))].into_iter().collect();
                answers.push(with_phase(a, &s["phase"]));
                break;
            }
        }
    }
    let doc = document(&answers, &state);
    for it in doc["open"].as_array().unwrap() {
        let phase = it.get("phase").cloned().unwrap_or(Value::Null);
        let s = doc["sites"].as_array().unwrap().iter().find(|x| x["template"] == it["site"] && x["phase"] == phase).unwrap();
        let name = py(&doc["templates"][it["site"].as_u64().unwrap() as usize]["template"]);
        let cands: Vec<String> = it["candidates"].as_array().unwrap().iter().map(py).collect();
        io.say("");
        io.say(&format!("site {}  {name}: {}; {} undecided", site_name(&it["site"], &phase), site_line(s), py(&it["question"])));
        io.say(&format!("  {}", py(&it["ask"])));
        for (i, c) in cands.iter().enumerate() {
            io.say(&format!("    {}. {c}: {}", i + 1, label(c)));
        }
        io.say(&format!("    {}. don't know", cands.len() + 1));
        let sc = &it["at_scale"];
        if let Some(c) = sc.get("choose") {
            io.say(&format!(
                "  at {} pass(es): a draw with replacement never reads {} of the set and re-reads {} of what it touches; a permutation reads every file each pass",
                py(&sc["epochs"]),
                pct(c["never_read_share"].as_f64().unwrap_or(0.0), 1),
                pct(c["repeat_share"].as_f64().unwrap_or(0.0), 1)
            ));
        } else {
            io.say(&format!("  at {} pass(es): {}", py(&sc["epochs"]), py(&sc["note"])));
        }
        io.say(&format!("  settles: {}", py(&it["settles"])));
        let n = loop {
            let Some(line) = io.ask(&format!("answer [1-{}]: ", cands.len() + 1)) else { finish!() };
            match number(line.trim()) {
                Some(k) if k >= 1 && k as usize <= cands.len() + 1 => break k as usize,
                _ => io.say(&format!("  a number from 1 to {}", cands.len() + 1)),
            }
        };
        let ans = cands.get(n - 1).cloned().unwrap_or_else(|| UNKNOWN.to_string());
        let a: Map<String, Value> = [("template".to_string(), json!(name)), ("question".to_string(), it["question"].clone()), ("answer".to_string(), json!(ans))].into_iter().collect();
        answers.push(with_phase(a, &phase));
        let Some(note) = io.ask("how do you know? (Enter to skip): ") else { finish!() };
        if !note.trim().is_empty() {
            answers.last_mut().unwrap()["note"] = json!(note.trim());
        }
    }
    finish!()
}

/// The answers file (`aeiou_answers: 1`): the trace answered, its hash, who answered and when, the
/// application stated or identified, and the answers.
pub fn write_answers(path: &str, answers: &[Value], trace: &str, trace_sha256: &str, by: &str, at: &str, application: &Value) -> Result<()> {
    let doc = json!({"aeiou_answers": ANSWERS_FORMAT, "trace": trace, "trace_sha256": trace_sha256, "by": by, "at": at, "application": application, "answers": answers});
    std::fs::write(path, serde_json::to_string_pretty(&doc)? + "\n").map_err(|e| anyhow!("{path}: {e}"))
}

/// The answers file: its header and its answers.
pub fn read_answers(path: &str) -> Result<(Value, Vec<Value>)> {
    let text = std::fs::read_to_string(path).map_err(|e| anyhow!("{path}: {e}"))?;
    let doc: Value = serde_json::from_str(&text).map_err(|e| anyhow!("{path}: {e}"))?;
    if doc.get("aeiou_answers").and_then(Value::as_u64) != Some(ANSWERS_FORMAT) {
        return Err(anyhow!("{path}: not an aeiou_answers: {ANSWERS_FORMAT} file"));
    }
    match doc.get("answers").and_then(Value::as_array) {
        Some(a) if a.iter().all(Value::is_object) => Ok((doc.clone(), a.clone())),
        _ => Err(anyhow!("{path}: answers must be a list of objects")),
    }
}
