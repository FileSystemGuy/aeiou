//! The application records (`applications/`, `DESIGN_REVIEW.md` §3.88, in Rust since §3.89
//! stage 3): what `aeiou-trace digest` knows about the applications whose traces it reads. Every
//! committed record is walked; the loader's refusals, the signature matching, the version
//! resolution, and the assertions a record makes are held to constructed trees (the Python
//! loader's tests, `builder/tests/test_applications.py`, ported).
use std::path::{Path, PathBuf};

use aeiou::applications::{self, Records};
use serde_json::{json, Map, Value};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../applications")
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let p = e.path();
        if p.is_dir() {
            copy_tree(&p, &to.join(e.file_name()));
        } else {
            std::fs::copy(&p, to.join(e.file_name())).unwrap();
        }
    }
}

/// A copy of the committed records to break.
fn tree(tag: &str) -> (crate::common::TempDir, PathBuf) {
    let d = crate::common::tmpdir("applications", tag);
    let t = d.join("applications");
    copy_tree(&repo(), &t);
    (d, t)
}

fn edit(p: &Path, old: &str, new: &str) {
    let text = std::fs::read_to_string(p).unwrap();
    assert!(text.contains(old), "{old}");
    std::fs::write(p, text.replacen(old, new, 1)).unwrap();
}

fn names(r: &Records, found: Vec<(&applications::Application, &str)>) -> Vec<String> {
    let _ = r;
    found.into_iter().map(|(a, _)| a.name.clone()).collect()
}

#[test]
fn the_shipped_records_are_the_checkouts() {
    let shipped = applications::shipped();
    let checkout = applications::load(&repo()).unwrap();
    assert_eq!(shipped.apps.keys().collect::<Vec<_>>(), checkout.apps.keys().collect::<Vec<_>>());
    for (name, a) in &checkout.apps {
        let b = &shipped.apps[name];
        assert_eq!(a.records.iter().map(|r| &r.version).collect::<Vec<_>>(), b.records.iter().map(|r| &r.version).collect::<Vec<_>>(), "{name}");
        assert_eq!(a.signatures, b.signatures, "{name}");
    }
}

#[test]
fn every_committed_record_loads() {
    let r = applications::load(&repo()).unwrap();
    for name in ["milvus", "qdrant", "pytorch-dcp", "lmcache", "safetensors", "diskann", "faiss"] {
        assert!(r.apps.contains_key(name), "{name}");
    }
    for (name, a) in &r.apps {
        assert!(&a.name == name && ["library", "service", "managed"].contains(&a.kind.as_str()) && !a.records.is_empty(), "{name}");
        for rec in &a.records {
            assert!(!rec.read.is_empty() && !rec.sources.is_empty(), "{name} {}", rec.version);
            for o in &rec.options {
                assert!(!o.choices.is_empty() && o.default.as_ref().is_none_or(|d| o.choices.contains(d)));
            }
        }
    }
    assert!(r.apps["faiss"].signatures.is_empty());
    assert_eq!(r.apps["pytorch-dcp"].mechanisms, ["checkpointing/dcp-save", "checkpointing/dcp-load"]);
}

#[test]
fn identification_by_path_signatures() {
    let r = applications::load(&repo()).unwrap();
    let found = r.identify(["ckpt/step_000001/__0_0.distcp", "ckpt/step_000001/.metadata"]);
    assert_eq!(found.iter().map(|(a, s)| (a.name.as_str(), *s)).collect::<Vec<_>>(), [("pytorch-dcp", "__*_*.distcp")]);
    assert_eq!(names(&r, r.identify([".metadata"])), ["pytorch-dcp"]); // a signature matches at the root too
    assert_eq!(names(&r, r.identify(["kv/meta-llama-Llama-3.1-8B-Instruct@1@0@3f2a9c0e1b7d4e6f@torch.bfloat16.pt"])), ["lmcache"]);
    assert_eq!(names(&r, r.identify(["files/insert_log/1/2/3/4/5", "a/b/sift_disk.index"])), ["diskann", "milvus"]);
    assert!(r.identify(["ivf/merged_index.ivfdata", "train/00000/sample_000000001.npz"]).is_empty());
    assert_eq!(names(&r, r.identify(["storage/collections/c/0/segments/u/vector_storage/0.dat"])), ["qdrant"]);
}

#[test]
fn version_resolution() {
    let r = applications::load(&repo()).unwrap();
    let dcp = &r.apps["pytorch-dcp"];
    let v = |x: Option<&str>| {
        let (rec, exact) = dcp.resolve(x);
        (rec.map(|r| r.version.clone()), exact)
    };
    assert_eq!(v(Some("2.14.1")), (Some("2.14".into()), true));
    assert_eq!(v(Some("2.15.0")), (Some("2.14".into()), true)); // no `until`: the record claims every later version until one says otherwise
    assert_eq!(v(Some("2.9")), (None, false)); // below the lowest record: nothing claims it
    assert_eq!(v(None), (Some("2.14".into()), false)); // unknown: the newest record, as a suggestion
    assert_eq!(v(Some("two")), (Some("2.14".into()), false));
    assert_eq!(applications::version_key("3.0.3"), Some(vec![3, 0, 3]));
    assert_eq!(applications::version_key("v1.20.0"), Some(vec![1, 20, 0]));
    assert_eq!(applications::version_key("1.20rc1"), Some(vec![1, 20]));
    assert_eq!(applications::version_key(" 2.1. "), Some(vec![2, 1]));
    assert_eq!(applications::version_key("latest"), None);
    assert_eq!(applications::version_key(""), None);
}

#[test]
fn a_range_with_an_end_and_the_nearest_lower_record() {
    let (_d, t) = tree("range");
    edit(&t.join("pytorch-dcp/2.14.toml"), "from = \"2.14\"\n", "from = \"2.14\"\nuntil = \"2.15\"\n");
    std::fs::write(t.join("pytorch-dcp/2.16.toml"), "from = \"2.16\"\nread = \"2026-10-29\"\nsources = [\"x\"]\n").unwrap();
    let r = applications::load(&t).unwrap();
    let dcp = &r.apps["pytorch-dcp"];
    assert_eq!(dcp.records.iter().map(|r| r.version.as_str()).collect::<Vec<_>>(), ["2.14", "2.16"]);
    let v = |x: Option<&str>| {
        let (rec, exact) = dcp.resolve(x);
        (rec.map(|r| r.version.clone()), exact)
    };
    assert_eq!(v(Some("2.14.9")), (Some("2.14".into()), true));
    assert_eq!(v(Some("2.15.2")), (Some("2.14".into()), false)); // between the ranges: the nearest lower, downgraded
    assert_eq!(v(Some("2.16")), (Some("2.16".into()), true));
    assert_eq!(v(None), (Some("2.16".into()), false));
}

#[test]
fn records_are_refused_with_every_problem_named() {
    let cases: &[(&str, &str, &str, &str)] = &[
        ("milvus/application.toml", "kind = \"service\"", "kind = \"cloud\"", "kind must be one of library, service, managed"),
        ("milvus/application.toml", "name = \"milvus\"", "name = \"milvos\"", "name must be the directory's, milvus"),
        ("milvus/application.toml", "display = \"Milvus\"\n", "", "display is required"),
        ("milvus/application.toml", "signatures = [", "signatures = \"x\"\nold = [", "signatures must be a list of strings"),
        ("milvus/3.0.toml", "from = \"3.0\"", "from = \"3.1\"", "from must be the file's stem, 3.0"),
        ("milvus/3.0.toml", "from = \"3.0\"\n", "from = \"3.0\"\nuntil = \"2.9\"\n", "until must be above from"),
        ("milvus/3.0.toml", "default = \"in-memory index\"", "default = \"GPU index\"", "option index: default must be one of its choices"),
        ("milvus/3.0.toml", "\"memory-mapped\" = \"mmap\"", "\"memory-mapped\" = \"swap\"", "option mmap_vector_field: blind spot swap is not in applications/attributes.toml"),
        ("milvus/3.0.toml", "\"memory-mapped\" = \"mmap\"", "\"mapped\" = \"mmap\"", "option mmap_vector_field: blind spot for a choice it does not have, mapped"),
        ("milvus/3.0.toml", "confidence = \"medium\"", "confidence = \"guess\"", "confidence must be high or medium"),
        ("milvus/3.0.toml", "kind = \"object\"", "kind = \"network\"", "blind spot network is not in applications/attributes.toml"),
        ("pytorch-dcp/2.14.toml", "question = \"size\"", "question = \"colour\"", "assert 1: question must be one of selection, size, order"),
        ("pytorch-dcp/2.14.toml", "answer = \"per-path\"", "answer = \"huge\"", "assert 1: answer must be one of per-path, drawn"),
        ("pytorch-dcp/2.14.toml", "template = \"__*_*.distcp\"\n", "template = \"__*_*.distcp\"\nwhen = { option = \"sync\", choice = \"fsync on\" }\n", "assert 1: when names no option of the record, sync"),
        ("pytorch-dcp/2.14.toml", "template = \"__*_*.distcp\"\n", "template = \"__*_*.distcp\"\nwhen = { option = \"sync_files\", choice = \"maybe\" }\n", "assert 1: when names no choice of option sync_files, maybe"),
        ("pytorch-dcp/2.14.toml", "question = \"size\"\nanswer = \"per-path\"", "fact = \"passes\"\nvalue = \"four\"", "assert 1: value must be a whole number of at least 1"),
        ("pytorch-dcp/2.14.toml", "question = \"size\"\nanswer = \"per-path\"", "fact = \"passes\"\nvalue = 0", "assert 1: value must be a whole number of at least 1"),
        ("pytorch-dcp/2.14.toml", "question = \"size\"\nanswer = \"per-path\"", "fact = \"sizes\"\nvalue = 4", "assert 1: fact must be one of candidates, passes"),
        ("pytorch-dcp/2.14.toml", "question = \"size\"\nanswer = \"per-path\"", "extra = \"nothing\"", "assert 1: either fact and value or question and answer"),
        ("pytorch-dcp/2.14.toml", "read = \"2026-10-29\"\n", "", "read is required"),
        ("pytorch-dcp/2.14.toml", "sources = [", "sources = \"x\"\nold = [", "sources must be a non-empty list of strings"),
        ("pytorch-dcp/2.14.toml", "name = \"sync_files\"\n", "", "option 2: name and question are required"),
        ("pytorch-dcp/2.14.toml", "choices = [\"fsync on\", \"fsync off\"]", "choices = [\"fsync on\"]", "option sync_files: choices must be a list of at least two strings"),
        ("safetensors/0.8.toml", "confidence = \"high\"", "confidence = \"sure\"", "blind spot 1: confidence must be high or medium"),
        ("pytorch-dcp/2.14.toml", "template = \"__*_*.distcp\"\n", "", "assert 1: template is required"),
        (
            "pytorch-dcp/2.14.toml",
            "source = \"workloads/checkpointing/dcp-load/kit/params.py (the plan read from .metadata)\"\nconfidence = \"high\"\n",
            "source = \"x\"\n",
            "assert 1: confidence must be high or medium",
        ),
        ("milvus/application.toml", "name = \"milvus\"", "name = = \"milvus\"", "TOML"),
        ("pytorch-dcp/2.14.toml", "sources = [", "sources = [[", "TOML"),
    ];
    for (i, (file, old, new, message)) in cases.iter().enumerate() {
        let (_d, t) = tree(&format!("refused-{i}"));
        edit(&t.join(file), old, new);
        let e = format!("{:#}", applications::load(&t).err().unwrap_or_else(|| panic!("{file}: {new} was accepted")));
        assert!(e.contains(message), "{file}: want {message}, got {e}");
        assert!(e.lines().any(|l| l.starts_with(&format!("{file}: "))), "{file} named: {e}");
    }
    // every problem at once, one a line
    let (_d, t) = tree("several");
    edit(&t.join("milvus/application.toml"), "kind = \"service\"", "kind = \"cloud\"");
    edit(&t.join("pytorch-dcp/2.14.toml"), "read = \"2026-10-29\"\n", "");
    let e = format!("{:#}", applications::load(&t).err().unwrap());
    // the record refused leaves its application none, which is said too
    assert_eq!(e.lines().collect::<Vec<_>>(), ["milvus/application.toml: kind must be one of library, service, managed", "pytorch-dcp/2.14.toml: read is required", "pytorch-dcp: no <version>.toml"]);
}

#[test]
fn a_directory_without_a_record_or_a_stray_file_is_refused() {
    let (_d, t) = tree("stray");
    std::fs::create_dir(t.join("nothing")).unwrap();
    assert!(format!("{:#}", applications::load(&t).err().unwrap()).contains("nothing: no application.toml"));
    std::fs::remove_dir(t.join("nothing")).unwrap();
    std::fs::write(t.join("milvus/notes.txt"), "x").unwrap();
    assert!(format!("{:#}", applications::load(&t).err().unwrap()).contains("milvus/notes.txt: not application.toml or a <version>.toml"));
    std::fs::remove_file(t.join("milvus/notes.txt")).unwrap();
    std::fs::create_dir(t.join("empty")).unwrap();
    std::fs::write(t.join("empty/application.toml"), "name = \"empty\"\ndisplay = \"E\"\nkind = \"library\"\nsummary = \"s\"\n").unwrap();
    assert!(format!("{:#}", applications::load(&t).err().unwrap()).contains("empty: no <version>.toml"));
    std::fs::remove_dir_all(t.join("empty")).unwrap();
    std::fs::remove_file(t.join("attributes.toml")).unwrap();
    assert!(format!("{:#}", applications::load(&t).err().unwrap()).contains("attributes.toml"));
    let absent = t.parent().unwrap().join("absent");
    assert_eq!(format!("{:#}", applications::load(&absent).err().unwrap()), format!("{}: no such directory of application records", absent.display()));
}

#[test]
fn assertions_and_blind_spots_of_a_record() {
    let (_d, t) = tree("assert");
    let p = t.join("pytorch-dcp/2.14.toml");
    let text = std::fs::read_to_string(&p).unwrap();
    std::fs::write(
        &p,
        text + "\n[[assert]]\ntemplate = \"__*_*.distcp\"\nfact = \"passes\"\nvalue = 3\nnote = \"only when fsync is off\"\nsource = \"made up for the test\"\nconfidence = \"medium\"\nwhen = { option = \"sync_files\", choice = \"fsync off\" }\n",
    )
    .unwrap();
    let r = applications::load(&t).unwrap();
    let rec = &r.apps["pytorch-dcp"].records[0];
    let templates = vec![
        ("ckpt/step_{a:06}/__{b}_0.distcp".to_string(), vec!["ckpt/step_000001/__0_0.distcp".to_string(), "ckpt/step_000001/__1_0.distcp".to_string()]),
        ("ckpt/step_{a:06}/.metadata".to_string(), vec!["ckpt/step_000001/.metadata".to_string()]),
    ];
    let got = rec.assertions(&Map::new(), &templates);
    assert_eq!(
        Value::Array(got),
        json!([
            {"template": "ckpt/step_{a:06}/__{b}_0.distcp", "question": "size", "answer": "per-path",
             "note": "a rank's shard holds the bytes of its plan's items; the bytes are a property of the file, the same at every load",
             "by": "PyTorch Distributed Checkpoint 2.14 record, read 2026-10-29 (high confidence)", "source": "workloads/checkpointing/dcp-load/kit/params.py (the plan read from .metadata)"},
            {"template": "ckpt/step_{a:06}/__{b}_0.distcp", "question": "label", "answer": "checkpoint",
             "note": "the shards are a training job's state, saved to survive a failure and loaded to resume",
             "by": "PyTorch Distributed Checkpoint 2.14 record, read 2026-10-29 (high confidence)", "source": "https://docs.pytorch.org/docs/stable/distributed.checkpoint.html"},
        ])
    );
    let off: Map<String, Value> = [("sync_files".to_string(), json!("fsync off"))].into_iter().collect();
    let got = rec.assertions(&off, &templates);
    assert_eq!(got.iter().map(|a| a.get("fact").or(a.get("question")).unwrap().as_str().unwrap()).collect::<Vec<_>>(), ["size", "label", "passes"]);
    assert_eq!(got[2]["value"], json!(3));
    assert!(got[2]["by"].as_str().unwrap().contains("medium confidence"));
    assert!(rec.assertions(&Map::new(), &[("other/{a}.bin".into(), vec!["other/1.bin".into()])]).is_empty());
    let milvus = &r.apps["milvus"].records[0];
    let chosen: Map<String, Value> = [("mmap_vector_field".to_string(), json!("memory-mapped")), ("mmap_vector_index".to_string(), json!("loaded into memory"))].into_iter().collect();
    let spots = r.blind_spots(milvus, &chosen);
    assert_eq!(spots.iter().map(|s| (s["kind"].as_str().unwrap(), s.get("option").and_then(Value::as_str))).collect::<Vec<_>>(), [("object", None), ("mmap", Some("mmap_vector_field"))]);
    assert!(spots[1]["note"].as_str().unwrap().starts_with("the application maps its files"));
    assert_eq!(milvus.provenance(), "Milvus 3.0 record, read 2026-10-29");
}
