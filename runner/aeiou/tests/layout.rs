//! Contract 0.2: container layouts (`format.layout`), unit and column handles, `stream`
//! access (the shuffle over shards), and the `fadvise` op, against hand-written ASTs whose
//! geometry is computed by hand here.

use std::collections::BTreeSet;

use aeiou::backend::BackendKind;
use aeiou::datagen::{datagen, DatagenOpts};
use aeiou::dryrun;
use aeiou::eval::{build_model, Config, Model, Params};
use aeiou::run::{self, RunOpts};
use aeiou::vm::{run_actor, Op, OpCtx, OpKind, Sink};

fn config(gpus: i64) -> Config {
    Config { seed: 1, gpus, overrides: vec![], sets: vec![], st_blksize: Default::default() }
}

#[derive(Default)]
struct Collect {
    ops: Vec<(i64, OpKind, String, i64, i64, i64)>,
}

impl<'m, 'a> Sink<'m, 'a> for Collect {
    fn op(&mut self, op: &Op, _ctx: &OpCtx) -> anyhow::Result<()> {
        self.ops.push((_ctx.actor, op.kind, op.path.to_string(), op.offset, op.len, op.bytes));
        Ok(())
    }
}

fn collect(text: &str, cfg: &Config, count: i64) -> Vec<(i64, OpKind, String, i64, i64, i64)> {
    let loaded = aeiou::load_str(text).expect("valid abstract");
    let params = Params::new(&loaded.ast, cfg).unwrap();
    let model = build_model(&loaded.ast, cfg, &params).unwrap();
    let mut all = Vec::new();
    for g in 0..count {
        all.extend(run_actor(&model, "gpu", g, count, Collect::default()).unwrap().ops);
    }
    all
}

/// A TFRecord-shaped shard dataset: 100 records of 1000 bytes in shards of 10, each record
/// framed by a 12-byte header and a 4-byte footer, read sequentially under `stream`.
const TFRECORD: &str = r#"{
  "ast": "0.12", "name": "stream_test",
  "params": {"batches": {"default": 10}, "xfer": {"default": 4096}},
  "datasets": {"shards": {"layout": "compound", "pattern": "s/{id:04}.tfrecord", "count": 100, "samples_per_file": 10,
     "size": {"const": 1000}, "seed": 7, "access": "stream",
     "format": {"class": "tfrecord", "reader": "tf.data", "framing": {"columns": [{"row_header": 12, "row_footer": 4, "weight": 1}]}}}},
  "actors": {"gpu": {"body": [
    {"loader": {"name": "q", "index": "b", "workers": 2, "prefetch": 1, "batches": {"param": "batches"}, "body": [
      {"loop": {"index": "j", "to": 1, "body": [
        {"let": {"name": "f", "value": {"consume": "shards"}}},
        {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}},
        {"read": {"file": {"ref": "f"}, "len": {"param": "xfer"}, "repeat": "until_eof"}},
        {"close": {"file": {"ref": "f"}}}
      ]}}
    ]}},
    {"loop": {"index": "s", "to": {"param": "batches"}, "body": [{"take": {"channel": "q"}}]}}
  ]}}
}"#;

#[test]
fn stream_consume_shuffles_shards_and_frames_records() {
    let ops = collect(TFRECORD, &config(2), 2);
    // 10 shards × (10 records × 1016 bytes) = 10160 bytes: three full reads of 4 KiB, a short one, an EOF read
    let opens: Vec<&(i64, OpKind, String, i64, i64, i64)> = ops.iter().filter(|o| o.1 == OpKind::Open).collect();
    assert_eq!(opens.len(), 20, "10 batches on each of 2 actors");
    for o in &ops {
        if o.1 == OpKind::Read {
            assert!(o.5 == 4096 || o.5 == 10160 - 2 * 4096 || o.5 == 0, "{o:?}");
        }
    }
    let reads = ops.iter().filter(|o| o.1 == OpKind::Read).count();
    assert_eq!(reads, 20 * 4);
    // the epoch is 10 shards ÷ (2 actors × 1 per batch) = 5 batches: every shard once per epoch,
    // across the two actors, and the two epochs use different orders
    let epoch = |e: i64| -> BTreeSet<String> {
        let mut per_actor_batches = std::collections::HashMap::new();
        let mut names = BTreeSet::new();
        for o in opens.iter() {
            let b = per_actor_batches.entry(o.0).or_insert(0i64);
            if *b / 5 == e {
                names.insert(o.2.clone());
            }
            *b += 1;
        }
        names
    };
    assert_eq!(epoch(0).len(), 10);
    assert_eq!(epoch(1).len(), 10);
    let order0: Vec<String> = opens.iter().take(5).map(|o| o.2.clone()).collect();
    let order1: Vec<String> = opens.iter().skip(5).take(5).map(|o| o.2.clone()).collect();
    assert_ne!(order0, order1, "epochs are permuted differently");
}

/// A Parquet-shaped file: 192 rows in row groups of 64; two column chunks per group (a
/// binary column with a 4-byte length prefix per value and a 26-byte page header, an int64
/// column of 8 bytes per row and a 22-byte page header); 4 magic bytes before the groups and a
/// footer of 200 + 120 per group after them.
const PARQUET: &str = r#"{
  "ast": "0.12", "name": "parquet_test",
  "params": {"groups": {"default": 3}},
  "datasets": {"t": {"layout": "compound", "pattern": "p/{id:04}.parquet", "count": 192, "samples_per_file": 192,
     "size": {"const": 100000}, "seed": 7, "access": "stream",
     "format": {"class": "parquet", "reader": "pyarrow", "version": "25.0.1", "framing": {
       "unit": 64, "file_header": 4, "file_footer": 200, "file_footer_per_unit": 120,
       "columns": [{"header": 26, "fixed": 4, "weight": 1.0}, {"header": 22, "fixed": 8, "weight": 0}],
       "writer": {"rows_per_group": 64}}}}},
  "actors": {"gpu": {"body": [
    {"let": {"name": "f", "value": {"file": {"dataset": "t", "id": 0}}}},
    {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}},
    {"read": {"file": {"ref": "f"}, "len": 65536, "offset": {"sub": [{"size": {"ref": "f"}}, 65536]}}},
    {"loop": {"index": "g", "to": {"units": {"ref": "f"}}, "body": [
      {"let": {"name": "u", "value": {"unit": {"of": {"ref": "f"}, "index": {"index": "g"}}}}},
      {"fadvise": {"file": {"ref": "f"}, "advice": "WILLNEED", "offset": {"offset": {"ref": "u"}}, "len": {"size": {"ref": "u"}}}},
      {"read": {"file": {"ref": "f"}, "len": {"size": {"ref": "u"}}, "offset": {"offset": {"ref": "u"}}}},
      {"let": {"name": "c", "value": {"column": {"of": {"ref": "u"}, "index": 1}}}},
      {"read": {"file": {"ref": "f"}, "len": {"size": {"ref": "c"}}, "offset": {"offset": {"ref": "c"}}}}
    ]}},
    {"close": {"file": {"ref": "f"}}}
  ]}}
}"#;

#[test]
fn framed_layout_geometry_by_hand() {
    let loaded = aeiou::load_str(PARQUET).unwrap();
    let cfg = config(1);
    let params = Params::new(&loaded.ast, &cfg).unwrap();
    let model: Model = build_model(&loaded.ast, &cfg, &params).unwrap();
    let ds = &model.datasets[0];
    let col0 = 26 + 64 * (4 + 100000); // 6400282
    let col1 = 22 + 64 * 8; // 534
    let unit = col0 + col1;
    assert_eq!(ds.units_in_file(0).unwrap(), 3);
    assert_eq!(ds.col_len(0, 0, 0).unwrap(), col0);
    assert_eq!(ds.col_len(0, 2, 1).unwrap(), col1);
    assert_eq!(ds.unit_len(0, 1).unwrap(), unit);
    assert_eq!(ds.unit_offset(0, 0).unwrap(), 4);
    assert_eq!(ds.unit_offset(0, 2).unwrap(), 4 + 2 * unit);
    assert_eq!(ds.col_offset_in_unit(0, 0, 1).unwrap(), col0);
    assert_eq!(ds.file_size(0).unwrap(), 4 + 3 * unit + 200 + 3 * 120);
    assert_eq!(ds.unit_of_sample(130).unwrap(), 2);
    let err = ds.sample_offset(5).unwrap_err().to_string();
    assert!(err.contains("split over 2 columns"), "{err}");
    assert!(ds.samples_in_unit(0, 3).is_err());

    let ops = collect(PARQUET, &cfg, 1);
    let size = 4 + 3 * unit + 200 + 3 * 120;
    let reads: Vec<_> = ops.iter().filter(|o| o.1 == OpKind::Read).collect();
    assert_eq!(reads[0].3, size - 65536, "footer read at EOF − 64 KiB");
    assert_eq!(reads[0].5, 65536);
    for g in 0..3i64 {
        let rg = &reads[1 + 2 * g as usize];
        assert_eq!((rg.3, rg.4, rg.5), (4 + g * unit, unit, unit), "row group {g}");
        let lab = &reads[2 + 2 * g as usize];
        assert_eq!((lab.3, lab.4), (4 + g * unit + col0, col1), "label chunk {g}");
    }
    let adv: Vec<_> = ops.iter().filter(|o| o.1 == OpKind::Fadvise).collect();
    assert_eq!(adv.len(), 3);
    assert_eq!((adv[1].3, adv[1].4), (4 + unit, unit));
}

/// Alignment: tar members (512-byte header, data padded to 512), the archive padded to
/// 10240 after two zero blocks.
const TAR: &str = r#"{
  "ast": "0.12", "name": "tar_test",
  "datasets": {"t": {"layout": "compound", "pattern": "w/{id:04}.tar", "count": 25, "samples_per_file": 10,
     "size": {"const": 1000}, "seed": 7, "access": "stream",
     "format": {"class": "webdataset", "framing": {"file_footer": 1024, "file_align": 10240,
       "columns": [{"row_header": 512, "row_align": 512, "weight": 1}]}}}},
  "actors": {"gpu": {"body": [
    {"let": {"name": "f", "value": {"file": {"dataset": "t", "id": 2}}}},
    {"let": {"name": "s", "value": {"pick": {"dataset": "t"}}}},
    {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}},
    {"read": {"file": {"ref": "f"}, "len": 1048576, "repeat": "until_eof"}},
    {"close": {"file": {"ref": "f"}}}
  ]}}
}"#;

#[test]
fn alignment_and_short_last_shard() {
    let loaded = aeiou::load_str(TAR).unwrap();
    let cfg = config(1);
    let params = Params::new(&loaded.ast, &cfg).unwrap();
    let model = build_model(&loaded.ast, &cfg, &params).unwrap();
    let ds = &model.datasets[0];
    // 10 members × 1536 + 1024 = 16384 → 20480; the last shard has 5 members: 7680 + 1024 → 10240
    assert_eq!(ds.file_size(0).unwrap(), 20480);
    assert_eq!(ds.file_size(2).unwrap(), 10240);
    assert_eq!(ds.files(), 3);
    // a sample's offset is the start of its payload: after its own 512-byte header
    assert_eq!(ds.sample_offset(0).unwrap(), 512);
    assert_eq!(ds.sample_offset(1).unwrap(), 1536 + 512);
    assert_eq!(ds.sample_offset(21).unwrap(), 1536 + 512);
    let ops = collect(TAR, &cfg, 1);
    let reads: Vec<_> = ops.iter().filter(|o| o.1 == OpKind::Read).collect();
    assert_eq!(reads.len(), 2);
    assert_eq!(reads[0].5, 10240);
    assert_eq!(reads[1].5, 0);
}

/// `fadvise` runs against a real file and is part of the fingerprint.
const FADVISE: &str = r#"{
  "ast": "0.12", "name": "fadvise_test",
  "datasets": {"d": {"layout": "whole", "access": "map", "pattern": "d/{id:04}", "count": 4, "size": {"const": 20000}, "seed": 3}},
  "actors": {"gpu": {"body": [
    {"loop": {"index": "i", "to": 4, "body": [
      {"let": {"name": "f", "value": {"file": {"dataset": "d", "id": {"index": "i"}}}}},
      {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}},
      {"fadvise": {"file": {"ref": "f"}, "advice": "WILLNEED", "len": {"size": {"ref": "f"}}}},
      {"fadvise": {"file": {"ref": "f"}, "advice": "SEQUENTIAL"}},
      {"read": {"file": {"ref": "f"}, "len": 20000}},
      {"fadvise": {"file": {"ref": "f"}, "advice": "DONTNEED", "offset": 0, "len": 20000}},
      {"close": {"file": {"ref": "f"}}}
    ]}}
  ]}}
}"#;

#[test]
fn fadvise_runs_and_is_fingerprinted() {
    let root = crate::common::tmpdir("layout", "fadvise");
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load_str(FADVISE).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(config(1)));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
    let opts = DatagenOpts { root: root.clone(), endpoints: Default::default(), writers: 2, dedupe: 1, compress: 1, datasets: vec![], rank: 0, ranks: 1, progress: std::time::Duration::from_secs(30), resume: None, resize: None, settle: None, sigint: false, ignore_space: false };
    datagen(loaded, cfg, params, model, &opts, None, &mut Vec::new()).unwrap();
    let dry = dryrun::run(model, 1, None).unwrap();
    assert_eq!(dry.total.total.ops, 4 * 6);
    let r = run::run(
        model,
        RunOpts {
            root: root.clone(),
            endpoints: Default::default(),
            backend: BackendKind::Sync,
            s3: aeiou::backend::S3Api::Blocking,
            buffer_bytes: 1 << 20,
        threads: 0,
            write_compress: 1,
            time_scale: 0.0,
            uring: Default::default(),
            mmap: Default::default(),
            mmap_consume: Default::default(),
            aio_depth: 0,
            clean_namespaces: false,
            expect_fingerprint: None,
            expect_dataset_ids: vec![],
            rank: 0,
            ranks: 1,
            rank_rotate: 0,
            max_gap: None,
            require_cold: false,
            drop_caches: false,
        },
        std::collections::HashMap::new(),
    )
    .unwrap();
    assert_eq!(r.stats.fingerprint, dry.total.fingerprint);
    assert_eq!(r.stats.counts.get(&OpKind::Fadvise).copied(), Some(12));
    // the advice is hashed: a different advice is a different workload
    let other = aeiou::load_str(&FADVISE.replace("\"SEQUENTIAL\"", "\"RANDOM\"")).unwrap();
    let p2 = Params::new(&other.ast, cfg).unwrap();
    let m2 = build_model(&other.ast, cfg, &p2).unwrap();
    assert_ne!(dryrun::run(&m2, 1, None).unwrap().total.fingerprint, dry.total.fingerprint);
}

#[test]
fn layout_rules_are_enforced() {
    let bad = PARQUET.replace("\"weight\": 1.0}, {\"header\": 22, \"fixed\": 8, \"weight\": 0}", "\"weight\": 0.5}, {\"header\": 22, \"fixed\": 8, \"weight\": 0}");
    let e = format!("{:#}", aeiou::load_str(&bad).err().expect("refused"));
    assert!(e.contains("weights sum to 0.5"), "{e}");
    let bad = PARQUET.replace("\"unit\": 64,", "\"unit\": 500,");
    let loaded = aeiou::load_str(&bad).unwrap();
    let cfg = config(1);
    let params = Params::new(&loaded.ast, &cfg).unwrap();
    let e = format!("{:#}", build_model(&loaded.ast, &cfg, &params).err().expect("refused"));
    assert!(e.contains("unit 500 must be in [1, samples_per_file = 192]"), "{e}");
    // a plain dataset keeps the packed layout: contract 0.1 geometry
    let plain = aeiou::load_str(FADVISE).unwrap();
    let p = Params::new(&plain.ast, &cfg).unwrap();
    let m = build_model(&plain.ast, &cfg, &p).unwrap();
    assert_eq!(m.datasets[0].file_size(1).unwrap(), 20000);
    assert_eq!(m.datasets[0].units_in_file(1).unwrap(), 1);
}

/// A Parquet dataset the writer can frame: 64-row groups of 100000-byte samples, 26-byte page
/// headers (the varint band of a 6.4 MB page), the footer of `parquet_stream`.
#[cfg(feature = "formats")]
const PARQUET_WRITE: &str = r#"{
  "ast": "0.12", "name": "parquet_write",
  "datasets": {"t": {"layout": "compound", "pattern": "p/{id:04}.parquet", "count": 384, "samples_per_file": 192,
     "size": {"const": 100000}, "seed": 7, "access": "stream",
     "format": {"class": "parquet", "reader": "pyarrow", "version": "25.0.1", "framing": {
       "unit": 64, "file_header": 4, "file_footer": 200, "file_footer_per_unit": 160,
       "columns": [{"header": 26, "fixed": 4, "weight": 1.0}, {"header": 22, "fixed": 8, "weight": 0}],
       "writer": {"rows_per_group": 64, "columns": [["image", "binary"], ["label", "int64"]]}}}}},
  "actors": {"gpu": {"body": [
    {"let": {"name": "f", "value": {"file": {"dataset": "t", "id": 0}}}},
    {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}},
    {"close": {"file": {"ref": "f"}}}
  ]}}
}"#;

/// Datagen of `text` into a fresh root: the error, if any, and every path left under the root.
#[cfg(feature = "formats")]
fn datagen_left(text: &str, tag: &str) -> (Option<String>, Vec<String>) {
    let root = crate::common::tmpdir("layout", tag);
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load_str(text).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(config(1)));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
    let opts = DatagenOpts { root: root.clone(), endpoints: Default::default(), writers: 1, dedupe: 1, compress: 1, datasets: vec![], rank: 0, ranks: 1, progress: std::time::Duration::from_secs(30), resume: None, resize: None, settle: None, sigint: false, ignore_space: true };
    let err = datagen(loaded, cfg, params, model, &opts, None, &mut Vec::new()).err().map(|e| format!("{e:#}"));
    let mut left: Vec<String> = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            left.push(p.strip_prefix(&*root).unwrap().display().to_string());
            if p.is_dir() {
                stack.push(p);
            }
        }
    }
    left.sort();
    (err, left)
}

/// A geometry the Parquet writer cannot frame is refused before datagen writes anything: no
/// manifest, no file, not even the first files the writer could make (RC1 review, bug 44).
#[cfg(feature = "formats")]
#[test]
fn parquet_geometry_refused_before_anything_is_written() {
    let (err, left) = datagen_left(PARQUET_WRITE, "pq-ok");
    assert_eq!(err, None);
    assert_eq!(left, ["p", "p/.aeiou-dataset.json", "p/0000.parquet", "p/0001.parquet"]);
    // a short last file: its one row group of 10 rows has a 23-byte page header, not the band's 26
    let (err, left) = datagen_left(&PARQUET_WRITE.replace("\"count\": 384", "\"count\": 202"), "pq-band");
    assert!(left.is_empty(), "left behind: {left:?}");
    let err = err.expect("refused");
    assert!(err.contains("dataset `t`: file p/0001.parquet: row group 0 column 0: a 23-byte page header, the layout predicts 26"), "{err}");
    // a 64-row page of 40 MB samples: past the 2 GiB a page header can say
    let big = PARQUET_WRITE.replace("\"count\": 384, \"samples_per_file\": 192", "\"count\": 64, \"samples_per_file\": 64").replace("\"const\": 100000", "\"const\": 40000000");
    let (err, left) = datagen_left(&big, "pq-big");
    assert!(left.is_empty(), "left behind: {left:?}");
    let err = err.expect("refused");
    assert!(err.contains("dataset `t`: file p/0000.parquet: a Parquet page of 2560000256 bytes: more than a page holds (2 GiB)"), "{err}");
}

/// A WebDataset of one 8 GiB sample per tar: past ustar's 11-digit octal size field.
#[cfg(feature = "formats")]
const TAR_WRITE: &str = r#"{
  "ast": "0.12", "name": "tar_write",
  "datasets": {"t": {"layout": "compound", "pattern": "w/{id:04}.tar", "count": 2, "samples_per_file": 1,
     "size": {"const": 8589934592}, "seed": 7, "access": "stream",
     "format": {"class": "webdataset", "reader": "tarfile (CPython)", "framing": {"file_footer": 1024, "file_align": 10240,
       "columns": [{"row_header": 512, "row_footer": 1024, "row_align": 512, "weight": 1}],
       "writer": {"members": [["jpg", null], ["cls", 4]]}}}}},
  "actors": {"gpu": {"body": [
    {"let": {"name": "f", "value": {"file": {"dataset": "t", "id": 0}}}},
    {"open": {"file": {"ref": "f"}, "flags": ["RDONLY"]}},
    {"close": {"file": {"ref": "f"}}}
  ]}}
}"#;

/// A tar member ustar's size field cannot say (8 GiB or more) is refused before datagen writes
/// anything, with the limit named, never a panic in the octal field (RC1 review).
#[cfg(feature = "formats")]
#[test]
fn tar_member_past_ustar_refused_before_anything_is_written() {
    let (err, left) = datagen_left(TAR_WRITE, "tar-big");
    assert!(left.is_empty(), "left behind: {left:?}");
    let err = err.expect("refused");
    assert!(err.contains("dataset `t`: file w/0000.tar: tar member `000000000.jpg` of 8589934592 bytes: more than ustar's 8 GiB − 1 (8589934591) size field"), "{err}");
}

/// A container dataset of `n` samples (a parameter, so a grow can change it) over `format`, and
/// a reader of every file, whole, to its end.
#[cfg(feature = "formats")]
fn container(pattern: &str, n: i64, spf: i64, size: i64, format: &str) -> String {
    format!(
        r#"{{"ast": "0.12", "name": "container_rt", "params": {{"n": {{"default": {n}}}}},
  "datasets": {{"d": {{"layout": "compound", "pattern": "{pattern}", "count": {{"param": "n"}}, "samples_per_file": {spf},
     "size": {{"const": {size}}}, "seed": 11, "access": "stream", "format": {format}}}}},
  "actors": {{"gpu": {{"body": [
    {{"loop": {{"index": "i", "to": {{"ceil_div": [{{"param": "n"}}, {spf}]}}, "body": [
      {{"let": {{"name": "f", "value": {{"file": {{"dataset": "d", "id": {{"index": "i"}}}}}}}}}},
      {{"open": {{"file": {{"ref": "f"}}, "flags": ["RDONLY"]}}}},
      {{"read": {{"file": {{"ref": "f"}}, "len": 65536, "repeat": "until_eof"}}}},
      {{"close": {{"file": {{"ref": "f"}}}}}}
    ]}}}}
  ]}}}}
}}"#
    )
}

#[cfg(feature = "formats")]
const FMT_TFRECORD: &str = r#"{"class": "tfrecord", "reader": "tf.data", "framing": {"columns": [{"row_header": 12, "row_footer": 4, "weight": 1}]}}"#;
#[cfg(feature = "formats")]
const FMT_WEBDATASET: &str = r#"{"class": "webdataset", "reader": "tarfile (CPython)", "framing": {"file_footer": 1024, "file_align": 10240,
  "columns": [{"row_header": 512, "row_footer": 1024, "row_align": 512, "weight": 1}], "writer": {"members": [["jpg", null], ["cls", 4]]}}}"#;
/// 4-row groups of 100-byte samples: page headers of 21 bytes (binary) and 19 (int64).
#[cfg(feature = "formats")]
const FMT_PARQUET: &str = r#"{"class": "parquet", "reader": "pyarrow", "version": "25.0.1", "framing": {"unit": 4, "file_header": 4, "file_footer": 2000,
  "columns": [{"header": 21, "fixed": 4, "weight": 1.0}, {"header": 19, "fixed": 8, "weight": 0}],
  "writer": {"rows_per_group": 4, "columns": [["image", "binary"], ["label", "int64"]]}}}"#;

/// The container cases: (tag, abstract of 25 samples, a short last file).
#[cfg(feature = "formats")]
fn containers() -> Vec<(&'static str, String)> {
    vec![
        ("tfrecord", container("s/{id:04}.tfrecord", 25, 10, 1000, FMT_TFRECORD)),
        ("webdataset", container("w/{id:04}.tar", 25, 10, 1000, FMT_WEBDATASET)),
        ("parquet", container("p/{id:04}.parquet", 25, 8, 100, FMT_PARQUET)),
    ]
}

/// Every file under `root` but the manifest, with its bytes.
#[cfg(feature = "formats")]
fn tree_bytes(root: &std::path::Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    let mut out = std::collections::BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if !p.ends_with(".aeiou-dataset.json") {
                out.insert(p.strip_prefix(root).unwrap().display().to_string(), std::fs::read(&p).unwrap());
            }
        }
    }
    out
}

/// Each writer of `aeiou datagen` (TFRecord, WebDataset, Parquet) makes files of the sizes the
/// layout computes, a short last file among them, that `aeiou run` reads to the dry run's
/// fingerprint, byte counts checked; and two datagens of one dataset make the same bytes.
#[cfg(feature = "formats")]
#[test]
fn container_datagen_reads_back_to_the_dry_run_fingerprint() {
    for (tag, text) in containers() {
        let root = crate::common::tmpdir("layout", &format!("rt-{tag}"));
        let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load_str(&text).unwrap()));
        let cfg: &'static Config = Box::leak(Box::new(config(1)));
        let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
        let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
        let ds = &model.datasets[0];
        let opts = DatagenOpts { root: root.clone(), endpoints: Default::default(), writers: 2, dedupe: 1, compress: 1, datasets: vec![], rank: 0, ranks: 1, progress: std::time::Duration::from_secs(30), resume: None, resize: None, settle: None, sigint: false, ignore_space: true };
        datagen(loaded, cfg, params, model, &opts, None, &mut Vec::new()).unwrap_or_else(|e| panic!("{tag}: {e:#}"));
        let files = tree_bytes(&root);
        assert_eq!(files.len() as i64, ds.files(), "{tag}");
        let mut total = 0;
        for f in 0..ds.files() {
            let size = files[&ds.file_path(f, None).unwrap().to_string()].len() as i64;
            assert_eq!(size, ds.file_size(f).unwrap(), "{tag}: file {f}");
            total += size;
        }
        assert!(ds.file_size(ds.files() - 1).unwrap() < ds.file_size(0).unwrap(), "{tag}: a short last file");
        let dry = dryrun::run(model, 1, None).unwrap();
        let r = run::run(
            model,
            RunOpts {
                root: root.clone(),
                endpoints: Default::default(),
                backend: BackendKind::Sync,
                s3: aeiou::backend::S3Api::Blocking,
                buffer_bytes: 1 << 20,
                threads: 0,
                write_compress: 1,
                time_scale: 0.0,
                uring: Default::default(),
                mmap: Default::default(),
                mmap_consume: Default::default(),
                aio_depth: 0,
                clean_namespaces: false,
                expect_fingerprint: None,
                expect_dataset_ids: vec![],
                rank: 0,
                ranks: 1,
                rank_rotate: 0,
                max_gap: None,
                require_cold: false,
                drop_caches: false,
            },
            std::collections::HashMap::new(),
        )
        .unwrap_or_else(|e| panic!("{tag}: {e:#}"));
        assert_eq!(r.stats.fingerprint, dry.total.fingerprint, "{tag}");
        assert_eq!(r.stats.bytes_read as i64, total, "{tag}");
        // the same dataset again, with one writer: the same bytes
        let again = crate::common::tmpdir("layout", &format!("rt-{tag}-again"));
        let opts = DatagenOpts { root: again.clone(), writers: 1, ..opts };
        datagen(loaded, cfg, params, model, &opts, None, &mut Vec::new()).unwrap();
        assert!(tree_bytes(&again) == files, "{tag}: a second datagen made other bytes");
    }
}

/// A container layout its writer does not make is refused before datagen writes anything,
/// naming the file and both sizes: a 10-byte TFRecord header where the writer's is 12, a
/// WebDataset framing without the blocks of its 4-byte member.
#[cfg(feature = "formats")]
#[test]
fn container_layout_drift_refused_before_anything_is_written() {
    let bad = container("s/{id:04}.tfrecord", 25, 10, 1000, &FMT_TFRECORD.replace("\"row_header\": 12", "\"row_header\": 10"));
    let (err, left) = datagen_left(&bad, "tfr-drift");
    assert!(left.is_empty(), "left behind: {left:?}");
    let err = err.expect("refused");
    assert!(err.contains("dataset `d`: file s/0000.tfrecord: the writer made 10160 bytes, the layout predicts 10140: the format class's layout does not match its writer"), "{err}");
    let bad = container("w/{id:04}.tar", 25, 10, 1000, &FMT_WEBDATASET.replace("\"row_footer\": 1024, ", ""));
    let (err, left) = datagen_left(&bad, "tar-drift");
    assert!(left.is_empty(), "left behind: {left:?}");
    let err = err.expect("refused");
    assert!(err.contains("dataset `d`: file w/0000.tar: the writer made 30720 bytes, the layout predicts 20480"), "{err}");
}

/// Datagen of an HDF5 dataset where no `aeiou-header` sits beside the test binary (None when
/// one does): the error and what is left under the root.
#[cfg(feature = "formats")]
fn hdf5_without_header_tool() -> Option<(Option<String>, Vec<String>)> {
    let beside = std::env::current_exe().unwrap().canonicalize().unwrap().with_file_name("aeiou-header");
    if beside.exists() {
        return None;
    }
    let h5 = container("h/{id:04}.h5", 10, 4, 300, r#"{"class": "hdf5", "reader": "h5py", "version": "test", "framing": {"file_header": 2048, "writer": {"dataset": "records", "shape": [300], "dtype": "u1"}}}"#);
    Some(datagen_left(&h5, "h5-notool"))
}

/// An HDF5 dataset needs `aeiou-header` beside the binary: without it, datagen is refused with
/// where the tool must go, before any file or manifest is written.
#[cfg(feature = "formats")]
#[test]
fn hdf5_without_its_header_tool_is_refused() {
    let Some((err, left)) = hdf5_without_header_tool() else { return };
    let err = err.expect("refused");
    assert!(err.contains("aeiou-header not found: aeiou-header makes HDF5 headers and must sit beside aeiou"), "{err}");
    assert!(left.iter().all(|p| p == "h"), "left behind: {left:?}");
}

/// The same refusal leaves nothing at all, as a geometry refusal does (bug 44's standard).
#[cfg(feature = "formats")]
#[test]
fn hdf5_without_its_header_tool_refused_before_anything_is_written() {
    let Some((err, left)) = hdf5_without_header_tool() else { return };
    assert!(err.is_some());
    assert!(left.is_empty(), "left behind: {left:?}");
}

/// `aeiou datagen` (the binary) with `args`: whether it succeeded, and what it printed.
#[cfg(feature = "formats")]
fn datagen_cli(args: &[&std::ffi::OsStr]) -> (bool, String) {
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_aeiou")).arg("datagen").args(args).args(["--writers", "2"]).env_remove("AEIOU_CONFIG").output().unwrap();
    (o.status.success(), format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)))
}

/// A grow and a shrink of a container corpus (`DESIGN_REVIEW.md` §3.70) leave the files a
/// datagen of the new count writes, byte for byte (the short last file made whole), and the
/// new count's id; a resume of the complete corpus has nothing to write.
#[cfg(feature = "formats")]
#[test]
fn container_grow_and_shrink_make_the_corpus_of_the_new_count() {
    for (tag, text) in containers() {
        let dir = crate::common::tmpdir("layout", &format!("resize-{tag}"));
        let ast = dir.join("container.ast.json");
        std::fs::write(&ast, &text).unwrap();
        let fresh = |n: i64| -> crate::common::TempDir {
            let r = crate::common::tmpdir("layout", &format!("fresh-{tag}-{n}"));
            let n = format!("n={n}");
            let (ok, out) = datagen_cli(&[ast.as_os_str(), "--root".as_ref(), r.as_os_str(), "--param".as_ref(), n.as_ref()]);
            assert!(ok, "{tag}: {out}");
            r
        };
        let (f45, f15) = (fresh(45), fresh(15));
        let root = fresh(25);
        let sub = std::fs::read_dir(&*root).unwrap().flatten().next().unwrap().path();
        let name = sub.file_name().unwrap().to_owned();
        let id = |r: &std::path::Path| aeiou::payload::Manifest::read(&r.join(&name)).unwrap();
        let (ok, out) = datagen_cli(&["--grow".as_ref(), "45".as_ref(), sub.as_os_str()]);
        assert!(ok, "{tag}: {out}");
        assert!(tree_bytes(&root) == tree_bytes(&f45), "{tag}: the grown corpus is not the 45-sample one");
        assert_eq!((id(&root).state, id(&root).id()), ("complete".to_string(), id(&f45).id()), "{tag}");
        let (ok, out) = datagen_cli(&["--shrink".as_ref(), "15".as_ref(), sub.as_os_str()]);
        assert!(ok, "{tag}: {out}");
        assert!(tree_bytes(&root) == tree_bytes(&f15), "{tag}: the shrunk corpus is not the 15-sample one");
        assert_eq!(id(&root).id(), id(&f15).id(), "{tag}");
        let (ok, out) = datagen_cli(&["--resume".as_ref(), sub.as_os_str()]);
        assert!(ok && out.contains("the manifest says `complete`; nothing to write"), "{tag}: {out}");
    }
}

/// `aeiou datagen` of `file_per_sample` (300 files, 30 to a directory, two writers) under
/// `strace -f -yy -e trace=<calls>`: the scratch directory (its root `root`) and the trace;
/// None when strace does not work here.
fn strace_datagen(tag: &str, calls: &str) -> Option<(crate::common::TempDir, String)> {
    if !std::process::Command::new("strace").args(["-o", "/dev/null", "true"]).status().is_ok_and(|s| s.success()) {
        eprintln!("skipped: no working strace");
        return None;
    }
    let dir = crate::common::tmpdir("layout", tag);
    let log = dir.join("strace.log");
    let o = std::process::Command::new("strace").args(["-f", "-yy", "-e", &format!("trace={calls}"), "-o"]).arg(&log).arg(env!("CARGO_BIN_EXE_aeiou"))
        .arg("datagen").arg(crate::common::workload_ast("file_per_sample.ast.json")).arg("--root").arg(dir.join("root"))
        .args(crate::common::file_per_sample_args()).args(["--param", "samples=300", "--param", "files_per_directory=30", "--writers", "2"]).env_remove("AEIOU_CONFIG").output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let text = std::fs::read_to_string(&log).unwrap();
    Some((dir, text))
}

/// (RC1 review, R67) A datagen makes a file's directory only when the open says it is missing:
/// 300 files in 10 directories were 300 `mkdir`s (290 of them EEXIST) and a `statx` each
/// beside the opens, every one a round trip on NFS.
#[test]
fn datagen_makes_each_directory_once_not_per_file() {
    let Some((dir, text)) = strace_datagen("mkdir-once", "mkdir,mkdirat,statx,newfstatat") else { return };
    let dirs = std::fs::read_dir(dir.join("root/train")).unwrap().flatten().filter(|e| e.path().is_dir()).count();
    assert_eq!(dirs, 10);
    let mkdirs = text.lines().filter(|l| l.contains(" mkdir(")).count();
    let stats = text.lines().filter(|l| l.contains(" statx(") || l.contains(" newfstatat(")).count();
    // the root and the dataset's directory, and each of the 10 at most once per writer (two
    // writers may both find a new one missing)
    assert!(mkdirs <= 30, "{mkdirs} mkdir for 10 directories:\n{text}");
    assert!(stats <= 60, "{stats} stat calls for 300 files");
}

/// (RC1 review, R68) Every write of the dataset manifest flushes its temporary file before the
/// rename and the directory after it, so a `complete` manifest outlives a crash.
#[test]
fn the_dataset_manifest_is_flushed_before_and_after_its_rename() {
    let Some((_dir, text)) = strace_datagen("manifest-fsync", "openat,fsync,fdatasync,rename,renameat,renameat2") else { return };
    // in order: the calls on the manifest's temporary file and on its directory
    let calls: Vec<&str> = text.lines().filter(|l| l.contains(".aeiou-dataset.json") || (l.contains("sync(") && l.contains("/root/train>"))).collect();
    let renames = calls.iter().filter(|l| l.contains("rename")).count();
    assert!(renames >= 2, "{renames} manifest writes:\n{text}");
    for (i, l) in calls.iter().enumerate().filter(|(_, l)| l.contains("rename")) {
        assert!(calls[..i].last().is_some_and(|p| p.contains("fsync(") && p.contains(".aeiou-dataset.json.tmp")), "no flush of the temporary file before\n{l}\n{}", calls.join("\n"));
        assert!(calls.get(i + 1).is_some_and(|n| n.contains("fsync(") && n.contains("/root/train>")), "no flush of the directory after\n{l}\n{}", calls.join("\n"));
    }
}

/// (RC1 review, R69) A chunked dataset's progress counts its files, the chunks its summary
/// counts: it said "0/24 files" of 24 ids that were 156 files.
#[test]
fn a_chunked_datasets_progress_counts_the_files_its_summary_does() {
    let dir = crate::common::tmpdir("layout", "chunked-progress");
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_aeiou")).arg("datagen").arg(crate::common::workload_ast("lmcache_local_disk.ast.json")).arg("--root").arg(&*dir)
        .args(["--param", "chunk_bytes=4096", "--param", "sys_prompts=12", "--param", "sub_prompts=12", "--progress-secs", "0.001"]).env_remove("AEIOU_CONFIG").output().unwrap();
    let out = format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
    assert!(o.status.success(), "{out}");
    // the summary: `dataset NAME: N file(s), ...`, one per dataset
    let files: u64 = out.lines().filter_map(|l| l.strip_prefix("dataset ")?.split_once(": ")?.1.split_once(" file(s), ")?.0.parse::<u64>().ok()).sum();
    assert!(files > 24, "{files} files of 24 ids:\n{out}");
    let progress: Vec<&str> = out.lines().filter(|l| l.starts_with("progress ")).collect();
    assert!(!progress.is_empty(), "{out}");
    assert!(progress.iter().all(|l| l.contains(&format!("/{files} files"))), "{out}");
}
