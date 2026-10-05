//! Contract 0.2: container layouts (`format.layout`), unit and column handles, `stream`
//! access (the shuffle over shards), and the `fadvise` op, against hand-written ASTs whose
//! geometry is computed by hand here.

use std::collections::BTreeSet;
use std::path::PathBuf;

use aeiou::backend::BackendKind;
use aeiou::datagen::{datagen, DatagenOpts};
use aeiou::dryrun;
use aeiou::eval::{build_model, Config, Model, Params};
use aeiou::run::{self, RunOpts};
use aeiou::vm::{run_actor, Op, OpCtx, OpKind, Sink};

fn config(gpus: i64) -> Config {
    Config { seed: 1, gpus, overrides: vec![], sets: vec![] }
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
  "ast": "0.5", "name": "stream_test",
  "params": {"batches": {"default": 10}, "xfer": {"default": 4096}},
  "datasets": {"shards": {"files": {"pattern": "s/{id:04}.tfrecord", "count": 100, "samples_per_file": 10,
     "size": {"const": 1000}, "seed": 7, "access": "stream",
     "format": {"class": "tfrecord", "reader": "tf.data", "layout": {"columns": [{"row_header": 12, "row_footer": 4, "weight": 1}]}}}}},
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
  "ast": "0.5", "name": "parquet_test",
  "params": {"groups": {"default": 3}},
  "datasets": {"t": {"files": {"pattern": "p/{id:04}.parquet", "count": 192, "samples_per_file": 192,
     "size": {"const": 100000}, "seed": 7, "access": "stream",
     "format": {"class": "parquet", "reader": "pyarrow", "version": "25.0.1", "layout": {
       "unit": 64, "file_header": 4, "file_footer": 200, "file_footer_per_unit": 120,
       "columns": [{"header": 26, "fixed": 4, "weight": 1.0}, {"header": 22, "fixed": 8, "weight": 0}],
       "writer": {"rows_per_group": 64}}}}}},
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
  "ast": "0.5", "name": "tar_test",
  "datasets": {"t": {"files": {"pattern": "w/{id:04}.tar", "count": 25, "samples_per_file": 10,
     "size": {"const": 1000}, "seed": 7, "access": "stream",
     "format": {"class": "webdataset", "layout": {"file_footer": 1024, "file_align": 10240,
       "columns": [{"row_header": 512, "row_align": 512, "weight": 1}]}}}}},
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
  "ast": "0.5", "name": "fadvise_test",
  "datasets": {"d": {"files": {"pattern": "d/{id:04}", "count": 4, "size": {"const": 20000}, "seed": 3}}},
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
    let root: PathBuf = std::env::temp_dir().join(format!("aeiou-layout-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let loaded: &'static aeiou::Loaded = Box::leak(Box::new(aeiou::load_str(FADVISE).unwrap()));
    let cfg: &'static Config = Box::leak(Box::new(config(1)));
    let params: &'static Params = Box::leak(Box::new(Params::new(&loaded.ast, cfg).unwrap()));
    let model: &'static Model<'static> = Box::leak(Box::new(build_model(&loaded.ast, cfg, params).unwrap()));
    let opts = DatagenOpts { root: root.clone(), threads: 2, dedupe: 1, compress: 1, datasets: vec![], rank: 0, ranks: 1 };
    datagen(loaded, cfg, params, model, &opts, None, &mut Vec::new()).unwrap();
    let dry = dryrun::run(model, 1, None).unwrap();
    assert_eq!(dry.total.total.ops, 4 * 6);
    let r = run::run(
        model,
        RunOpts {
            root: root.clone(),
            backend: BackendKind::Sync,
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
    let _ = std::fs::remove_dir_all(&root);
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
