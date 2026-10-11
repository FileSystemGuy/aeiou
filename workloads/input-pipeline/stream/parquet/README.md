# `parquet_stream`: Streaming training over Parquet shards

The `tf.data` shape of `../tfrecord` over Parquet read by `pyarrow.parquet.ParquetFile`: per shard, the footer read, then `WILLNEED` and one `pread` per coalesced run of row groups (full projection; the format class's `read_all(columns=[…])` is the projected variant). The script `parquet_stream.py` says what it models and what it cuts; `GRAMMAR_OPTIONS.md` §6.5 and `PROJECT_BRIEF.md` §6 item 15 say why
this shape is here.

Not traced yet: there is no case, and the abstract's defaults (`aeiou dry-run parquet_stream.ast.json
--gpus 1`) stand for it. Its corpus is a container dataset that the Python writer makes:

```
aeiou-datagen parquet_stream.ast.json --root /mnt/sut/aeiou [--param k=v]…
aeiou run parquet_stream.ast.json --root /mnt/sut/aeiou --gpus 2 [--param k=v]…
```
