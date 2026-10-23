# `parquet_stream`: Streaming training over Parquet shards

The `tf.data` shape of `../tfrecord` over Parquet read by `pyarrow.parquet.ParquetFile`: per shard, the footer read, then `WILLNEED` and one `pread` per coalesced run of row groups (full projection; the format class's `read_all(columns=[…])` is the projected variant). The script `parquet_stream.py` says what it models and what it cuts; `GRAMMAR_OPTIONS.md` §6.5 and `PROJECT_BRIEF.md` §6 item 15 say why
this shape is here.

Not traced yet. Its job's values and the model's on a GPU are required (`DESIGN_REVIEW.md`
§3.78); its one case, `cases/imagenet-rows`, is of kind `unsettled` and holds the placeholders its
defaults held until 2026-10-22, never measured (`PROJECT_BRIEF.md` §6 item 36 traces it). Its
corpus is a container dataset that `aeiou datagen` writes, built with the cargo feature
`formats` as the release is:

```
aeiou datagen --case imagenet-rows --root /mnt/sut/aeiou [--param samples=N]
aeiou run --case imagenet-rows --root /mnt/sut/aeiou --gpus 2 [--param k=v]…
```
