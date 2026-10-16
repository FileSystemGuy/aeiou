# `tfrecord_stream`: Streaming training over TFRecord shards

`tf.data` lists the shards, shuffles the list, and interleaves `cycle` shards at a time, each read front to back by `TFRecordDataset` in positioned reads of its buffer size to the end of the file; records are batched from the interleaved stream, and the shuffle buffer is memory. A loader worker's unit of work is a shard. `../parquet` is the same shape over Parquet. The script `tfrecord_stream.py` says what it models and what it cuts; `GRAMMAR_OPTIONS.md` §6.5 and `PROJECT_BRIEF.md` §6 item 15 say why
this shape is here.

Not traced yet: there is no case, and the abstract's defaults (`aeiou dry-run
tfrecord_stream.ast.json --gpus 1`) stand for it. Its corpus is a container dataset that `aeiou
datagen` writes, built with the cargo feature `formats` as the release is (`aeiou-datagen`, the
Python writer, writes it too):

```
aeiou datagen tfrecord_stream.ast.json --root /mnt/sut/aeiou [--param k=v]…
aeiou run tfrecord_stream.ast.json --root /mnt/sut/aeiou --gpus 2 [--param k=v]…
```
