# `imagenet-records`: a case of `tfrecord_stream`, untraced

Its kind is not settled. `params.json` holds the values `tfrecord_stream`'s defaults held until
2026-10-22, when the job's values and the model's on a GPU became required (`DESIGN_REVIEW.md`
§3.78 points 1 and 4). They are placeholders: not measured, and untraced. No run of the
application (tf.data's TFRecordDataset) stands behind them, and there is no capture. The case keeps the
mechanism in the catalog and gives its abstract a lock to run until a traced example case
replaces it (`PROJECT_BRIEF.md` §6 item 36).

One GPU (`--gpus 1`). The mechanism's README (`../../README.md`) says what the abstract models.
