# Application records

What `aeiou-trace digest` knows about the applications whose traces it reads (`DESIGN_REVIEW.md`
§3.88). A person who uses a product may not know its storage design; the digest recognizes the
product from the paths in the trace, asks "is this X, and which version?", asks only the few
options of X that change its storage workload, and fills from the record what a short trace
cannot show. Every fact a record supplies is recorded in the digest as an assertion with the
record as its provenance, weaker than the owner's own answer, and is checked against the trace
like any answer: the trace always wins.

One directory per application, named as a user would pick it from a list, lowercase and
hyphenated. Inside:

- `application.toml`: what does not change across versions. `name` (the directory's), `display`,
  `kind` (`library`, `service`, `managed`, defined in `attributes.toml`), `summary`, `home`,
  `source` (the repository), `signatures` (globs matched against each path under the trace's
  root and against its part after any `/`, so `__*_*.distcp` matches a shard at any depth; `*`
  crosses `/`; any match identifies the application), and `mechanisms`
  (the catalog mechanisms it maps to, as `category/[class/]mechanism`).
- `<version>.toml`, one per version range the facts were read for, named by the lowest version
  it covers (`2.5.toml`): `from` (the file's stem), optional `until` (exclusive), `read` (the
  date), `sources` (what was read), then `[[option]]` entries, each a question with `choices`,
  an optional `default`, its `source` and `confidence`, and `blind_spots`, a table from a choice
  to the blind spot it causes (`attributes.toml`); `[[blind_spot]]` entries that hold whatever
  the options; and `[[assert]]` entries, what the record answers for the digest: a `template`
  glob (matched against the paths of a template), either `fact` and `value` or `question` and
  `answer` in the digest's own vocabulary, a `note`, the `source` and `confidence`, and an
  optional `when = { option = "...", choice = "..." }`.

The digest picks the record whose range contains the version the person names; with none, it
takes the nearest lower record and every one of its assertions becomes a suggestion the person
confirms rather than a fact it records. `builder/tests/test_applications.py` walks every record.
The directory ships in the wheel (`builder/pyproject.toml`), and `--applications DIR` points the
digest at a checkout or at extra records.
