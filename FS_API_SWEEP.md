# The filesystem API sweep: every Linux call and flag against every layer (2026-11-02)

Asked for by the user on 2026-11-02, after a `truncate` and an `O_TRUNC` were found ignored and a
`close` dropped (`DESIGN_REVIEW.md` §3.92): collect every operation and flag of the Linux
filesystem APIs and check that each one is honored at every layer of the tools, where
"honored" is one of the statuses below. This file is the register the fixes are worked from:
the findings are numbered once and never renumbered; each one's **status** line is updated in
the commit that fixes it (the commit named), and nothing is deleted. The decisions the user
made on the report are `DESIGN_REVIEW.md` §3.93; the work is brief items 57 (the bugs) and
58 (the new ops).

## 1. Method

- **Inventory**, from the primary sources on the development machine, not from memory: the
  373 syscalls of `asm/unistd_64.h` (x86_64) and the 355 of `asm-generic/unistd.h` (aarch64),
  each assigned to one of 17 filesystem families (F1 open/create, F2 descriptor lifetime, F3
  read, F4 write, F5 position, F6 durability, F7 copy/clone, F8 space, F9 metadata read, F10
  metadata write, F11 namespace, F12 directory read, F13 locks and notification, F14 advice,
  F15 asynchronous I/O, F16 descriptors not on a path, F17 process context) or to "not
  filesystem", so the partition can be checked; the flags of `linux/fcntl.h`,
  `asm-generic/fcntl.h`, `bits/fcntl-linux.h`, `linux/fs.h`, `linux/falloc.h`, `linux/stat.h`,
  `linux/openat2.h`, `linux/aio_abi.h`, `linux/io_uring.h`, `asm-generic/mman-common.h`,
  `linux/fadvise.h`, `linux/xattr.h`, `linux/close_range.h`; and the S3 API's operations (O1)
  for the object side. Kept with the probe beside the repo (`~/MLPerfStorage/posix-sweep/`:
  `inventory.md`, `findings.md`, `probe.c`, `attribute.py`).
- **Five read-only audits**, one per layer, each classifying every inventory entry: the strace
  input (`strace.rs`), the trace file and its replay (`export.rs`, `trace.rs`, the lanes in
  `run.rs`/`uring.rs`), the abstract language and its eight backends (schema, builder,
  `ast.rs`, `validate.rs`, `vm.rs`, `backend.rs`, `aio.rs`, `uring.rs`, `object.rs`,
  `run.rs`), the object side (`objectlog.rs`, `object.rs`, datagen's writer), and the
  analysis tools (`metrics.rs`, `digest.rs`, `compare.rs`, `emit.rs`, the Python consumers).
- **Measured, not only read**: a C program issuing 32 groups of the inventory's calls on files
  under a root, run under the capture recipe's `strace -f -ttt -T -yy -v`, each exported op
  attributed back to the call that made it; the audits' own probes under strace 6.8, small
  `aeiou run` fixtures, and a real MinIO (RELEASE.2025-09-07) audit log from object_store
  0.14.1 and curl. Every top finding was re-read in the code before it was accepted.

**Statuses.** HONORED (carried with its meaning); NO-IO-EFFECT (correctly ignored, with the
reason); NOTED (not replayed, the user told by name); REFUSED (loudly); and the bug class:
SILENT-DROP (ignored with no note) and SILENT-ALTER (carried with its meaning changed);
INEXPRESSIBLE (the abstract language has no op for it, so the trace tools have nowhere to put
it).

**Confidence.** High for the syscall and flag surface of the strace input and of the abstract
language (the inventory is a checked partition; the input side was audited by reading and by
running, and independent passes found the same bugs: the probe and two audits each found S-2,
S-3, S-5). Medium for object logs (MinIO verified against a real log; AWS S3 server access
logs, aws-chunked uploads, and replication traffic untested). Lower for behavior only real NFS
shows (none here), aarch64 (checked by name only), other strace versions (the `(deleted)`
placement has moved between versions), and multi-process interactions beyond fork and dup.

**Clean.** Every backend's dispatch matches all 19 op kinds exhaustively (`issue_blocking`,
`uring::issue`, `object::start`); the fingerprint hashes every kind (`op_hash`); `truncate`
and every `fallocate` mode are carried end to end (§3.92).

## 2. Findings

Group **A**: a bug with no design question, fixed as found (brief item 57). Group **B**: a
representation the user decided on 2026-11-02 (§3.93; brief item 58). Group **O**: open, a
work item of its own.

### Calls dropped with no note (the truncate class)

- **S-1** (A) `strace.rs:1119` `if !handled(name) { return; }` drops every call outside a
  fixed list with no note, even on a path under the root, where `man/aeiou-trace.1.md` and
  `runner/REFERENCE.md` §13 promise "dropped and counted by name". Among them: `sendfile`,
  `copy_file_range`, `splice` (data: `shutil.copyfile`, `cp`), `readahead`, `syncfs`,
  `sync_file_range`, `access`/`faccessat*`, `readlink*`, `statfs`/`fstatfs`, every xattr call,
  `chmod`/`chown`/`utime*`, `link*`/`symlink*`/`mknod*`, `flock`, `close_range`, `io_destroy`,
  `io_cancel`, `name_to_handle_at`, `open_by_handle_at`. *Status: fixed 2026-11-02: a call
  outside the list that names a path under the root, by a descriptor (inside a struct too) or
  a string where a path argument stands, is noted `not_counted:<call>` in the metrics and the
  export header, a split one (`<unfinished ...>`) as well; three corpus goldens gained
  `not_counted:readlink` and nothing else; test
  `export_notes_by_name_every_call_on_the_root_it_does_not_count`. `close_range` and
  `io_destroy` name no path: S-12.*
- **S-2** (A) `fcntl` other than `F_DUPFD*` returns with no note (`strace.rs:1302`): `F_SETFL`
  turning `O_DIRECT`/`O_APPEND`/`O_NOATIME` on after the open, `F_SETLK`/`F_SETLKW`/`F_GETLK`,
  the OFD locks, leases, `F_SET_RW_HINT`. *Status: fixed 2026-11-02 with S-1: noted
  `not_counted:fcntl:<command>`, `F_SETFL` with its flags; none for `F_GETFD`, `F_SETFD`,
  `F_GETFL`, and the owner and signal commands.*
- **S-3** (A) An `io_uring_setup` on a traced task is not noted: the application's ring I/O is
  invisible to strace and nothing says so. *Status: fixed 2026-11-02 with S-1: noted
  `not_counted:io_uring_setup` in every instance.*
- **S-4** (A) The capture recipe's `-e trace=%file,%desc,%network,%process,io_*` cannot see
  `msync`, `madvise`, `munmap`, `mremap`, `mlock*` (`%memory`) or `sync()` (no class); the
  `metrics` help text's filter also lacks `%network` (SCM_RIGHTS). *Status: fixed 2026-11-02:
  the recipe (man page, `--help`, READMEs, references; the workloads' READMEs keep the command
  each capture was taken with) names `sync`, `msync`, `madvise`, `munmap`, `mremap`, `mlock`,
  `mlock2`, and `io_cancel`, not `%memory` (every `brk` and `mprotect`); each is noted by name,
  a call on a mapping when its address is in a range an `mmap` of a root file returned
  (followed through `mremap`, trimmed by `munmap`); test
  `export_notes_the_calls_on_a_mapping_of_the_root_and_sync`. `%network` stays the addition the
  man page names for an application that passes descriptors over a socket (2026-11-02,
  §3.92), not the default: a socket-heavy job's trace would fill with its messages.*

### Parsing that loses or corrupts calls

- **S-5** (A) strace 6.8 prints a descriptor of an unlinked file as `fd<path>(deleted)`, with
  `(deleted)` outside the brackets; `fd_arg` needs the token to end in `>` and returns
  nothing, so every call on an unlinked-open file or an `O_TMPFILE` file is lost silently
  (Python `tempfile.TemporaryFile`). *Status: fixed 2026-11-02: `fd_arg` takes `(deleted)` after
  the brackets as inside them (tests `descriptors_and_their_paths`,
  `export_keeps_calls_on_an_unlinked_file`); an `O_TMPFILE` file's calls are now exported, its
  open still the directory's until S-10 (fixed the same day).*
- **S-6** (A) `iocb_at` (`strace.rs:600-648`): an `IOCB_CMD_FSYNC`/`FDSYNC`/`NOOP`/`POLL` has
  no `aio_nbytes`/`aio_offset` and a `PREADV`/`PWRITEV` has `aio_buf=[iovec]`, so the search
  runs into the next request (its size taken, the request swallowed, the path overrunning into
  it and planted in the descriptor table: a false "trace incomplete" warning and synthesized
  opens); a request with `aio_rw_flags=` (printed before the opcode) is skipped. Measured: 1 of
  4 and 0 of 2 requests exported. The unit test builds an FSYNC in a format strace never
  prints. *Status: fixed 2026-11-02: each `{...}` is read by its fields (`iocb`), a vectored
  request counted as the read or write it is by its iovec's bytes, and only the first `ret`
  requests issued, where the counted ones were cut to `ret` (an unsubmitted read after a
  sync was exported); tests `aio_arrays` (every kind as strace 6.8 prints it, probed) and
  `export_reads_every_aio_request_as_strace_prints_it`. `aio_rw_flags` still not carried
  (S-40).*
- **S-7** (A) `openat2`: the `{flags=…, mode=…, resolve=…}` struct is read as the flags and the
  size as the mode: `WRONLY|CREAT|TRUNC` exported `["RDONLY","CREAT"]`, mode 020; the run fails
  with EBADF. `RESOLVE_*` lost. *Status: fixed 2026-11-02: flags and mode read from the struct,
  a `resolve` other than 0 noted by value (`not_exported:openat2:resolve …`); test
  `export_reads_openat2_and_creat`, the lines probed under strace 6.8.*
- **S-8** (A) `creat` loses its mode (read from `a[pi+2]`; creat's is `a[1]`). *Status: fixed
  2026-11-02, with S-7 (same test).*
- **S-9** (A) `pwritev2`/`preadv2` `RWF_*` (`a[4]`) never read: an `RWF_APPEND` write is
  exported at its given offset (0) instead of the end of the file. *Status: fixed 2026-11-02:
  `RWF_APPEND` appends as `O_APPEND` does, for the one call; and an append moves the position
  only when the call names no offset, where a `pwrite` on an `O_APPEND` file moved it to the
  end (a later read placed there); test
  `export_places_appends_at_the_end_and_moves_the_position_as_the_kernel_does`.* (`RWF_DSYNC`,
  `RWF_SYNC`, `RWF_NOWAIT`, `RWF_HIPRI` carried: B, S-40.)
- **S-10** (A) `O_PATH` exported as a full `RDONLY` open, `O_TMPFILE` as an `RDWR` open of the
  directory (replay: EISDIR or a refusal as a missing input); both counted only in the unnamed
  `open_flags_dropped`. Dropped flags to be named, `O_PATH`/`O_TMPFILE` noted and not replayed
  as another call until B. *Status: fixed 2026-11-02: both opens left out and noted
  (`not_exported:open:O_PATH`, `…:O_TMPFILE`), the calls on the descriptor noted as on an open
  not exported (no longer "descriptor without open"), and a dropped flag noted by name
  (`open_flags_dropped:O_NONBLOCK`; three corpus goldens renamed that note and changed nothing
  else); test `export_leaves_out_o_path_and_o_tmpfile_opens_and_names_dropped_flags`.*

### Closes (the close class)

- **S-11** (A) Closing one of two descriptors of one open file description (dup, inherited over
  fork) exports a close of the shared open while the other is still open (`strace.rs:1290`);
  Python `mmap` does `F_DUPFD_CLOEXEC` and closes the duplicate. *Status: decided 2026-11-02
  (§3.93 note): the trace file names descriptors (a `dup` line, a descriptor per forked lane),
  every close line a real `close(2)`; with S-12 and S-14, contract 0.15. Found while fixing:
  the replay's close of an open at its last close line waits on a later use in its own lane
  when the trace shows fewer closes than descriptors, a hang.*
- **S-12** (A) Implicit closes never recorded: `exit`/`exit_group` and `+++ exited`, `dup2`/
  `dup3` onto an open descriptor, a successful `execve` closing `O_CLOEXEC` descriptors,
  `close_range`. On NFS each is a flush and a CLOSE. *Status: decided 2026-11-02 with S-11:
  each recorded as a close line of its descriptor.*
- **S-13** (A) `close(2)`'s return is thrown away on every POSIX backend and the trace node
  (`OwnedFd` drop; `run.rs:1350`, `uring.rs` inline): NFS reports deferred ENOSPC/EDQUOT/EIO
  at close, so a failed checkpoint is scored a success and an `expect` on a close can never
  match. *Status: fixed 2026-11-02: `backend::close` closes the last reference explicitly and
  returns `close(2)`'s error, used by the actor's `close` (every backend, objects included)
  and the trace node's last close on both paths; test `a_close_returns_the_error_of_close`.*
- **S-14** (A, decided §3.93) A close of a descriptor inherited over a fork issues no
  `close(2)` (`run.rs:1352`; the trace node closes only at an open's last close line,
  `uring.rs:1985`): each forked worker's close is a `->flush` (NFS, FUSE_FLUSH). *Status: open;
  built with S-11 (the descriptor per forked lane), contract 0.15.*

### Failed calls

- **S-15** (A, decided §3.93) A traced error that does not recur on replay is accepted
  silently (`run.rs:1396` `Ok` branch never reads `op.expect`; a write that failed with ENOSPC
  replays as a full write; an `O_EXCL` lock file is created). To warn loudly: the SUTs differ.
  *Status: fixed 2026-11-02: counted per op and errno (`Stats::unreproduced_errors`, merged
  across ranks, in the JSON report) and a `WARNING:` line after the run; test
  `a_traced_error_that_does_not_recur_is_warned_of`.*
- **S-16** (A) Object log: a 304 Not Modified becomes EIO (`objectlog.rs:42-49, 360`); a failed
  CompleteMultipartUpload or Abort is exported as a successful close (`objectlog.rs:382` passes
  no error). *Status: open.*

### Bookkeeping that ignores what an op means

- **S-17** (A) An `O_CREAT` open of a file that already existed is listed in the trace file's
  `creates` (`export.rs:389`); the gate then refuses, and its advice, `--clean-namespaces`,
  deletes the application's input (Python `'a'`, h5py `'a'`, an S3 PUT overwrite). A create to
  be listed only when the trace proves the file new. *Status: fixed 2026-11-02, the rule
  narrowed while fixing: a create is listed when the trace proves the path absent when it
  began (first appearance an `O_EXCL` create, a `mkdir`, or an ENOENT) or when its first
  appearance destroys what it held (`O_CREAT|O_TRUNC`, a rename onto it, an S3 PUT), so a
  rerun never truncates the last run's file (§3.92) and the clean removes nothing the
  application keeps; an `O_CREAT` open that keeps the content (`O_APPEND`, `O_RDWR`, write-only
  without `O_TRUNC`), unproven, is not, nor a path the trace used before creating it. "Only
  when proven new" literally cut the corpus's creates from 110 to 5. Twelve corpus goldens
  lose such creates, nothing else; test
  `export_lists_a_create_only_where_the_trace_proves_the_path_new`.*
- **S-18** (A) Locality history is keyed by path hash and never reset or moved by an `O_TRUNC`
  open, `truncate`, `ftruncate`, a punched hole, `unlink`, or `rename` (`metrics.rs:777-791`,
  both sides): false reuse for a file rewritten under its name, the write-tmp-rename-read link
  lost. *Status: open.* (Changes measured shapes: the catalog is re-measured.)
- **S-19** (A) The VM's size of a file (`as_written`, `until_eof`, `lseek END`) ignores
  `truncate`, `ftruncate`, and an extending `fallocate`, and sums rewrites without `TRUNC`
  whatever their offset (`vm.rs:1683, 1850-1871`). *Status: open.*
- **S-20** (A) The namespace manifest never records an `rmdir` (`run.rs:1638`; `uring.rs:788`
  `_ => Ok(0)`), and subtracts removals without regard to order (`run.rs:2849, 3003`): a path
  unlinked and created again is dropped from the manifest. *Status: open.*
- **S-21** (A) An open's explicit mode 0 becomes 0644 (`run.rs:1550`, `uring.rs:606`); absent and
  0 hash the same (`vm.rs:1711`). *Status: fixed 2026-11-02: an explicit 0 carries a mark of
  its own in the mode field (`vm::open_aux`, `open_mode`), so it creates mode 0 on every path
  and hashes apart from no mode, every other open's `aux` unchanged (no committed abstract or
  trace names mode 0); test `an_explicit_mode_0_is_mode_0`.*
- **S-22** (A) `Whence::from_code`, `IoctlRequest::from_code`, `Advice::from_code` end in `_ =>`
  a default (SET, TCGETS, NORMAL): a new variant missing its arm would silently become the
  default. To be exhaustive. *Status: fixed 2026-11-02: every code has its arm and any other
  panics (the VM makes the codes; every caller reads its own kind's `aux`); test
  `op_codes_come_back_as_the_values_they_were_made_from`, which had pinned the default.*
- **S-23** (A) A trace `truncate` under `--posix mmap` never updates the lane's mapping
  (`run.rs:1622`, `a.fds` is empty in a lane; REFERENCE §13 says otherwise): SIGBUS risk.
  *Status: open.*
- **S-24** (A) A per-open `O_DIRECT` under `--posix mmap` is read through the page cache with
  nothing said (`validate.rs:568` refuses only `--cache direct`). *Status: open.*
- **S-25** (A) A rename from outside the root into it is dropped with no note, and the name then
  counts as an input that must exist (`strace.rs:1630`). *Status: open.*
- **S-26** (A) The digest's rename moves state for the exact path only: a directory rename
  leaves its files behind, a rename over a file is not a removal (`digest.rs:949-965`).
  *Status: open.*
- **S-27** (A) `renameat2` `RENAME_EXCHANGE`/`NOREPLACE`/`WHITEOUT` exported as a plain rename
  (an exchange destroys a name). Noted and not exported until B (S-40). *Status: open.*
- **S-28** (A) `lseek` `SEEK_DATA`/`SEEK_HOLE` are exported as `DATA`/`HOLE`, which the runner's
  `Whence` refuses, so the whole trace file is unusable. Exported only once B adds them (S-40);
  until then noted and left out. *Status: open.*

### The object-log reader

- **S-29** (A) Multipart parts written in log (completion) order, `partNumber` ignored, a failed
  or retried part advancing the offset (`objectlog.rs:390-397`). *Status: open.*
- **S-30** (A) A suffix range `bytes=-N` (Parquet footers) and a `partNumber` GET exported as
  reads at 0 with no note (`objectlog.rs:192-204`). *Status: open.*
- **S-31** (A) Two multipart uploads to one key: the second's start overwrites the first's
  (`objectlog.rs:434`), whose open is never closed. *Status: open.*
- **S-32** (A) An upload never completed is closed at t0 on lane 0, before its open
  (`objectlog.rs:441-447`); NewMultipartUpload is an `O_TRUNC` open at the start where S3 keeps
  the old object until Complete; an Abort unlinks an object that existed. *Status: open.*
- **S-33** (A) Each ListObjects page is its own listing (continuation, `max-keys`, `delimiter`
  ignored; a non-directory prefix becomes a directory): replayed as about N² requests.
  *Status: open.*
- **S-34** (B) DeleteObjects (batch), CopyObject, UploadPartCopy are counted, never exported:
  their unlinks and their targets lost. Copy is S-40's copy op. *Status: open.*
- **S-35** (A) Unchecked, to verify: replicated PUTs (MinIO `trigger`, replication headers)
  counted as client writes; aws-chunked bodies (`X-Amz-Decoded-Content-Length` unread); old
  MinIO `timeToResponse`; S3 access logs not sorted by time. *Status: open.*
- **S-36** (A) Object engine: a read-only open of a missing key succeeds (ENOENT at the first
  read), a HEAD 404 replays as HEAD + LIST, `BLKGETSIZE64` succeeds, and `SYNC`/`DSYNC`/
  `DIRECT` are ignored without being listed in REFERENCE §16. *Status: open.*

### What `compare` never judges

- **S-37** (B, decided §3.93) An op kind is judged only as its share of all ops at 0.05
  (`compare.rs:151-153`): a kind under 5 % can vanish (the dcp-save capture passes with its 6
  `fsync`s removed). A kind present on one side only is to be a difference. *Status: open.*
- **S-38** (B, decided §3.93) Open flags (`DIRECT`, `SYNC`, `DSYNC`, `APPEND`, `EXCL`, access
  mode), fadvise advice, `lseek` whence, ioctl request are never compared (the digest reads
  only `TRUNC`, `digest.rs:850`; its checkpoint `sync` verdict counts only `fsync`/`fdatasync`).
  To be compared. *Status: open.*
- **S-39** (B, decided §3.93) The documents' notes (`calls_not_exported`, `opens_synthesized`,
  `mmap_as_read`, …) never reach the verdict. To be shown in it. *Status: open.* Also in this
  layer (A): templates only B has are not differences (`compare.rs:446`); `--self` widens a
  row's tolerance by the gap between document types (`compare.rs:350`: fan-out's becomes 1.1);
  `--emit-trace` flattens `submit` groups (`metrics.rs:962`); the digest's metrics and
  templates disagree on mmap, synthesized opens, and unexported calls; `catalog.py:74`'s
  `METADATA` leaves out `fsync`/`fdatasync`, `attributes.toml:292`'s text leaves out
  `truncate`.

### What the language cannot express (decided: all of them, §3.93)

- **S-40** (B) New ops of the abstract and the trace file, contract 0.15: copy
  (`copy_file_range`, `sendfile`, `splice` between files, `FICLONE`/`FICLONERANGE`/
  `FIDEDUPERANGE`, S3 CopyObject/UploadPartCopy); `link`/`linkat` (with `O_TMPFILE`'s
  publish), `symlink`, `readlink`; setattr (`chmod`, `chown`, `utimensat`) and xattrs (get,
  set, list, remove); locks (`flock`, `fcntl` record and OFD locks); `syncfs` and
  `sync_file_range`; `msync`; `readahead` and `madvise`; `access`; `statfs`; `lstat` and the
  `statx` sync flags; `SEEK_DATA`/`SEEK_HOLE`; `renameat2` flags; `RWF_*` flags; `O_PATH`,
  `O_TMPFILE`; `F_SETFL`; IOCB FSYNC/FDSYNC/PREADV/PWRITEV in `submit` groups; batch deletes.
  *Status: open.*

### Open work items

- **S-41** (O) mmap: a mapping is exported as one read of its whole length whatever its
  protection, flags, or the pages touched (writes through a shared mapping lost, LMDB's
  `map_size` refused at the gate); the strace metrics count nothing for it and say nothing.
  The page references that cause I/O are not in an strace: how to capture them (eBPF in the
  kernel, brief item 54) is to be designed first. Stays open (the user, 2026-11-02).
  *Status: open.*
