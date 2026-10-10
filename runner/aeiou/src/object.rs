//! The object engine (`DESIGN_REVIEW.md` §3.65): the ops of a dataset or namespace declared
//! `protocol: s3`, issued to an S3 store through Apache `object_store`, the library chosen
//! over `s3dlio` by measurement (the same entry). It is built with the cargo feature
//! `object`; the default build has no tokio and refuses an object endpoint.
//!
//! **Where.** `--endpoint NAME=s3://BUCKET[/PREFIX]` places a name's root at a prefix of a
//! bucket, as a directory endpoint places it in a directory (`endpoint.rs`): the abstract's
//! path `ROOT/a/b` is the key `PREFIX/a/b`. The store is the one the standard environment
//! names (`AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`,
//! `AWS_ALLOW_HTTP`, read by `AmazonS3Builder::from_env`), with path-style requests unless
//! `AWS_VIRTUAL_HOSTED_STYLE_REQUEST` says otherwise.
//!
//! **The mapping** (§3.65's table): `open` for reading sends nothing (the handle is the key);
//! a read at an offset is one ranged `GET` streamed into the actor's buffer, a range past the
//! end of the object a read of 0 bytes, a range over the end a short read; `stat` and `fstat`
//! are a `HEAD`, and for a path that is no object a `LIST` of the prefix below it, which
//! succeeds (size 0) when there is one, as `stat` of a directory does; `readdir` is a `LIST`
//! with the delimiter `/`, its objects and common prefixes the entries; `lseek`, `fadvise`,
//! `fsync`, and `fdatasync` are local and send nothing, and `ioctl` is local with a regular
//! file's answers (`TCGETS` is `ENOTTY`). An `open` for writing begins an upload (`Upload`):
//! the writes, each at the end of what the handle has written (V18), fill a part of
//! `--object-part-mib` MiB from the payload, a full part is sent as a part of a multipart
//! upload before the write returns, and `close` sends the object, one `PUT` when it fits in
//! a part and the last part and the completion otherwise, so an object exists from its
//! `close`; `fstat` of a handle being written is its written size, sent nothing. `unlink` is
//! a `DELETE`, `rename` a copy and a `DELETE` (not atomic), `mkdir` and `rmdir` send nothing
//! (a prefix is no object), and `ftruncate`, `fallocate`, and `O_APPEND` are refused by V18
//! before a run (`validate.rs`), and here with `EOPNOTSUPP` should one come.
//!
//! **Threads.** One multi-thread tokio runtime per process, built on the first object call
//! with `--object-threads` workers (`set_threads`), parked until the process exits. An op is
//! begun on the actor's thread (`start`: what is local, the payload of a write, the actor's
//! tables) and what it sends is a future that owns what it needs; its answer is settled on
//! the actor's thread again (`settle`). Under `--s3 blocking` the actor's thread runs the
//! future with `Handle::block_on` (`issue`), so the request is built and signed there and the
//! workers drive the connections; under `--s3 async` an event loop (`uring.rs`: one of
//! `io_uring` or `libaio` beside POSIX names, one with no ring for S3 names alone) `spawn`s
//! it instead, so the workers do all of it, and goes on with its other actors until the
//! answer wakes it. The sockets are tokio's (`epoll`) whatever the POSIX API. Either way an actor has one op in flight. An op on an
//! upload another op has on the runtime (two actors writing one object at once) is `EBUSY`.

use std::io;
use std::sync::Arc;

/// The library the engine is built on, as the run prints it (pinned in `Cargo.toml`).
pub const LIBRARY_VERSION: &str = "0.14.2";

/// Default worker threads of the engine's runtime (`--object-threads`).
pub const DEFAULT_THREADS: usize = 2;

/// The part size of a multipart upload: an object larger than this is sent in parts of it
/// (8 MiB, as `s3dlio`; `object_store`'s own writer uses 10 MiB). `aeiou datagen` uses it;
/// a run uses `--object-part-mib`.
pub const DEFAULT_PART: u64 = 8 << 20;

/// The bounds S3 sets on a part: at least 5 MiB (but the last), at most 5 GiB.
pub const MIN_PART: u64 = 5 << 20;
pub const MAX_PART: u64 = 5 << 30;

static PART: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(DEFAULT_PART);

/// The part size of the run's uploads (`--object-part-mib`), set before the first open.
pub fn set_part_size(n: u64) {
    PART.store(n, std::sync::atomic::Ordering::Relaxed);
}

pub fn part_size() -> u64 {
    PART.load(std::sync::atomic::Ordering::Relaxed)
}

/// An error of the engine as the run sees it: an errno for the statement's `expect` list
/// and the structural check, with the store's message.
#[derive(Debug)]
pub struct ObjectError {
    pub errno: i32,
    pub message: String,
}

impl std::fmt::Display for ObjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ObjectError {}

pub fn error(errno: i32, message: impl Into<String>) -> io::Error {
    io::Error::new(io::Error::from_raw_os_error(errno).kind(), ObjectError { errno, message: message.into() })
}

/// The errno of an error: the OS's, or the object engine's.
pub fn errno_of(e: &io::Error) -> Option<i32> {
    e.raw_os_error().or_else(|| e.get_ref().and_then(|i| i.downcast_ref::<ObjectError>()).map(|o| o.errno))
}

/// A file of the abstract that lives in an object store, as an actor holds it open: the
/// store, the key's part below the endpoint's prefix, and, when it was opened for writing,
/// its upload (`None` inside while an op of the handle has it on the runtime).
pub struct Handle {
    pub store: Arc<Store>,
    pub rest: String,
    pub upload: Option<std::sync::Mutex<Option<Upload>>>,
}

pub use imp::{block, set_threads, spawn, threads, Reader, Store, Upload};

/// A future of the engine: what an op sends and receives, run by the runtime.
pub type Fut<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

/// Where a read's bytes go: the actor's buffer, lent while the actor's thread waits (a
/// thread-per-actor API), or a buffer the op owns while it is in flight (an event loop),
/// handed back with the answer; `.1` is the length read into it.
pub enum Sink<'a> {
    Lent(&'a mut [u8]),
    Owned(Vec<u8>, usize),
}

impl Sink<'_> {
    fn bytes(&mut self) -> &mut [u8] {
        match self {
            Sink::Lent(b) => b,
            Sink::Owned(v, n) => &mut v[..*n],
        }
    }

    fn owned(self) -> Option<Vec<u8>> {
        match self {
            Sink::Lent(_) => None,
            Sink::Owned(v, _) => Some(v),
        }
    }
}

/// What an op sent brings back to the actor's thread, which settles it (`settle`).
pub enum Answer {
    Done(io::Result<i64>),
    Read(io::Result<i64>, Option<Vec<u8>>),
    Removed(io::Result<()>),
    Renamed(io::Result<()>),
    /// A write that sent parts: the upload goes back to its handle.
    Wrote(Arc<crate::backend::OpenFile>, Upload, io::Result<i64>),
}

/// An op begun on the actor's thread: done there (nothing to send), or a future to run.
pub enum Step<'a> {
    Done(io::Result<i64>),
    Send(Fut<'a, Answer>),
}

/// Is `uri` an object endpoint (rather than a directory)?
pub fn is_uri(uri: &str) -> bool {
    uri.contains("://")
}

/// The flags of an open that would change the store.
fn writes(flags: u64) -> bool {
    use crate::ast::OpenFlag;
    let bit = |f: OpenFlag| flags & (1 << (f as u8)) != 0;
    bit(OpenFlag::WRONLY) || bit(OpenFlag::RDWR) || bit(OpenFlag::CREAT) || bit(OpenFlag::TRUNC) || bit(OpenFlag::APPEND)
}

/// An op on an upload another op has on the runtime: two actors writing one object at once
/// (a sub-actor and the actor it was forked from), which an upload written in order (V18)
/// cannot carry.
fn busy(op: &crate::vm::Op) -> io::Error {
    error(libc::EBUSY, format!("{} {}: another op of this upload is in flight; an object is written by one actor at a time (V18)", op.kind.name(), op.path))
}

/// Issue `op`, whose path lives at `rest` of `store`, on the actor's thread, waiting for
/// what it sends (a thread-per-actor API); `Ok(n)` is what the op returned, as from
/// `run::issue_blocking`.
pub(crate) fn issue(store: &Arc<Store>, rest: String, sh: &crate::run::Shared, a: &mut crate::run::ActorState, rbuf: &mut crate::run::Ring, op: &crate::vm::Op) -> io::Result<i64> {
    let sink = (op.kind == crate::vm::OpKind::Read).then(|| Sink::Lent(rbuf.slice(op.len as usize)));
    match start(store, rest, sh, a, op, sink) {
        Step::Done(r) => r,
        Step::Send(f) => settle(a, op, block(f)).0,
    }
}

/// Begin `op`, whose path lives at `rest` of `store`, the object engine's way (see the module
/// comment): what is local is done here, on the actor's thread, and what goes to the store
/// is the future returned, which holds no reference to the actor (`sink`, a read's buffer,
/// aside). The future's answer is `settle`d on the actor's thread.
pub(crate) fn start<'a>(store: &Arc<Store>, rest: String, sh: &crate::run::Shared, a: &mut crate::run::ActorState, op: &crate::vm::Op, sink: Option<Sink<'a>>) -> Step<'a> {
    use crate::ast::OpenFlag;
    use crate::vm::OpKind;
    let refused = |what: &str| Step::Done(Err(error(libc::EOPNOTSUPP, format!("{} {}: {what} has no object form (V18, DESIGN_REVIEW.md §3.65)", op.kind.name(), op.path))));
    macro_rules! fd {
        () => {
            match a.fd(op.path) {
                Ok(f) => f,
                Err(e) => return Step::Done(Err(e)),
            }
        };
    }
    match op.kind {
        OpKind::Open => {
            let flags = op.aux & 0xffff_ffff;
            let bit = |f: OpenFlag| flags & (1 << (f as u8)) != 0;
            if bit(OpenFlag::APPEND) || bit(OpenFlag::RDWR) || bit(OpenFlag::EXCL) {
                return refused("O_APPEND, O_RDWR, or O_EXCL");
            }
            let upload = writes(flags).then(|| std::sync::Mutex::new(Some(store.upload(&rest))));
            let file = crate::backend::OpenFile::object(Handle { store: store.clone(), rest, upload });
            a.opened(sh, op.path, op.aux, file);
            Step::Done(Ok(0))
        }
        OpKind::Close => {
            // the owner's close sends the object; a sub-actor closing what it inherited does not
            let up = a.fds.own.get(op.path).cloned().and_then(|f| f.as_object().and_then(|h| h.upload.as_ref()).map(|c| c.lock().unwrap().take()));
            if let Err(e) = a.close(op.path) {
                return Step::Done(Err(e));
            }
            match up {
                None => Step::Done(Ok(0)),
                Some(None) => Step::Done(Err(busy(op))),
                Some(Some(up)) => Step::Send(Box::pin(async move { Answer::Done(up.finish().await.map(|_| 0)) })),
            }
        }
        OpKind::Read => {
            let f = fd!();
            if f.as_object().is_some_and(|h| h.upload.is_some()) {
                return Step::Done(Err(io::Error::from_raw_os_error(libc::EBADF)));
            }
            let mut sink = sink.expect("a read is begun with its buffer");
            let read = store.read(&rest, op.offset as u64);
            Step::Send(Box::pin(async move {
                let r = read.into(sink.bytes()).await.map(|n| n as i64);
                Answer::Read(r, sink.owned())
            }))
        }
        OpKind::Write => {
            let f = fd!();
            let Some(cell) = f.as_object().and_then(|h| h.upload.as_ref()) else {
                return Step::Done(Err(io::Error::from_raw_os_error(libc::EBADF)));
            };
            let Some(mut up) = cell.lock().unwrap().take() else { return Step::Done(Err(busy(op))) };
            let seed = crate::payload::object_seed(op.seed, op.path);
            let filler = &mut a.filler;
            let written = up.written();
            let parts = up.fill(op.offset as u64, op.len as u64, &mut |at, b| filler.fill_range(|blk| crate::payload::block_seed(seed, 0, blk), at, b));
            match parts {
                Ok(parts) if !parts.is_empty() => {
                    let len = op.len;
                    let file = f.clone();
                    Step::Send(Box::pin(async move {
                        let r = up.send(parts).await.map(|_| len);
                        Answer::Wrote(file, up, r)
                    }))
                }
                r => {
                    *cell.lock().unwrap() = Some(up);
                    Step::Done(match r {
                        Ok(_) => Ok(op.len),
                        Err(e) if e.raw_os_error() == Some(libc::ESPIPE) => Err(error(libc::ESPIPE, format!("write {} at {}: the object has {written} bytes written; an upload is written in order from 0 (V18)", op.path, op.offset))),
                        Err(e) => Err(e),
                    })
                }
            }
        }
        OpKind::Lseek | OpKind::Fadvise | OpKind::Fsync | OpKind::Fdatasync => Step::Done(a.fd(op.path).map(|_| 0)),
        OpKind::Ioctl => {
            fd!();
            Step::Done(match crate::ast::IoctlRequest::from_code(op.aux) {
                crate::ast::IoctlRequest::TCGETS => Err(io::Error::from_raw_os_error(libc::ENOTTY)),
                _ => Ok(0),
            })
        }
        OpKind::Fstat => {
            let f = fd!();
            match f.as_object().and_then(|h| h.upload.as_ref()) {
                Some(c) => Step::Done(c.lock().unwrap().as_ref().map(|u| u.written() as i64).ok_or_else(|| busy(op))),
                None => {
                    let f = store.stat(&rest);
                    Step::Send(Box::pin(async move { Answer::Done(f.await) }))
                }
            }
        }
        OpKind::Stat => {
            let f = store.stat(&rest);
            Step::Send(Box::pin(async move { Answer::Done(f.await) }))
        }
        OpKind::Readdir => {
            fd!();
            let f = store.list_dir(&rest);
            Step::Send(Box::pin(async move { Answer::Done(f.await.map(|n| n as i64)) }))
        }
        OpKind::Unlink => {
            let f = store.delete(&rest);
            Step::Send(Box::pin(async move { Answer::Removed(f.await) }))
        }
        OpKind::Rename => {
            let to = op.path2.unwrap_or("");
            let Some((to_store, to_rest)) = sh.opts.endpoints.object(to) else {
                return Step::Done(Err(error(libc::EXDEV, format!("rename {} to {to}: the target is not in the object store", op.path))));
            };
            if !Arc::ptr_eq(store, &to_store) {
                return Step::Done(Err(error(libc::EXDEV, format!("rename {} to {to}: the target is in another object store", op.path))));
            }
            let f = store.rename(&rest, &to_rest);
            Step::Send(Box::pin(async move { Answer::Renamed(f.await) }))
        }
        OpKind::Mkdir | OpKind::Rmdir => Step::Done(Ok(0)),
        OpKind::Ftruncate => refused("ftruncate"),
        OpKind::Fallocate => refused("fallocate"),
    }
}

/// The answer of `op`'s future, on the actor's thread: what the op returned, and the read's
/// buffer when it owned one.
pub(crate) fn settle(a: &mut crate::run::ActorState, op: &crate::vm::Op, answer: Answer) -> (io::Result<i64>, Option<Vec<u8>>) {
    match answer {
        Answer::Done(r) => (r, None),
        Answer::Read(r, buf) => (r, buf),
        Answer::Removed(r) => {
            if r.is_ok() {
                a.removed.push(op.path.to_string());
            }
            (r.map(|_| 0), None)
        }
        Answer::Renamed(r) => {
            if r.is_ok() {
                a.removed.push(op.path.to_string());
                a.created.push((op.path2.unwrap_or("").to_string(), a.actor));
            }
            (r.map(|_| 0), None)
        }
        Answer::Wrote(file, up, r) => {
            if let Some(cell) = file.as_object().and_then(|h| h.upload.as_ref()) {
                *cell.lock().unwrap() = Some(up);
            }
            (r, None)
        }
    }
}

#[cfg(feature = "object")]
mod imp {
    use std::io;
    use std::ops::Range;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::OnceLock;

    use futures::StreamExt;
    use object_store::aws::{AmazonS3, AmazonS3Builder};
    use object_store::path::Path;
    use object_store::{GetOptions, GetRange, MultipartUpload, ObjectStore, ObjectStoreExt, PutPayload};

    use super::{error, Fut};

    static THREADS: AtomicUsize = AtomicUsize::new(super::DEFAULT_THREADS);
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

    /// The workers of the runtime the first object call builds (later calls change nothing).
    pub fn set_threads(n: usize) {
        THREADS.store(n.max(1), Ordering::Relaxed);
    }

    pub fn threads() -> usize {
        THREADS.load(Ordering::Relaxed)
    }

    fn runtime() -> &'static tokio::runtime::Runtime {
        RUNTIME.get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(threads())
                .thread_name("aeiou-object")
                .enable_all()
                .build()
                .expect("building the object engine's runtime")
        })
    }

    /// Run `f` on the runtime, the calling thread waiting for it (never one of the
    /// runtime's own threads, where `block_on` panics).
    pub fn block<F: std::future::Future>(f: F) -> F::Output {
        runtime().handle().block_on(f)
    }

    /// Run `f` on the runtime without waiting; `done` gets its answer on a worker.
    pub fn spawn(f: super::Fut<'static, super::Answer>, done: Box<dyn FnOnce(super::Answer) + Send>) {
        runtime().spawn(async move { done(f.await) });
    }

    /// An object being written through one handle: the part being filled, what has been
    /// written, and the multipart upload once a part has gone out. Dropped unfinished (a
    /// run that failed), it aborts the upload, so the store keeps no parts.
    pub struct Upload {
        s3: AmazonS3,
        key: Path,
        show: String,
        part: usize,
        buf: Vec<u8>,
        written: u64,
        multi: Option<Box<dyn MultipartUpload>>,
    }

    impl Upload {
        /// The object's size so far: every byte written through the handle.
        pub fn written(&self) -> u64 {
            self.written
        }

        /// `len` bytes at `offset`, which must be the end of what is written (else
        /// `ESPIPE`), made by `fill(offset, bytes)` into the part being filled; returns the
        /// parts that filled, for `send` before the write returns.
        pub fn fill(&mut self, offset: u64, len: u64, fill: &mut dyn FnMut(u64, &mut [u8])) -> io::Result<Vec<Vec<u8>>> {
            if offset != self.written {
                return Err(io::Error::from_raw_os_error(libc::ESPIPE));
            }
            let mut full = Vec::new();
            let mut at = 0u64;
            while at < len {
                if self.buf.capacity() == 0 {
                    self.buf = Vec::with_capacity(self.part);
                }
                let take = ((len - at) as usize).min(self.part - self.buf.len());
                let from = self.buf.len();
                self.buf.resize(from + take, 0);
                fill(offset + at, &mut self.buf[from..]);
                at += take as u64;
                self.written += take as u64;
                if self.buf.len() == self.part {
                    full.push(std::mem::take(&mut self.buf));
                }
            }
            Ok(full)
        }

        /// Send `parts` in order, the first one starting the multipart upload.
        pub async fn send(&mut self, parts: Vec<Vec<u8>>) -> io::Result<()> {
            for data in parts {
                if self.multi.is_none() {
                    self.multi = Some(self.s3.put_multipart(&self.key).await.map_err(|e| io_err(&self.show, e))?);
                }
                let up = self.multi.as_mut().expect("started");
                up.put_part(PutPayload::from(data)).await.map_err(|e| io_err(&self.show, e))?;
            }
            Ok(())
        }

        /// The `close`: one `PUT` of what is written when no part has gone out, else the
        /// last part and the completion.
        pub async fn finish(mut self) -> io::Result<()> {
            if self.multi.is_none() {
                let data = PutPayload::from(std::mem::take(&mut self.buf));
                return self.s3.put(&self.key, data).await.map(|_| ()).map_err(|e| io_err(&self.show, e));
            }
            if !self.buf.is_empty() {
                let last = std::mem::take(&mut self.buf);
                self.send(vec![last]).await?;
            }
            let mut up = self.multi.take().expect("started");
            match up.complete().await {
                Ok(_) => Ok(()),
                Err(e) => {
                    let _ = up.abort().await;
                    Err(io_err(&self.show, e))
                }
            }
        }
    }

    impl Drop for Upload {
        fn drop(&mut self) {
            if let Some(mut up) = self.multi.take() {
                // on one of the runtime's own threads (an answer no loop took), where
                // `block_on` panics, the abort goes on the runtime
                match tokio::runtime::Handle::try_current() {
                    Ok(h) => drop(h.spawn(async move { up.abort().await })),
                    Err(_) => drop(block(up.abort())),
                }
            }
        }
    }

    /// A ranged `GET` about to be sent: `into` streams the bytes the object has at
    /// `[offset, offset + buf.len())` into `buf` (fewer at its end, none past it).
    pub struct Reader {
        s3: AmazonS3,
        key: Path,
        show: String,
        offset: u64,
    }

    impl Reader {
        pub async fn into(self, buf: &mut [u8]) -> io::Result<usize> {
            if buf.is_empty() {
                return Ok(0);
            }
            let range: Range<u64> = self.offset..self.offset + buf.len() as u64;
            let opts = GetOptions { range: Some(GetRange::Bounded(range)), ..Default::default() };
            let r = match self.s3.get_opts(&self.key, opts).await {
                Ok(r) => r,
                Err(e) if not_satisfiable(&e) => return Ok(0),
                Err(e) => return Err(io_err(&self.show, e)),
            };
            let mut stream = r.into_stream();
            let mut n = 0usize;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| io_err(&self.show, e))?;
                let take = chunk.len().min(buf.len() - n);
                buf[n..n + take].copy_from_slice(&chunk[..take]);
                n += take;
            }
            Ok(n)
        }
    }

    /// One endpoint's store: a bucket and the prefix the name's root is placed at.
    pub struct Store {
        uri: String,
        s3: AmazonS3,
        prefix: String,
    }

    fn io_err(what: &str, e: object_store::Error) -> io::Error {
        let errno = match &e {
            object_store::Error::NotFound { .. } => libc::ENOENT,
            object_store::Error::AlreadyExists { .. } => libc::EEXIST,
            object_store::Error::PermissionDenied { .. } | object_store::Error::Unauthenticated { .. } => libc::EACCES,
            object_store::Error::NotSupported { .. } | object_store::Error::NotImplemented { .. } => libc::EOPNOTSUPP,
            _ => libc::EIO,
        };
        error(errno, format!("{what}: {e}"))
    }

    /// A ranged `GET` whose start is at or past the end of the object: the server's 416.
    fn not_satisfiable(e: &object_store::Error) -> bool {
        let s = e.to_string();
        s.contains("416") || s.contains("Range Not Satisfiable") || s.contains("InvalidRange")
    }

    async fn any_below(s3: &AmazonS3, key: &Path) -> io::Result<bool> {
        let prefix = if key.as_ref().is_empty() { None } else { Some(key) };
        match s3.list(prefix).next().await {
            None => Ok(false),
            Some(Ok(_)) => Ok(true),
            Some(Err(e)) => Err(io_err(key.as_ref(), e)),
        }
    }

    impl Store {
        /// `s3://BUCKET[/PREFIX]`, the store from the environment.
        pub fn open(uri: &str) -> anyhow::Result<Store> {
            let Some(rest) = uri.strip_prefix("s3://") else {
                anyhow::bail!("{uri}: the object engine knows `s3://BUCKET[/PREFIX]` only");
            };
            let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
            if bucket.is_empty() {
                anyhow::bail!("{uri}: no bucket");
            }
            let mut b = AmazonS3Builder::from_env().with_bucket_name(bucket);
            if std::env::var_os("AWS_VIRTUAL_HOSTED_STYLE_REQUEST").is_none() {
                b = b.with_virtual_hosted_style_request(false);
            }
            let s3 = b.build().map_err(|e| anyhow::anyhow!("{uri}: {e}"))?;
            Ok(Store { uri: uri.trim_end_matches('/').to_string(), s3, prefix: prefix.trim_matches('/').to_string() })
        }

        pub fn uri(&self) -> &str {
            &self.uri
        }

        /// The key of `rest` (a path below the endpoint's root, `""` for the root itself).
        fn key(&self, rest: &str) -> Path {
            let rest = rest.trim_matches('/');
            match (self.prefix.is_empty(), rest.is_empty()) {
                (true, _) => Path::from(rest),
                (false, true) => Path::from(self.prefix.as_str()),
                (false, false) => Path::from(format!("{}/{rest}", self.prefix)),
            }
        }

        /// Where `rest` is, for messages.
        pub fn show(&self, rest: &str) -> String {
            if rest.is_empty() { self.uri.clone() } else { format!("{}/{}", self.uri, rest.trim_matches('/')) }
        }

        /// The ranged `GET` of a read at `offset` of `rest`, owning what it needs.
        pub fn read(&self, rest: &str, offset: u64) -> Reader {
            Reader { s3: self.s3.clone(), key: self.key(rest), show: self.show(rest), offset }
        }

        /// `HEAD`: the object's size; for no object, size 0 when a `LIST` finds a key below
        /// it (a directory), `ENOENT` otherwise.
        pub fn stat(&self, rest: &str) -> Fut<'static, io::Result<i64>> {
            let (s3, key, show) = (self.s3.clone(), self.key(rest), self.show(rest));
            Box::pin(async move {
                match s3.head(&key).await {
                    Ok(m) => Ok(m.size as i64),
                    Err(object_store::Error::NotFound { .. }) => {
                        if any_below(&s3, &key).await? { Ok(0) } else { Err(error(libc::ENOENT, format!("{show}: no such object"))) }
                    }
                    Err(e) => Err(io_err(&show, e)),
                }
            })
        }

        /// `LIST` with the delimiter `/` below `rest`: its objects and common prefixes,
        /// not counting names that begin with `.aeiou` (`schema/README.md` §6).
        pub fn list_dir(&self, rest: &str) -> Fut<'static, io::Result<usize>> {
            let (s3, key, show) = (self.s3.clone(), self.key(rest), self.show(rest));
            Box::pin(async move {
                let prefix = if key.as_ref().is_empty() { None } else { Some(&key) };
                let l = s3.list_with_delimiter(prefix).await.map_err(|e| io_err(&show, e))?;
                let hidden = |p: &Path| p.filename().is_some_and(|f| f.starts_with(".aeiou"));
                Ok(l.objects.iter().filter(|o| !hidden(&o.location)).count() + l.common_prefixes.iter().filter(|p| !hidden(p)).count())
            })
        }

        /// Is there any key below `rest`?
        pub fn non_empty(&self, rest: &str) -> io::Result<bool> {
            let key = self.key(rest);
            block(any_below(&self.s3, &key))
        }

        /// The whole object, or `None` when there is none.
        pub fn get(&self, rest: &str) -> io::Result<Option<Vec<u8>>> {
            let key = self.key(rest);
            block(async {
                match self.s3.get(&key).await {
                    Ok(r) => r.bytes().await.map(|b| Some(b.to_vec())).map_err(|e| io_err(&self.show(rest), e)),
                    Err(object_store::Error::NotFound { .. }) => Ok(None),
                    Err(e) => Err(io_err(&self.show(rest), e)),
                }
            })
        }

        /// An upload of the object at `rest`, in parts of `--object-part-mib`; nothing is
        /// sent until a part fills or the handle closes.
        pub fn upload(&self, rest: &str) -> Upload {
            Upload { s3: self.s3.clone(), key: self.key(rest), show: self.show(rest), part: super::part_size() as usize, buf: Vec::new(), written: 0, multi: None }
        }

        /// `DELETE` (a missing key is no error, as S3 has it).
        pub fn delete(&self, rest: &str) -> Fut<'static, io::Result<()>> {
            let (s3, key, show) = (self.s3.clone(), self.key(rest), self.show(rest));
            Box::pin(async move { s3.delete(&key).await.map_err(|e| io_err(&show, e)) })
        }

        /// A copy on the server and a `DELETE` of the source.
        pub fn rename(&self, from: &str, to: &str) -> Fut<'static, io::Result<()>> {
            let (s3, f, t, show) = (self.s3.clone(), self.key(from), self.key(to), self.show(from));
            Box::pin(async move {
                s3.copy(&f, &t).await.map_err(|e| io_err(&show, e))?;
                s3.delete(&f).await.map_err(|e| io_err(&show, e))
            })
        }

        /// Every key below `rest` (a recursive `LIST`), as paths below the endpoint's prefix.
        pub fn keys_below(&self, rest: &str) -> io::Result<Vec<String>> {
            let key = self.key(rest);
            let strip = if self.prefix.is_empty() { String::new() } else { format!("{}/", self.prefix) };
            block(async {
                let prefix = if key.as_ref().is_empty() { None } else { Some(&key) };
                let mut out = Vec::new();
                let mut l = self.s3.list(prefix);
                while let Some(m) = l.next().await {
                    let m = m.map_err(|e| io_err(&self.show(rest), e))?;
                    let k = m.location.as_ref();
                    out.push(k.strip_prefix(strip.as_str()).unwrap_or(k).to_string());
                }
                out.sort();
                Ok(out)
            })
        }

        /// `DELETE` every key of `rests` (paths below the endpoint's prefix), in batches.
        pub fn delete_all(&self, rests: &[String]) -> io::Result<()> {
            let keys: Vec<object_store::Result<Path>> = rests.iter().map(|r| Ok(self.key(r))).collect();
            block(async {
                let mut s = self.s3.delete_stream(futures::stream::iter(keys).boxed());
                while let Some(r) = s.next().await {
                    r.map_err(|e| io_err(&self.uri, e))?;
                }
                Ok(())
            })
        }

        /// One `PUT` of the whole object.
        pub fn put(&self, rest: &str, data: Vec<u8>) -> io::Result<()> {
            let key = self.key(rest);
            block(async { self.s3.put(&key, PutPayload::from(data)).await.map(|_| ()).map_err(|e| io_err(&self.show(rest), e)) })
        }

        /// An object of `size` bytes whose content `fill(offset, part)` writes, part by part:
        /// one `PUT` when it fits in one part, a multipart upload in parts of `part` bytes
        /// otherwise, each part sent before the next is made.
        pub fn put_parts(&self, rest: &str, size: u64, part: u64, fill: &mut dyn FnMut(u64, &mut [u8])) -> io::Result<()> {
            let make = |fill: &mut dyn FnMut(u64, &mut [u8]), at: u64, len: u64| {
                let mut v = vec![0u8; len as usize];
                fill(at, &mut v);
                v
            };
            if size <= part {
                return self.put(rest, make(fill, 0, size));
            }
            let key = self.key(rest);
            let show = self.show(rest);
            block(async {
                let mut up = self.s3.put_multipart(&key).await.map_err(|e| io_err(&show, e))?;
                let mut at = 0u64;
                while at < size {
                    let len = (size - at).min(part);
                    if let Err(e) = up.put_part(PutPayload::from(make(fill, at, len))).await {
                        let _ = up.abort().await;
                        return Err(io_err(&show, e));
                    }
                    at += len;
                }
                up.complete().await.map(|_| ()).map_err(|e| io_err(&show, e))
            })
        }
    }
}

#[cfg(not(feature = "object"))]
mod imp {
    use std::io;

    use super::Fut;

    pub fn set_threads(_: usize) {}

    pub fn threads() -> usize {
        0
    }

    const NONE: &str = "no object engine in this build, so no op of one";

    pub fn block<F: std::future::Future>(_: F) -> F::Output {
        unreachable!("{NONE}")
    }

    pub fn spawn(_: Fut<'static, super::Answer>, _: Box<dyn FnOnce(super::Answer) + Send>) {
        unreachable!("{NONE}")
    }

    /// No store: this build has no object engine, and `open` says so.
    pub enum Store {}

    pub enum Upload {}

    pub struct Reader;

    impl Reader {
        pub async fn into(self, _: &mut [u8]) -> io::Result<usize> {
            unreachable!("{NONE}")
        }
    }

    impl Upload {
        pub fn written(&self) -> u64 {
            match *self {}
        }
        pub fn fill(&mut self, _: u64, _: u64, _: &mut dyn FnMut(u64, &mut [u8])) -> io::Result<Vec<Vec<u8>>> {
            match *self {}
        }
        pub async fn send(&mut self, _: Vec<Vec<u8>>) -> io::Result<()> {
            match *self {}
        }
        pub async fn finish(self) -> io::Result<()> {
            match self {}
        }
    }

    impl Store {
        pub fn open(uri: &str) -> anyhow::Result<Store> {
            anyhow::bail!("{uri}: this aeiou was built without the object engine (cargo feature `object`, DESIGN_REVIEW.md §3.65)")
        }
        pub fn uri(&self) -> &str {
            match *self {}
        }
        pub fn show(&self, _: &str) -> String {
            match *self {}
        }
        pub fn read(&self, _: &str, _: u64) -> Reader {
            match *self {}
        }
        pub fn stat(&self, _: &str) -> Fut<'static, io::Result<i64>> {
            match *self {}
        }
        pub fn list_dir(&self, _: &str) -> Fut<'static, io::Result<usize>> {
            match *self {}
        }
        pub fn non_empty(&self, _: &str) -> io::Result<bool> {
            match *self {}
        }
        pub fn get(&self, _: &str) -> io::Result<Option<Vec<u8>>> {
            match *self {}
        }
        pub fn put(&self, _: &str, _: Vec<u8>) -> io::Result<()> {
            match *self {}
        }
        pub fn upload(&self, _: &str) -> Upload {
            match *self {}
        }
        pub fn delete(&self, _: &str) -> Fut<'static, io::Result<()>> {
            match *self {}
        }
        pub fn rename(&self, _: &str, _: &str) -> Fut<'static, io::Result<()>> {
            match *self {}
        }
        pub fn keys_below(&self, _: &str) -> io::Result<Vec<String>> {
            match *self {}
        }
        pub fn delete_all(&self, _: &[String]) -> io::Result<()> {
            match *self {}
        }
        pub fn put_parts(&self, _: &str, _: u64, _: u64, _: &mut dyn FnMut(u64, &mut [u8])) -> io::Result<()> {
            match *self {}
        }
    }
}
