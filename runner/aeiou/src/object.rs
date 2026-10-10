//! The object engine (`DESIGN_REVIEW.md` §3.65): the ops of a dataset or namespace declared
//! `protocol: object`, issued to an S3 store through Apache `object_store`, the library chosen
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
//! **The mapping** (§3.65's table), reads only so far: `open` sends nothing (the handle is
//! the key); a read at an offset is one ranged `GET` streamed into the actor's buffer, a
//! range past the end of the object a read of 0 bytes, a range over the end a short read;
//! `stat` and `fstat` are a `HEAD`, and for a path that is no object a `LIST` of the prefix
//! below it, which succeeds (size 0) when there is one, as `stat` of a directory does;
//! `readdir` is a `LIST` with the delimiter `/`, its objects and common prefixes the entries;
//! `lseek`, `fadvise`, `fsync`, and `fdatasync` are local and send nothing, and `ioctl` is
//! local with a regular file's answers (`TCGETS` is `ENOTTY`). Every op that would change the
//! store fails with `EROFS` until the object writes are built.
//!
//! **Threads.** One multi-thread tokio runtime per process, built on the first object call
//! with `--object-threads` workers (`set_threads`), parked until the process exits. An
//! actor's thread runs each of its ops to completion with `Handle::block_on`, so an actor has
//! one op in flight, as under `sync`: the request is built and signed on the actor's thread,
//! and the workers drive the connections. The event loops (`io_uring`, `libaio`) refuse
//! object names until a completion of the runtime can wake a loop.

use std::io;
use std::sync::Arc;

/// The library the engine is built on, as the run prints it (pinned in `Cargo.toml`).
pub const LIBRARY_VERSION: &str = "0.14.2";

/// Default worker threads of the engine's runtime (`--object-threads`).
pub const DEFAULT_THREADS: usize = 2;

/// The part size of a multipart upload: an object larger than this is sent in parts of it
/// (8 MiB, as `s3dlio`; `object_store`'s own writer uses 10 MiB).
pub const DEFAULT_PART: u64 = 8 << 20;

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
/// store and the key's part below the endpoint's prefix.
pub struct Handle {
    pub store: Arc<Store>,
    pub rest: String,
}

pub use imp::{set_threads, threads, Store};

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

/// Issue `op`, whose path lives at `rest` of `store`, the object engine's way (see the module
/// comment); `Ok(n)` is what the op returned, as from `run::issue_blocking`.
pub(crate) fn issue(store: &Arc<Store>, rest: String, sh: &crate::run::Shared, a: &mut crate::run::ActorState, rbuf: &mut crate::run::Ring, op: &crate::vm::Op) -> io::Result<i64> {
    use crate::vm::OpKind;
    let erofs = || Err(error(libc::EROFS, format!("{} {}: the object engine does not write yet (DESIGN_REVIEW.md §3.65)", op.kind.name(), op.path)));
    match op.kind {
        OpKind::Open => {
            if writes(op.aux & 0xffff_ffff) {
                return erofs();
            }
            let file = crate::backend::OpenFile::object(Handle { store: store.clone(), rest });
            a.opened(sh, op.path, op.aux, file);
            Ok(0)
        }
        OpKind::Close => a.close(op.path).map(|_| 0),
        OpKind::Read => {
            a.fd(op.path)?;
            let buf = rbuf.slice(op.len as usize);
            store.read_into(&rest, op.offset as u64, buf).map(|n| n as i64)
        }
        OpKind::Lseek | OpKind::Fadvise | OpKind::Fsync | OpKind::Fdatasync => a.fd(op.path).map(|_| 0),
        OpKind::Ioctl => {
            a.fd(op.path)?;
            match crate::ast::IoctlRequest::from_code(op.aux) {
                crate::ast::IoctlRequest::TCGETS => Err(io::Error::from_raw_os_error(libc::ENOTTY)),
                _ => Ok(0),
            }
        }
        OpKind::Fstat => {
            a.fd(op.path)?;
            store.stat(&rest)
        }
        OpKind::Stat => store.stat(&rest),
        OpKind::Readdir => {
            a.fd(op.path)?;
            store.list_dir(&rest).map(|n| n as i64)
        }
        OpKind::Write | OpKind::Unlink | OpKind::Ftruncate | OpKind::Fallocate | OpKind::Mkdir | OpKind::Rmdir | OpKind::Rename => erofs(),
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
    use object_store::{GetOptions, GetRange, ObjectStore, ObjectStoreExt, PutPayload};

    use super::error;

    static THREADS: AtomicUsize = AtomicUsize::new(super::DEFAULT_THREADS);
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

    /// The workers of the runtime the first object call builds (later calls change nothing).
    pub fn set_threads(n: usize) {
        THREADS.store(n.max(1), Ordering::Relaxed);
    }

    pub fn threads() -> usize {
        THREADS.load(Ordering::Relaxed)
    }

    fn block<F: std::future::Future>(f: F) -> F::Output {
        let rt = RUNTIME.get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(threads())
                .thread_name("aeiou-object")
                .enable_all()
                .build()
                .expect("building the object engine's runtime")
        });
        rt.handle().block_on(f)
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

        /// A ranged `GET` of `[offset, offset + buf.len())` streamed into `buf`: the bytes the
        /// object has there (fewer at its end, none past it).
        pub fn read_into(&self, rest: &str, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
            if buf.is_empty() {
                return Ok(0);
            }
            let key = self.key(rest);
            let range: Range<u64> = offset..offset + buf.len() as u64;
            block(async {
                let opts = GetOptions { range: Some(GetRange::Bounded(range)), ..Default::default() };
                let r = match self.s3.get_opts(&key, opts).await {
                    Ok(r) => r,
                    Err(e) if not_satisfiable(&e) => return Ok(0),
                    Err(e) => return Err(io_err(&self.show(rest), e)),
                };
                let mut stream = r.into_stream();
                let mut n = 0usize;
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.map_err(|e| io_err(&self.show(rest), e))?;
                    let take = chunk.len().min(buf.len() - n);
                    buf[n..n + take].copy_from_slice(&chunk[..take]);
                    n += take;
                }
                Ok(n)
            })
        }

        /// `HEAD`: the object's size; for no object, size 0 when a `LIST` finds a key below
        /// it (a directory), `ENOENT` otherwise.
        pub fn stat(&self, rest: &str) -> io::Result<i64> {
            let key = self.key(rest);
            block(async {
                match self.s3.head(&key).await {
                    Ok(m) => Ok(m.size as i64),
                    Err(object_store::Error::NotFound { .. }) => {
                        if self.any_below(&key).await? { Ok(0) } else { Err(error(libc::ENOENT, format!("{}: no such object", self.show(rest)))) }
                    }
                    Err(e) => Err(io_err(&self.show(rest), e)),
                }
            })
        }

        async fn any_below(&self, key: &Path) -> io::Result<bool> {
            let prefix = if key.as_ref().is_empty() { None } else { Some(key) };
            match self.s3.list(prefix).next().await {
                None => Ok(false),
                Some(Ok(_)) => Ok(true),
                Some(Err(e)) => Err(io_err(key.as_ref(), e)),
            }
        }

        /// `LIST` with the delimiter `/` below `rest`: its objects and common prefixes,
        /// not counting names that begin with `.aeiou` (`schema/README.md` §6).
        pub fn list_dir(&self, rest: &str) -> io::Result<usize> {
            let key = self.key(rest);
            block(async {
                let prefix = if key.as_ref().is_empty() { None } else { Some(&key) };
                let l = self.s3.list_with_delimiter(prefix).await.map_err(|e| io_err(&self.show(rest), e))?;
                let hidden = |p: &Path| p.filename().is_some_and(|f| f.starts_with(".aeiou"));
                Ok(l.objects.iter().filter(|o| !hidden(&o.location)).count() + l.common_prefixes.iter().filter(|p| !hidden(p)).count())
            })
        }

        /// Is there any key below `rest`?
        pub fn non_empty(&self, rest: &str) -> io::Result<bool> {
            let key = self.key(rest);
            block(self.any_below(&key))
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

    pub fn set_threads(_: usize) {}

    pub fn threads() -> usize {
        0
    }

    /// No store: this build has no object engine, and `open` says so.
    pub enum Store {}

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
        pub fn read_into(&self, _: &str, _: u64, _: &mut [u8]) -> io::Result<usize> {
            match *self {}
        }
        pub fn stat(&self, _: &str) -> io::Result<i64> {
            match *self {}
        }
        pub fn list_dir(&self, _: &str) -> io::Result<usize> {
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
        pub fn put_parts(&self, _: &str, _: u64, _: u64, _: &mut dyn FnMut(u64, &mut [u8])) -> io::Result<()> {
            match *self {}
        }
    }
}
