//! The `seq` value (contract 0.16, `runner/REFERENCE.md` §2): a sequence of blocks, each
//! `(origin, offset)`, and the keys of its whole chunks.
//!
//! A sequence is a persistent tree: a leaf is a run of one origin's blocks, a `Cat` one
//! sequence after another, shared, never copied. A prompt built as an earlier prompt's prefix
//! and new blocks is one new node over the earlier one, however long the lineage; a flat list
//! of runs would hold one run a step of it (a main chain of 1,500 extensions, 1,500 runs a
//! prompt, every prompt kept by the timeline that published it).
//!
//! A chunk's hash is chained: `h(c) = xxh3_64(blocks of chunk c, seed h(c − 1))`, `h(−1) = 0`,
//! each block its origin then its offset, 64-bit little-endian. Two sequences give one hash for
//! chunk `c` exactly when they hold the same blocks up to its end, however they were built. A
//! node at the start of a sequence (one on its left spine) keeps the hashes of the chunks that
//! end inside it and not inside its left child, so a sequence's keys cost the chunks its newest
//! node adds, plus one step down the spine a node. Every walk is a loop, never a recursion
//! through the tree, and dropping a tree drops it the same way.

use std::sync::{Arc, OnceLock};
use xxhash_rust::xxh3::xxh3_64_with_seed;

pub struct Seq {
    len: i64,
    kind: Kind,
    /// The hashes of the chunks of `per` blocks this node holds the end of and its left child
    /// does not, valid when the node starts a sequence; the first `per` asked keeps it.
    memo: OnceLock<(i64, Vec<u64>)>,
}

enum Kind {
    Leaf { origin: u64, start: i64 },
    Cat(Arc<Seq>, Arc<Seq>),
}

impl std::fmt::Debug for Seq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "seq({} blocks)", self.len)
    }
}

impl Drop for Seq {
    fn drop(&mut self) {
        let mut stack = Vec::new();
        if let Kind::Cat(a, b) = std::mem::replace(&mut self.kind, Kind::Leaf { origin: 0, start: 0 }) {
            stack.push(a);
            stack.push(b);
        }
        while let Some(n) = stack.pop() {
            if let Ok(mut n) = Arc::try_unwrap(n) {
                if let Kind::Cat(a, b) = std::mem::replace(&mut n.kind, Kind::Leaf { origin: 0, start: 0 }) {
                    stack.push(a);
                    stack.push(b);
                }
            }
        }
    }
}

fn node(len: i64, kind: Kind) -> Arc<Seq> {
    Arc::new(Seq { len, kind, memo: OnceLock::new() })
}

/// `n` blocks of `origin`, offsets `0..n`.
pub fn run(origin: u64, n: i64) -> Arc<Seq> {
    node(n, Kind::Leaf { origin, start: 0 })
}

impl Seq {
    pub fn len(&self) -> i64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// `a` then `b`.
pub fn cat(a: &Arc<Seq>, b: &Arc<Seq>) -> Arc<Seq> {
    if a.len == 0 {
        return b.clone();
    }
    if b.len == 0 {
        return a.clone();
    }
    node(a.len + b.len, Kind::Cat(a.clone(), b.clone()))
}

/// The first `k` blocks of `s`, `0 <= k <= len`: the nodes left of the cut shared.
pub fn prefix(s: &Arc<Seq>, k: i64) -> Arc<Seq> {
    debug_assert!((0..=s.len).contains(&k));
    let mut lefts = Vec::new();
    let mut n = s.clone();
    let mut k = k;
    let mut out = loop {
        if k == n.len {
            break n;
        }
        if k == 0 {
            break run(0, 0);
        }
        let next = match &n.kind {
            Kind::Leaf { origin, start } => break node(k, Kind::Leaf { origin: *origin, start: *start }),
            Kind::Cat(a, b) => {
                if k <= a.len {
                    a.clone()
                } else {
                    lefts.push(a.clone());
                    k -= a.len;
                    b.clone()
                }
            }
        };
        n = next;
    };
    while let Some(a) = lefts.pop() {
        out = cat(&a, &out);
    }
    out
}

/// Blocks `from..to` of `s`, appended to `out` as their bytes (origin, then offset).
fn blocks(s: &Arc<Seq>, from: i64, to: i64, out: &mut Vec<u8>) {
    // (node, the block the node starts at)
    let mut stack = vec![(s.clone(), 0i64)];
    let mut want = from;
    while let Some((n, at)) = stack.pop() {
        if want >= to {
            break;
        }
        if at + n.len <= want {
            continue;
        }
        match &n.kind {
            Kind::Leaf { origin, start } => {
                let end = to.min(at + n.len);
                for p in want..end {
                    out.extend_from_slice(&origin.to_le_bytes());
                    out.extend_from_slice(&(start + p - at).to_le_bytes());
                }
                want = end;
            }
            Kind::Cat(a, b) => {
                stack.push((b.clone(), at + a.len));
                stack.push((a.clone(), at));
            }
        }
    }
}

/// The chained hashes of `s`'s whole chunks of `per` blocks, `per > 0`.
pub fn chunk_hashes(s: &Arc<Seq>, per: i64) -> Vec<u64> {
    debug_assert!(per > 0);
    // the left spine, top first
    let mut spine = vec![s.clone()];
    loop {
        let next = match &spine.last().unwrap().kind {
            Kind::Cat(a, _) => a.clone(),
            Kind::Leaf { .. } => break,
        };
        spine.push(next);
    }
    let mut out: Vec<u64> = Vec::with_capacity((s.len / per) as usize);
    let mut buf = Vec::with_capacity(16 * per as usize);
    let mut left = 0i64;
    for n in spine.iter().rev() {
        // the chunks ending in this node and not in its left child: those past `left`'s whole ones
        let first = left / per;
        let last = n.len / per;
        if let Some((p, own)) = n.memo.get() {
            if *p == per {
                out.extend_from_slice(own);
                left = n.len;
                continue;
            }
        }
        let mut own = Vec::with_capacity((last - first) as usize);
        for c in first..last {
            buf.clear();
            blocks(n, c * per, (c + 1) * per, &mut buf);
            let seed = out.last().copied().unwrap_or(0);
            let h = xxh3_64_with_seed(&buf, seed);
            out.push(h);
            own.push(h);
        }
        let _ = n.memo.set((per, own));
        left = n.len;
    }
    out
}

/// A chunk's key as the contract gives it: its hash's top 63 bits, never negative.
pub fn key(h: u64) -> i64 {
    (h >> 1) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blocks of `s` as a flat list, to hold the tree to.
    fn flat(s: &Arc<Seq>) -> Vec<(u64, i64)> {
        let mut b = Vec::new();
        blocks(s, 0, s.len, &mut b);
        b.chunks(16).map(|c| (u64::from_le_bytes(c[..8].try_into().unwrap()), i64::from_le_bytes(c[8..].try_into().unwrap()))).collect()
    }

    fn reference(b: &[(u64, i64)], per: usize) -> Vec<u64> {
        let mut out: Vec<u64> = Vec::new();
        for c in b.chunks_exact(per) {
            let bytes: Vec<u8> = c.iter().flat_map(|(o, p)| o.to_le_bytes().into_iter().chain(p.to_le_bytes())).collect();
            out.push(xxh3_64_with_seed(&bytes, out.last().copied().unwrap_or(0)));
        }
        out
    }

    #[test]
    fn a_tree_holds_its_blocks_and_its_hashes_match_a_flat_list() {
        let mut s = run(1, 3);
        let mut f = flat(&s);
        for i in 0..400 {
            // extend, cut, or reuse an earlier prefix, as a session's prompts do
            let fresh = run(100 + i, 1 + (i as i64 * 7) % 5);
            s = match i % 4 {
                0 | 1 => cat(&s, &fresh),
                2 => cat(&prefix(&s, s.len - 1), &fresh),
                _ => cat(&prefix(&s, s.len * 2 / 3), &fresh),
            };
            let mut g = f.clone();
            match i % 4 {
                0 | 1 => {}
                2 => g.truncate(g.len() - 1),
                _ => g.truncate(f.len() * 2 / 3),
            }
            g.extend(flat(&fresh));
            f = g;
            assert_eq!(flat(&s), f);
            for per in [1, 4] {
                assert_eq!(chunk_hashes(&s, per), reference(&f, per as usize));
            }
        }
        // a deep chain drops without recursion
        let mut d = run(7, 1);
        for i in 0..200_000 {
            d = cat(&d, &run(8 + i, 1));
        }
        assert_eq!(chunk_hashes(&d, 1000).len(), 200);
        drop(d);
    }
}
