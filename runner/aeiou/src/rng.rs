//! Positional randomness and the keyed permutation. Nothing here has state: every value is a
//! function of its arguments, which is what makes the op stream a pure function of
//! `(seed, actor, site, indices)` (`NAPKIN_MATH.md` §4.1, `DESIGN_REVIEW.md` §2.3).
//!
//! Definitions fixed by the runner (the schema leaves them to it, `schema/README.md` §5):
//! - a **key** is `xxh3_64(bytes, seed)` over the little-endian words `actor, site, i₁ … iₙ`;
//! - the **words** of a key are the SplitMix64 sequence started at the key, so a draw that
//!   needs several uniforms (a mixture arm and its value, Box–Muller) takes them in order;
//! - the **permutation** over `[0, N)` is a 4-round Feistel network on the smallest 2^k ≥ N,
//!   unbalanced when k is odd, with cycle-walking, keyed by a 64-bit key.

use xxhash_rust::xxh3::xxh3_64_with_seed;

/// The key of a positional draw. `indices` are the enclosing loop indices, outermost first.
pub fn position_key(seed: u64, actor: u64, site: u64, indices: &[i64]) -> u64 {
    let mut buf = Vec::with_capacity(16 + 8 * indices.len());
    buf.extend_from_slice(&actor.to_le_bytes());
    buf.extend_from_slice(&site.to_le_bytes());
    for i in indices {
        buf.extend_from_slice(&i.to_le_bytes());
    }
    xxh3_64_with_seed(&buf, seed)
}

/// The key of a per-id dataset draw (sizes, layouts): `(dataset seed, id)`, no site.
pub fn dataset_key(dataset_seed: u64, id: i64) -> u64 {
    xxh3_64_with_seed(&id.to_le_bytes(), dataset_seed)
}

/// A named sub-key: the permutation key of a dataset for an epoch, the rank order of a
/// dataset, and the like. `label` keeps different uses of one seed apart.
pub fn labeled_key(seed: u64, label: &str, parts: &[u64]) -> u64 {
    let mut buf = Vec::with_capacity(label.len() + 8 * parts.len() + 1);
    buf.extend_from_slice(label.as_bytes());
    buf.push(0);
    for p in parts {
        buf.extend_from_slice(&p.to_le_bytes());
    }
    xxh3_64_with_seed(&buf, seed)
}

/// Hash of a site's JSON pointer.
pub fn site_hash(pointer: &str) -> u64 {
    xxh3_64_with_seed(pointer.as_bytes(), 0x5174_e5e1_7e5e_0000)
}

/// The word sequence of a key: SplitMix64 (Steele, Lea, Flood 2014).
#[derive(Clone, Copy, Debug)]
pub struct Words(u64);

impl Words {
    pub fn new(key: u64) -> Self {
        Words(key)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix64(self.0)
    }

    /// Uniform in `[0, 1)` with 53 bits of resolution.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `(0, 1]`, for logarithms.
    #[inline]
    pub fn next_f64_open(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `[lo, hi)`. Modulo reduction: the bias is 2⁻⁶⁴ × range, irrelevant
    /// for reproducibility and for any range this program uses.
    #[inline]
    pub fn next_range(&mut self, lo: i64, hi: i64) -> i64 {
        debug_assert!(hi > lo);
        let span = (hi as i128 - lo as i128) as u64;
        lo.wrapping_add((self.next_u64() % span) as i64)
    }

    /// A standard normal deviate (Box–Muller, one of the pair).
    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64_open();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// The SplitMix64 finalizer (also a fine 64-bit mixer on its own).
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A keyed bijection on `[0, n)`: 4-round Feistel on the smallest power of two ≥ n, unbalanced
/// when the bit count is odd, cycle-walked (`NAPKIN_MATH.md` §2.1).
#[derive(Clone, Copy, Debug)]
pub struct Perm {
    n: u64,
    key: u64,
    left_bits: u32,
    right_bits: u32,
}

const ROUNDS: usize = 4;

impl Perm {
    pub fn new(n: u64, key: u64) -> Self {
        assert!(n > 0, "permutation over an empty domain");
        let bits = if n <= 2 { 2 } else { 64 - (n - 1).leading_zeros() }; // smallest 2^k ≥ n, k ≥ 2
        let right_bits = bits / 2;
        let left_bits = bits - right_bits;
        Perm { n, key, left_bits, right_bits }
    }

    pub fn len(&self) -> u64 {
        self.n
    }

    #[inline]
    fn round(&self, r: usize, half: u64) -> u64 {
        mix64(self.key ^ (r as u64).rotate_left(48) ^ half.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    #[inline]
    fn encrypt_once(&self, x: u64) -> u64 {
        let (mut lb, mut rb) = (self.left_bits, self.right_bits);
        let mut l = x >> rb;
        let mut r = x & ((1u64 << rb) - 1);
        for i in 0..ROUNDS {
            let f = self.round(i, r) & ((1u64 << lb) - 1);
            let nl = r;
            let nr = l ^ f;
            l = nl;
            r = nr;
            std::mem::swap(&mut lb, &mut rb);
        }
        (l << rb) | r
    }

    /// The image of `x`, `x < n`.
    #[inline]
    pub fn apply(&self, x: u64) -> u64 {
        debug_assert!(x < self.n);
        let mut y = self.encrypt_once(x);
        while y >= self.n {
            y = self.encrypt_once(y);
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perm_is_a_bijection_at_small_and_odd_sizes() {
        for n in [1u64, 2, 3, 5, 7, 8, 100, 1000, 1301, 4096, 65537] {
            let p = Perm::new(n, 0xdead_beef ^ n);
            let mut seen = vec![false; n as usize];
            for x in 0..n {
                let y = p.apply(x);
                assert!(y < n);
                assert!(!seen[y as usize], "n={n}: {y} hit twice");
                seen[y as usize] = true;
            }
        }
    }

    #[test]
    fn perm_is_a_bijection_at_fifty_million() {
        // Spike 2, correctness half: N = 50M through a bitmap.
        let n = 50_000_000u64;
        let p = Perm::new(n, 0x5eed);
        let mut bits = vec![0u64; (n as usize + 63) / 64];
        for x in 0..n {
            let y = p.apply(x) as usize;
            let (w, b) = (y / 64, y % 64);
            assert_eq!(bits[w] >> b & 1, 0, "{y} hit twice");
            bits[w] |= 1 << b;
        }
    }

    #[test]
    fn perm_has_no_constant_stride() {
        let p = Perm::new(1 << 20, 42);
        let d: Vec<i64> = (0..8).map(|x| p.apply(x + 1) as i64 - p.apply(x) as i64).collect();
        assert!(d.windows(2).any(|w| w[0] != w[1]), "{d:?}");
    }

    #[test]
    fn words_are_stable() {
        let mut w = Words::new(1);
        assert_eq!(w.next_u64(), mix64(1u64.wrapping_add(0x9E37_79B9_7F4A_7C15)));
        let f = Words::new(7).next_f64();
        assert!((0.0..1.0).contains(&f));
    }
}
