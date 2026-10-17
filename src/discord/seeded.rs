//! The draws behind the fixed novelty commands (`/hotdog-rater`, `/kobold-name`,
//! `/lick-test`, `/hoard-value`, `/kobold-age`, `/judge`; design.md §6): the same
//! input gives the same result forever, but only someone holding the key
//! (`discord.novelty_key`) can compute it (a key of 0, the default, is no secret).
//!
//! The spec, so it can be reproduced elsewhere (`openssl`, Python):
//!
//! - `K` is the config's 64-bit key, as 8 big-endian bytes (so in the order its
//!   hex digits are written: `0x0001_0203_0405_0607` is bytes 00 01 … 07).
//! - Each command has its own AES-256 key: `K_cmd = SHA-256(K ‖
//!   "increbot/<tag>/v1")` (the tag as ASCII bytes after the key bytes).
//! - The input `x` is a `u64`: the Discord user ID, or for anything else (a
//!   pair of IDs, a judged text) the first 8 bytes, little-endian, of the
//!   SHA-256 of its UTF-8 text.
//! - Block `n` (0, 1, …) is `AES-256-ECB(K_cmd, x as 8 LE bytes ‖ n as 8 LE
//!   bytes)`; its 16 bytes are draws `2n` and `2n + 1`, each a little-endian `u64`.
//!
//! **Changing the key, a tag or this scheme changes everyone's results**; the
//! known-answer test pins the scheme and the commands' pinned tests their use
//! of it (both under a test key, not the deployed one). Bump a tag's `v1` to
//! re-roll one command on purpose.

use aes::Aes256;
use aes::cipher::{BlockEncrypt, KeyInit, generic_array::GenericArray};
use sha2::{Digest, Sha256};

/// The novelty key. Its `Debug` never shows the value.
#[derive(Clone)]
pub struct Key(u64);

impl std::fmt::Debug for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Key(…)")
    }
}

impl Key {
    pub fn new(key: u64) -> Key {
        Key(key)
    }
}

/// First 8 bytes, little-endian, of the SHA-256 of `text`.
pub fn text_input(text: &str) -> u64 {
    // Array-destructuring `d` (rather than slicing it and going through
    // `TryFrom<&[u8]>`) means there's no length to get wrong: the compiler
    // knows `d` is 32 bytes and this takes exactly the first 8.
    let [a0, a1, a2, a3, a4, a5, a6, a7, ..]: [u8; 32] = Sha256::digest(text.as_bytes()).into();
    u64::from_le_bytes([a0, a1, a2, a3, a4, a5, a6, a7])
}

/// The draws for one command and input.
pub struct Stream {
    cipher: Aes256,
    x: u64,
    block: u64,
    buf: [u64; 2],
    used: usize,
}

impl Stream {
    pub fn new(key: &Key, tag: &str, x: u64) -> Stream {
        let mut h = Sha256::new();
        h.update(key.0.to_be_bytes());
        h.update(format!("increbot/{tag}/v1").as_bytes());
        Stream {
            cipher: Aes256::new(&h.finalize()),
            x,
            block: 0,
            buf: [0; 2],
            used: 2,
        }
    }

    pub fn next(&mut self) -> u64 {
        if self.used == 2 {
            let mut b = [0u8; 16];
            b[..8].copy_from_slice(&self.x.to_le_bytes());
            b[8..].copy_from_slice(&self.block.to_le_bytes());
            let mut b = GenericArray::from(b);
            self.cipher.encrypt_block(&mut b);
            // Same reasoning as `text_input`: destructure the known-16-byte
            // array instead of slicing it through `TryFrom<&[u8]>` twice.
            let [l0, l1, l2, l3, l4, l5, l6, l7, h0, h1, h2, h3, h4, h5, h6, h7]: [u8; 16] =
                b.into();
            self.buf = [
                u64::from_le_bytes([l0, l1, l2, l3, l4, l5, l6, l7]),
                u64::from_le_bytes([h0, h1, h2, h3, h4, h5, h6, h7]),
            ];
            self.block += 1;
            self.used = 0;
        }
        self.used += 1;
        self.buf[self.used - 1]
    }

    /// Uniform in (0, 1), from the top 53 bits.
    pub fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Approximately standard normal (Irwin–Hall, 12 uniforms; within ±6): only
    /// `+`, `-`, `*` and exact conversions, so no libm `ln`/`cos` that could
    /// differ by platform.
    pub fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.unit()).sum::<f64>() - 6.0
    }
}

/// The key the tests use, never the deployed one.
#[cfg(test)]
pub fn test_key() -> Key {
    Key::new(0x0001_0203_0405_0607)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scheme in the module docs, checked against values computed outside
    /// this code (Python `hashlib` for `K_cmd`, `openssl enc -aes-256-ecb` for
    /// the blocks).
    #[test]
    fn known_answers() {
        let mut s = Stream::new(&test_key(), "hotdog", 80351110224678912);
        assert_eq!([s.next(), s.next(), s.next()], KAT_HOTDOG);
        assert_eq!(text_input("a hotdog"), KAT_TEXT);
    }

    const KAT_HOTDOG: [u64; 3] = [11408088994561714817, 16430784477261156457, 13498045913007883646];
    const KAT_TEXT: u64 = 10799725900042007765;

    #[test]
    fn tags_and_inputs_are_independent() {
        let k = test_key();
        let first = |tag, x| Stream::new(&k, tag, x).next();
        assert_ne!(first("hotdog", 1), first("kobold-age", 1));
        assert_ne!(first("hotdog", 1), first("hotdog", 2));
        let other = Key::new(0xabab_abab_abab_abab);
        assert_ne!(first("hotdog", 1), Stream::new(&other, "hotdog", 1).next());
    }

    #[test]
    fn debug_hides_the_key() {
        assert_eq!(format!("{:?}", test_key()), "Key(…)");
    }
}
