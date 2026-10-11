//! Text normalisation and hashed feature extraction.
//!
//! Shared by the library and the trainer so both see identical features.

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

const SALT_TOKEN: u8 = 1;
const SALT_BIGRAM: u8 = 7;
const MIN_NGRAM: usize = 2;
const MAX_NGRAM: usize = 5;

/// Lower-cases, maps every non-alphanumeric character to a space and splits.
pub fn normalize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_alphanumeric() {
            current.extend(ch.to_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

fn hash(salt: u8, bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET ^ salt as u64;
    h = h.wrapping_mul(FNV_PRIME);

    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }

    h
}

/// Hashed feature ids for `text`, each in `0..buckets` (`buckets` must be a power of two).
///
/// Features: whole token, character n-grams of `<token>` (n = 2..=5) and token bigrams.
pub fn features(text: &str, buckets: u32) -> Vec<u32> {
    let mask = (buckets - 1) as u64;
    let tokens = normalize(text);
    let mut out = Vec::with_capacity(tokens.len() * 24);

    for token in &tokens {
        out.push((hash(SALT_TOKEN, token.as_bytes()) & mask) as u32);

        let marked: Vec<char> = std::iter::once('<')
            .chain(token.chars())
            .chain(std::iter::once('>'))
            .collect();
        let mut buf = String::new();

        for n in MIN_NGRAM..=MAX_NGRAM {
            if marked.len() < n {
                break;
            }

            for window in marked.windows(n) {
                buf.clear();
                buf.extend(window);
                out.push((hash(n as u8 + 8, buf.as_bytes()) & mask) as u32);
            }
        }
    }

    for pair in tokens.windows(2) {
        let joined = format!("{} {}", pair[0], pair[1]);
        out.push((hash(SALT_BIGRAM, joined.as_bytes()) & mask) as u32);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn normalizes_case_and_punctuation() {
        assert_eq!(
            normalize("  Blue-Green, ish!"),
            vec!["blue", "green", "ish"]
        );
        assert!(normalize("...").is_empty());
    }

    #[test]
    fn deterministic_and_in_range() {
        let a = features("dark teal", 1 << 10);
        let b = features("dark teal", 1 << 10);

        assert_eq!(a, b);
        assert!(a.iter().all(|&f| f < 1024));
        assert!(!a.is_empty());
    }

    #[test]
    fn typo_shares_most_features() {
        let overlap = |a: &str, b: &str| {
            let fa: HashSet<u32> = features(a, 1 << 20).into_iter().collect();
            let fb: HashSet<u32> = features(b, 1 << 20).into_iter().collect();

            fa.intersection(&fb).count() as f32 / fa.len() as f32
        };

        assert!(overlap("green", "gren") > 0.4);
        assert!(overlap("purple", "purpel") > 0.4);
        assert!(overlap("blue", "bleu") > 0.15);
        assert!(overlap("blue", "table") < 0.2);
    }

    #[test]
    fn empty_text_has_no_features() {
        assert!(features("", 1 << 10).is_empty());
    }
}
