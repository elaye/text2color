//! Data augmentation: typos and noise words. Shared by training and evaluation.

use rand::seq::IndexedRandom;
use rand::{Rng, RngExt};

const LETTERS: &[u8] = b"abcdefghijklmnopqrstuvwxyz";

/// Applies one random edit (delete, insert, substitute or transpose) to a token of at least
/// four characters. Returns the text unchanged if no token qualifies.
pub fn typo<R: Rng>(text: &str, rng: &mut R) -> String {
    let mut tokens: Vec<Vec<u8>> = text.split(' ').map(|t| t.as_bytes().to_vec()).collect();
    let candidates: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| t.len() >= 4 && t.is_ascii())
        .map(|(i, _)| i)
        .collect();

    let Some(&idx) = candidates.choose(rng) else {
        return text.to_string();
    };

    let t = &mut tokens[idx];
    let pos = rng.random_range(0..t.len());
    let letter = *LETTERS.choose(rng).unwrap();

    match rng.random_range(0..4) {
        0 => {
            t.remove(pos);
        }
        1 => t.insert(pos, letter),
        2 => t[pos] = letter,
        _ => {
            let other = if pos + 1 < t.len() { pos + 1 } else { pos - 1 };
            t.swap(pos, other);
        }
    }

    tokens
        .iter()
        .map(|t| String::from_utf8_lossy(t).into_owned())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Prepends or appends a noise word.
pub fn with_noise<R: Rng>(text: &str, noise: &str, rng: &mut R) -> String {
    if rng.random_bool(0.5) {
        format!("{noise} {text}")
    } else {
        format!("{text} {noise}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn typo_changes_long_tokens_only() {
        let mut rng = StdRng::seed_from_u64(1);

        assert_eq!(typo("red", &mut rng), "red");

        let changed = (0..20)
            .filter(|_| typo("purple", &mut rng) != "purple")
            .count();

        assert!(changed >= 15);
    }
}
