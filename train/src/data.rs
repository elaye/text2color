//! Raw survey parsing, aggregation and the derived dataset files.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use text2color::color::rgb_to_oklab;
use text2color::text::normalize;

pub const MAX_TOKENS: usize = 5;
pub const MAX_CHARS: usize = 40;
const MIN_COUNT: u32 = 2;
const SPAM_THRESHOLD: f64 = 0.5;

#[derive(Default)]
struct Agg {
    n: u32,
    sum: [f64; 3],
    sumsq: [f64; 3],
}

impl Agg {
    fn push(&mut self, lab: [f32; 3]) {
        self.n += 1;

        for ((sum, sumsq), v) in self.sum.iter_mut().zip(self.sumsq.iter_mut()).zip(lab) {
            *sum += v as f64;
            *sumsq += (v as f64).powi(2);
        }
    }

    fn mean(&self) -> [f64; 3] {
        self.sum.map(|s| s / self.n as f64)
    }

    /// Mean squared distance from the mean, summed over the three axes.
    fn variance(&self) -> f64 {
        let m = self.mean();

        (0..3)
            .map(|i| self.sumsq[i] / self.n as f64 - m[i] * m[i])
            .sum::<f64>()
            .max(0.0)
    }
}

/// Splits `INSERT INTO "t" VALUES(...)` into table name and the raw value list.
fn insert_parts(line: &str) -> Option<(&str, &str)> {
    let rest = line.strip_prefix("INSERT INTO \"")?;
    let (table, rest) = rest.split_once("\" VALUES(")?;
    let values = rest.strip_suffix(");")?;

    Some((table, values))
}

/// Parses an SQL string literal that is the last field of the value list.
fn trailing_string(values: &str) -> Option<String> {
    let start = values.find('\'')?;
    let body = values[start + 1..].strip_suffix('\'')?;

    Some(body.replace("''", "'"))
}

fn user_is_usable(values: &str) -> Option<(u32, bool)> {
    let fields: Vec<&str> = values.split(',').collect();
    let id: u32 = fields.first()?.parse().ok()?;
    let colorblind = fields.get(8).map(|v| *v == "1").unwrap_or(false);
    let spam: f64 = fields.last()?.parse().unwrap_or(0.0);

    Some((id, !colorblind && spam < SPAM_THRESHOLD))
}

fn answer(values: &str) -> Option<(u32, [u8; 3], String)> {
    let mut it = values.splitn(7, ',');
    let _id = it.next()?;
    let user: u32 = it.next()?.parse().ok()?;
    let _ts = it.next()?;
    let r: u8 = it.next()?.parse().ok()?;
    let g: u8 = it.next()?.parse().ok()?;
    let b: u8 = it.next()?.parse().ok()?;
    let name = trailing_string(it.next()?)?;

    Some((user, [r, g, b], name))
}

/// Canonical form of a survey answer or query string, or `None` if it is unusable.
pub fn canonical(text: &str) -> Option<String> {
    let tokens = normalize(text);

    if tokens.is_empty() || tokens.len() > MAX_TOKENS {
        return None;
    }

    let joined = tokens.join(" ");

    (joined.chars().count() <= MAX_CHARS).then_some(joined)
}

/// Reads the raw dump and word list, writes `positives.tsv` and `negatives.txt` into `derived`.
pub fn prepare(raw: &Path, derived: &Path) -> Result<()> {
    let dump = File::open(raw.join("mainsurvey_sqldump.txt")).context("open survey dump")?;
    let mut users: HashMap<u32, bool> = HashMap::new();
    let mut aggs: HashMap<String, Agg> = HashMap::new();
    let mut global = Agg::default();
    let mut dropped = 0u64;

    for line in BufReader::new(dump).lines() {
        let line = line?;
        let Some((table, values)) = insert_parts(&line) else {
            continue;
        };

        match table {
            "users" => {
                if let Some((id, ok)) = user_is_usable(values) {
                    users.insert(id, ok);
                }
            }
            "answers" => {
                let Some((user, rgb, name)) = answer(values) else {
                    continue;
                };

                if !users.get(&user).copied().unwrap_or(false) {
                    dropped += 1;
                    continue;
                }

                let Some(key) = canonical(&name) else {
                    continue;
                };
                let lab = rgb_to_oklab(rgb);

                global.push(lab);
                aggs.entry(key).or_default().push(lab);
            }
            _ => {}
        }
    }

    let sigma_rand = global.variance().sqrt();
    println!(
        "answers kept {} dropped {} distinct strings {} sigma_rand {:.4}",
        global.n,
        dropped,
        aggs.len(),
        sigma_rand
    );

    fs::create_dir_all(derived)?;

    let mut rows: Vec<(&String, &Agg)> = aggs.iter().filter(|(_, a)| a.n >= MIN_COUNT).collect();
    rows.sort_by(|a, b| b.1.n.cmp(&a.1.n).then_with(|| a.0.cmp(b.0)));

    let mut out = BufWriter::new(File::create(derived.join("positives.tsv"))?);
    writeln!(out, "# sigma_rand={sigma_rand}")?;

    for (name, agg) in &rows {
        let m = agg.mean();
        writeln!(
            out,
            "{}\t{}\t{:.5}\t{:.5}\t{:.5}\t{:.5}",
            name,
            agg.n,
            m[0],
            m[1],
            m[2],
            agg.variance().sqrt()
        )?;
    }

    println!("positives (n >= {MIN_COUNT}): {}", rows.len());

    let words = fs::read_to_string(raw.join("google-20k.txt")).context("open word list")?;
    let mut negatives = BufWriter::new(File::create(derived.join("negatives.txt"))?);
    let mut kept = 0;

    for word in words.lines() {
        let Some(key) = canonical(word) else { continue };

        if !aggs.contains_key(&key) {
            writeln!(negatives, "{key}")?;
            kept += 1;
        }
    }

    println!("negatives: {kept}");

    Ok(())
}

/// One aggregated survey string.
#[derive(Debug, Clone)]
pub struct Positive {
    pub text: String,
    pub count: u32,
    pub oklab: [f32; 3],
    pub sigma: f32,
}

pub struct Derived {
    pub sigma_rand: f32,
    pub positives: Vec<Positive>,
    pub negatives: Vec<String>,
}

pub fn load_derived(derived: &Path) -> Result<Derived> {
    let text = fs::read_to_string(derived.join("positives.tsv"))
        .context("read positives.tsv (run prepare)")?;
    let mut lines = text.lines();
    let header = lines.next().unwrap_or_default();

    let Some(sigma_rand) = header
        .strip_prefix("# sigma_rand=")
        .and_then(|v| v.parse().ok())
    else {
        bail!("positives.tsv is missing the sigma_rand header");
    };

    let mut positives = Vec::new();

    for line in lines {
        let f: Vec<&str> = line.split('\t').collect();

        if f.len() != 6 {
            bail!("malformed positives row: {line}");
        }

        positives.push(Positive {
            text: f[0].to_string(),
            count: f[1].parse()?,
            oklab: [f[2].parse()?, f[3].parse()?, f[4].parse()?],
            sigma: f[5].parse()?,
        });
    }

    let negatives: Vec<String> = fs::read_to_string(derived.join("negatives.txt"))
        .context("read negatives.txt")?
        .lines()
        .map(str::to_string)
        .collect();

    Ok(Derived {
        sigma_rand,
        positives,
        negatives,
    })
}

/// Deterministic 10 % hold-out split keyed on the text.
pub fn is_holdout(text: &str) -> bool {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;

    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }

    h.is_multiple_of(10)
}

/// Maps answer spread and count to the confidence target in `0..=1`.
///
/// Spread is shrunk toward the random-color spread with `prior` pseudo-answers, then a logistic
/// centred at ratio `r0` with width `tau` is applied, plus `kappa` per decade of answer count.
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceMap {
    pub prior: f32,
    pub r0: f32,
    pub tau: f32,
    pub kappa: f32,
}

impl Default for ConfidenceMap {
    fn default() -> Self {
        ConfidenceMap {
            prior: 2.0,
            r0: 0.80,
            tau: 0.04,
            kappa: 0.3,
        }
    }
}

impl ConfidenceMap {
    pub fn target(&self, sigma: f32, n: u32, sigma_rand: f32) -> f32 {
        let var_post = (n as f32 * sigma * sigma + self.prior * sigma_rand * sigma_rand)
            / (n as f32 + self.prior);
        let ratio = var_post.sqrt() / sigma_rand;
        let z = (self.r0 - ratio) / self.tau + self.kappa * (n as f32 / 100.0).log10();

        1.0 / (1.0 + (-z).exp())
    }
}

/// The 949 xkcd names with their reference colors, for evaluation.
pub fn load_xkcd(raw: &Path) -> Result<Vec<(String, [u8; 3])>> {
    let text = fs::read_to_string(raw.join("xkcd-rgb.txt")).context("read xkcd-rgb.txt")?;
    let mut out = Vec::new();

    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }

        let mut it = line.split('\t');
        let (Some(name), Some(hex)) = (it.next(), it.next()) else {
            continue;
        };
        let hex = hex.trim_start_matches('#');

        if hex.len() != 6 {
            continue;
        }

        let v = u32::from_str_radix(hex, 16)?;
        out.push((name.to_string(), [(v >> 16) as u8, (v >> 8) as u8, v as u8]));
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_answer_with_escaped_quote() {
        let (table, values) = insert_parts(
            r#"INSERT INTO "answers" VALUES(1,1,1267418734.0,72,100,175,'it''s blue');"#,
        )
        .unwrap();

        assert_eq!(table, "answers");
        assert_eq!(
            answer(values).unwrap(),
            (1, [72, 100, 175], "it's blue".to_string())
        );
    }

    #[test]
    fn parses_user_flags() {
        let (_, values) = insert_parts(r#"INSERT INTO "users" VALUES(7,'1dbq',1267419070.0,'0.0.0.0',NULL,'CRT',9300,NULL,0,1,'yellow/bright mustard',0.024);"#).unwrap();

        assert_eq!(user_is_usable(values), Some((7, true)));

        let (_, spam) = insert_parts(r#"INSERT INTO "users" VALUES(8,'x',1.0,'0.0.0.0',NULL,'LCD',NULL,NULL,1,1,'gray/charcoal',0.9);"#).unwrap();

        assert_eq!(user_is_usable(spam), Some((8, false)));
    }

    #[test]
    fn confidence_target_shape() {
        let map = ConfidenceMap::default();
        let basic = map.target(0.168, 270_000, 0.234);
        let tight = map.target(0.09, 6000, 0.234);
        let junk = map.target(0.20, 150, 0.234);
        let rare = map.target(0.0, 2, 0.234);

        assert!(basic > 0.9, "{basic}");
        assert!(tight > 0.99, "{tight}");
        assert!(junk < 0.3, "{junk}");
        assert!(rare > 0.5 && rare < 0.95, "{rare}");
    }
}
