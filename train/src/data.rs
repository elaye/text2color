//! Raw survey parsing, colour-bin histograms and the derived dataset files.

use crate::bins::{colour_bins, nearest};
use crate::labels::{Labels, Palette};
use anyhow::{Context, Result, bail};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use text2color::color::rgb_to_oklab;
use text2color::text::normalize;

pub const MAX_TOKENS: usize = 5;
pub const MAX_CHARS: usize = 40;
const MIN_COUNT: u32 = 2;
const SPAM_THRESHOLD: f64 = 0.5;
/// Answers needed before a string's own histogram is a usable estimate of human agreement.
pub const RELIABLE_COUNT: u32 = 30;
/// Pseudo-answers a word's colour label counts for in its target.
pub const LABEL_STRENGTH: f32 = 20.0;

/// Bin counts of the answers given for one string, sparse and sorted by bin.
pub type Histogram = Vec<(u16, u32)>;

fn merge(a: &Histogram, b: &Histogram) -> Histogram {
    let mut counts: HashMap<u16, u32> = a.iter().copied().collect();

    for &(bin, n) in b {
        *counts.entry(bin).or_default() += n;
    }

    let mut out: Histogram = counts.into_iter().collect();
    out.sort_unstable();
    out
}

fn total(h: &Histogram) -> u32 {
    h.iter().map(|&(_, n)| n).sum()
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

/// Reads `english<TAB>french` pairs. Lines starting with `#` and pairs with an empty or
/// unusable French side are skipped.
pub fn load_translations(path: &Path) -> Result<Vec<(String, String)>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = Vec::new();

    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let Some((en, fr)) = line.split_once('\t') else {
            continue;
        };
        let (Some(en), Some(fr)) = (canonical(en), canonical(fr)) else {
            continue;
        };

        if en != fr {
            out.push((en, fr));
        }
    }

    Ok(out)
}

/// Words of a frequency list, one per line, optionally followed by a count.
pub fn word_list(path: &Path, limit: usize) -> Result<Vec<String>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;

    Ok(text
        .lines()
        .filter_map(|l| l.split_whitespace().next())
        .filter_map(canonical)
        .take(limit)
        .collect())
}

/// Reads the raw dump and translations, writes `bins.tsv` and `strings.tsv` into `derived`.
pub fn prepare(raw: &Path, translations: &Path, bins: usize, derived: &Path) -> Result<()> {
    let dump = File::open(raw.join("mainsurvey_sqldump.txt")).context("open survey dump")?;
    let mut users: HashMap<u32, bool> = HashMap::new();
    let mut ids: HashMap<String, u32> = HashMap::new();
    let mut names: Vec<String> = Vec::new();
    let mut answers: Vec<(u32, [u8; 3])> = Vec::new();
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
                let id = *ids.entry(key).or_insert_with_key(|k| {
                    names.push(k.clone());
                    names.len() as u32 - 1
                });

                answers.push((id, rgb));
            }
            _ => {}
        }
    }

    println!(
        "answers kept {} dropped {dropped} distinct strings {}",
        answers.len(),
        names.len()
    );

    let centroids = colour_bins(bins);
    let mut bin_of: HashMap<[u8; 3], u16> = HashMap::new();
    let mut pairs: Vec<(u32, u16)> = Vec::with_capacity(answers.len());
    let mut prior = vec![0u64; bins];

    for &(id, rgb) in &answers {
        let bin = *bin_of
            .entry(rgb)
            .or_insert_with(|| nearest(&centroids, rgb_to_oklab(rgb)) as u16);

        prior[bin as usize] += 1;
        pairs.push((id, bin));
    }

    pairs.sort_unstable();

    let mut hists: Vec<Histogram> = vec![Vec::new(); names.len()];

    for chunk in pairs.chunk_by(|a, b| a == b) {
        let (id, bin) = chunk[0];
        hists[id as usize].push((bin, chunk.len() as u32));
    }

    // A translated string inherits the answers of its English source, on top of any answers
    // the survey already has for it.
    let mut translated = 0;

    for (en, fr) in load_translations(translations)? {
        let Some(&src) = ids.get(&en) else {
            continue;
        };
        let source = hists[src as usize].clone();
        let dst = *ids.entry(fr).or_insert_with_key(|k| {
            names.push(k.clone());
            hists.push(Vec::new());
            names.len() as u32 - 1
        });

        hists[dst as usize] = merge(&hists[dst as usize], &source);
        translated += 1;
    }

    println!("translations applied: {translated}");

    fs::create_dir_all(derived)?;

    let mut out = BufWriter::new(File::create(derived.join("bins.tsv"))?);
    let prior_total: u64 = prior.iter().sum();

    for (c, n) in centroids.iter().zip(&prior) {
        writeln!(
            out,
            "{:.6}\t{:.6}\t{:.6}\t{:.8}",
            c[0],
            c[1],
            c[2],
            *n as f64 / prior_total as f64
        )?;
    }

    let mut rows: Vec<(&String, &Histogram, u32)> = names
        .iter()
        .zip(&hists)
        .map(|(name, h)| (name, h, total(h)))
        .filter(|r| r.2 >= MIN_COUNT)
        .collect();
    rows.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(b.0)));

    let mut out = BufWriter::new(File::create(derived.join("strings.tsv"))?);

    for (name, hist, n) in &rows {
        let cells: Vec<String> = hist.iter().map(|(b, c)| format!("{b}:{c}")).collect();
        writeln!(out, "{name}\t{n}\t{}", cells.join(" "))?;
    }

    println!("strings (n >= {MIN_COUNT}): {}", rows.len());

    Ok(())
}

/// A string with the answers observed for it, possibly none.
#[derive(Debug, Clone)]
pub struct Observed {
    pub text: String,
    pub hist: Histogram,
}

impl Observed {
    pub fn count(&self) -> u32 {
        total(&self.hist)
    }
}

pub struct Derived {
    pub centroids: Vec<[f32; 3]>,
    /// Share of all survey answers in each bin.
    pub prior: Vec<f32>,
    /// Survey strings with their answers.
    pub strings: Vec<Observed>,
    /// Labelled words without survey answers, most frequent first.
    pub vocabulary: Vec<String>,
    pub labels: Option<Labels>,
}

impl Derived {
    /// The answers in `hist` as a distribution.
    pub fn answers(&self, hist: &[(u16, u32)]) -> Vec<f32> {
        self.target("", hist, 0.0)
    }

    /// Pseudo-answers behind the target of `text` besides its survey answers.
    pub fn pseudo_count(&self, text: &str, alpha: f32) -> f32 {
        let labelled = self.labels.as_ref().is_some_and(|l| l.contains(text));

        alpha + if labelled { LABEL_STRENGTH } else { 0.0 }
    }

    /// Target distribution for `text`: its survey answers, plus [`LABEL_STRENGTH`] pseudo-answers
    /// spread like its calibrated label if it has one, plus `alpha` pseudo-answers spread like
    /// the prior, normalised.
    pub fn target(&self, text: &str, hist: &[(u16, u32)], alpha: f32) -> Vec<f32> {
        let mut t: Vec<f32> = self.prior.iter().map(|p| alpha * p).collect();

        if let Some(label) = self.labels.as_ref().and_then(|l| l.distribution(text)) {
            for (t, q) in t.iter_mut().zip(label) {
                *t += LABEL_STRENGTH * q;
            }
        }

        for &(bin, n) in hist {
            t[bin as usize] += n as f32;
        }

        let sum: f32 = t.iter().sum();
        t.iter_mut().for_each(|v| *v /= sum);
        t
    }
}

fn parse_histogram(field: &str) -> Result<Histogram> {
    field
        .split_whitespace()
        .map(|cell| {
            let (b, n) = cell.split_once(':').context("bin:count")?;

            Ok((b.parse()?, n.parse()?))
        })
        .collect()
}

/// Loads the prepared survey data and, if `labels` is given, the colour labels of words, whose
/// calibration is fitted on the survey strings they share.
pub fn load_derived(derived: &Path, labels: Option<(&[PathBuf], &Palette)>) -> Result<Derived> {
    let mut centroids = Vec::new();
    let mut prior = Vec::new();

    for line in fs::read_to_string(derived.join("bins.tsv"))
        .context("read bins.tsv (run prepare)")?
        .lines()
    {
        let f: Vec<f32> = line.split('\t').map(str::parse).collect::<Result<_, _>>()?;

        if f.len() != 4 {
            bail!("malformed bins row: {line}");
        }

        centroids.push([f[0], f[1], f[2]]);
        prior.push(f[3]);
    }

    let mut strings = Vec::new();

    for line in fs::read_to_string(derived.join("strings.tsv"))
        .context("read strings.tsv (run prepare)")?
        .lines()
    {
        let f: Vec<&str> = line.split('\t').collect();

        if f.len() != 3 {
            bail!("malformed strings row: {line}");
        }

        strings.push(Observed {
            text: f[0].to_string(),
            hist: parse_histogram(f[2])?,
        });
    }

    let labels = labels
        .map(|(paths, palette)| Labels::load(paths, palette, &strings, &prior))
        .transpose()?;
    let answered: HashSet<&str> = strings.iter().map(|s| s.text.as_str()).collect();
    let vocabulary = labels.as_ref().map_or_else(Vec::new, |l| {
        l.words()
            .filter(|w| !answered.contains(w))
            .map(str::to_string)
            .collect()
    });

    Ok(Derived {
        centroids,
        prior,
        strings,
        vocabulary,
        labels,
    })
}

fn split_hash(text: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;

    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }

    h
}

/// Deterministic 10 % hold-out split keyed on the text.
pub fn is_holdout(text: &str) -> bool {
    split_hash(text).is_multiple_of(10)
}

/// Half of the hold-out set, used to fit the temperature; evaluation uses the other half.
pub fn is_calibration(text: &str) -> bool {
    split_hash(text).is_multiple_of(20)
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
    fn histograms_merge_by_bin() {
        let merged = merge(&vec![(1, 2), (5, 1)], &vec![(5, 3), (7, 1)]);

        assert_eq!(merged, vec![(1, 2), (5, 4), (7, 1)]);
        assert_eq!(total(&merged), 7);
        assert_eq!(parse_histogram("1:2 5:4 7:1").unwrap(), merged);
    }

    #[test]
    fn target_shrinks_toward_the_prior() {
        let d = Derived {
            centroids: vec![[0.0; 3]; 2],
            prior: vec![0.5, 0.5],
            strings: Vec::new(),
            vocabulary: Vec::new(),
            labels: None,
        };

        assert_eq!(d.target("", &[], 2.0), vec![0.5, 0.5]);
        assert_eq!(d.target("", &[(0, 8)], 2.0), vec![0.9, 0.1]);
    }
}
