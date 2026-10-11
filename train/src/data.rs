//! Raw survey parsing, aggregation and the derived dataset files.

use crate::augment::base_forms;
use anyhow::{Context, Result, bail};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use text2color::color::rgb_to_oklab;
use text2color::text::normalize;

pub const MAX_TOKENS: usize = 5;
pub const MAX_CHARS: usize = 40;
const MIN_COUNT: u32 = 2;
const WORDS_PER_LANGUAGE: usize = 20_000;
const SPAM_THRESHOLD: f64 = 0.5;
/// RGB is binned into 2^CELL_BITS steps per channel to estimate how dominant a name is per cell.
const CELL_BITS: u32 = 4;
/// Exponent on the dominance weight; 1 is a soft centroid, large values approach the argmax cell.
const DOMINANCE_GAMMA: i32 = 4;
/// Survey strings with at least this many answers count as colour words whose inflections are
/// kept out of the negatives.
const COLOUR_WORD_COUNT: u32 = 20;

/// Weighted color mean plus unweighted spread of the answers for one string.
///
/// The mean of all swatches given a name is the centroid of the region where the name is used,
/// which for names at the gamut boundary ("black", "red") lies well inside the gamut. Weighting
/// each answer by a power of how dominant its name is in that swatch's cell moves the estimate
/// toward the color where the name is most probable, and reduces to the plain mean for strings
/// whose answers are spread one per cell.
#[derive(Default, Clone)]
struct Agg {
    n: u32,
    sum: [f64; 3],
    sumsq: [f64; 3],
    wsum: f64,
    wsum_lab: [f64; 3],
}

impl Agg {
    fn push(&mut self, lab: [f32; 3], weight: f64) {
        self.n += 1;
        self.wsum += weight;

        for (i, &v) in lab.iter().enumerate() {
            let v = v as f64;
            self.sum[i] += v;
            self.sumsq[i] += v * v;
            self.wsum_lab[i] += weight * v;
        }
    }

    fn merge(&mut self, other: &Agg) {
        self.n += other.n;
        self.wsum += other.wsum;

        for i in 0..3 {
            self.sum[i] += other.sum[i];
            self.sumsq[i] += other.sumsq[i];
            self.wsum_lab[i] += other.wsum_lab[i];
        }
    }

    fn mean(&self) -> [f64; 3] {
        self.wsum_lab.map(|s| s / self.wsum)
    }

    /// Mean squared distance from the unweighted mean, summed over the three axes.
    fn variance(&self) -> f64 {
        (0..3)
            .map(|i| {
                let m = self.sum[i] / self.n as f64;
                self.sumsq[i] / self.n as f64 - m * m
            })
            .sum::<f64>()
            .max(0.0)
    }
}

fn cell(rgb: [u8; 3]) -> usize {
    let shift = 8 - CELL_BITS;

    rgb.iter()
        .fold(0, |acc, &c| (acc << CELL_BITS) | (c >> shift) as usize)
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

/// One curated concept: its surface forms in both languages, the survey strings giving its
/// colour, and how broadly speakers agree on that colour.
struct Concept {
    forms: Vec<String>,
    colours: Vec<String>,
    agreement: f32,
}

fn split_list(field: &str) -> Vec<String> {
    field.split(',').filter_map(canonical).collect()
}

/// Reads `english forms<TAB>french forms<TAB>colours<TAB>agreement` lines.
fn load_concepts(path: &Path) -> Result<Vec<Concept>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = Vec::new();

    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let f: Vec<&str> = line.split('\t').collect();

        if f.len() != 4 {
            bail!("malformed association row: {line}");
        }

        let mut forms = split_list(f[0]);
        forms.extend(split_list(f[1]));

        let colours = split_list(f[2]);
        let agreement: f32 = f[3].trim().parse()?;

        if forms.is_empty() || colours.is_empty() || !(0.0..=1.0).contains(&agreement) {
            bail!("incomplete association row: {line}");
        }

        out.push(Concept {
            forms,
            colours,
            agreement,
        });
    }

    Ok(out)
}

/// Words of a frequency list, one per line, optionally followed by a count.
fn word_list(path: &Path, limit: usize) -> Result<Vec<String>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;

    Ok(text
        .lines()
        .filter_map(|l| l.split_whitespace().next())
        .filter_map(canonical)
        .take(limit)
        .collect())
}

/// Reads the raw dump, word lists, translations and associations, writes `positives.tsv`,
/// `associations.tsv` and `negatives.txt` into `derived`.
pub fn prepare(raw: &Path, translations: &Path, associations: &Path, derived: &Path) -> Result<()> {
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

    let mut cell_total = vec![0u32; 1 << (3 * CELL_BITS)];
    let mut name_cell: HashMap<(u32, u16), u32> = HashMap::new();

    for &(id, rgb) in &answers {
        let c = cell(rgb);
        cell_total[c] += 1;
        *name_cell.entry((id, c as u16)).or_default() += 1;
    }

    let mut aggs: Vec<Agg> = (0..names.len()).map(|_| Agg::default()).collect();
    let mut global = Agg::default();

    for &(id, rgb) in &answers {
        let c = cell(rgb);
        let weight =
            (name_cell[&(id, c as u16)] as f64 / cell_total[c] as f64).powi(DOMINANCE_GAMMA);
        let lab = rgb_to_oklab(rgb);

        global.push(lab, 1.0);
        aggs[id as usize].push(lab, weight);
    }

    let sigma_rand = global.variance().sqrt();
    println!(
        "answers kept {} dropped {} distinct strings {} sigma_rand {:.4}",
        global.n,
        dropped,
        names.len(),
        sigma_rand
    );

    // A translated string inherits the answers of its English source, on top of any answers
    // the survey already has for it.
    let mut translated = 0;

    for (en, fr) in load_translations(translations)? {
        let Some(&src) = ids.get(&en) else {
            continue;
        };
        let source = aggs[src as usize].clone();
        let dst = *ids.entry(fr).or_insert_with_key(|k| {
            names.push(k.clone());
            aggs.push(Agg::default());
            names.len() as u32 - 1
        });

        aggs[dst as usize].merge(&source);
        translated += 1;
    }

    println!("translations applied: {translated}");

    fs::create_dir_all(derived)?;

    let mut assoc_out = BufWriter::new(File::create(derived.join("associations.tsv"))?);
    let mut assoc_forms: HashSet<String> = HashSet::new();

    for concept in load_concepts(associations)? {
        let mut sum = [0f64; 3];

        for colour in &concept.colours {
            let agg = ids
                .get(colour)
                .map(|&id| &aggs[id as usize])
                .filter(|a| a.n >= MIN_COUNT)
                .with_context(|| format!("association colour {colour:?} is not a survey string"))?;
            let m = agg.mean();

            for i in 0..3 {
                sum[i] += m[i] / concept.colours.len() as f64;
            }
        }

        for form in &concept.forms {
            if !assoc_forms.insert(form.clone()) {
                bail!("association form {form:?} appears twice");
            }

            writeln!(
                assoc_out,
                "{form}\t{:.5}\t{:.5}\t{:.5}\t{}",
                sum[0], sum[1], sum[2], concept.agreement
            )?;
        }
    }

    println!("association forms: {}", assoc_forms.len());

    let mut rows: Vec<(&String, &Agg)> = names
        .iter()
        .zip(&aggs)
        .filter(|(name, a)| a.n >= MIN_COUNT && !assoc_forms.contains(*name))
        .collect();
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

    // Interleaved by rank so that any prefix of the list is bilingual.
    let english = word_list(&raw.join("google-20k.txt"), WORDS_PER_LANGUAGE)?;
    let french = word_list(&raw.join("french-50k.txt"), WORDS_PER_LANGUAGE)?;
    let colour_words: HashSet<&str> = names
        .iter()
        .zip(&aggs)
        .filter(|(_, a)| a.n >= COLOUR_WORD_COUNT)
        .map(|(name, _)| name.as_str())
        .chain(assoc_forms.iter().map(String::as_str))
        .collect();
    let is_colour_inflection = |word: &str| {
        base_forms(word)
            .iter()
            .any(|base| colour_words.contains(base.as_str()))
    };
    let mut negatives = BufWriter::new(File::create(derived.join("negatives.txt"))?);
    let mut seen = HashSet::new();

    for i in 0..english.len().max(french.len()) {
        for word in english.get(i).into_iter().chain(french.get(i)) {
            let excluded =
                ids.contains_key(word) || assoc_forms.contains(word) || is_colour_inflection(word);

            if !excluded && seen.insert(word.clone()) {
                writeln!(negatives, "{word}")?;
            }
        }
    }

    println!("negatives: {}", seen.len());

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

/// One surface form of a curated association.
#[derive(Debug, Clone)]
pub struct Association {
    pub text: String,
    pub oklab: [f32; 3],
    pub agreement: f32,
}

pub struct Derived {
    pub sigma_rand: f32,
    pub positives: Vec<Positive>,
    pub associations: Vec<Association>,
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

    let mut associations = Vec::new();
    let assoc_text = fs::read_to_string(derived.join("associations.tsv"))
        .context("read associations.tsv (run prepare)")?;

    for line in assoc_text.lines() {
        let f: Vec<&str> = line.split('\t').collect();

        if f.len() != 5 {
            bail!("malformed associations row: {line}");
        }

        associations.push(Association {
            text: f[0].to_string(),
            oklab: [f[1].parse()?, f[2].parse()?, f[3].parse()?],
            agreement: f[4].parse()?,
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
        associations,
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
    fn dominant_cells_pull_the_mean() {
        let mut agg = Agg::default();

        agg.push([0.2, 0.0, 0.0], 0.9);
        agg.push([0.6, 0.0, 0.0], 0.1);

        assert!((agg.mean()[0] - 0.24).abs() < 1e-6);
        assert!((agg.variance().sqrt() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn cell_index_covers_the_cube() {
        assert_eq!(cell([0, 0, 0]), 0);
        assert_eq!(cell([255, 255, 255]), (1 << (3 * CELL_BITS)) - 1);
        assert_eq!(cell([16, 0, 0]), 1 << (2 * CELL_BITS));
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
