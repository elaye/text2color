//! Colour labels for words, written by Claude Code subagents: the batches to label, validation
//! and merging of the returned files, and the distributions the labels stand for.

use crate::data::{Derived, Observed, RELIABLE_COUNT, canonical, load_translations, word_list};
use anyhow::{Context, Result, bail};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use text2color::color::{delta_e, rgb_to_oklab};
use text2color::readout::Readout;

pub const NONE: &str = "none";
pub const WORDS_PER_LANGUAGE: usize = 50_000;
const BATCH: usize = 500;
/// Answers a palette name needs for its survey histogram to stand for it.
const PALETTE_MIN_COUNT: u32 = 100;

pub struct Palette {
    names: Vec<String>,
    index: HashMap<String, usize>,
}

impl Palette {
    pub fn load(path: &Path) -> Result<Palette> {
        let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let names: Vec<String> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(str::to_string)
            .collect();
        let index = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        Ok(Palette { names, index })
    }
}

/// One sample's answer for a word: shares of palette names and of no consistent colour,
/// summing to 1, and the single colour given as hex.
#[derive(Debug, Clone)]
pub struct Label {
    pub word: String,
    pub shares: Vec<(usize, f32)>,
    pub none: f32,
    pub hex: Option<[u8; 3]>,
}

impl Label {
    fn shares_text(&self, palette: &Palette) -> String {
        let mut cells: Vec<String> = self
            .shares
            .iter()
            .map(|&(i, s)| format!("{}:{}", palette.names[i], (s * 100.0).round()))
            .collect();

        cells.push(format!("{NONE}:{}", (self.none * 100.0).round()));
        cells.join(",")
    }
}

fn parse_hex(field: &str) -> Result<Option<[u8; 3]>> {
    if field == "-" {
        return Ok(None);
    }

    let digits = field.strip_prefix('#').unwrap_or(field);

    if digits.len() != 6 {
        bail!("hex `{field}` is not #rrggbb or -");
    }

    let v = u32::from_str_radix(digits, 16).with_context(|| format!("hex `{field}`"))?;

    Ok(Some([(v >> 16) as u8, (v >> 8) as u8, v as u8]))
}

/// Parses `word<TAB>name:share,…,none:share<TAB>#rrggbb`. Shares may be off by a few points;
/// they are normalised.
pub fn parse_line(line: &str, palette: &Palette) -> Result<Label> {
    let fields: Vec<&str> = line.split('\t').collect();

    if fields.len() != 3 {
        bail!("expected 3 tab-separated fields, found {}", fields.len());
    }

    let mut shares: HashMap<usize, f32> = HashMap::new();
    let mut none = 0f32;

    for cell in fields[1].split(',') {
        let (name, share) = cell
            .rsplit_once(':')
            .with_context(|| format!("`{cell}` is not name:share"))?;
        let name = name.trim().to_lowercase();
        let share: f32 = share
            .trim()
            .parse()
            .with_context(|| format!("share in `{cell}`"))?;

        if share < 0.0 {
            bail!("negative share in `{cell}`");
        }

        if name == NONE {
            none += share;
            continue;
        }

        let Some(&i) = palette.index.get(&name) else {
            bail!("`{name}` is not in the palette");
        };

        *shares.entry(i).or_default() += share;
    }

    let total = none + shares.values().sum::<f32>();

    if !(95.0..=105.0).contains(&total) {
        bail!("shares add up to {total}, not 100");
    }

    let mut shares: Vec<(usize, f32)> = shares.into_iter().map(|(i, s)| (i, s / total)).collect();
    shares.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

    Ok(Label {
        word: fields[0].to_string(),
        shares,
        none: none / total,
        hex: parse_hex(fields[2].trim())?,
    })
}

/// Problems with a returned file, in a form an agent can act on. Empty when the file is valid.
pub fn validate(batch: &[String], text: &str, palette: &Palette) -> (Vec<Label>, Vec<String>) {
    let expected: HashSet<&str> = batch.iter().map(String::as_str).collect();
    let mut seen = HashSet::new();
    let mut labels = Vec::new();
    let mut errors = Vec::new();

    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        match parse_line(line, palette) {
            Ok(label) if !expected.contains(label.word.as_str()) => errors.push(format!(
                "line {}: `{}` is not in the batch (spell words exactly as given)",
                n + 1,
                label.word
            )),
            Ok(label) if !seen.insert(label.word.clone()) => {
                errors.push(format!("line {}: `{}` appears twice", n + 1, label.word))
            }
            Ok(label) => labels.push(label),
            Err(e) => errors.push(format!("line {}: {e:#}: {line}", n + 1)),
        }
    }

    let missing: Vec<&str> = batch
        .iter()
        .map(String::as_str)
        .filter(|w| !seen.contains(*w))
        .collect();

    if !missing.is_empty() {
        errors.push(format!(
            "{} words missing: {}",
            missing.len(),
            missing.join(", ")
        ));
    }

    (labels, errors)
}

fn read_words(path: &Path) -> Result<Vec<String>> {
    Ok(fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect())
}

fn write_batches(dir: &Path, words: &[String], size: usize) -> Result<()> {
    let batches = dir.join("batches");

    if batches.exists() {
        bail!(
            "{} exists; remove it to rebuild the batches",
            batches.display()
        );
    }

    fs::create_dir_all(&batches)?;

    for (i, chunk) in words.chunks(size).enumerate() {
        fs::write(batches.join(format!("{i:03}.txt")), chunk.join("\n") + "\n")?;
    }

    println!(
        "wrote {} batches of up to {size} words to {}",
        words.len().div_ceil(size),
        batches.display()
    );

    Ok(())
}

/// The most frequent words of each language, interleaved by rank so any prefix is bilingual.
/// Numbers and single letters are dropped; a word in both lists appears once.
pub fn vocabulary(raw: &Path) -> Result<Vec<(String, &'static str)>> {
    let english = word_list(&raw.join("english-50k.txt"), WORDS_PER_LANGUAGE)?;
    let french = word_list(&raw.join("french-50k.txt"), WORDS_PER_LANGUAGE)?;
    let mut seen = HashSet::new();
    let mut out = Vec::new();

    for i in 0..english.len().max(french.len()) {
        let pair = [(english.get(i), "en"), (french.get(i), "fr")];

        for (word, language) in pair {
            let Some(word) = word else {
                continue;
            };
            let usable = word.chars().count() > 1
                && word.chars().any(char::is_alphabetic)
                && !word.chars().any(|c| c.is_ascii_digit());

            if usable && seen.insert(word.clone()) {
                out.push((word.clone(), language));
            }
        }
    }

    Ok(out)
}

/// Batches for the whole vocabulary, in frequency order.
pub fn full_batches(raw: &Path, dir: &Path) -> Result<()> {
    let words: Vec<String> = vocabulary(raw)?.into_iter().map(|(w, _)| w).collect();

    write_batches(dir, &words, BATCH)
}

pub struct PilotArgs<'a> {
    pub raw: &'a Path,
    pub translations: &'a Path,
    pub probes: &'a Path,
    pub palette: &'a Palette,
    pub survey: usize,
    pub per_language: usize,
    pub batches: usize,
    pub seed: u64,
}

/// Pilot batches: the most answered survey names outside the palette, random vocabulary words
/// of each language and the probe words, shuffled together so no batch is all colour names.
/// `kinds.tsv` records where each word came from.
pub fn pilot_batches(d: &Derived, a: &PilotArgs, dir: &Path) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(a.seed);
    let translated: HashSet<String> = load_translations(a.translations)?
        .into_iter()
        .map(|(_, fr)| fr)
        .collect();
    let mut kinds: Vec<(String, String)> = Vec::new();
    let mut taken = HashSet::new();
    let mut take = |word: &str, kind: &str, kinds: &mut Vec<(String, String)>| {
        if taken.insert(word.to_string()) {
            kinds.push((word.to_string(), kind.to_string()));
        }
    };
    let mut survey: Vec<_> = d
        .strings
        .iter()
        .filter(|s| !a.palette.index.contains_key(&s.text) && !translated.contains(&s.text))
        .collect();

    survey.sort_by_key(|s| std::cmp::Reverse(s.count()));

    for s in survey.iter().take(a.survey) {
        take(&s.text, "survey", &mut kinds);
    }

    for word in read_words(a.probes)?.iter().filter_map(|w| canonical(w)) {
        take(&word, "probe", &mut kinds);
    }

    let mut vocabulary = vocabulary(a.raw)?;
    vocabulary.shuffle(&mut rng);

    for language in ["en", "fr"] {
        let words = vocabulary.iter().filter(|(_, l)| *l == language);

        for (word, _) in words.take(a.per_language) {
            take(word, language, &mut kinds);
        }
    }

    kinds.shuffle(&mut rng);
    fs::create_dir_all(dir)?;

    let words: Vec<String> = kinds.iter().map(|(w, _)| w.clone()).collect();

    write_batches(dir, &words, words.len().div_ceil(a.batches))?;

    let table: String = kinds.iter().map(|(w, k)| format!("{w}\t{k}\n")).collect();

    fs::write(dir.join("kinds.tsv"), table)?;
    println!("{} words: {}", kinds.len(), count_kinds(&kinds));

    Ok(())
}

fn count_kinds(kinds: &[(String, String)]) -> String {
    let mut counts: Vec<(String, usize)> = Vec::new();

    for (_, k) in kinds {
        match counts.iter_mut().find(|(c, _)| c == k) {
            Some((_, n)) => *n += 1,
            None => counts.push((k.clone(), 1)),
        }
    }

    counts
        .iter()
        .map(|(k, n)| format!("{k} {n}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn batch_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir.join("batches"))
        .with_context(|| format!("read {}/batches", dir.display()))?
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<_, _>>()?;

    files.sort();
    Ok(files)
}

/// Validates every returned sample of `model` in `dir` and merges the valid ones into `out`, one
/// line per word and sample. Lists the samples still missing or invalid.
pub fn check(dir: &Path, model: &str, samples: usize, palette: &Palette, out: &Path) -> Result<()> {
    let mut merged = String::new();
    let mut missing = Vec::new();
    let mut invalid = Vec::new();
    let mut done = 0;
    let batches = batch_files(dir)?;

    for batch in &batches {
        let words = read_words(batch)?;
        let stem = batch.file_stem().unwrap().to_string_lossy();

        for sample in 1..=samples {
            let path = dir.join(model).join(format!("{stem}.{sample}.tsv"));
            let Ok(text) = fs::read_to_string(&path) else {
                missing.push(path.display().to_string());
                continue;
            };
            let (labels, errors) = validate(&words, &text, palette);

            if !errors.is_empty() {
                invalid.push(format!("{} ({} problems)", path.display(), errors.len()));
                continue;
            }

            for l in labels {
                let hex = l.hex.map_or("-".to_string(), |c| {
                    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
                });

                writeln!(
                    merged,
                    "{}\t{sample}\t{}\t{hex}",
                    l.word,
                    l.shares_text(palette)
                )?;
            }

            done += 1;
        }
    }

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(out, merged)?;
    println!(
        "{model}: {done}/{} samples valid, merged into {}",
        batches.len() * samples,
        out.display()
    );

    for (what, list) in [("missing", &missing), ("invalid", &invalid)] {
        if !list.is_empty() {
            println!("{what}:\n  {}", list.join("\n  "));
        }
    }

    Ok(())
}

/// Merged labels by word, in file order, one entry per sample.
pub fn load_labels(path: &Path, palette: &Palette) -> Result<Vec<(String, Vec<Label>)>> {
    let mut out: Vec<(String, Vec<Label>)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;

    for line in text.lines() {
        let Some((word, rest)) = line.split_once('\t') else {
            continue;
        };
        let Some((_, rest)) = rest.split_once('\t') else {
            continue;
        };
        let label = parse_line(&format!("{word}\t{rest}"), palette)
            .with_context(|| format!("{}: {line}", path.display()))?;

        let i = *index.entry(label.word.clone()).or_insert_with(|| {
            out.push((label.word.clone(), Vec::new()));
            out.len() - 1
        });

        out[i].1.push(label);
    }

    Ok(out)
}

/// Turns labels into distributions over the colour bins: each palette name stands for the
/// answers the survey collected for it, and the `none` share for the answers to everything.
pub struct Encoder {
    names: Vec<Vec<f32>>,
    prior: Vec<f32>,
}

impl Encoder {
    pub fn new(strings: &[Observed], prior: &[f32], palette: &Palette) -> Result<Encoder> {
        let by_text: HashMap<&str, _> = strings.iter().map(|s| (s.text.as_str(), s)).collect();
        let mut names = Vec::with_capacity(palette.names.len());

        for name in &palette.names {
            let Some(s) = by_text.get(name.as_str()) else {
                bail!("palette name `{name}` has no survey answers");
            };

            if s.count() < PALETTE_MIN_COUNT {
                bail!("palette name `{name}` has only {} answers", s.count());
            }

            let mut dist = vec![0f32; prior.len()];

            for &(bin, n) in &s.hist {
                dist[bin as usize] = n as f32 / s.count() as f32;
            }

            names.push(dist);
        }

        Ok(Encoder {
            names,
            prior: prior.to_vec(),
        })
    }

    /// Mean distribution of the samples.
    pub fn distribution(&self, samples: &[Label]) -> Vec<f32> {
        let mut p = vec![0f32; self.prior.len()];
        let w = 1.0 / samples.len() as f32;

        for label in samples {
            let parts = label
                .shares
                .iter()
                .map(|&(i, s)| (&self.names[i], s))
                .chain([(&self.prior, label.none)]);

            for (dist, share) in parts {
                for (p, q) in p.iter_mut().zip(dist) {
                    *p += w * share * q;
                }
            }
        }

        p
    }
}

/// Moves `p` toward the prior by `mix` after sharpening it with `power`.
fn calibrate(p: &[f32], prior: &[f32], mix: f32, power: f32) -> Vec<f32> {
    let sharpened: Vec<f32> = p.iter().map(|v| v.max(1e-12).powf(power)).collect();
    let sum: f32 = sharpened.iter().sum();

    sharpened
        .iter()
        .zip(prior)
        .map(|(s, q)| (1.0 - mix) * s / sum + mix * q)
        .collect()
}

fn cross_entropy(answers: &[f32], p: &[f32]) -> f64 {
    answers
        .iter()
        .zip(p)
        .filter(|(a, _)| **a > 0.0)
        .map(|(a, p)| -(*a as f64) * (p.max(1e-12) as f64).ln())
        .sum()
}

fn quantile(values: &mut [f32], q: f32) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }

    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f32 * q).round() as usize]
}

fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len() as f32;
    let (mx, my) = (xs.iter().sum::<f32>() / n, ys.iter().sum::<f32>() / n);
    let cov: f32 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sx: f32 = xs.iter().map(|x| (x - mx).powi(2)).sum::<f32>().sqrt();
    let sy: f32 = ys.iter().map(|y| (y - my).powi(2)).sum::<f32>().sqrt();

    cov / (sx * sy)
}

fn hex(lab: [f32; 3]) -> String {
    let [r, g, b] = text2color::color::oklab_to_rgb(lab);

    format!("#{r:02x}{g:02x}{b:02x}")
}

struct Labelled {
    model: String,
    labels: HashMap<String, Vec<Label>>,
    mix: f32,
    power: f32,
}

impl Labelled {
    fn distribution(&self, enc: &Encoder, word: &str) -> Option<Vec<f32>> {
        self.calibrated(enc, word, self.mix, self.power)
    }

    fn calibrated(&self, enc: &Encoder, word: &str, mix: f32, power: f32) -> Option<Vec<f32>> {
        let samples = self.labels.get(word)?;

        Some(calibrate(
            &enc.distribution(samples),
            &enc.prior,
            mix,
            power,
        ))
    }
}

/// Mix toward the prior and sharpening power minimising the cross-entropy to the human answers
/// of the survey words.
fn fit_calibration(enc: &Encoder, pairs: &[(Vec<f32>, Vec<f32>)]) -> (f32, f32) {
    let mut best = (0.0, 1.0, f64::INFINITY);

    for power in (5..=20).map(|i| i as f32 / 10.0) {
        for mix in (0..=19).map(|i| i as f32 / 20.0) {
            let ce: f64 = pairs
                .iter()
                .map(|(answers, p)| cross_entropy(answers, &calibrate(p, &enc.prior, mix, power)))
                .sum();

            if ce < best.2 {
                best = (mix, power, ce);
            }
        }
    }

    (best.0, best.1)
}

/// Colour labels of words, calibrated against the survey, as used for training targets.
pub struct Labels {
    enc: Encoder,
    words: Vec<(String, Vec<Label>)>,
    index: HashMap<String, usize>,
    mix: f32,
    power: f32,
}

impl Labels {
    /// Loads merged label files, pooling the samples each gives a word, and fits their
    /// calibration on the survey strings with enough answers that were also labelled.
    pub fn load(
        paths: &[PathBuf],
        palette: &Palette,
        strings: &[Observed],
        prior: &[f32],
    ) -> Result<Labels> {
        let enc = Encoder::new(strings, prior, palette)?;
        let mut words: Vec<(String, Vec<Label>)> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();

        for path in paths {
            for (word, samples) in load_labels(path, palette)? {
                match index.get(&word) {
                    Some(&i) => words[i].1.extend(samples),
                    None => {
                        index.insert(word.clone(), words.len());
                        words.push((word, samples));
                    }
                }
            }
        }

        let samples: usize = words.iter().map(|(_, s)| s.len()).sum();
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = strings
            .iter()
            .filter(|s| s.count() >= RELIABLE_COUNT)
            .filter_map(|s| {
                let samples = &words[*index.get(&s.text)?].1;
                let answers = s
                    .hist
                    .iter()
                    .fold(vec![0f32; prior.len()], |mut a, &(bin, n)| {
                        a[bin as usize] = n as f32 / s.count() as f32;
                        a
                    });

                Some((answers, enc.distribution(samples)))
            })
            .collect();

        if pairs.is_empty() {
            bail!("no labelled word has survey answers to calibrate against");
        }

        let (mix, power) = fit_calibration(&enc, &pairs);

        println!(
            "labels: {} words, {samples} samples, calibration on {} survey strings: mix {mix:.2}, power {power:.1}",
            words.len(),
            pairs.len()
        );

        Ok(Labels {
            enc,
            words,
            index,
            mix,
            power,
        })
    }

    pub fn contains(&self, word: &str) -> bool {
        self.index.contains_key(word)
    }

    /// Labelled words in file order.
    pub fn words(&self) -> impl Iterator<Item = &str> {
        self.words.iter().map(|(w, _)| w.as_str())
    }

    /// Calibrated label distribution of `word`, if it was labelled.
    pub fn distribution(&self, word: &str) -> Option<Vec<f32>> {
        let samples = &self.words[*self.index.get(word)?].1;

        Some(calibrate(
            &self.enc.distribution(samples),
            &self.enc.prior,
            self.mix,
            self.power,
        ))
    }
}

/// Report on a labelled run: label colours and confidences against the human answers for the survey words,
/// hex values against palette names, agreement between models and samples, and a sheet of
/// words to check by eye.
pub fn report(
    d: &Derived,
    palette: &Palette,
    readout: &Readout,
    dir: &Path,
    label_files: &[(String, PathBuf)],
) -> Result<()> {
    let enc = Encoder::new(&d.strings, &d.prior, palette)?;
    let kinds: Vec<(String, String)> = fs::read_to_string(dir.join("kinds.tsv"))
        .context("read kinds.tsv")?
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(w, k)| (w.to_string(), k.to_string()))
        .collect();
    let human: HashMap<&str, Vec<f32>> = d
        .strings
        .iter()
        .map(|s| (s.text.as_str(), d.answers(&s.hist)))
        .collect();
    let survey: Vec<&str> = kinds
        .iter()
        .filter(|(_, k)| k == "survey")
        .map(|(w, _)| w.as_str())
        .collect();
    let mut models = Vec::new();

    for (model, path) in label_files {
        let labels: HashMap<String, Vec<Label>> = load_labels(path, palette)?.into_iter().collect();
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = survey
            .iter()
            .filter_map(|w| Some((human[w].clone(), enc.distribution(labels.get(*w)?))))
            .collect();
        let (mix, power) = if pairs.is_empty() {
            (0.0, 1.0)
        } else {
            fit_calibration(&enc, &pairs)
        };

        models.push(Labelled {
            model: model.clone(),
            labels,
            mix,
            power,
        });
    }

    if !survey.is_empty() {
        println!(
            "\nsurvey words: label readout vs the readout of the human answers ({} words)",
            survey.len()
        );
        println!(
            "  {:<8} {:>11} {:>6} {:>6} {:>8} {:>9} {:>6}   {:>8} {:>6}",
            "model", "calibration", "dE med", "p90", "conf r", "gate agr", "CE", "hex dE", "p90"
        );

        for m in &models {
            let settings = [
                (0.0, 1.0, "raw".to_string()),
                (m.mix, m.power, format!("{:.2}/{:.1}", m.mix, m.power)),
            ];

            for (mix, power, name) in settings {
                let mut de = Vec::new();
                let mut hex_de = Vec::new();
                let (mut conf_l, mut conf_h) = (Vec::new(), Vec::new());
                let mut gate = 0;
                let mut ce = 0f64;

                for w in &survey {
                    let Some(p) = m.calibrated(&enc, w, mix, power) else {
                        continue;
                    };
                    let (colour_h, c_h) = readout.read(&human[w]);
                    let (colour_l, c_l) = readout.read(&p);

                    de.push(delta_e(colour_h, colour_l));
                    conf_l.push(c_l);
                    conf_h.push(c_h);
                    gate += usize::from((c_l >= 0.5) == (c_h >= 0.5));
                    ce += cross_entropy(&human[w], &p);

                    let hexes: Vec<[f32; 3]> = m.labels[*w]
                        .iter()
                        .filter_map(|s| s.hex.map(rgb_to_oklab))
                        .collect();

                    if !hexes.is_empty() {
                        let mean = [0, 1, 2]
                            .map(|i| hexes.iter().map(|h| h[i]).sum::<f32>() / hexes.len() as f32);

                        hex_de.push(delta_e(colour_h, mean));
                    }
                }

                let n = de.len();

                println!(
                    "  {:<8} {:>11} {:>6.3} {:>6.3} {:>8.2} {:>8.0}% {:>6.2}   {:>8.3} {:>6.3}",
                    m.model,
                    name,
                    quantile(&mut de.clone(), 0.5),
                    quantile(&mut de, 0.9),
                    pearson(&conf_l, &conf_h),
                    100.0 * gate as f32 / n.max(1) as f32,
                    ce / n.max(1) as f64,
                    quantile(&mut hex_de.clone(), 0.5),
                    quantile(&mut hex_de, 0.9),
                );
            }
        }
    }

    println!(
        "\nshare of words at confidence >= 0.5 (calibrated), and samples disagreeing on the gate"
    );

    for m in &models {
        let mut line = format!("  {:<8}", m.model);

        for kind in ["survey", "en", "fr", "probe"] {
            let words: Vec<&str> = kinds
                .iter()
                .filter(|(_, k)| k == kind)
                .map(|(w, _)| w.as_str())
                .filter(|w| m.labels.contains_key(*w))
                .collect();
            let confident = words
                .iter()
                .filter(|w| readout.read(&m.distribution(&enc, w).unwrap()).1 >= 0.5)
                .count();
            let split = words
                .iter()
                .filter(|w| {
                    let gates: HashSet<bool> = m.labels[**w]
                        .iter()
                        .map(|s| readout.read(&enc.distribution(std::slice::from_ref(s))).1 >= 0.5)
                        .collect();

                    gates.len() > 1
                })
                .count();

            let _ = write!(
                line,
                "  {kind} {:>3.0}% (split {:>2.0}%)",
                100.0 * confident as f32 / words.len().max(1) as f32,
                100.0 * split as f32 / words.len().max(1) as f32
            );
        }

        println!("{line}");
    }

    if let [a, b] = models.as_slice() {
        let words: Vec<&str> = kinds
            .iter()
            .map(|(w, _)| w.as_str())
            .filter(|w| a.labels.contains_key(*w) && b.labels.contains_key(*w))
            .collect();
        let mut gate_differs = 0;
        let mut de = Vec::new();

        for w in &words {
            let (ca, pa) = readout.read(&a.distribution(&enc, w).unwrap());
            let (cb, pb) = readout.read(&b.distribution(&enc, w).unwrap());

            if (pa >= 0.5) != (pb >= 0.5) {
                gate_differs += 1;
            } else if pa >= 0.5 {
                de.push(delta_e(ca, cb));
            }
        }

        println!(
            "\n{} vs {} on {} words: gate differs {:.1}%, colour dE where both pass median {:.3} p90 {:.3}",
            a.model,
            b.model,
            words.len(),
            100.0 * gate_differs as f32 / words.len().max(1) as f32,
            quantile(&mut de.clone(), 0.5),
            quantile(&mut de, 0.9)
        );
    }

    let mut sheet = String::from("word\tkind");

    for m in &models {
        let _ = write!(sheet, "\t{0} hex\t{0} conf\t{0} label", m.model);
    }

    sheet.push('\n');

    let mut rng = StdRng::seed_from_u64(7);
    let mut picks: Vec<&(String, String)> = kinds.iter().filter(|(_, k)| k == "probe").collect();
    let mut others: Vec<&(String, String)> = kinds
        .iter()
        .filter(|(_, k)| k == "en" || k == "fr")
        .collect();

    others.shuffle(&mut rng);
    picks.extend(
        others
            .into_iter()
            .take(100usize.saturating_sub(picks.len())),
    );

    for (w, k) in picks {
        let _ = write!(sheet, "{w}\t{k}");

        for m in &models {
            match m.distribution(&enc, w) {
                Some(p) => {
                    let (colour, conf) = readout.read(&p);
                    let _ = write!(
                        sheet,
                        "\t{}\t{conf:.2}\t{}",
                        hex(colour),
                        m.labels[w][0].shares_text(palette)
                    );
                }
                None => sheet.push_str("\t\t\t"),
            }
        }

        sheet.push('\n');
    }

    fs::write(dir.join("sheet.tsv"), sheet)?;
    println!(
        "\nsheet of words to check by eye: {}",
        dir.join("sheet.tsv").display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn palette() -> Palette {
        let names: Vec<String> = ["blue", "sky blue", "white"].map(String::from).to_vec();
        let index = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        Palette { names, index }
    }

    #[test]
    fn parses_and_normalises_shares() {
        let l = parse_line("mer\tblue:50,Sky Blue:20,none:28\t#1f5fa8", &palette()).unwrap();
        let total: f32 = l.shares.iter().map(|s| s.1).sum::<f32>() + l.none;

        assert_eq!(l.shares[0].0, 0);
        assert!((total - 1.0).abs() < 1e-6);
        assert_eq!(l.hex, Some([0x1f, 0x5f, 0xa8]));
        assert!(
            parse_line("famille\tnone:100\t-", &palette())
                .unwrap()
                .hex
                .is_none()
        );
    }

    #[test]
    fn rejects_bad_lines() {
        let p = palette();

        assert!(parse_line("mer\tazure:100\t-", &p).is_err());
        assert!(parse_line("mer\tblue:50,none:20\t-", &p).is_err());
        assert!(parse_line("mer\tblue:100", &p).is_err());
        assert!(parse_line("mer\tblue:100\t#12", &p).is_err());
    }

    #[test]
    fn validation_reports_missing_and_foreign_words() {
        let batch = vec!["mer".to_string(), "neige".to_string()];
        let (labels, errors) = validate(&batch, "mer\tblue:100\t-\nsea\tblue:100\t-\n", &palette());

        assert_eq!(labels.len(), 1);
        assert_eq!(errors.len(), 2);
        assert!(errors[1].contains("neige"));
    }
}
