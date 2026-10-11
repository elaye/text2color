//! Evaluation of an exported model through the library's inference path.

use crate::augment::{strip_accents, typo};
use crate::data::{
    Derived, RELIABLE_COUNT, is_calibration, is_holdout, load_translations, load_xkcd,
};
use crate::train::auc;
use anyhow::{Context, Result};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use text2color::Model;
use text2color::color::{delta_e, rgb_to_oklab};
use text2color::readout::Readout;

/// Radii compared in the agreement-radius sweep.
const RADII: [f32; 6] = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2];
const BASIC_TERMS: &[&str] = &[
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white",
    "grey", "rouge", "jaune", "vert", "bleu", "violet", "rose", "marron", "noir", "blanc", "gris",
];

fn load_probes(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;

    Ok(text.lines().map(str::to_string).collect())
}

pub fn probes(model: &Model, path: &Path) -> Result<()> {
    let texts = load_probes(path)?;
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    for (p, r) in refs.iter().zip(model.predict_many(&refs)) {
        println!(
            "  {:<20} {}  rgb({:>3},{:>3},{:>3})  conf {:.2}",
            p,
            r.hex(),
            r.rgb[0],
            r.rgb[1],
            r.rgb[2],
            r.confidence
        );
    }

    Ok(())
}

fn distributions(model: &Model, texts: &[&str]) -> Vec<Vec<f32>> {
    texts
        .chunks(512)
        .flat_map(|chunk| model.distributions(chunk))
        .map(|p| p.unwrap_or_default())
        .collect()
}

fn normalised(d: &Derived, hist: &crate::data::Histogram) -> Vec<f32> {
    d.target(hist, 0.0)
}

/// Predicted confidence against the agreement measured from the answers themselves, on
/// hold-out strings outside the calibration fold with enough answers, in deciles of predicted
/// confidence.
fn reliability_report(model: &Model, d: &Derived) {
    let rows: Vec<_> = d
        .strings
        .iter()
        .filter(|s| is_holdout(&s.text) && !is_calibration(&s.text) && s.count() >= RELIABLE_COUNT)
        .collect();
    let texts: Vec<&str> = rows.iter().map(|s| s.text.as_str()).collect();
    let mut pairs: Vec<(f32, f32)> = rows
        .iter()
        .zip(model.predict_many(&texts))
        .map(|(s, p)| {
            (
                p.confidence,
                model.readout().read(&normalised(d, &s.hist)).1,
            )
        })
        .collect();

    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    println!(
        "reliability on {} hold-out strings with >= {RELIABLE_COUNT} answers (predicted vs measured agreement):",
        pairs.len()
    );

    for decile in pairs.chunks(pairs.len().div_ceil(10).max(1)) {
        let n = decile.len() as f32;
        let predicted: f32 = decile.iter().map(|p| p.0).sum::<f32>() / n;
        let measured: f32 = decile.iter().map(|p| p.1).sum::<f32>() / n;

        println!(
            "  predicted {predicted:.2}  measured {measured:.2}  ({} strings)",
            decile.len()
        );
    }
}

/// Confidence of basic colour terms, xkcd names and unanswered words for several radii.
fn radius_sweep(model: &Model, d: &Derived, xkcd: &[(String, [u8; 3])]) {
    let vocab: Vec<&str> = d
        .vocabulary
        .iter()
        .filter(|w| is_holdout(w))
        .map(String::as_str)
        .collect();
    let names: Vec<&str> = xkcd.iter().map(|(n, _)| n.as_str()).collect();
    let (basic, names, vocab) = (
        distributions(model, BASIC_TERMS),
        distributions(model, &names),
        distributions(model, &vocab),
    );
    let centroids = model.readout().centroids().to_vec();
    let share = |r: &Readout, ps: &[Vec<f32>]| {
        ps.iter().filter(|p| r.read(p).1 >= 0.5).count() as f32 / ps.len() as f32
    };

    println!("radius sweep (model trained at its own radius; confidence re-read at each radius):");

    for radius in RADII {
        let r = Readout::new(centroids.clone(), radius, model.config().colour_radius);
        let mean_basic: f32 = basic.iter().map(|p| r.read(p).1).sum::<f32>() / basic.len() as f32;

        println!(
            "  r {radius:.3}: basic terms mean {mean_basic:.2}, xkcd names >= 0.5: {:.0}%, unanswered words >= 0.5: {:.1}%",
            100.0 * share(&r, &names),
            100.0 * share(&r, &vocab)
        );
    }
}

/// Splits a concatenated CSS name ("darkslategray") into the most probable sequence of common
/// survey words under a unigram model. Names that cannot be split are kept whole.
fn segment(name: &str, words: &HashMap<&str, f64>) -> String {
    let n = name.len();
    let mut best: Vec<Option<(f64, usize)>> = vec![None; n + 1];
    best[0] = Some((0.0, 0));

    for end in 1..=n {
        for start in 0..end {
            let (Some((score, _)), Some(&logp)) = (best[start], words.get(&name[start..end]))
            else {
                continue;
            };
            let score = score + logp;

            if best[end].is_none_or(|(s, _)| score > s) {
                best[end] = Some((score, start));
            }
        }
    }

    if best[n].is_none() {
        return name.to_string();
    }

    let mut parts = Vec::new();
    let mut end = n;

    while end > 0 {
        let start = best[end].unwrap().1;
        parts.push(&name[start..end]);
        end = start;
    }

    parts.reverse();
    parts.join(" ")
}

/// Error against the 148 CSS named colors, which the model never saw as targets.
fn css_report(model: &Model, d: &Derived) {
    let mut counts: HashMap<&str, f64> = HashMap::new();

    for s in &d.strings {
        for token in s.text.split(' ').filter(|t| t.len() >= 3) {
            *counts.entry(token).or_default() += s.count() as f64;
        }
    }

    let total: f64 = counts.values().sum();
    let words: HashMap<&str, f64> = counts
        .into_iter()
        .filter(|(_, c)| *c >= 20.0)
        .map(|(w, c)| (w, (c / total).ln()))
        .collect();
    let mut rows: Vec<(String, [u8; 3], text2color::Prediction, f32)> = palette::named::entries()
        .map(|(name, c)| {
            let text = segment(name, &words);
            let p = model.predict(&text);
            let rgb = [c.red, c.green, c.blue];

            (text, rgb, p, delta_e(p.oklab, rgb_to_oklab(rgb)))
        })
        .collect();
    rows.sort_by(|a, b| b.3.total_cmp(&a.3));

    let errs: Vec<f32> = rows.iter().rev().map(|r| r.3).collect();
    let close = errs.iter().filter(|&&e| e < 0.1).count();

    println!(
        "css 148 names: median dE {:.4}, p90 {:.4}, {close} within 0.1",
        errs[errs.len() / 2],
        errs[errs.len() * 9 / 10]
    );

    for (text, css, p, e) in &rows {
        println!(
            "  {:<24} {}  css #{:02x}{:02x}{:02x}  dE {:.3}  conf {:.2}",
            text,
            p.hex(),
            css[0],
            css[1],
            css[2],
            e,
            p.confidence
        );
    }
}

/// Error of the French translations of the xkcd names, with and without accents.
fn french_report(model: &Model, xkcd: &[(String, [u8; 3])], translations: &Path) -> Result<()> {
    let table: HashMap<String, String> = load_translations(translations)?.into_iter().collect();
    let mut accented = Vec::new();
    let mut plain = Vec::new();

    for (name, rgb) in xkcd {
        let Some(fr) = table.get(name) else {
            continue;
        };
        let target = rgb_to_oklab(*rgb);

        accented.push(delta_e(model.predict(fr).oklab, target));
        plain.push(delta_e(model.predict(&strip_accents(fr)).oklab, target));
    }

    accented.sort_by(f32::total_cmp);
    plain.sort_by(f32::total_cmp);

    let n = accented.len();

    println!(
        "xkcd names in french ({n}): median dE {:.4}, p90 {:.4}; without accents median {:.4}, p90 {:.4}",
        accented[n / 2],
        accented[n * 9 / 10],
        plain[n / 2],
        plain[n * 9 / 10]
    );

    Ok(())
}

pub fn report(
    model: &Model,
    d: &Derived,
    raw: &Path,
    translations: &Path,
    probe_path: &Path,
    alpha: f32,
) -> Result<()> {
    let mut err_sum = 0f32;
    let mut err_n = 0;
    let mut scored = Vec::new();

    for s in d.strings.iter().filter(|s| is_holdout(&s.text)) {
        let (colour, confidence) = model.readout().read(&d.target(&s.hist, alpha));

        if confidence > 0.5 {
            let p = model.predict(&s.text);

            err_sum += delta_e(p.oklab, colour);
            err_n += 1;
            scored.push((p.confidence, true));
        }
    }

    let unanswered: Vec<&str> = d
        .vocabulary
        .iter()
        .filter(|w| is_holdout(w))
        .map(String::as_str)
        .collect();
    let unanswered = model.predict_many(&unanswered);
    let passing = unanswered.iter().filter(|p| p.confidence >= 0.5).count();

    scored.extend(unanswered.iter().map(|p| (p.confidence, false)));

    println!(
        "holdout confident strings: {err_n}, mean dE {:.4}",
        err_sum / err_n.max(1) as f32
    );
    println!(
        "confidence AUC (holdout confident strings vs unanswered words): {:.4}",
        auc(&mut scored)
    );
    println!(
        "holdout unanswered words at confidence >= 0.5: {passing}/{} ({:.2}%)",
        unanswered.len(),
        100.0 * passing as f32 / unanswered.len().max(1) as f32
    );

    reliability_report(model, d);

    let xkcd = load_xkcd(raw)?;

    radius_sweep(model, d, &xkcd);

    // The same readout on the human answers themselves: the floor for the model's xkcd error.
    let answers: HashMap<&str, &crate::data::Histogram> = d
        .strings
        .iter()
        .map(|s| (s.text.as_str(), &s.hist))
        .collect();
    let mut floor: Vec<f32> = xkcd
        .iter()
        .filter_map(|(name, rgb)| {
            let hist = answers.get(name.as_str())?;
            let (colour, _) = model.readout().read(&normalised(d, hist));

            Some(delta_e(colour, rgb_to_oklab(*rgb)))
        })
        .collect();
    floor.sort_by(f32::total_cmp);

    println!(
        "xkcd names, readout of the human answers ({}): median dE {:.4}, p90 {:.4}",
        floor.len(),
        floor[floor.len() / 2],
        floor[floor.len() * 9 / 10]
    );

    let mut errs: Vec<f32> = xkcd
        .iter()
        .map(|(name, rgb)| delta_e(model.predict(name).oklab, rgb_to_oklab(*rgb)))
        .collect();
    errs.sort_by(f32::total_cmp);

    println!(
        "xkcd 949 names (seen in training): median dE {:.4}, p90 {:.4}",
        errs[errs.len() / 2],
        errs[errs.len() * 9 / 10]
    );

    french_report(model, &xkcd, translations)?;
    css_report(model, d);

    let mut rng = StdRng::seed_from_u64(7);
    let mut robust = 0;
    let mut tried = 0;

    for (name, _) in &xkcd {
        let clean = model.predict(name);
        let typed = typo(name, &mut rng);

        if typed == *name {
            continue;
        }

        let noisy = model.predict(&typed);
        tried += 1;

        if delta_e(clean.oklab, noisy.oklab) < 0.1 && noisy.confidence >= 0.5 {
            robust += 1;
        }
    }

    println!(
        "typo robustness on xkcd names: {:.1}% ({robust}/{tried})",
        100.0 * robust as f32 / tried.max(1) as f32
    );

    let texts = load_probes(probe_path)?;
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let started = Instant::now();
    let rounds = 200;

    for _ in 0..rounds {
        for p in &refs {
            std::hint::black_box(model.predict(p));
        }
    }

    let per_query = started.elapsed().as_secs_f64() / (rounds * refs.len()) as f64;
    println!("latency: {:.1} us/query", per_query * 1e6);

    println!("probes:");
    probes(model, probe_path)
}
