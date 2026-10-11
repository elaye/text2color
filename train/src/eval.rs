//! Evaluation of an exported model through the library's inference path.

use crate::augment::{strip_accents, typo};
use crate::data::{ConfidenceMap, Derived, is_holdout, load_translations, load_xkcd};
use crate::train::auc;
use anyhow::Result;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use text2color::Model;
use text2color::color::{delta_e, rgb_to_oklab};

pub const PROBES: &[&str] = &[
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "grey",
    "bleu",
    "gren",
    "purpel",
    "yelow",
    "ornage",
    "dark blue",
    "light blue",
    "pale pink",
    "hot pink",
    "neon green",
    "sky",
    "grass",
    "blood",
    "lemon",
    "chocolate",
    "ocean",
    "sunset",
    "table",
    "monday",
    "running",
    "asdfgh",
    "a dark blue",
    "blue car",
    "the color red",
    "blue green",
    "teal",
    "rouge",
    "vert",
    "jaune",
    "violet",
    "marron",
    "gris",
    "bleu clair",
    "vert foncé",
    "vert fonce",
    "rose pâle",
    "bleu marine",
    "ciel",
    "herbe",
    "sang",
    "citron",
    "chocolat",
    "lundi",
    "voiture bleue",
    "",
];

pub fn probes(model: &Model) {
    for (p, r) in PROBES.iter().zip(model.predict_many(PROBES)) {
        println!(
            "  {:<16} {}  rgb({:>3},{:>3},{:>3})  conf {:.2}",
            p,
            r.hex(),
            r.rgb[0],
            r.rgb[1],
            r.rgb[2],
            r.confidence
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

    for p in &d.positives {
        for token in p.text.split(' ').filter(|t| t.len() >= 3) {
            *counts.entry(token).or_default() += p.count as f64;
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
    map: ConfidenceMap,
) -> Result<()> {
    let mut err_sum = 0f32;
    let mut err_n = 0;
    let mut scored = Vec::new();

    for p in d.positives.iter().filter(|p| is_holdout(&p.text)) {
        let c = map.target(p.sigma, p.count, d.sigma_rand);
        let r = model.predict(&p.text);

        if c > 0.5 {
            err_sum += delta_e(r.oklab, p.oklab);
            err_n += 1;
            scored.push((r.confidence, true));
        }
    }

    for w in d.negatives.iter().filter(|w| is_holdout(w)) {
        scored.push((model.predict(w).confidence, false));
    }

    println!(
        "holdout confident positives: {err_n}, mean dE {:.4}",
        err_sum / err_n.max(1) as f32
    );
    println!(
        "confidence AUC (holdout positives c>0.5 vs negatives): {:.4}",
        auc(&mut scored)
    );

    let xkcd = load_xkcd(raw)?;
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

    let started = Instant::now();
    let rounds = 2000;

    for _ in 0..rounds {
        for p in PROBES {
            std::hint::black_box(model.predict(p));
        }
    }

    let per_query = started.elapsed().as_secs_f64() / (rounds * PROBES.len()) as f64;
    println!("latency: {:.1} us/query", per_query * 1e6);

    println!("probes:");
    probes(model);

    Ok(())
}
