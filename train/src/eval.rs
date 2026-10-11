//! Evaluation of an exported model through the library's inference path.

use crate::augment::typo;
use crate::data::{ConfidenceMap, Derived, is_holdout, load_xkcd};
use crate::train::auc;
use anyhow::Result;
use rand::SeedableRng;
use rand::rngs::StdRng;
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

pub fn report(model: &Model, d: &Derived, raw: &Path, map: ConfidenceMap) -> Result<()> {
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
