//! Training loop.

use crate::augment::{strip_accents, typo, with_noise};
use crate::data::{Derived, Histogram, RELIABLE_COUNT, is_calibration, is_holdout};
use anyhow::Result;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::{IndexedRandom, SliceRandom};
use std::time::Instant;
use text2color::Model;
use text2color::color::delta_e;
use text2color::net::NetConfig;

#[derive(Debug, Clone)]
pub struct Config {
    pub net: NetConfig,
    pub epochs: usize,
    pub batch: usize,
    pub lr: f64,
    pub weight_decay: f32,
    /// Pseudo-answers spread like the prior, added to every string's answers.
    pub alpha: f32,
    pub pair_fraction: f32,
    pub typo_p: f64,
    pub noise_p: f64,
    pub accent_p: f64,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct Example {
    pub text: String,
    pub hist: Histogram,
}

impl Example {
    fn count(&self) -> u32 {
        self.hist.iter().map(|&(_, n)| n).sum()
    }
}

pub struct Split {
    pub train: Vec<Example>,
    pub holdout: Vec<Example>,
}

pub fn build_examples(d: &Derived, cfg: &Config, rng: &mut StdRng) -> Split {
    let mut all: Vec<Example> = d
        .strings
        .iter()
        .map(|s| Example {
            text: s.text.clone(),
            hist: s.hist.clone(),
        })
        .collect();
    let unanswered = |text: String| Example {
        text,
        hist: Vec::new(),
    };

    all.extend(d.vocabulary.iter().cloned().map(unanswered));

    let pairs = (d.vocabulary.len() as f32 * cfg.pair_fraction) as usize;

    for _ in 0..pairs {
        let a = d.vocabulary.choose(rng).unwrap();
        let b = d.vocabulary.choose(rng).unwrap();
        all.push(unanswered(format!("{a} {b}")));
    }

    let (holdout, train) = all.into_iter().partition(|e| is_holdout(&e.text));

    Split { train, holdout }
}

/// Augmented text and the share of the target kept; the rest moves to the prior, since a typo
/// or an extra word makes the intended colour less certain.
fn augment(e: &Example, noise_words: &[String], cfg: &Config, rng: &mut StdRng) -> (String, f32) {
    let mut text = e.text.clone();
    let mut keep = 1.0;

    if !text.is_ascii() && rng.random_bool(cfg.accent_p) {
        text = strip_accents(&text);
    }

    if rng.random_bool(cfg.typo_p) {
        let typed = typo(&text, rng);

        if typed != text {
            text = typed;
            keep *= 0.9;
        }
    }

    if rng.random_bool(cfg.noise_p) {
        text = with_noise(&text, noise_words.choose(rng).unwrap(), rng);
        keep *= 0.7;
    }

    (text, keep)
}

struct Targets {
    dist: Tensor<2>,
    weight: Tensor<2>,
}

fn targets(d: &Derived, examples: &[(&Example, f32)], alpha: f32, device: &Device) -> Targets {
    let n = examples.len();
    let bins = d.prior.len();
    let mut dist = Vec::with_capacity(n * bins);
    let mut weight = Vec::with_capacity(n);

    for (e, keep) in examples {
        let t = d.target(&e.hist, alpha);

        dist.extend(
            t.iter()
                .zip(&d.prior)
                .map(|(t, p)| keep * t + (1.0 - keep) * p),
        );
        // Between weighting each string equally and each answer equally: a bare "blue" with
        // 270k answers must outweigh the many phrases that share its n-grams.
        weight.push((1.0 + e.count() as f32 + alpha).sqrt());
    }

    let total: f32 = weight.iter().sum();
    weight.iter_mut().for_each(|w| *w /= total);

    Targets {
        dist: Tensor::from_data(TensorData::new(dist, [n, bins]), device),
        weight: Tensor::from_data(TensorData::new(weight, [n, 1]), device),
    }
}

/// Weighted cross-entropy between the target distribution and the predicted one.
fn loss(logits: Tensor<2>, t: &Targets) -> Tensor<1> {
    let ce = (t.dist.clone() * log_softmax(logits, 1)).sum_dim(1).neg();

    (ce * t.weight.clone()).sum()
}

/// Mean Oklab error on hold-out strings whose target is confident, and AUC of the predicted
/// confidence for those strings against unanswered ones.
pub fn evaluate(model: &Model, d: &Derived, holdout: &[Example], alpha: f32) -> (f32, f32) {
    let mut err_sum = 0f32;
    let mut err_n = 0;
    let mut scored: Vec<(f32, bool)> = Vec::new();

    for chunk in holdout.chunks(512) {
        let texts: Vec<&str> = chunk.iter().map(|e| e.text.as_str()).collect();

        for (e, p) in chunk.iter().zip(model.predict_many(&texts)) {
            if e.hist.is_empty() {
                scored.push((p.confidence, false));
                continue;
            }

            let (colour, confidence) = model.readout().read(&d.target(&e.hist, alpha));

            if confidence > 0.5 {
                err_sum += delta_e(p.oklab, colour);
                err_n += 1;
                scored.push((p.confidence, true));
            }
        }
    }

    (err_sum / err_n.max(1) as f32, auc(&mut scored))
}

/// Temperature minimising the cross-entropy between the human answers and the predicted
/// distribution, on calibration strings with enough answers to measure agreement.
pub fn fit_temperature(model: &Model, d: &Derived) -> f32 {
    let rows: Vec<_> = d
        .strings
        .iter()
        .filter(|s| is_calibration(&s.text) && s.count() >= RELIABLE_COUNT)
        .collect();
    let texts: Vec<&str> = rows.iter().map(|s| s.text.as_str()).collect();
    let logits: Vec<Vec<f32>> = texts
        .chunks(512)
        .flat_map(|chunk| model.logits(chunk))
        .map(Option::unwrap_or_default)
        .collect();
    let answers: Vec<Vec<f32>> = rows.iter().map(|s| d.target(&s.hist, 0.0)).collect();
    let cross_entropy = |t: f32| -> f64 {
        logits
            .iter()
            .zip(&answers)
            .map(|(z, a)| {
                let p = text2color::softmax(z, t);

                a.iter()
                    .zip(&p)
                    .map(|(a, p)| -(*a as f64) * (*p as f64).max(1e-12).ln())
                    .sum::<f64>()
            })
            .sum()
    };

    (20..=300)
        .map(|i| i as f32 / 100.0)
        .min_by(|a, b| cross_entropy(*a).total_cmp(&cross_entropy(*b)))
        .expect("non-empty grid")
}

pub fn auc(scored: &mut [(f32, bool)]) -> f32 {
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));

    let pos = scored.iter().filter(|s| s.1).count() as f64;
    let neg = scored.len() as f64 - pos;
    let mut rank_sum = 0f64;

    for (i, s) in scored.iter().enumerate() {
        if s.1 {
            rank_sum += (i + 1) as f64;
        }
    }

    if pos == 0.0 || neg == 0.0 {
        return f32::NAN;
    }

    ((rank_sum - pos * (pos + 1.0) / 2.0) / (pos * neg)) as f32
}

/// Trains on `device` and returns the inference model (on the plain device) and the hold-out set.
pub fn train(d: &Derived, cfg: &Config, device: &Device) -> Result<(Model, Vec<Example>)> {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let split = build_examples(d, cfg, &mut rng);
    let noise_words: Vec<String> = d.vocabulary.iter().take(2000).cloned().collect();

    println!(
        "train {} examples ({} with answers), holdout {}",
        split.train.len(),
        split.train.iter().filter(|e| !e.hist.is_empty()).count(),
        split.holdout.len()
    );

    let train_device = device.clone().autodiff();
    train_device.seed(cfg.seed);

    let mut net = cfg
        .net
        .init(&train_device)
        .with_centroids(&d.centroids, &train_device);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(cfg.weight_decay)
        .init();
    let steps_per_epoch = split.train.len().div_ceil(cfg.batch);
    let total_steps = steps_per_epoch * cfg.epochs;
    let mut step = 0usize;
    let mut order: Vec<usize> = (0..split.train.len()).collect();

    for epoch in 0..cfg.epochs {
        let started = Instant::now();
        order.shuffle(&mut rng);

        let mut sum_loss = 0f32;

        for chunk in order.chunks(cfg.batch) {
            let lr = cfg.lr * (1.0 - step as f64 / total_steps as f64).max(0.02);
            let mut texts = Vec::with_capacity(chunk.len());
            let mut pairs = Vec::with_capacity(chunk.len());

            for &i in chunk {
                let e = &split.train[i];
                let (text, keep) = augment(e, &noise_words, cfg, &mut rng);
                texts.push(text);
                pairs.push((e, keep));
            }

            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let batch = net.batch(&refs, &train_device);
            let t = targets(d, &pairs, cfg.alpha, &train_device);
            let loss = loss(net.forward(batch), &t);
            let grads = GradientsParams::from_grads(loss.backward(), &net);

            net = optim.step(lr, net, grads);

            sum_loss += loss.into_scalar::<f32>();
            step += 1;
        }

        let model = Model::new(net.valid(), cfg.net, device.clone());
        let (err, auc) = evaluate(&model, d, &split.holdout, cfg.alpha);

        println!(
            "epoch {:>2}  loss {:.4}  holdout dE {:.4}  auc {:.4}  ({:.0}s)",
            epoch + 1,
            sum_loss / steps_per_epoch as f32,
            err,
            auc,
            started.elapsed().as_secs_f32()
        );
    }

    Ok((
        Model::new(net.valid(), cfg.net, device.clone()),
        split.holdout,
    ))
}
