//! Training loop.

use crate::augment::{inflect, strip_accents, typo, with_noise};
use crate::data::{ConfidenceMap, Derived, is_holdout};
use anyhow::Result;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::activation::relu;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::{IndexedRandom, SliceRandom};
use std::time::Instant;
use text2color::Model;
use text2color::color::{AB_SCALE, delta_e};
use text2color::net::NetConfig;

#[derive(Debug, Clone)]
pub struct Config {
    pub net: NetConfig,
    pub epochs: usize,
    pub batch: usize,
    pub lr: f64,
    pub weight_decay: f32,
    pub confidence: ConfidenceMap,
    pub neg_weight: f32,
    pub pair_fraction: f32,
    pub assoc_weight: f32,
    pub assoc_repeat: usize,
    pub typo_p: f64,
    pub inflect_p: f64,
    pub noise_p: f64,
    pub accent_p: f64,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct Example {
    pub text: String,
    /// Scaled Oklab target, meaningless when `has_color` is false.
    pub target: [f32; 3],
    pub confidence: f32,
    pub weight: f32,
    pub has_color: bool,
}

pub struct Split {
    pub train: Vec<Example>,
    pub holdout: Vec<Example>,
}

pub fn build_examples(d: &Derived, cfg: &Config, rng: &mut StdRng) -> Split {
    let mut all = Vec::new();

    for p in &d.positives {
        all.push(Example {
            text: p.text.clone(),
            target: [p.oklab[0], p.oklab[1] * AB_SCALE, p.oklab[2] * AB_SCALE],
            confidence: cfg.confidence.target(p.sigma, p.count, d.sigma_rand),
            weight: (1.0 + p.count as f32).ln(),
            has_color: true,
        });
    }

    let negative = |text: String| Example {
        text,
        target: [0.0; 3],
        confidence: 0.0,
        weight: cfg.neg_weight,
        has_color: false,
    };

    for w in &d.negatives {
        all.push(negative(w.clone()));
    }

    let pairs = (d.negatives.len() as f32 * cfg.pair_fraction) as usize;

    for _ in 0..pairs {
        let a = d.negatives.choose(rng).unwrap();
        let b = d.negatives.choose(rng).unwrap();
        all.push(negative(format!("{a} {b}")));
    }

    let (holdout, mut train): (Vec<Example>, Vec<Example>) =
        all.into_iter().partition(|e| is_holdout(&e.text));

    // Curated associations are few and are the only source for their words, so they are never
    // held out and are repeated to weigh against the survey rows.
    for a in &d.associations {
        for _ in 0..cfg.assoc_repeat {
            train.push(Example {
                text: a.text.clone(),
                target: [a.oklab[0], a.oklab[1] * AB_SCALE, a.oklab[2] * AB_SCALE],
                confidence: a.agreement,
                weight: cfg.assoc_weight,
                has_color: true,
            });
        }
    }

    Split { train, holdout }
}

fn augment(e: &Example, noise_words: &[String], cfg: &Config, rng: &mut StdRng) -> (String, f32) {
    let mut text = e.text.clone();
    let mut confidence = e.confidence;

    if !text.is_ascii() && rng.random_bool(cfg.accent_p) {
        text = strip_accents(&text);
    }

    if rng.random_bool(cfg.inflect_p) {
        let inflected = inflect(&text, rng);

        if inflected != text {
            text = inflected;
            confidence *= 0.9;
        }
    }

    if rng.random_bool(cfg.typo_p) {
        let typed = typo(&text, rng);

        if typed != text {
            text = typed;
            confidence *= 0.9;
        }
    }

    if e.has_color && rng.random_bool(cfg.noise_p) {
        text = with_noise(&text, noise_words.choose(rng).unwrap(), rng);
        confidence *= 0.7;
    }

    (text, confidence)
}

struct Targets {
    color: Tensor<2>,
    conf: Tensor<2>,
    color_w: Tensor<2>,
    conf_w: Tensor<2>,
}

fn targets(examples: &[(&Example, f32)], device: &Device) -> Targets {
    let n = examples.len();
    let mut color = Vec::with_capacity(n * 3);
    let mut conf = Vec::with_capacity(n);
    let mut color_w = Vec::with_capacity(n);
    let mut conf_w = Vec::with_capacity(n);

    for (e, c) in examples {
        color.extend(e.target);
        conf.push(*c);
        color_w.push(if e.has_color {
            e.weight * e.confidence.max(0.05)
        } else {
            0.0
        });
        conf_w.push(e.weight);
    }

    let total_color: f32 = color_w.iter().sum::<f32>().max(1e-6);
    let total_conf: f32 = conf_w.iter().sum::<f32>().max(1e-6);

    for w in &mut color_w {
        *w /= total_color;
    }

    for w in &mut conf_w {
        *w /= total_conf;
    }

    Targets {
        color: Tensor::from_data(TensorData::new(color, [n, 3]), device),
        conf: Tensor::from_data(TensorData::new(conf, [n, 1]), device),
        color_w: Tensor::from_data(TensorData::new(color_w, [n, 1]), device),
        conf_w: Tensor::from_data(TensorData::new(conf_w, [n, 1]), device),
    }
}

/// Weighted Huber on the color dims plus weighted soft-target BCE on the confidence logit.
fn loss(out: Tensor<2>, t: &Targets) -> (Tensor<1>, Tensor<1>) {
    const DELTA: f64 = 0.1;

    let diff = (out.clone().narrow(1, 0, 3) - t.color.clone()).abs();
    let quadratic = diff.clone().powi_scalar(2) * (0.5 / DELTA);
    let linear = diff.clone() - 0.5 * DELTA;
    let huber = linear
        .mask_where(diff.lower_elem(DELTA), quadratic)
        .sum_dim(1);
    let color_loss = (huber * t.color_w.clone()).sum();

    let z = out.narrow(1, 3, 1);
    let bce = relu(z.clone()) - z.clone() * t.conf.clone() + (z.abs().neg().exp() + 1.0).log();
    let conf_loss = (bce * t.conf_w.clone()).sum();

    (color_loss, conf_loss)
}

/// Mean Oklab error on confident hold-out positives and AUC of confidence (positives with
/// target > 0.5 vs negatives).
pub fn evaluate(model: &Model, holdout: &[Example]) -> (f32, f32) {
    let mut err_sum = 0f32;
    let mut err_n = 0;
    let mut scored: Vec<(f32, bool)> = Vec::new();

    for chunk in holdout.chunks(512) {
        let texts: Vec<&str> = chunk.iter().map(|e| e.text.as_str()).collect();

        for (e, p) in chunk.iter().zip(model.predict_many(&texts)) {
            if e.has_color && e.confidence > 0.5 {
                let t = [e.target[0], e.target[1] / AB_SCALE, e.target[2] / AB_SCALE];
                err_sum += delta_e(p.oklab, t);
                err_n += 1;
                scored.push((p.confidence, true));
            } else if !e.has_color {
                scored.push((p.confidence, false));
            }
        }
    }

    (err_sum / err_n.max(1) as f32, auc(&mut scored))
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
    let noise_words: Vec<String> = d.negatives.iter().take(2000).cloned().collect();

    println!(
        "train {} examples ({} positives), holdout {}",
        split.train.len(),
        split.train.iter().filter(|e| e.has_color).count(),
        split.holdout.len()
    );

    let train_device = device.clone().autodiff();
    train_device.seed(cfg.seed);

    let mut net = cfg.net.init(&train_device);
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

        let mut sum_color = 0f32;
        let mut sum_conf = 0f32;

        for chunk in order.chunks(cfg.batch) {
            let lr = cfg.lr * (1.0 - step as f64 / total_steps as f64).max(0.02);
            let mut texts = Vec::with_capacity(chunk.len());
            let mut pairs = Vec::with_capacity(chunk.len());

            for &i in chunk {
                let e = &split.train[i];
                let (text, c) = augment(e, &noise_words, cfg, &mut rng);
                texts.push(text);
                pairs.push((e, c));
            }

            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let batch = net.batch(&refs, &train_device);
            let t = targets(&pairs, &train_device);
            let (color_loss, conf_loss) = loss(net.forward(batch), &t);
            let total = color_loss.clone() + conf_loss.clone();
            let grads = GradientsParams::from_grads(total.backward(), &net);

            net = optim.step(lr, net, grads);

            sum_color += color_loss.into_scalar::<f32>();
            sum_conf += conf_loss.into_scalar::<f32>();
            step += 1;
        }

        let model = Model::new(net.valid(), device.clone());
        let (err, auc) = evaluate(&model, &split.holdout);

        println!(
            "epoch {:>2}  color {:.4}  conf {:.4}  holdout dE {:.4}  auc {:.4}  ({:.0}s)",
            epoch + 1,
            sum_color / steps_per_epoch as f32,
            sum_conf / steps_per_epoch as f32,
            err,
            auc,
            started.elapsed().as_secs_f32()
        );
    }

    Ok((Model::new(net.valid(), device.clone()), split.holdout))
}
