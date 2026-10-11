mod augment;
mod data;
mod eval;
mod train;

use anyhow::{Context, Result};
use burn::prelude::*;
use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use text2color::net::NetConfig;

#[derive(Parser)]
#[command(about = "Data preparation, training and evaluation for text2color")]
struct Cli {
    #[arg(long, default_value = "data/raw")]
    raw: PathBuf,
    #[arg(long, default_value = "data/derived")]
    derived: PathBuf,
    /// `english<TAB>french` pairs applied to the survey strings
    #[arg(long, default_value = "train/translations/en-fr.tsv")]
    translations: PathBuf,
    /// Curated word-colour associations with agreement scores
    #[arg(long, default_value = "train/associations.tsv")]
    associations: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Parse the raw survey dump into data/derived
    Prepare,
    /// Train and export a model
    Train(TrainArgs),
    /// Evaluate an exported model through the library
    Eval {
        #[arg(long, default_value = "model/text2color.bpk")]
        model: PathBuf,
        #[command(flatten)]
        confidence: ConfidenceArgs,
    },
}

#[derive(clap::Args)]
struct TrainArgs {
    #[arg(long, default_value = "model/text2color.bpk")]
    out: PathBuf,
    #[arg(long, default_value_t = 17)]
    log2_buckets: u32,
    #[arg(long, default_value_t = 32)]
    dim: usize,
    #[arg(long, default_value_t = 64)]
    hidden: usize,
    #[arg(long, default_value_t = 20)]
    epochs: usize,
    #[arg(long, default_value_t = 128)]
    batch: usize,
    #[arg(long, default_value_t = 2e-3)]
    lr: f64,
    #[arg(long, default_value_t = 0.01)]
    weight_decay: f32,
    #[command(flatten)]
    confidence: ConfidenceArgs,
    #[arg(long, default_value_t = 3.0)]
    neg_weight: f32,
    /// Random negative word pairs, as a fraction of the negative word count
    #[arg(long, default_value_t = 0.5)]
    pair_fraction: f32,
    /// Loss weight of a curated association example
    #[arg(long, default_value_t = 4.0)]
    assoc_weight: f32,
    /// Copies of each curated association per epoch
    #[arg(long, default_value_t = 6)]
    assoc_repeat: usize,
    #[arg(long, default_value_t = 0.3)]
    typo_p: f64,
    /// Probability of inflecting a word (French gender and number, English -er/-y)
    #[arg(long, default_value_t = 0.15)]
    inflect_p: f64,
    #[arg(long, default_value_t = 0.15)]
    noise_p: f64,
    /// Probability of dropping accents from a training example
    #[arg(long, default_value_t = 0.5)]
    accent_p: f64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    cuda: bool,
}

fn device(cuda: bool) -> Result<Device> {
    if !cuda {
        return Ok(Device::flex());
    }

    #[cfg(feature = "cuda")]
    {
        Ok(Device::cuda(0))
    }
    #[cfg(not(feature = "cuda"))]
    {
        anyhow::bail!("rebuild with --features cuda to use the GPU")
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Prepare => data::prepare(&cli.raw, &cli.translations, &cli.associations, &cli.derived),
        Cmd::Train(a) => {
            let d = data::load_derived(&cli.derived)?;
            let cfg = train::Config {
                net: NetConfig {
                    log2_buckets: a.log2_buckets,
                    dim: a.dim,
                    hidden: a.hidden,
                },
                epochs: a.epochs,
                batch: a.batch,
                lr: a.lr,
                weight_decay: a.weight_decay,
                confidence: a.confidence.map(),
                neg_weight: a.neg_weight,
                pair_fraction: a.pair_fraction,
                assoc_weight: a.assoc_weight,
                assoc_repeat: a.assoc_repeat,
                typo_p: a.typo_p,
                inflect_p: a.inflect_p,
                noise_p: a.noise_p,
                accent_p: a.accent_p,
                seed: a.seed,
            };
            let dev = device(a.cuda)?;
            let (model, holdout) = train::train(&d, &cfg, &dev)?;
            let (err, auc) = train::evaluate(&model, &holdout);

            println!("final: holdout dE {err:.4}  auc {auc:.4}");
            model
                .save(&a.out, cfg.net)
                .with_context(|| format!("write {}", a.out.display()))?;
            println!(
                "wrote {} ({} bytes, f16)",
                a.out.display(),
                fs::metadata(&a.out)?.len()
            );

            let reloaded = load_model(&a.out)?;
            eval::probes(&reloaded);

            Ok(())
        }
        Cmd::Eval { model, confidence } => {
            let m = load_model(&model)?;
            let d = data::load_derived(&cli.derived)?;

            eval::report(&m, &d, &cli.raw, &cli.translations, confidence.map())
        }
    }
}

#[derive(clap::Args)]
struct ConfidenceArgs {
    /// Pseudo-answers pulling rare strings toward the random-color spread
    #[arg(long, default_value_t = 2.0)]
    prior: f32,
    /// Spread ratio at which the confidence target is 0.5
    #[arg(long, default_value_t = 0.80)]
    r0: f32,
    /// Width of the logistic on the spread ratio
    #[arg(long, default_value_t = 0.04)]
    tau: f32,
    /// Added to the logit per decade of answer count above 100
    #[arg(long, default_value_t = 0.3)]
    kappa: f32,
}

impl ConfidenceArgs {
    fn map(&self) -> data::ConfidenceMap {
        data::ConfidenceMap {
            prior: self.prior,
            r0: self.r0,
            tau: self.tau,
            kappa: self.kappa,
        }
    }
}

/// Loads weights plus the `.cfg` sidecar written next to them.
fn load_model(path: &std::path::Path) -> Result<text2color::Model> {
    let cfg_path = path.with_extension("cfg");
    let cfg_text =
        fs::read_to_string(&cfg_path).with_context(|| format!("read {}", cfg_path.display()))?;
    let cfg =
        NetConfig::parse(&cfg_text).with_context(|| format!("parse {}", cfg_path.display()))?;

    text2color::Model::from_file(path, cfg).map_err(|e| anyhow::anyhow!("{e}"))
}
