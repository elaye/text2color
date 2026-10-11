mod augment;
mod bins;
mod data;
mod eval;
mod labels;
mod train;

use anyhow::{Context, Result};
use burn::prelude::*;
use clap::{Parser, Subcommand};
use std::fs;
use std::path::{Path, PathBuf};
use text2color::net::NetConfig;
use text2color::readout::Readout;

/// Oklab distance within which two colours count as the same answer.
const RADIUS: f32 = 0.125;
/// Neighbourhood of the peak averaged into the returned colour.
const COLOUR_RADIUS: f32 = 0.05;

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
    /// Texts printed with their predictions after training and evaluation, one per line
    #[arg(long, default_value = "train/probes.txt")]
    probes: PathBuf,
    /// Merged colour labels of words, used in training targets
    #[arg(long, default_value = "train/labels/full-sonnet.tsv")]
    labels: PathBuf,
    /// Colour names a label may use
    #[arg(long, default_value = "train/labels/palette.txt")]
    palette: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Parse the raw survey dump into colour-bin histograms in data/derived
    Prepare {
        /// Number of colour bins
        #[arg(long, default_value_t = 1024)]
        bins: usize,
    },
    /// Train and export a model
    Train(TrainArgs),
    /// Evaluate an exported model through the library
    Eval {
        #[arg(long, default_value = "model/text2color.bpk")]
        model: PathBuf,
        /// Pseudo-answers spread like the prior, as in training
        #[arg(long, default_value_t = 2.0)]
        alpha: f32,
    },
    /// Colour labels for words, written by Claude Code subagents
    Labels {
        /// Working directory holding a run's batches and returned files
        #[arg(long, default_value = "data/labels-work")]
        work: PathBuf,
        #[command(subcommand)]
        cmd: LabelsCmd,
    },
}

#[derive(Subcommand)]
enum LabelsCmd {
    /// Batches of the whole vocabulary, in frequency order, in <work>/full
    Batches,
    /// Pilot batches in <work>/pilot: survey names, random vocabulary words and the probes
    Pilot {
        #[arg(long, default_value_t = 500)]
        survey: usize,
        #[arg(long, default_value_t = 450)]
        per_language: usize,
        #[arg(long, default_value_t = 3)]
        batches: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Check one returned file against its batch
    Validate { batch: PathBuf, result: PathBuf },
    /// Validate a model's returned files for a run and merge them into train/labels/<run>-<model>.tsv
    Check {
        run: String,
        model: String,
        #[arg(long, default_value_t = 3)]
        samples: usize,
    },
    /// Compare a run's labels from each model with the survey answers and with each other
    Report {
        run: String,
        #[arg(required = true)]
        models: Vec<String>,
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
    #[arg(long, default_value_t = 256)]
    hidden: usize,
    #[arg(long, default_value_t = RADIUS)]
    radius: f32,
    #[arg(long, default_value_t = COLOUR_RADIUS)]
    colour_radius: f32,
    #[arg(long, default_value_t = 20)]
    epochs: usize,
    #[arg(long, default_value_t = 128)]
    batch: usize,
    #[arg(long, default_value_t = 2e-3)]
    lr: f64,
    #[arg(long, default_value_t = 0.01)]
    weight_decay: f32,
    /// Pseudo-answers spread like the prior, added to every string's answers
    #[arg(long, default_value_t = 2.0)]
    alpha: f32,
    /// Random word pairs without answers, as a fraction of the vocabulary
    #[arg(long, default_value_t = 0.5)]
    pair_fraction: f32,
    #[arg(long, default_value_t = 0.3)]
    typo_p: f64,
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

    let with_labels = || -> Result<data::Derived> {
        let palette = labels::Palette::load(&cli.palette)?;

        data::load_derived(&cli.derived, Some((&cli.labels, &palette)))
    };

    match cli.cmd {
        Cmd::Prepare { bins } => data::prepare(&cli.raw, &cli.translations, bins, &cli.derived),
        Cmd::Train(a) => {
            let d = with_labels()?;
            let cfg = train::Config {
                net: NetConfig {
                    log2_buckets: a.log2_buckets,
                    dim: a.dim,
                    hidden: a.hidden,
                    bins: d.centroids.len(),
                    radius: a.radius,
                    colour_radius: a.colour_radius,
                    temperature: 1.0,
                },
                epochs: a.epochs,
                batch: a.batch,
                lr: a.lr,
                weight_decay: a.weight_decay,
                alpha: a.alpha,
                pair_fraction: a.pair_fraction,
                typo_p: a.typo_p,
                noise_p: a.noise_p,
                accent_p: a.accent_p,
                seed: a.seed,
            };
            let dev = device(a.cuda)?;
            let (mut model, holdout) = train::train(&d, &cfg, &dev)?;
            let temperature = train::fit_temperature(&model, &d);

            model.set_temperature(temperature);

            let (err, auc) = train::evaluate(&model, &d, &holdout, cfg.alpha);

            println!("final: temperature {temperature:.2}  holdout dE {err:.4}  auc {auc:.4}");
            model
                .save(&a.out)
                .with_context(|| format!("write {}", a.out.display()))?;
            println!(
                "wrote {} ({} bytes, f16)",
                a.out.display(),
                fs::metadata(&a.out)?.len()
            );

            let reloaded = load_model(&a.out)?;
            eval::probes(&reloaded, &cli.probes)
        }
        Cmd::Eval { model, alpha } => {
            let m = load_model(&model)?;
            let d = with_labels()?;

            eval::report(&m, &d, &cli.raw, &cli.translations, &cli.probes, alpha)
        }
        Cmd::Labels { work, cmd } => {
            let palette = labels::Palette::load(&cli.palette)?;

            labels_cmd(
                &cli.raw,
                &cli.derived,
                &cli.translations,
                &cli.probes,
                &work,
                &palette,
                cmd,
            )
        }
    }
}

fn labels_cmd(
    raw: &Path,
    derived: &Path,
    translations: &Path,
    probes: &Path,
    work: &Path,
    palette: &labels::Palette,
    cmd: LabelsCmd,
) -> Result<()> {
    match cmd {
        LabelsCmd::Batches => labels::full_batches(raw, &work.join("full")),
        LabelsCmd::Pilot {
            survey,
            per_language,
            batches,
            seed,
        } => {
            let d = data::load_derived(derived, None)?;
            let args = labels::PilotArgs {
                raw,
                translations,
                probes,
                palette,
                survey,
                per_language,
                batches,
                seed,
            };

            labels::pilot_batches(&d, &args, &work.join("pilot"))
        }
        LabelsCmd::Validate { batch, result } => {
            let words: Vec<String> = fs::read_to_string(&batch)
                .with_context(|| format!("read {}", batch.display()))?
                .lines()
                .map(str::to_string)
                .collect();
            let text = fs::read_to_string(&result)
                .with_context(|| format!("read {}", result.display()))?;
            let (_, errors) = labels::validate(&words, &text, palette);

            if errors.is_empty() {
                println!("ok");
                return Ok(());
            }

            for e in &errors {
                println!("{e}");
            }

            anyhow::bail!("{} problems", errors.len())
        }
        LabelsCmd::Check {
            run,
            model,
            samples,
        } => {
            let out = PathBuf::from(format!("train/labels/{run}-{model}.tsv"));

            labels::check(&work.join(&run), &model, samples, palette, &out)
        }
        LabelsCmd::Report { run, models } => {
            let d = data::load_derived(derived, None)?;
            let readout = Readout::new(d.centroids.clone(), RADIUS, COLOUR_RADIUS);
            let files: Vec<(String, PathBuf)> = models
                .into_iter()
                .map(|m| {
                    let path = PathBuf::from(format!("train/labels/{run}-{m}.tsv"));
                    (m, path)
                })
                .collect();

            labels::report(&d, palette, &readout, &work.join(&run), &files)
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
