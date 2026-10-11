//! Map a word or short phrase to a color, with a confidence score.
//!
//! ```
//! let p = text2color::predict("dark teal");
//! println!("{} rgb={:?} confidence={:.2}", p.hex(), p.rgb, p.confidence);
//! ```
//!
//! The model is a small hashed character-n-gram network that predicts, for any text, the
//! distribution of colours people would pick for it. It tolerates typos ("bleu"), understands
//! modifiers ("pale", "neon") and associations ("grass", "blood"). The colour is the densest part
//! of that distribution and `confidence` is the share of people who would pick a colour close to
//! it: near 1 for unambiguous colour names, low for text without a consistent colour.

pub mod color;
pub mod net;
pub mod readout;
#[doc(hidden)]
pub mod text;

use burn::prelude::*;
use burn::store::{BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot};
use color::oklab_to_rgb;
use net::{Net, NetConfig};
use readout::Readout;
use std::fmt;
use std::path::Path;
use std::sync::OnceLock;

// `include_bytes!` only guarantees byte alignment; burnpack reads tensors in place and needs more.
#[repr(C, align(64))]
struct Aligned<T: ?Sized>(T);

static EMBEDDED_WEIGHTS: &Aligned<[u8]> = &Aligned(*include_bytes!("../model/text2color.bpk"));
static EMBEDDED_CONFIG: &str = include_str!("../model/text2color.cfg");
static EMBEDDED: OnceLock<Model> = OnceLock::new();

/// Result of a prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    /// 8-bit sRGB, clamped to gamut.
    pub rgb: [u8; 3],
    /// The same color in Oklab `[L, a, b]`, before gamut clamping.
    pub oklab: [f32; 3],
    /// Share of people who would pick a color close to this one for the text, in `0..=1`.
    /// Above 0.5, most people agree on the color; text without a consistent color scores low.
    pub confidence: f32,
}

impl Prediction {
    /// CSS hex string such as `#1f77b4`.
    pub fn hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.rgb[0], self.rgb[1], self.rgb[2])
    }
}

/// Error returned when weights cannot be loaded.
#[derive(Debug, Clone)]
pub struct LoadError(String);

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cannot load text2color weights: {}", self.0)
    }
}

impl std::error::Error for LoadError {}

/// A loaded network ready for inference on the CPU.
pub struct Model {
    net: Net,
    readout: Readout,
    config: NetConfig,
    device: Device,
}

impl Model {
    /// The model compiled into the library.
    pub fn embedded() -> &'static Model {
        EMBEDDED.get_or_init(|| {
            let config = NetConfig::parse(EMBEDDED_CONFIG).expect("embedded config is valid");

            Model::from_static(&EMBEDDED_WEIGHTS.0, config).expect("embedded weights are valid")
        })
    }

    /// Wraps an already initialised network living on `device`, read out as `config` says.
    pub fn new(net: Net, config: NetConfig, device: Device) -> Model {
        let readout = Readout::new(net.centroids(), config.radius, config.colour_radius);

        Model {
            net,
            readout,
            config,
            device,
        }
    }

    pub fn config(&self) -> NetConfig {
        self.config
    }

    /// Sets the temperature applied to the logits, as fitted after training.
    pub fn set_temperature(&mut self, temperature: f32) {
        self.config.temperature = temperature;
    }

    /// Loads burnpack weights embedded in the binary.
    pub fn from_static(bytes: &'static [u8], config: NetConfig) -> Result<Model, LoadError> {
        let store = BurnpackStore::from_static(bytes);

        Model::load(store, config)
    }

    /// Loads burnpack weights from a file written by the trainer.
    pub fn from_file(path: impl AsRef<Path>, config: NetConfig) -> Result<Model, LoadError> {
        let store = BurnpackStore::from_file(path.as_ref());

        Model::load(store, config)
    }

    fn load(store: BurnpackStore, config: NetConfig) -> Result<Model, LoadError> {
        let device = Device::flex();
        let mut net = config.init(&device);
        let mut store = store.with_from_adapter(HalfPrecisionAdapter::new());

        net.load_from(&mut store)
            .map_err(|e| LoadError(e.to_string()))?;

        Ok(Model::new(net, config, device))
    }

    /// Saves the weights as f16 burnpack plus a `.cfg` sidecar with the shape and readout.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), LoadError> {
        let path = path.as_ref();
        let mut store = BurnpackStore::from_file(path)
            .with_to_adapter(HalfPrecisionAdapter::new())
            .overwrite(true);

        self.net
            .save_into(&mut store)
            .map_err(|e| LoadError(e.to_string()))?;
        std::fs::write(path.with_extension("cfg"), self.config.to_text())
            .map_err(|e| LoadError(e.to_string()))?;

        Ok(())
    }

    pub fn readout(&self) -> &Readout {
        &self.readout
    }

    /// Probability of each colour bin for each text. Texts without any feature yield `None`.
    pub fn distributions(&self, texts: &[&str]) -> Vec<Option<Vec<f32>>> {
        self.logits(texts)
            .into_iter()
            .map(|z| z.map(|z| softmax(&z, self.config.temperature)))
            .collect()
    }

    /// Bin logits for each text, before the temperature. Texts without any feature yield `None`.
    pub fn logits(&self, texts: &[&str]) -> Vec<Option<Vec<f32>>> {
        if texts.is_empty() {
            return Vec::new();
        }

        let batch = self.net.batch(texts, &self.device);
        let non_empty = batch.non_empty.clone();
        let out = self.net.forward(batch).into_data();
        let values = out.as_slice::<f32>().expect("f32 outputs");
        let bins = values.len() / texts.len();

        values
            .chunks(bins)
            .zip(non_empty)
            .map(|(row, ok)| ok.then(|| row.to_vec()))
            .collect()
    }

    /// Predicts the color described by `text`.
    pub fn predict(&self, text: &str) -> Prediction {
        self.predict_many(&[text]).remove(0)
    }

    /// Predicts several texts in one pass, cheaper than calling [`Model::predict`] in a loop.
    pub fn predict_many(&self, texts: &[&str]) -> Vec<Prediction> {
        self.distributions(texts)
            .into_iter()
            .map(|p| match p {
                Some(p) => {
                    let (oklab, confidence) = self.readout.read(&p);

                    Prediction {
                        rgb: oklab_to_rgb(oklab),
                        oklab,
                        confidence,
                    }
                }
                None => Prediction {
                    rgb: [128, 128, 128],
                    oklab: [0.6, 0.0, 0.0],
                    confidence: 0.0,
                },
            })
            .collect()
    }
}

/// Softmax of `logits / temperature`.
pub fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits
        .iter()
        .map(|z| ((z - max) / temperature).exp())
        .collect();
    let sum: f32 = exp.iter().sum();

    exp.into_iter().map(|e| e / sum).collect()
}

/// Predicts a color with the model embedded in the library.
pub fn predict(text: &str) -> Prediction {
    Model::embedded().predict(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: NetConfig = NetConfig {
        log2_buckets: 10,
        dim: 8,
        hidden: 6,
        bins: 3,
        radius: 0.1,
        colour_radius: 0.05,
        temperature: 1.0,
    };

    fn tiny_model() -> Model {
        let device = Device::flex();
        let centroids = [[0.2, 0.0, 0.0], [0.6, 0.1, 0.0], [0.9, 0.0, -0.1]];
        let net = TINY.init(&device).with_centroids(&centroids, &device);

        Model::new(net, TINY, device)
    }

    #[test]
    fn save_and_reload_gives_identical_outputs() {
        let model = tiny_model();
        let dir = std::env::temp_dir().join(format!("text2color-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("net.bpk");

        model.save(&path).unwrap();

        let reloaded = Model::from_file(
            &path,
            NetConfig::parse(&std::fs::read_to_string(path.with_extension("cfg")).unwrap())
                .unwrap(),
        )
        .unwrap();
        let probes = ["blue", "dark teal", "x y z"];

        for (a, b) in model
            .distributions(&probes)
            .into_iter()
            .zip(reloaded.distributions(&probes))
        {
            let (a, b) = (a.unwrap(), b.unwrap());

            for k in 0..TINY.bins {
                assert!((a[k] - b[k]).abs() < 1e-2, "{a:?} vs {b:?}");
            }
        }

        for (a, b) in model
            .readout()
            .centroids()
            .iter()
            .zip(reloaded.readout().centroids())
        {
            assert!(color::delta_e(*a, *b) < 1e-3, "{a:?} vs {b:?}");
        }

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn temperature_sharpens_or_flattens() {
        let sharp = softmax(&[1.0, 0.0], 0.5);
        let flat = softmax(&[1.0, 0.0], 2.0);

        assert!(sharp[0] > flat[0] && flat[0] > 0.5);
        assert!((sharp.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn empty_text_is_gray_with_zero_confidence() {
        let p = tiny_model().predict("!!!");

        assert_eq!(p.confidence, 0.0);
        assert_eq!(p.rgb, [128, 128, 128]);
    }
}
