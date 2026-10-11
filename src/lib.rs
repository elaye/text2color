//! Map a word or short phrase to a color, with a confidence score.
//!
//! ```
//! let p = text2color::predict("dark teal");
//! println!("{} rgb={:?} confidence={:.2}", p.hex(), p.rgb, p.confidence);
//! ```
//!
//! The model is a small hashed character-n-gram network trained on the xkcd color survey,
//! so it tolerates typos ("bleu"), understands modifiers ("pale", "neon") and common
//! associations ("grass", "blood"). `confidence` is how predictable the color is from the text:
//! near 1 for unambiguous color names, near 0 for text that does not describe a color.

pub mod color;
pub mod net;
#[doc(hidden)]
pub mod text;

use burn::prelude::*;
use burn::store::{BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot};
use color::{AB_SCALE, oklab_to_rgb};
use net::{Net, NetConfig, OUTPUTS};
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
    /// How predictable the color is from the text, in `0..=1`.
    /// 1 means people agree on the color this text names; 0 means it is not a color description.
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

    /// Wraps an already initialised network living on `device`.
    pub fn new(net: Net, device: Device) -> Model {
        Model { net, device }
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

        Ok(Model { net, device })
    }

    /// Saves the weights as f16 burnpack plus a `.cfg` sidecar describing the shape.
    pub fn save(&self, path: impl AsRef<Path>, config: NetConfig) -> Result<(), LoadError> {
        let path = path.as_ref();
        let mut store = BurnpackStore::from_file(path)
            .with_to_adapter(HalfPrecisionAdapter::new())
            .overwrite(true);

        self.net
            .save_into(&mut store)
            .map_err(|e| LoadError(e.to_string()))?;
        std::fs::write(path.with_extension("cfg"), config.to_text())
            .map_err(|e| LoadError(e.to_string()))?;

        Ok(())
    }

    /// Raw network outputs `[L, a * AB_SCALE, b * AB_SCALE, confidence logit]` for each text.
    /// Texts without any feature yield `None`.
    pub fn raw_outputs(&self, texts: &[&str]) -> Vec<Option<[f32; OUTPUTS]>> {
        if texts.is_empty() {
            return Vec::new();
        }

        let batch = self.net.batch(texts, &self.device);
        let non_empty = batch.non_empty.clone();
        let out = self.net.forward(batch).into_data();
        let values = out.as_slice::<f32>().expect("f32 outputs");

        values
            .as_chunks::<OUTPUTS>()
            .0
            .iter()
            .zip(non_empty)
            .map(|(row, ok)| ok.then_some(*row))
            .collect()
    }

    /// Predicts the color described by `text`.
    pub fn predict(&self, text: &str) -> Prediction {
        self.predict_many(&[text]).remove(0)
    }

    /// Predicts several texts in one pass, cheaper than calling [`Model::predict`] in a loop.
    pub fn predict_many(&self, texts: &[&str]) -> Vec<Prediction> {
        self.raw_outputs(texts)
            .into_iter()
            .map(|out| match out {
                Some(out) => {
                    let oklab = [out[0], out[1] / AB_SCALE, out[2] / AB_SCALE];
                    let confidence = 1.0 / (1.0 + (-out[3]).exp());

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

/// Predicts a color with the model embedded in the library.
pub fn predict(text: &str) -> Prediction {
    Model::embedded().predict(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_reload_gives_identical_outputs() {
        let config = NetConfig {
            log2_buckets: 10,
            dim: 8,
            hidden: 6,
        };
        let device = Device::flex();
        let model = Model::new(config.init(&device), device);
        let dir = std::env::temp_dir().join(format!("text2color-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("net.bpk");

        model.save(&path, config).unwrap();

        let reloaded = Model::from_file(
            &path,
            NetConfig::parse(&std::fs::read_to_string(path.with_extension("cfg")).unwrap())
                .unwrap(),
        )
        .unwrap();
        let probes = ["blue", "dark teal", "x y z"];

        for (a, b) in model
            .raw_outputs(&probes)
            .into_iter()
            .zip(reloaded.raw_outputs(&probes))
        {
            let (a, b) = (a.unwrap(), b.unwrap());

            for k in 0..OUTPUTS {
                assert!((a[k] - b[k]).abs() < 1e-2, "{a:?} vs {b:?}");
            }
        }

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn empty_text_is_gray_with_zero_confidence() {
        let config = NetConfig {
            log2_buckets: 10,
            dim: 8,
            hidden: 6,
        };
        let device = Device::flex();
        let model = Model::new(config.init(&device), device);
        let p = model.predict("!!!");

        assert_eq!(p.confidence, 0.0);
        assert_eq!(p.rgb, [128, 128, 128]);
    }
}
