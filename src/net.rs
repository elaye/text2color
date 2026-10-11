//! The network: a hashed character-n-gram embedding bag, one hidden layer, four outputs.
//!
//! This single definition is used both for training (see the `text2color-train` crate) and
//! for inference in this library.

use crate::text::features;
use burn::nn::{Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;

/// Network outputs: Oklab `L`, scaled `a`, scaled `b`, confidence logit.
pub const OUTPUTS: usize = 4;

/// Shape of the network. Must match the weights being loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetConfig {
    pub log2_buckets: u32,
    pub dim: usize,
    pub hidden: usize,
}

impl NetConfig {
    pub fn buckets(&self) -> u32 {
        1 << self.log2_buckets
    }

    pub fn init(&self, device: &Device) -> Net {
        let initializer = Initializer::Normal {
            mean: 0.0,
            std: 0.1,
        };

        Net {
            emb: EmbeddingConfig::new(self.buckets() as usize, self.dim)
                .with_initializer(initializer)
                .init(device),
            l1: LinearConfig::new(self.dim, self.hidden).init(device),
            l2: LinearConfig::new(self.hidden, OUTPUTS).init(device),
        }
    }

    /// `key=value` lines, stored next to the weights.
    pub fn to_text(&self) -> String {
        format!(
            "log2_buckets={}\ndim={}\nhidden={}\n",
            self.log2_buckets, self.dim, self.hidden
        )
    }

    pub fn parse(text: &str) -> Option<Self> {
        let mut cfg = NetConfig {
            log2_buckets: 0,
            dim: 0,
            hidden: 0,
        };

        for line in text.lines() {
            let (key, value) = line.split_once('=')?;

            match key.trim() {
                "log2_buckets" => cfg.log2_buckets = value.trim().parse().ok()?,
                "dim" => cfg.dim = value.trim().parse().ok()?,
                "hidden" => cfg.hidden = value.trim().parse().ok()?,
                _ => return None,
            }
        }

        (cfg.log2_buckets > 0 && cfg.dim > 0 && cfg.hidden > 0).then_some(cfg)
    }
}

#[derive(Module, Debug)]
pub struct Net {
    emb: Embedding,
    l1: Linear,
    l2: Linear,
}

/// Hashed feature ids of a batch of texts, padded to a rectangle with a mask.
pub struct Batch {
    pub ids: Tensor<2, Int>,
    pub mask: Tensor<3>,
    pub count: Tensor<2>,
    /// Which rows had at least one feature.
    pub non_empty: Vec<bool>,
}

impl Net {
    pub fn buckets(&self) -> u32 {
        self.emb.weight.dims()[0] as u32
    }

    pub fn batch(&self, texts: &[&str], device: &Device) -> Batch {
        let buckets = self.buckets();
        let feats: Vec<Vec<u32>> = texts.iter().map(|t| features(t, buckets)).collect();
        let width = feats.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let n = texts.len();
        let mut ids = vec![0i32; n * width];
        let mut mask = vec![0f32; n * width];
        let mut count = vec![1f32; n];

        for (i, f) in feats.iter().enumerate() {
            for (j, &id) in f.iter().enumerate() {
                ids[i * width + j] = id as i32;
                mask[i * width + j] = 1.0;
            }

            count[i] = f.len().max(1) as f32;
        }

        Batch {
            ids: Tensor::from_data(TensorData::new(ids, [n, width]), device),
            mask: Tensor::from_data(TensorData::new(mask, [n, width, 1]), device),
            count: Tensor::from_data(TensorData::new(count, [n, 1]), device),
            non_empty: feats.iter().map(|f| !f.is_empty()).collect(),
        }
    }

    /// Raw outputs, shape `[batch, OUTPUTS]`.
    pub fn forward(&self, batch: Batch) -> Tensor<2> {
        let rows = self.emb.forward(batch.ids);
        let [n, _, d] = rows.dims();
        let pooled = (rows * batch.mask).sum_dim(1).reshape([n, d]) / batch.count;
        let h = relu(self.l1.forward(pooled));

        self.l2.forward(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_round_trips_through_text() {
        let cfg = NetConfig {
            log2_buckets: 12,
            dim: 8,
            hidden: 6,
        };

        assert_eq!(NetConfig::parse(&cfg.to_text()), Some(cfg));
        assert_eq!(NetConfig::parse("dim=3"), None);
    }

    #[test]
    fn forward_shapes_and_empty_rows() {
        let device = Device::flex();
        let net = NetConfig {
            log2_buckets: 10,
            dim: 8,
            hidden: 6,
        }
        .init(&device);
        let batch = net.batch(&["dark blue", "", "x"], &device);

        assert_eq!(batch.non_empty, vec![true, false, true]);

        let out = net.forward(batch);

        assert_eq!(out.dims(), [3, OUTPUTS]);
    }
}
