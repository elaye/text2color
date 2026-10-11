//! The network: a hashed character-n-gram embedding bag, one hidden layer, and a distribution
//! over colour bins whose centroids are stored with the weights.
//!
//! This single definition is used both for training (see the `text2color-train` crate) and
//! for inference in this library.

use crate::text::features;
use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;

/// Shape of the network and its readout. Must match the weights being loaded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetConfig {
    pub log2_buckets: u32,
    pub dim: usize,
    pub hidden: usize,
    /// Number of colour bins.
    pub bins: usize,
    /// Oklab distance within which two colours count as the same answer.
    pub radius: f32,
    /// Neighbourhood of the peak averaged into the returned colour.
    pub colour_radius: f32,
    /// Divides the logits before the softmax; fitted after training so that confidence
    /// matches measured agreement.
    pub temperature: f32,
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
            l2: LinearConfig::new(self.hidden, self.bins).init(device),
            centroids: Param::from_tensor(Tensor::zeros([self.bins, 3], device))
                .set_require_grad(false),
        }
    }

    /// `key=value` lines, stored next to the weights.
    pub fn to_text(&self) -> String {
        format!(
            "log2_buckets={}\ndim={}\nhidden={}\nbins={}\nradius={}\ncolour_radius={}\ntemperature={}\n",
            self.log2_buckets,
            self.dim,
            self.hidden,
            self.bins,
            self.radius,
            self.colour_radius,
            self.temperature
        )
    }

    pub fn parse(text: &str) -> Option<Self> {
        let mut cfg = NetConfig {
            log2_buckets: 0,
            dim: 0,
            hidden: 0,
            bins: 0,
            radius: 0.0,
            colour_radius: 0.0,
            temperature: 0.0,
        };

        for line in text.lines() {
            let (key, value) = line.split_once('=')?;

            match key.trim() {
                "log2_buckets" => cfg.log2_buckets = value.trim().parse().ok()?,
                "dim" => cfg.dim = value.trim().parse().ok()?,
                "hidden" => cfg.hidden = value.trim().parse().ok()?,
                "bins" => cfg.bins = value.trim().parse().ok()?,
                "radius" => cfg.radius = value.trim().parse().ok()?,
                "colour_radius" => cfg.colour_radius = value.trim().parse().ok()?,
                "temperature" => cfg.temperature = value.trim().parse().ok()?,
                _ => return None,
            }
        }

        let valid = cfg.log2_buckets > 0
            && cfg.dim > 0
            && cfg.hidden > 0
            && cfg.bins > 0
            && cfg.radius > 0.0
            && cfg.colour_radius > 0.0
            && cfg.temperature > 0.0;

        valid.then_some(cfg)
    }
}

#[derive(Module, Debug)]
pub struct Net {
    emb: Embedding,
    l1: Linear,
    l2: Linear,
    /// Oklab centre of each bin; fixed, not trained.
    centroids: Param<Tensor<2>>,
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

    /// Oklab centre of each bin.
    pub fn centroids(&self) -> Vec<[f32; 3]> {
        let data = self.centroids.val().into_data();

        data.as_slice::<f32>()
            .expect("f32 centroids")
            .as_chunks::<3>()
            .0
            .to_vec()
    }

    /// Replaces the bin centres, which the trainer computes before training.
    pub fn with_centroids(mut self, centroids: &[[f32; 3]], device: &Device) -> Net {
        let flat: Vec<f32> = centroids.iter().flatten().copied().collect();
        let tensor = Tensor::from_data(TensorData::new(flat, [centroids.len(), 3]), device);

        self.centroids = Param::from_tensor(tensor).set_require_grad(false);
        self
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

    /// Bin logits, shape `[batch, bins]`.
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

    const TINY: NetConfig = NetConfig {
        log2_buckets: 10,
        dim: 8,
        hidden: 6,
        bins: 5,
        radius: 0.1,
        colour_radius: 0.05,
        temperature: 1.0,
    };

    #[test]
    fn config_round_trips_through_text() {
        let cfg = NetConfig {
            log2_buckets: 12,
            radius: 0.125,
            ..TINY
        };

        assert_eq!(NetConfig::parse(&cfg.to_text()), Some(cfg));
        assert_eq!(NetConfig::parse("dim=3"), None);
    }

    #[test]
    fn forward_shapes_and_empty_rows() {
        let device = Device::flex();
        let net = TINY.init(&device);
        let batch = net.batch(&["dark blue", "", "x"], &device);

        assert_eq!(batch.non_empty, vec![true, false, true]);

        let out = net.forward(batch);

        assert_eq!(out.dims(), [3, TINY.bins]);
    }

    #[test]
    fn centroids_are_replaceable() {
        let device = Device::flex();
        let centroids = [[0.1, 0.2, 0.3]; 5];
        let net = TINY.init(&device).with_centroids(&centroids, &device);

        assert_eq!(net.centroids(), centroids.to_vec());
    }
}
