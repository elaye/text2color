//! Colour bins: k-means centres in Oklab.

use linfa::DatasetBase;
use linfa::traits::Fit;
use linfa_clustering::KMeans;
use ndarray::Array2;
use text2color::color::{delta_e, rgb_to_oklab};

/// Grid steps per sRGB channel. The survey swatches were uniform in sRGB, so clustering a uniform
/// grid gives bins that each hold a similar share of the answers.
const GRID: u32 = 32;

/// `k` bin centres from k-means over a uniform sRGB grid, sorted by lightness.
pub fn colour_bins(k: usize) -> Vec<[f32; 3]> {
    let step = |i: u32| (i * 255 / (GRID - 1)) as u8;
    let mut points = Vec::with_capacity((GRID * GRID * GRID * 3) as usize);

    for r in 0..GRID {
        for g in 0..GRID {
            for b in 0..GRID {
                points.extend(rgb_to_oklab([step(r), step(g), step(b)]).map(f64::from));
            }
        }
    }

    let data = Array2::from_shape_vec((points.len() / 3, 3), points).expect("n x 3 points");
    let model = KMeans::params(k)
        .max_n_iterations(300)
        .tolerance(1e-6)
        .fit(&DatasetBase::from(data))
        .expect("k-means fits");
    let mut centroids: Vec<[f32; 3]> = model
        .centroids()
        .rows()
        .into_iter()
        .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
        .collect();

    centroids.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    centroids
}

/// Index of the centroid closest to `lab`.
pub fn nearest(centroids: &[[f32; 3]], lab: [f32; 3]) -> usize {
    (0..centroids.len())
        .min_by(|&a, &b| delta_e(centroids[a], lab).total_cmp(&delta_e(centroids[b], lab)))
        .expect("at least one centroid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bins_cover_black_and_white() {
        let bins = colour_bins(256);
        let black = rgb_to_oklab([0, 0, 0]);
        let white = rgb_to_oklab([255, 255, 255]);

        assert_eq!(bins.len(), 256);
        assert!(delta_e(bins[nearest(&bins, black)], black) < 0.15);
        assert!(delta_e(bins[nearest(&bins, white)], white) < 0.1);
        assert!(bins.windows(2).all(|w| w[0][0] <= w[1][0]));
    }
}
