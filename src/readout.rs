//! Reading a colour and a confidence off a distribution over colour bins.
//!
//! Shared by the library and the trainer so predictions and targets are read the same way.

use crate::color::delta_e;

/// Weight of lightness differences in the agreement distance: the same colour in a lighter or
/// darker shade ("navy" and "sky" for "blue") counts as closer than a change of hue or chroma.
pub const LIGHTNESS_WEIGHT: f32 = 0.5;

/// Oklab distance with lightness differences scaled by [`LIGHTNESS_WEIGHT`].
pub fn agreement_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dl = (a[0] - b[0]) * LIGHTNESS_WEIGHT;

    (dl * dl + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn within(
    centroids: &[[f32; 3]],
    radius: f32,
    distance: fn([f32; 3], [f32; 3]) -> f32,
) -> Vec<Vec<usize>> {
    centroids
        .iter()
        .map(|&c| {
            centroids
                .iter()
                .enumerate()
                .filter(|(_, other)| distance(c, **other) <= radius)
                .map(|(j, _)| j)
                .collect()
        })
        .collect()
}

/// Colour bins plus, for each bin, the bins within the agreement radius and within the
/// smaller radius used to place the colour.
#[derive(Debug, Clone)]
pub struct Readout {
    centroids: Vec<[f32; 3]>,
    agreement: Vec<Vec<usize>>,
    nearby: Vec<Vec<usize>>,
}

impl Readout {
    /// `radius` is the [`agreement_distance`] within which two colours count as the same
    /// answer; `colour_radius` is the Oklab neighbourhood of the peak averaged into the colour.
    pub fn new(centroids: Vec<[f32; 3]>, radius: f32, colour_radius: f32) -> Readout {
        Readout {
            agreement: within(&centroids, radius, agreement_distance),
            nearby: within(&centroids, colour_radius, delta_e),
            centroids,
        }
    }

    pub fn centroids(&self) -> &[[f32; 3]] {
        &self.centroids
    }

    /// Oklab colour and confidence of a probability distribution over the bins.
    ///
    /// The broad mode is the bin whose agreement neighbourhood holds the most mass. Within that
    /// neighbourhood, the peak is the bin with the most mass close by, where the answers are
    /// densest; searching the whole distribution for it instead could jump to a smaller second
    /// mode. The colour is the mean of the bins close to the peak, and the confidence is the mass
    /// in the peak's agreement neighbourhood: the share of answers that agree with the returned
    /// colour.
    pub fn read(&self, p: &[f32]) -> ([f32; 3], f32) {
        let mass = |bins: &[usize]| bins.iter().map(|&j| p[j]).sum::<f32>();
        let argmax = |bins: &mut dyn Iterator<Item = usize>, of: &[Vec<usize>]| {
            bins.map(|j| (j, mass(&of[j])))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("at least one bin")
        };
        let (mode, _) = argmax(&mut (0..p.len()), &self.agreement);
        let (peak, near_mass) = argmax(&mut self.agreement[mode].iter().copied(), &self.nearby);
        let mut colour = [0f32; 3];

        for &j in &self.nearby[peak] {
            for (c, v) in colour.iter_mut().zip(self.centroids[j]) {
                *c += p[j] * v;
            }
        }

        (
            colour.map(|v| v / near_mass.max(1e-12)),
            mass(&self.agreement[peak]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line() -> Readout {
        Readout::new(
            vec![
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            0.15,
            0.15,
        )
    }

    #[test]
    fn peaked_distribution_is_confident() {
        let (colour, confidence) = line().read(&[0.45, 0.45, 0.05, 0.05]);

        assert!((confidence - 0.9).abs() < 1e-6);
        assert!((colour[0] - 0.05).abs() < 1e-6);
    }

    #[test]
    fn colour_comes_from_near_the_peak() {
        let r = Readout::new(
            vec![
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            0.25,
            0.05,
        );
        let (colour, confidence) = r.read(&[0.6, 0.2, 0.2, 0.0]);

        assert!((confidence - 1.0).abs() < 1e-6);
        assert!(colour[0].abs() < 1e-6);
    }

    #[test]
    fn small_second_mode_does_not_capture_the_colour() {
        let r = Readout::new(
            vec![
                [0.0, 0.0, 0.0],
                [0.04, 0.0, 0.0],
                [0.08, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            0.2,
            0.05,
        );
        let (colour, confidence) = r.read(&[0.25, 0.25, 0.25, 0.25]);

        assert!((confidence - 0.75).abs() < 1e-6);
        assert!(colour[0] < 0.1, "{colour:?}");
    }

    #[test]
    fn lightness_counts_half() {
        assert!((agreement_distance([0.2, 0.0, 0.0], [0.6, 0.0, 0.0]) - 0.2).abs() < 1e-6);
        assert!((agreement_distance([0.5, 0.1, 0.0], [0.5, 0.0, 0.0]) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn flat_distribution_is_not() {
        let (_, confidence) = line().read(&[0.25; 4]);

        assert!((confidence - 0.5).abs() < 1e-6);

        let (colour, confidence) = line().read(&[0.0, 0.0, 0.3, 0.7]);

        assert!((confidence - 0.7).abs() < 1e-6);
        assert!((colour[0] - 1.0).abs() < 1e-6);
    }
}
