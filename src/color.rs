//! sRGB <-> Oklab helpers on top of `palette`.

use palette::{Clamp, FromColor, IntoColor, Oklab, Srgb};

/// Multiplier applied to Oklab a/b so all three regression targets are roughly unit range.
pub const AB_SCALE: f32 = 3.0;

/// Converts 8-bit sRGB to Oklab `[L, a, b]`.
pub fn rgb_to_oklab(rgb: [u8; 3]) -> [f32; 3] {
    let lab: Oklab = Srgb::new(rgb[0], rgb[1], rgb[2])
        .into_format::<f32>()
        .into_color();

    [lab.l, lab.a, lab.b]
}

/// Converts Oklab to 8-bit sRGB, clamping out-of-gamut values.
pub fn oklab_to_rgb(lab: [f32; 3]) -> [u8; 3] {
    let rgb: Srgb<u8> = Srgb::from_color(Oklab::new(lab[0], lab[1], lab[2]))
        .clamp()
        .into_format();

    [rgb.red, rgb.green, rgb.blue]
}

/// Euclidean distance in Oklab (a just-noticeable difference is roughly 0.02).
pub fn delta_e(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_and_black() {
        let w = rgb_to_oklab([255, 255, 255]);
        let k = rgb_to_oklab([0, 0, 0]);

        assert!((w[0] - 1.0).abs() < 1e-3 && w[1].abs() < 1e-3 && w[2].abs() < 1e-3);
        assert!(k[0].abs() < 1e-3);
    }

    #[test]
    fn round_trip() {
        for rgb in [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [17, 200, 99],
            [128, 128, 128],
        ] {
            let back = oklab_to_rgb(rgb_to_oklab(rgb));

            for i in 0..3 {
                assert!(
                    (back[i] as i16 - rgb[i] as i16).abs() <= 1,
                    "{rgb:?} -> {back:?}"
                );
            }
        }
    }

    #[test]
    fn out_of_gamut_is_clamped() {
        let rgb = oklab_to_rgb([0.5, 0.9, 0.0]);

        assert_eq!(rgb[0], 255);
    }
}
