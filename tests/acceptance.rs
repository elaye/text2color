//! Semantic expectations against the shipped model. Thresholds are deliberately loose: the
//! model is learned, so these guard behaviour, not exact values.

use text2color::{Prediction, predict};

fn hue(p: &Prediction) -> f32 {
    p.oklab[2].atan2(p.oklab[1]).to_degrees().rem_euclid(360.0)
}

fn chroma(p: &Prediction) -> f32 {
    p.oklab[1].hypot(p.oklab[2])
}

fn hue_within(text: &str, lo: f32, hi: f32) {
    let p = predict(text);
    let h = hue(&p);
    let ok = if lo <= hi {
        (lo..=hi).contains(&h)
    } else {
        h >= lo || h <= hi
    };

    assert!(
        ok && chroma(&p) > 0.04,
        "{text}: hue {h:.0} chroma {:.3} {}",
        chroma(&p),
        p.hex()
    );
}

fn confident(text: &str, min: f32) {
    let c = predict(text).confidence;

    assert!(c >= min, "{text}: confidence {c:.2} < {min}");
}

fn unconfident(text: &str, max: f32) {
    let c = predict(text).confidence;

    assert!(c <= max, "{text}: confidence {c:.2} > {max}");
}

fn same_hue(a: &str, b: &str, tolerance: f32) {
    let (pa, pb) = (predict(a), predict(b));
    let diff = (hue(&pa) - hue(&pb)).abs();
    let diff = diff.min(360.0 - diff);

    assert!(
        diff <= tolerance,
        "{a} ({}) vs {b} ({}): hue differs by {diff:.0}",
        pa.hex(),
        pb.hex()
    );
}

// Confidence is the share of people who would pick a colour close to the prediction. Callers
// trust it from 0.5; a prediction carrying no information still scores about 0.2.
const GATE: f32 = 0.5;
const NOT_A_COLOUR: f32 = 0.35;

// Approximate Oklab hue sectors, in degrees.
const RED: (f32, f32) = (10.0, 45.0);
const ORANGE: (f32, f32) = (45.0, 75.0);
const YELLOW: (f32, f32) = (90.0, 125.0);
const GREEN: (f32, f32) = (125.0, 170.0);
const BLUE: (f32, f32) = (230.0, 280.0);
const PURPLE: (f32, f32) = (290.0, 330.0);
const PINK: (f32, f32) = (330.0, 20.0);

#[test]
fn basic_color_terms() {
    hue_within("red", RED.0, RED.1);
    hue_within("orange", ORANGE.0, ORANGE.1);
    hue_within("yellow", YELLOW.0, YELLOW.1);
    hue_within("green", GREEN.0, GREEN.1);
    hue_within("blue", BLUE.0, BLUE.1);
    hue_within("purple", PURPLE.0, PURPLE.1);
    hue_within("pink", PINK.0, PINK.1);

    for word in [
        "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white",
        "grey", "gray",
    ] {
        confident(word, GATE);
    }

    assert!(predict("black").oklab[0] < 0.35);
    assert!(predict("white").oklab[0] > 0.85);
    assert!(chroma(&predict("grey")) < 0.04);
}

#[test]
fn typos_keep_the_hue() {
    same_hue("bleu", "blue", 25.0);
    same_hue("gren", "green", 25.0);
    same_hue("purpel", "purple", 25.0);
    same_hue("yelow", "yellow", 25.0);
    same_hue("ornage", "orange", 25.0);

    for word in ["bleu", "gren", "purpel", "yelow", "ornage"] {
        confident(word, GATE);
    }
}

#[test]
fn modifiers_shift_lightness_and_chroma() {
    let dark = predict("dark blue").oklab[0];
    let plain = predict("blue").oklab[0];
    let light = predict("light blue").oklab[0];

    assert!(
        dark < plain && plain < light,
        "L: dark {dark:.2} plain {plain:.2} light {light:.2}"
    );
    assert!(chroma(&predict("pale pink")) < chroma(&predict("hot pink")));
    same_hue("dark blue", "blue", 30.0);
}

#[test]
fn common_associations() {
    hue_within("sky", 200.0, 280.0);
    hue_within("grass", GREEN.0, GREEN.1);
    hue_within("blood", RED.0, RED.1);
    hue_within("lemon", YELLOW.0, YELLOW.1);
    hue_within("chocolate", 20.0, 75.0);
    assert!(predict("chocolate").oklab[0] < 0.5);

    for word in ["sky", "grass", "blood", "lemon", "chocolate"] {
        confident(word, GATE);
    }
}

#[test]
fn non_colors_are_rejected() {
    for text in [
        "table",
        "tomorrow",
        "running",
        "asdfgh",
        "",
        "   ",
        "meeting at the office tomorrow",
    ] {
        unconfident(text, NOT_A_COLOUR);
    }
}

#[test]
fn noise_words_are_tolerated() {
    let bare = predict("dark blue");
    let noisy = predict("a dark blue");

    assert!(text2color::color::delta_e(bare.oklab, noisy.oklab) < 0.1);
    same_hue("blue car", "blue", 30.0);
}

#[test]
fn hex_formatting() {
    let p = predict("red");

    assert_eq!(p.hex().len(), 7);
    assert!(p.hex().starts_with('#'));
}

#[test]
fn french_color_terms() {
    hue_within("rouge", RED.0, RED.1);
    hue_within("jaune", YELLOW.0, YELLOW.1);
    hue_within("vert", GREEN.0, GREEN.1);
    hue_within("bleu", BLUE.0, BLUE.1);
    hue_within("violet", PURPLE.0, PURPLE.1);
    hue_within("rose", PINK.0, PINK.1);
    same_hue("vert foncé", "dark green", 30.0);
    same_hue("vert fonce", "vert foncé", 15.0);
    same_hue("bleu marine", "navy blue", 30.0);

    for word in ["rouge", "jaune", "vert", "bleu", "noir", "blanc", "gris"] {
        confident(word, GATE);
    }

    assert!(predict("noir").oklab[0] < 0.35);
    assert!(predict("blanc").oklab[0] > 0.85);
    assert!(predict("bleu foncé").oklab[0] < predict("bleu clair").oklab[0]);
}

#[test]
fn french_associations_and_rejections() {
    hue_within("herbe", GREEN.0, GREEN.1);
    hue_within("sang", RED.0, RED.1);
    hue_within("citron", YELLOW.0, YELLOW.1);

    for text in ["lundi", "demain", "la réunion de demain au bureau"] {
        unconfident(text, NOT_A_COLOUR);
    }
}

fn blue_grey(text: &str) {
    let p = predict(text);
    let (h, c) = (hue(&p), chroma(&p));

    assert!(
        (200.0..=290.0).contains(&h) && (0.01..=0.1).contains(&c),
        "{text}: hue {h:.0} chroma {c:.3} {}",
        p.hex()
    );
}

#[test]
fn weather_associations() {
    for word in ["ensoleillé", "sunny"] {
        hue_within(word, 80.0, YELLOW.1);
    }

    for word in ["pluvieux", "rainy"] {
        blue_grey(word);
    }

    for word in ["chaud", "hot"] {
        hue_within(word, RED.0, ORANGE.1);
    }

    for word in ["frais", "cool"] {
        hue_within(word, 180.0, 280.0);
    }

    for word in [
        "ensoleillé",
        "sunny",
        "pluvieux",
        "rainy",
        "chaud",
        "hot",
        "frais",
        "cool",
    ] {
        confident(word, GATE);
    }
}

#[test]
fn everyday_words_stay_below_the_gate() {
    for word in [
        "famille", "amour", "graine", "horloge", "family", "love", "seed", "clock",
    ] {
        unconfident(word, GATE);
    }
}
