# text2color

Map a word or short phrase to a color, with a confidence score. Small enough for a
Raspberry Pi: one embedded 8 MB model, a few microseconds per query, no I/O.

```rust
let p = text2color::predict("dark teal");
println!("{} rgb={:?} oklab={:?} confidence={:.2}", p.hex(), p.rgb, p.oklab, p.confidence);
```

```
$ just query bleu
bleu    #3177cf  rgb( 49,119,207)  confidence 0.81
```

## What it does

- **Color names and modifiers**: "navy", "pale pink", "neon green", "reddish orange".
- **Typos**: "bleu", "purpel", "ornage" land on the intended color.
- **Associations**: "grass", "blood", "sky", "lemon", "chocolate".
- **Confidence**: how predictable a color is from the text. Near 1 for an unambiguous color
  name, lower for broad or noisy descriptions ("blue car"), near 0 for text that is not a color
  ("table", "monday").

There is no lexicon and no grammar. Everything above is learned from data.

## How it works

The model is a hashed character-n-gram network in the spirit of fastText, defined once as a
[burn](https://burn.dev) module (`src/net.rs`) and used unchanged for training and inference.
Inference runs on burn's pure-Rust `flex` CPU backend; weights are embedded as an f16 burnpack.

1. Text is lower-cased and split on anything non-alphanumeric.
2. Each token yields its whole-token feature plus character n-grams (2 to 5) of `<token>`;
   adjacent tokens add a bigram feature. Features are hashed into 2^17 buckets.
3. Bucket embeddings (32 dims) are mean-pooled, then a 64-unit ReLU layer predicts
   Oklab `L, a, b` and a confidence logit.

Oklab is used internally because distances there are perceptual, so the regression loss,
the averaging of survey answers and the spread-based confidence target all mean what they
should. The output is converted to sRGB with [`palette`](https://crates.io/crates/palette).

### Training data

- The [xkcd color survey](https://blog.xkcd.com/2010/05/03/color-survey-results/): 3.4 M
  free-text names people typed for random swatches. Answers from users flagged as likely spam
  or colorblind are dropped. Each distinct normalized string with at least two answers becomes a
  training row with its mean Oklab color and spread.
- The confidence target is a logistic on the string's spread relative to random swatches, plus
  a small term for how often the string was used. Strings everybody agrees on score high;
  "idk" and "ugly" score low.
- Negatives: words from a 20 k English frequency list that never appear in the survey, with
  confidence target 0 and no color loss.
- Augmentation during training: random typos (keep the color, confidence × 0.9) and noise
  words prepended or appended (keep the color, confidence × 0.7).

The 949-name xkcd list (CC0) is used only for evaluation. The raw survey dump is published by
xkcd without an explicit licence statement; only learned weights ship in this crate.

## Numbers

From `just eval` on the shipped model (x86_64 desktop, single thread):

| metric | value |
|---|---|
| hold-out mean Oklab error (confident strings) | 0.077 |
| xkcd 949 names, median / p90 Oklab error | 0.050 / 0.107 |
| confidence AUC, color strings vs non-words | 0.989 |
| typo robustness (one random edit keeps the color) | 84.6 % |
| latency per query | 3 µs |
| weights file (f16) | 8.4 MB |

A just-noticeable Oklab difference is about 0.02. Halving the hash table (`--log2-buckets 16`)
halves the file size at a small accuracy cost.

## Retraining

Requires `just`, `curl` and a Rust toolchain. Training runs on the CPU in well under an hour;
build the trainer with `--features cuda` and pass `--cuda` to use an NVIDIA GPU.

```
just fetch      # downloads ~90 MB into data/raw
just prepare    # aggregates the survey into data/derived (20 s)
just train      # writes model/text2color.bpk + .cfg; see `just train --help` for hyper-parameters
just eval       # metrics and a probe list, through the library's own inference path
just test
```

## Licence

MIT. See [LICENSE](LICENSE).
