# text2color

Map an English or French word or short phrase to a color, with a confidence score. Small
enough for a Raspberry Pi: one embedded 9 MB model, under a millisecond per query, no I/O.

```rust
let p = text2color::predict("dark teal");
println!("{} rgb={:?} oklab={:?} confidence={:.2}", p.hex(), p.rgb, p.oklab, p.confidence);
```

```
$ just query bleu
bleu    #236dd0  rgb( 35,109,208)  confidence 0.82
```

## What it does

- **Color names and modifiers**: "navy", "pale pink", "neon green", "reddish orange".
- **French**: "vert foncé", "bleu marine", "rose pâle", "herbe", with or without accents.
- **Typos**: "bleu", "purpel", "ornage" land on the intended color.
- **Associations**: "grass", "blood", "ciel", "rouille", "ensoleillé", "rainy", "hot", with
  inflections ("bleues", "rusty") and missing accents ("ecarlate").
- **Confidence**: the share of people who would pick a color close to the returned one. A clear
  color name scores 0.8 to 1, a word most people would color the same way ("sand", "neige")
  above 0.5, and text with no particular color ("table", "famille", "love") around 0.2, which is
  what a guess at the most common answer scores. Callers typically trust a prediction from 0.5.

There is no lexicon and no grammar in the library. Everything above is learned from data.

## How it works

The model predicts, for any text, the distribution of colors people would pick for it, over
1024 color bins (k-means centres in Oklab). It is a hashed character-n-gram network in the
spirit of fastText, defined once as a [burn](https://burn.dev) module (`src/net.rs`) and used
unchanged for training and inference. Inference runs on burn's pure-Rust `flex` CPU backend;
weights, bin centres and settings are embedded as an f16 burnpack and a small `.cfg`.

1. Text is lower-cased and split on anything non-alphanumeric.
2. Each token yields its whole-token feature plus character n-grams (2 to 5) of `<token>`;
   adjacent tokens add a bigram feature. Features are hashed into 2^17 buckets.
3. Bucket embeddings (32 dims) are mean-pooled, then a 256-unit ReLU layer gives one logit per
   color bin.

The color and the confidence are both read off the distribution (`src/readout.rs`). Two
colors *agree* when their Oklab distance, with lightness counted at half weight, is under
0.125: "navy" and "sky blue" are both acceptable answers for "blue", "teal" is not. The broad
mode is the bin whose agreeing neighbourhood holds the most probability; within it, the color
is the mean of the bins around the densest point, and the confidence is the probability that
agrees with that color. The output is converted to sRGB with
[`palette`](https://crates.io/crates/palette).

### Training data

Every training string has a target distribution built the same way:

> survey answers + 20 × calibrated color label (if the word has one) + 2 × all answers pooled

- **The [xkcd color survey](https://blog.xkcd.com/2010/05/03/color-survey-results/)**: 3.4 M
  free-text names people typed for random swatches. Answers from users flagged as likely spam
  or colorblind are dropped. Each distinct normalized string with at least two answers keeps
  the histogram of its answers over the color bins.
- **French**: `train/translations/en-fr.tsv` holds machine translations of the 4000
  most-answered survey strings. Each French string inherits the answers of its English source,
  merged with any answers it already had (the survey contains some French, "bleu" and "rouge"
  among them).
- **Color labels for everyday words**: the 50 000 most frequent words of English and of French
  (82 886 after merging) were labelled once, offline, by Claude Sonnet subagents in Claude Code,
  following `train/labels/prompt.md`. For each word the label gives the share of people who
  would picture each of about 180 survey color names (`train/labels/palette.txt`), and the
  share who would picture no particular color. Sensory associations count (temperature,
  weather, light); symbolic ones do not ("love" is not red). A palette name stands for all the
  answers the survey collected for it, and the "no color" share for all answers pooled. The
  labels are calibrated against the survey on the 693 labelled words with at least 30 survey
  answers. The merged labels are committed in `train/labels/full-sonnet.tsv`; a pilot comparing
  Haiku and Sonnet labels against the survey is in `train/labels/pilot-*.tsv`.
- **Weighting**: each string weighs the square root of its total answers and pseudo-answers, so
  a bare "blue" with 270 k answers outweighs the many phrases sharing its n-grams.
- **Augmentation**: accents dropped on half the accented examples, random typos and noise
  words, each moving part of the target toward the pooled answers (10 % and 30 %), since they
  make the intended color less certain. Random pairs of words get the pooled answers as target.
- **Temperature**: one softmax temperature is fitted on a calibration fold of the survey so
  that confidences match measured agreement.

The 949-name xkcd list (CC0) and the 148 CSS named colors (from `palette::named`) are used only
for evaluation. CSS names are split into survey words first ("darkslategray" becomes "dark slate
gray"). The raw survey dump is published by xkcd without an explicit licence statement; only
learned weights ship in this crate.

## Numbers

From `just eval` on the shipped model (x86_64 desktop, single thread):

| metric | value |
|---|---|
| hold-out mean Oklab error (survey strings people agree on) | 0.088 |
| xkcd 949 names, median / p90 Oklab error | 0.044 / 0.094 |
| xkcd names in French, median / p90 Oklab error | 0.044 / 0.094 |
| CSS 148 named colors (unseen), median / p90 Oklab error | 0.093 / 0.226 |
| hold-out labelled words, gate agrees with the label | 96.5 % |
| hold-out words without a color passing 0.5 | 1.7 % |
| typo robustness (one random edit keeps the color) | 70.5 % |
| latency per query | 82 µs |
| weights file (f16) | 8.9 MB |

A just-noticeable Oklab difference is about 0.02. Halving the hash table (`--log2-buckets 16`)
halves the file size at a small accuracy cost.

## Retraining

Requires `just`, `curl` and a Rust toolchain. Training runs on the CPU in under an hour;
build the trainer with `--features cuda` and pass `--cuda` to use an NVIDIA GPU.

```
just fetch      # downloads ~90 MB into data/raw
just prepare    # color bins and survey histograms into data/derived
just train      # writes model/text2color.bpk + .cfg; see `just train --help` for hyper-parameters
just eval       # metrics and a probe list, through the library's own inference path
just test
```

Relabelling words is only needed to change the prompt or the vocabulary. `just labels batches`
splits the vocabulary into batches under `data/labels-work/full`, Claude Code subagents write
one result file per batch, each checked with `labels validate`, and `just labels check full
sonnet --samples 1` merges them and lists any batch still missing.

## Licence

MIT. See [LICENSE](LICENSE).
