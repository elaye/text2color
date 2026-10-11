# Colour labels

You are labelling words with the colour people picture for them. The labels train a small model
that turns any text into a colour, with a confidence that says how many people would agree.

## Task

For every word in the batch file, estimate what a large group of ordinary English and French
speakers would answer if each was shown the word alone and asked: *"What colour is this?
Pick one colour, or say that it has none."*

Answer with the share of people, in percent, who would picture each colour, and the share who
would say the word has no particular colour.

- Use only names from the palette file. Pick at most three names; merge close shades into the
  nearest single name rather than splitting them.
- `none` is the share who would not picture any consistent colour. For most words (`family`,
  `horloge`, `seed`, `run`, `the`, `pourquoi`) that is 100 or close to it.
- Shares are whole numbers and add up to 100.
- Judge the colour people **see or feel** in the word:
  - the physical colour of the thing it names: `sea` is blue, `rouille` is rust, `charcoal` is
    black or dark grey, `banana` is yellow;
  - sensory associations, which most people share: temperature (`scorching` is red or orange,
    `glacial` is icy blue), weather and light (`sunny` is yellow, `stormy` is dark grey, `dusk`
    is deep blue), and how light or dark something is (`pale`, `gloomy`).
- Symbolic or cultural associations do not count: `love`, `anger`, `danger`, `envy`, `money`,
  `luck` and `christmas` get `none`, not red or green.
- A colour is only a real answer if most people would converge on it. Things that come in many
  colours (`car`, `shirt`, `flower`, `paint`) are mostly `none`.
- Words can be English or French, inflected (`bleue`, `vertes`, `ensoleillée`, `darker`,
  `rusty`) or missing their accents (`ensoleille`, `dore`, `ecarlate`). Read them as the word
  they come from. A word that exists in both languages (`rose`, `bride`, `pain`) mixes both
  senses by how often each is meant.
- Names of people, places and brands get `none` unless the name itself evokes one colour.
- After the shares, give the hex value (`#rrggbb`) of the single colour most people would
  picture, or `-` if `none` is above 50.

Do not look anything up and do not write code to produce labels: answer from your own knowledge,
word by word.

## Output

One line per word of the batch, in the batch's order, with the word spelled exactly as in the
batch, and nothing else in the file:

```
word<TAB>name:share,name:share,none:share<TAB>#rrggbb
```

Examples:

```
sea	blue:40,ocean blue:25,teal:15,none:20	#1f5fa8
neige	white:90,none:10	#fafafa
rouille	rust:65,reddish brown:20,none:15	#a8501e
sunny	yellow:70,pale yellow:10,none:20	#ffd83a
famille	none:100	-
lave	red orange:45,orange:25,none:30	#e2481c
olive	olive green:60,olive:20,none:20	#6f7a2a
love	none:95,red:5	-
```

## Checking your file

When the file is written, run:

```
target/release/text2color-train labels validate <batch file> <your file>
```

Fix every problem it reports and run it again until it prints `ok`.
