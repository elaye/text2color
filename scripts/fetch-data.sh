#!/usr/bin/env bash
# Downloads the training/evaluation datasets into data/raw (not committed).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
raw="$root/data/raw"
mkdir -p "$raw"
cd "$raw"

fetch() {
    local url="$1" out="$2"
    if [ -s "$out" ]; then
        echo "have $out"
    else
        echo "fetching $url"
        curl -fsSL --retry 3 -o "$out.part" "$url"
        mv "$out.part" "$out"
    fi
}

fetch "https://xkcd.com/color/colorsurvey.tar.gz" colorsurvey.tar.gz
fetch "https://xkcd.com/color/rgb.txt" xkcd-rgb.txt
fetch "https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt" google-20k.txt
fetch "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/fr/fr_50k.txt" french-50k.txt

if [ ! -s mainsurvey_sqldump.txt ]; then
    echo "extracting colorsurvey.tar.gz"
    tar xzf colorsurvey.tar.gz
fi

ls -la "$raw"
