default:
    @just --list

# Download raw datasets into data/raw
fetch:
    scripts/fetch-data.sh

# Parse and aggregate the raw survey into data/derived
prepare:
    cargo run --release -p text2color-train -- prepare

# Train and export model/text2color.bpk (+ .cfg sidecar)
train *args:
    cargo run --release -p text2color-train -- train {{args}}

# Evaluate model/text2color.bpk through the library inference path
eval *args:
    cargo run --release -p text2color-train -- eval {{args}}

test:
    cargo test --release --workspace

# Query the embedded model
query *args:
    cargo run --release --example query -- {{args}}

lint:
    cargo clippy --workspace --all-targets -- -D warnings
    cargo fmt --all --check
