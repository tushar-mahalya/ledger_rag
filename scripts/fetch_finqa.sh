#!/usr/bin/env bash
# Download the FinQA dataset (MIT license) at a pinned commit and verify checksums.
set -euo pipefail
COMMIT=0f16e2867befa6840783e58be38c9efb9229d742
DEST=${1:-data/finqa}
mkdir -p "$DEST"
for split in train dev test; do
  curl -sSfL -o "$DEST/$split.json" \
    "https://raw.githubusercontent.com/czyssrs/FinQA/$COMMIT/dataset/$split.json"
done
cd "$DEST"
sha256sum -c - <<'SUMS'
49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6  train.json
a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51  dev.json
831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc  test.json
SUMS
