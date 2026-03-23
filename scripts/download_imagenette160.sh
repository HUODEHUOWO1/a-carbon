#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-./data/vision_proxy}"
mkdir -p "$OUT_DIR"
curl -L "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz" -o "$OUT_DIR/imagenette2-160.tgz"
tar -xzf "$OUT_DIR/imagenette2-160.tgz" -C "$OUT_DIR"
echo "Extracted to $OUT_DIR"
