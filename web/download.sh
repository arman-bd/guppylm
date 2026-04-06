#!/usr/bin/env bash
# Download model.onnx and tokenizer.json from HuggingFace into this directory.

set -e

REPO="arman-bd/guppylm-9M"
BASE="https://huggingface.co/${REPO}/resolve/main"
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Downloading from ${REPO}..."

curl -fSL "${BASE}/model.onnx" -o "${DIR}/model.onnx"
echo "  model.onnx     $(du -h "${DIR}/model.onnx" | cut -f1)"

curl -fSL "${BASE}/tokenizer.json" -o "${DIR}/tokenizer.json"
echo "  tokenizer.json $(du -h "${DIR}/tokenizer.json" | cut -f1)"

echo "Done. Run: cd web && python -m http.server 8080"
