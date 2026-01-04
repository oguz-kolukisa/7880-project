#!/bin/bash
# Training script for Pascal-5i with ResNet50
# Handles both 1-shot and 5-shot experiments for fold 0
# Total: 2 training runs (2 shots × 1 fold)

set -e

# Configuration - UPDATE THESE PATHS
DATA_ROOT="./data"
OUTPUT_DIR="./output"
DEVICE="cuda"
HF_REPO="okolukisa1/7880-project"
HF_TOKEN="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"  # Set your Hugging Face token here or use environment variable

# Authenticate with Hugging Face Hub
echo "Checking Hugging Face authentication..."
if [ -n "$HF_TOKEN" ]; then
    echo "Using HF_TOKEN environment variable"
    hf auth login --token "$HF_TOKEN" || echo "Warning: HF authentication failed, --push-to-hub will not work"
else
    echo "No HF_TOKEN found, checking existing authentication"
    hf auth whoami || echo "Warning: Not authenticated with HF, --push-to-hub will not work"
fi

# Hugging Face Hub integration enabled
PUSH_ARGS="--push-to-hub --hf-repo $HF_REPO"

echo "======================================"
echo "Pascal-5i + ResNet50 Training"
echo "======================================"
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "======================================"

python src/replicate_abcb.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --datasets pascal5i \
    --backbones resnet50 \
    --folds 0 \
    --log-level INFO \
    $PUSH_ARGS

echo "======================================"
echo "Pascal-5i + ResNet50 Complete!"
echo "======================================"
