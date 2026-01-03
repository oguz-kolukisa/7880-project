# Evaluation-Only Mode Usage

## Overview
The `--eval-only` flag allows you to evaluate existing trained models without retraining them.

## Usage Examples

### 1. Evaluate All Existing Checkpoints
```bash
python src/replicate_abcb.py \
    --data-root ./data \
    --output-dir ./output \
    --eval-only
```

### 2. Evaluate with Debug Mode (Limited Steps)
```bash
python src/replicate_abcb.py \
    --data-root ./data \
    --output-dir ./output \
    --eval-only \
    --debug
```

### 3. Evaluate and Push Results to Hugging Face
```bash
python src/replicate_abcb.py \
    --data-root ./data \
    --output-dir ./output \
    --eval-only \
    --push-to-hub \
    --hf-repo your-username/your-repo
```

## Behavior

**With `--eval-only` flag:**
- ✓ Loads existing checkpoints from `output-dir/{dataset}_{backbone}_{shots}shot/fold{fold}/model.pt`
- ✓ Runs evaluation on validation dataset
- ✓ Updates `replication_results.json` with new scores
- ✗ **Skips training entirely** (even if checkpoint doesn't exist)
- ⚠️ Warns and skips if checkpoint not found

**Without `--eval-only` flag (default behavior):**
- ✓ Loads checkpoint if exists
- ✓ Trains model if checkpoint doesn't exist
- ✓ Evaluates after training/loading

## Expected Directory Structure

```
output/
├── replication_results.json
├── pascal5i_resnet50_1shot/
│   ├── fold0/
│   │   ├── model.pt          ← Required for --eval-only
│   │   └── training_metrics.json
│   ├── fold1/
│   │   └── model.pt
│   ├── fold2/
│   │   └── model.pt
│   └── fold3/
│       └── model.pt
├── pascal5i_resnet50_5shot/
│   └── ...
└── ...
```

## Tips

1. **Skip Already Evaluated Models:**
   - Results are cached in `replication_results.json`
   - Already evaluated folds are skipped automatically
   - Delete specific entries from JSON to re-evaluate

2. **Force Re-evaluation:**
   ```bash
   # Remove all results
   rm output/replication_results.json
   
   # Then run eval-only
   python src/replicate_abcb.py --data-root ./data --output-dir ./output --eval-only
   ```

3. **Evaluate Specific Configurations:**
   Edit the script's `datasets`, `backbones`, `shots_list`, and `folds` variables to limit scope.

## Error Messages

- **"checkpoint not found"**: The model file doesn't exist at expected path
- **"Results exist for ... skipping"**: Already evaluated (check `replication_results.json`)
- **"dataset not available"**: Dataset loading failed

## Modes Comparison

| Mode | Training | Evaluation | Use Case |
|------|----------|------------|----------|
| Default | ✓ (if needed) | ✓ | Full pipeline |
| `--eval-only` | ✗ | ✓ | Test pretrained models |
| `--prepare-only` | ✗ | ✗ | Cache datasets |
| `--debug` | ✓ (2 steps) | ✓ (2 steps) | Quick testing |
