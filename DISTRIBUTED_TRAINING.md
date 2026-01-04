# Distributed Training Guide

This guide explains how to distribute the replication experiments across multiple servers for parallel training.

## Total Combinations

With the current setup, you have **32 total combinations**:
- 2 datasets (pascal5i, coco20i)
- 2 backbones (resnet50, resnet101)
- 2 shot counts (1-shot, 5-shot)
- 4 folds (0, 1, 2, 3)

Total: 2 × 2 × 2 × 4 = **32 training runs**

## Strategy Recommendations

### Option 1: Split by Dataset and Backbone (4 servers)
Best for balanced workload distribution:

**Server 1:**
```bash
python src/replicate_abcb.py \
    --data-root /path/to/data \
    --output-dir /path/to/output \
    --datasets pascal5i \
    --backbones resnet50
```

**Server 2:**
```bash
python src/replicate_abcb.py \
    --data-root /path/to/data \
    --output-dir /path/to/output \
    --datasets pascal5i \
    --backbones resnet101
```

**Server 3:**
```bash
python src/replicate_abcb.py \
    --data-root /path/to/data \
    --output-dir /path/to/output \
    --datasets coco20i \
    --backbones resnet50
```

**Server 4:**
```bash
python src/replicate_abcb.py \
    --data-root /path/to/data \
    --output-dir /path/to/output \
    --datasets coco20i \
    --backbones resnet101
```

Each server handles 8 combinations (2 shots × 4 folds).

### Option 2: Split by Folds (4 servers)
Good if you want to parallelize within the same model:

**Server 1 (Fold 0):**
```bash
python src/replicate_abcb.py \
    --data-root /path/to/data \
    --output-dir /path/to/output \
    --folds 0
```

**Server 2 (Fold 1):**
```bash
python src/replicate_abcb.py \
    --data-root /path/to/data \
    --output-dir /path/to/output \
    --folds 1
```

**Server 3 (Fold 2):**
```bash
python src/replicate_abcb.py \
    --data-root /path/to/data \
    --output-dir /path/to/output \
    --folds 2
```

**Server 4 (Fold 3):**
```bash
python src/replicate_abcb.py \
    --data-root /path/to/data \
    --output-dir /path/to/output \
    --folds 3
```

Each server handles 8 combinations (2 datasets × 2 backbones × 2 shots).

### Option 3: Fine-grained Split (8 servers)
Maximum parallelization:

**Server 1:** `--datasets pascal5i --backbones resnet50 --shots 1`
**Server 2:** `--datasets pascal5i --backbones resnet50 --shots 5`
**Server 3:** `--datasets pascal5i --backbones resnet101 --shots 1`
**Server 4:** `--datasets pascal5i --backbones resnet101 --shots 5`
**Server 5:** `--datasets coco20i --backbones resnet50 --shots 1`
**Server 6:** `--datasets coco20i --backbones resnet50 --shots 5`
**Server 7:** `--datasets coco20i --backbones resnet101 --shots 1`
**Server 8:** `--datasets coco20i --backbones resnet101 --shots 5`

Each server handles 4 folds.

### Option 4: Ultra Fine-grained (32 servers)
One combination per server:

```bash
# Server 1
python src/replicate_abcb.py --datasets pascal5i --backbones resnet50 --shots 1 --folds 0

# Server 2
python src/replicate_abcb.py --datasets pascal5i --backbones resnet50 --shots 1 --folds 1

# ... and so on for all 32 combinations
```

## Shared Storage Setup

### Using NFS/Network Storage
All servers should mount the same network storage for:
1. **Data directory** (read-only): Share the dataset across servers
2. **Output directory** (read-write): All servers write to the same output folder

The script handles concurrent writes safely because:
- Each combination writes to its own subdirectory
- Results are merged in `replication_results.json` after each completion
- Checkpoint paths are unique per combination

```bash
# Mount shared storage on each server
mount -t nfs server:/shared/data /path/to/data
mount -t nfs server:/shared/output /path/to/output
```

### Using Cloud Storage (S3, GCS, etc.)
If using cloud storage:
1. Download data to each server locally (faster training)
2. Sync output directory after each fold completion:

```bash
# After training completes on each server
aws s3 sync /path/to/output s3://bucket/output --exclude "*.pt"  # Sync results
aws s3 sync /path/to/output s3://bucket/output --include "*.pt"  # Sync checkpoints
```

## Merging Results

If you run servers independently and need to merge results:

```python
import json
from pathlib import Path

def merge_results(output_dirs):
    merged = {"results": {}}
    
    for output_dir in output_dirs:
        results_file = Path(output_dir) / "replication_results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
                for key, value in data.get("results", {}).items():
                    if key not in merged["results"]:
                        merged["results"][key] = {}
                    merged["results"][key].update(value)
    
    # Calculate means
    for key in merged["results"]:
        if "mean" not in merged["results"][key]:
            fold_scores = [v["miou"] for k, v in merged["results"][key].items() 
                          if k.isdigit() and "miou" in v]
            if fold_scores:
                merged["results"][key]["mean"] = {
                    "miou": sum(fold_scores) / len(fold_scores)
                }
    
    with open("merged_results.json", "w") as f:
        json.dump(merged, f, indent=2)

# Usage
merge_results([
    "/server1/output",
    "/server2/output",
    "/server3/output",
    "/server4/output"
])
```

## Monitoring Progress

Each server logs its progress. To monitor:

```bash
# Check what's currently running on a server
tail -f train.log

# Count completed folds
find output -name "model.pt" | wc -l

# Check results so far
cat output/replication_results.json
```

## Resource Estimation

Per training run (approximate):
- **Time**: 30-60 minutes per fold (depending on GPU)
- **GPU Memory**: ~10-12 GB
- **Disk**: ~500 MB per checkpoint, ~50 MB for cached datasets

For all 32 combinations:
- **Total time**: ~32 hours on 1 GPU, ~1-2 hours on 32 GPUs
- **Total storage**: ~20 GB (checkpoints + datasets)

## Example Complete Setup

```bash
# Preparation (run once on one server)
python src/replicate_abcb.py \
    --data-root /shared/data \
    --output-dir /shared/output \
    --download \
    --prepare-only

# Server 1 (pascal5i + resnet50)
python src/replicate_abcb.py \
    --data-root /shared/data \
    --output-dir /shared/output \
    --datasets pascal5i \
    --backbones resnet50 \
    --log-level INFO \
    > server1.log 2>&1 &

# Server 2 (pascal5i + resnet101)
python src/replicate_abcb.py \
    --data-root /shared/data \
    --output-dir /shared/output \
    --datasets pascal5i \
    --backbones resnet101 \
    --log-level INFO \
    > server2.log 2>&1 &

# Server 3 (coco20i + resnet50)
python src/replicate_abcb.py \
    --data-root /shared/data \
    --output-dir /shared/output \
    --datasets coco20i \
    --backbones resnet50 \
    --log-level INFO \
    > server3.log 2>&1 &

# Server 4 (coco20i + resnet101)
python src/replicate_abcb.py \
    --data-root /shared/data \
    --output-dir /shared/output \
    --datasets coco20i \
    --backbones resnet101 \
    --log-level INFO \
    > server4.log 2>&1 &
```

## Troubleshooting

**Issue**: Servers overwriting each other's work
- **Solution**: Results are saved incrementally with unique paths per combination. No overwrites should occur.

**Issue**: Checkpoint conflicts
- **Solution**: Each fold has its own directory. If you see conflicts, ensure output directories are properly shared.

**Issue**: Different servers have different results.json
- **Solution**: Use shared storage or merge results afterward using the merge script above.

## Recommended: 4-Server Split by Dataset+Backbone

For most use cases, **Option 1** (4 servers split by dataset+backbone) is optimal because:
- Each server is independent (no conflicts)
- Balanced workload (~8 combinations each)
- Easy to set up and monitor
- Natural grouping (same model family together)
