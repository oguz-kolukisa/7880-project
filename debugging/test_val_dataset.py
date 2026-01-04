"""Quick test to verify validation dataset also works."""
import torch
import logging
from src.pascal5i import Pascal5iDataset

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Test validation dataset
val_ds = Pascal5iDataset(
    root="/mnt/j/Workspace/7880-project/data",
    fold=0,
    train=False,
    shots=1,
    queries=1,
    episodes=3,
    seed=321,
)

logging.info(f"Val label set: {val_ds.class_ids}")

for i in range(3):
    sample = val_ds[i]
    class_id = sample["class_id"].item()
    support_mask = sample["support_masks"]
    query_mask = sample["query_masks"]
    
    logging.info(f"Sample {i}: class={class_id}, support_mask fg={support_mask.float().mean():.3f}, query_mask fg={query_mask.float().mean():.3f}")
