"""Debug dataset - check what the dataset actually returns."""
import torch
import logging
from src.pascal5i import Pascal5iDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    fold = 0
    
    # Build train dataset
    logging.info(f"Building Pascal5i TRAIN dataset for fold {fold}")
    train_ds = Pascal5iDataset(
        root="/mnt/j/Workspace/7880-project/data",
        fold=fold,
        train=True,
        shots=1,
        queries=1,
        episodes=5,
        seed=123,
    )
    
    logging.info(f"Train label set: {train_ds.class_ids}")
    logging.info(f"Val label set: {train_ds.reader.val_label_set}")
    
    # Get a few samples
    for i in range(3):
        logging.info(f"\n=== Train Sample {i} ===")
        sample = train_ds[i]
        class_id = sample["class_id"].item()
        support_mask = sample["support_masks"]
        query_mask = sample["query_masks"]
        
        logging.info(f"Class ID: {class_id}")
        logging.info(f"Support mask shape: {support_mask.shape}, unique: {torch.unique(support_mask).tolist()}")
        logging.info(f"Query mask shape: {query_mask.shape}, unique: {torch.unique(query_mask).tolist()}")
        
        # Check the raw target from reader
        support_idx = i % len(train_ds.reader)
        img, target = train_ds.reader[support_idx]
        logging.info(f"Raw target from reader: unique values = {torch.unique(target).tolist()}")
        logging.info(f"Does target contain class_id {class_id}? {(target == class_id).any().item()}")
        
        # Try manually building the mask
        manual_mask = (target == class_id).long()
        logging.info(f"Manual mask (target == {class_id}): unique = {torch.unique(manual_mask).tolist()}, fg_ratio = {manual_mask.float().mean().item():.4f}")

if __name__ == "__main__":
    main()
