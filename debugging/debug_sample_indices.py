"""Debug dataset - check _sample_indices method."""
import torch
import logging
import random
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
    
    # Manually test _sample_indices
    rng = random.Random(123)
    class_id = rng.choice(train_ds.class_ids)
    
    logging.info(f"\n=== Testing _sample_indices for class_id={class_id} ===")
    candidates = list(train_ds.reader.get_img_containing_class(class_id))
    logging.info(f"Number of candidate images: {len(candidates)}")
    logging.info(f"First 10 candidate indices: {candidates[:10]}")
    
    support_indices, query_indices = train_ds._sample_indices(rng, class_id)
    logging.info(f"Support indices: {support_indices}")
    logging.info(f"Query indices: {query_indices}")
    
    # Check what's in these images
    for idx in support_indices:
        img, target = train_ds.reader[idx]
        logging.info(f"\nSupport image idx={idx}:")
        logging.info(f"  Target unique values: {torch.unique(target).tolist()}")
        logging.info(f"  Contains class {class_id}? {(target == class_id).any().item()}")
        
        # Build binary mask
        binary_mask = train_ds._build_binary_mask(target, class_id)
        logging.info(f"  Binary mask unique: {torch.unique(binary_mask).tolist()}, fg_ratio: {binary_mask.float().mean().item():.4f}")
    
    for idx in query_indices:
        img, target = train_ds.reader[idx]
        logging.info(f"\nQuery image idx={idx}:")
        logging.info(f"  Target unique values: {torch.unique(target).tolist()}")
        logging.info(f"  Contains class {class_id}? {(target == class_id).any().item()}")
        
        # Build binary mask
        binary_mask = train_ds._build_binary_mask(target, class_id)
        logging.info(f"  Binary mask unique: {torch.unique(binary_mask).tolist()}, fg_ratio: {binary_mask.float().mean().item():.4f}")

if __name__ == "__main__":
    main()
