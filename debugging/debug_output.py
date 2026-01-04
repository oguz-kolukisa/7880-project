#!/usr/bin/env python3
"""Debug script to validate model output and IoU calculation."""

import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

# Add project to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.abcb import ABCB
from src.pascal5i import Pascal5iDataset
from src.train_abcb import binary_miou_from_logits, unpack_episode


def debug_output():
    """Test model output shapes and IoU calculation."""
    logging.info("Starting model output debug")
    
    # Load a small dataset
    logging.info("Loading Pascal5i dataset (fold=0, train=False, episodes=2)")
    val_ds = Pascal5iDataset(root="./data", fold=0, train=False, episodes=2, seed=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    
    # Create model
    logging.info("Creating ABCB model with ResNet-50 backbone")
    model = ABCB(backbone_name="resnet50", pretrained_backbone=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    device = next(model.parameters()).device
    logging.info(f"Model device: {device}")
    
    # Test forward pass
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            logging.info(f"\n=== Batch {batch_idx} ===")
            support_img, support_mask, query_img, query_mask = unpack_episode(batch)
            
            logging.info(f"Raw batch shapes:")
            logging.info(f"  support_img: {support_img.shape}, dtype={support_img.dtype}")
            logging.info(f"  support_mask: {support_mask.shape}, dtype={support_mask.dtype}")
            logging.info(f"  query_img: {query_img.shape}, dtype={query_img.dtype}")
            logging.info(f"  query_mask: {query_mask.shape}, dtype={query_mask.dtype}")
            logging.info(f"  query_mask unique values: {torch.unique(query_mask)}")
            
            # Move to device
            support_img = support_img.to(device)
            support_mask = support_mask.to(device)
            query_img = query_img.to(device)
            query_mask = query_mask.to(device)
            
            # Forward pass
            try:
                out = model(
                    query_img=query_img,
                    support_img=support_img,
                    support_mask=support_mask,
                    return_all=False,
                )
                
                logits = out["logits"]
                logging.info(f"\nModel output:")
                logging.info(f"  logits shape: {logits.shape}")
                logging.info(f"  logits dtype: {logits.dtype}")
                logging.info(f"  logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                
                # Test IoU calculation
                logging.info(f"\nTesting IoU calculation:")
                iou = binary_miou_from_logits(logits, query_mask)
                logging.info(f"  IoU: {iou:.4f}")
                
                # Manual check: compute prediction
                if logits.shape[1] == 2:
                    pred = logits.argmax(dim=1)
                else:
                    pred = logits[:, 0]
                
                gt = (query_mask[:, 0] > 0.5).long() if query_mask.dim() == 4 else query_mask.long()
                
                logging.info(f"  Prediction shape: {pred.shape}, dtype: {pred.dtype}")
                logging.info(f"  GT shape: {gt.shape}, dtype: {gt.dtype}")
                logging.info(f"  Pred unique values: {torch.unique(pred)}")
                logging.info(f"  GT unique values: {torch.unique(gt)}")
                
                inter = ((pred == 1) & (gt == 1)).sum().item()
                union = ((pred == 1) | (gt == 1)).sum().item()
                logging.info(f"  Manual IoU: inter={inter}, union={union}, iou={inter/(union+1e-6):.4f}")
                
                if batch_idx >= 1:
                    break
                    
            except Exception as e:
                logging.error(f"Error during forward pass: {e}", exc_info=True)
                break
    
    logging.info("\n=== Debug complete ===")


if __name__ == "__main__":
    debug_output()
