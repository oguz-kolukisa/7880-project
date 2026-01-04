#!/usr/bin/env python3
"""Diagnostic script to identify why IoU is not improving during training."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn.functional as F
from src.train_abcb import abcb_loss, binary_miou_from_logits

print("=" * 70)
print("TRAINING DIAGNOSTICS: Loss vs IoU")
print("=" * 70)

# Simulate different prediction scenarios
B, H, W = 2, 100, 100
gt = torch.randint(0, 2, (B, 1, H, W)).float()  # Ground truth

def perfect_pred():
    logits = torch.zeros(B, 2, H, W)
    gt_mask = gt[:, 0]
    logits[:, 0] = torch.where(gt_mask == 0, 5.0, -5.0)  # Class 0 high where gt=0
    logits[:, 1] = torch.where(gt_mask == 1, 5.0, -5.0)  # Class 1 high where gt=1
    return logits

scenarios = {
    "1. Perfect prediction": perfect_pred,
    
    "2. All background (0)": lambda: torch.stack([torch.ones(B, H, W) * 5.0, 
                                                   torch.ones(B, H, W) * -5.0], dim=1),
    
    "3. All foreground (1)": lambda: torch.stack([torch.ones(B, H, W) * -5.0, 
                                                   torch.ones(B, H, W) * 5.0], dim=1),
    
    "4. Random predictions": lambda: torch.randn(B, 2, H, W),
    
    "5. Close to random (50/50)": lambda: torch.randn(B, 2, H, W) * 0.1,
}

print("\nGround Truth Statistics:")
print(f"  Total pixels: {B * H * W}")
print(f"  Foreground (1): {(gt == 1).sum().item()} pixels ({100 * (gt == 1).float().mean().item():.1f}%)")
print(f"  Background (0): {(gt == 0).sum().item()} pixels ({100 * (gt == 0).float().mean().item():.1f}%)")
print()

for name, pred_fn in scenarios.items():
    logits = pred_fn()
    
    # Calculate loss
    P_list = [logits]
    Phat_list = [logits]
    loss = abcb_loss(P_list, Phat_list, gt, lam=0.2)
    
    # Calculate IoU
    iou = binary_miou_from_logits(logits, gt)
    
    # Get predictions
    pred = logits.argmax(dim=1)
    fg_pixels = (pred == 1).sum().item()
    bg_pixels = (pred == 0).sum().item()
    
    print(f"{name}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  IoU:  {iou:.4f}")
    print(f"  Pred: FG={fg_pixels} ({100*fg_pixels/(B*H*W):.1f}%), BG={bg_pixels} ({100*bg_pixels/(B*H*W):.1f}%)")
    print()

print("=" * 70)
print("COMMON ISSUES & SOLUTIONS:")
print("=" * 70)
print("""
1. **Model predicting mostly one class (all 0s or all 1s)**
   - Symptom: Loss decreasing but IoU stays at 0
   - Cause: Class imbalance or model collapse
   - Solution:
     * Check class balance in dataset
     * Use weighted loss (e.g., focal loss)
     * Lower learning rate
     * Check if support masks are properly loaded

2. **Loss and IoU both stuck**
   - Symptom: Loss plateaus, IoU doesn't improve
   - Cause: Model not learning useful features
   - Solution:
     * Verify data augmentation not too aggressive
     * Check if backbone is frozen (should be trainable)
     * Verify support/query masks are correct
     * Try smaller crop size or less augmentation

3. **Loss decreasing but IoU fluctuating near 0**
   - Symptom: Loss goes down but IoU stays 0-0.1
   - Cause: Model learning to predict background
   - Solution:
     * Check if ground truth has enough foreground pixels
     * Verify query_mask format matches predictions
     * Add IoU-based loss component
     * Use larger λ for auxiliary loss

4. **Gradient issues**
   - Symptom: Loss becomes NaN or stays constant
   - Cause: Gradient explosion/vanishing
   - Solution:
     * Check gradient norms (already clipped at 10.0)
     * Lower learning rate
     * Check for NaN in data

5. **Validation different from training**
   - Symptom: Training loss improves but val IoU doesn't
   - Cause: Overfitting to augmented data
   - Solution:
     * Reduce augmentation strength
     * Check if validation uses same preprocessing
""")

print("\nRECOMMENDED DEBUGGING STEPS:")
print("-" * 70)
print("""
1. Add prediction visualization during training:
   - Save prediction masks from first few batches
   - Compare with ground truth visually

2. Monitor class distribution:
   - Log % of foreground predictions per batch
   - Check if model is predicting varied outputs

3. Check loss components:
   - Log P_list and Phat_list losses separately
   - See which one is dominant

4. Verify data loading:
   - Print support_mask and query_mask statistics
   - Check unique values and shapes
   - Ensure masks are {0, 1} not {0, 255}

5. Add IoU to training loop:
   - Calculate IoU during training (not just validation)
   - Monitor if training IoU is improving
""")
