#!/usr/bin/env python3
"""Quick debug: test model output without forward pass."""

import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

# Test IoU calculation directly
print("=== Testing IoU Calculation ===\n")

# Create dummy logits and GT
B, H, W = 2, 473, 473
logits = torch.randn(B, 2, H, W)  # 2-class output
query_mask = torch.randint(0, 2, (B, 1, H, W)).float()  # Binary GT

print(f"Logits shape: {logits.shape}, range: [{logits.min():.2f}, {logits.max():.2f}]")
print(f"Query mask shape: {query_mask.shape}, dtype: {query_mask.dtype}")
print(f"Query mask unique values: {torch.unique(query_mask)}")
print()

# Simulate binary_miou_from_logits
pred = logits.argmax(dim=1)  # [B, H, W]
gt = (query_mask[:, 0] > 0.5).long()  # [B, H, W]

print(f"Prediction shape: {pred.shape}, unique values: {torch.unique(pred)}")
print(f"GT shape: {gt.shape}, unique values: {torch.unique(gt)}")
print()

inter = ((pred == 1) & (gt == 1)).sum().item()
union = ((pred == 1) | (gt == 1)).sum().item()
iou = inter / (union + 1e-6)

print(f"Intersection (pred=1 AND gt=1): {inter}")
print(f"Union (pred=1 OR gt=1): {union}")
print(f"IoU = {inter} / {union} = {iou:.6f}")
print()

# Test upsampling
print("=== Testing Shape Mismatch Handling ===\n")
logits_low = torch.randn(2, 2, 237, 237)  # Lower res
query_mask_hi = torch.randint(0, 2, (2, 1, 473, 473)).float()  # Higher res

print(f"Logits: {logits_low.shape}, Query mask: {query_mask_hi.shape}")

# Upsample logits
logits_up = torch.nn.functional.interpolate(logits_low, size=query_mask_hi.shape[-2:], mode="nearest")
print(f"Upsampled logits: {logits_up.shape}")
print("✓ Upsampling works correctly")
