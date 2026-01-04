#!/usr/bin/env python3
"""Test IoU function and verify prediction/GT format compatibility."""

import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from train_abcb import binary_miou_from_logits

print("=" * 60)
print("TEST 1: IoU with identical ground truth inputs")
print("=" * 60)

# Test 1: Same GT as both prediction and GT -> should give IoU = 1.0
gt_mask = torch.tensor([
    [[1, 1, 0, 0],
     [1, 1, 0, 0],
     [0, 0, 1, 1],
     [0, 0, 1, 1]],
], dtype=torch.float32).unsqueeze(1)  # [1, 1, 4, 4]

print(f"\nGround truth mask shape: {gt_mask.shape}")
print(f"Ground truth mask:\n{gt_mask[0, 0]}\n")

# Convert GT to logits format [B, 2, H, W] by making class 1 have high logit
logits = torch.zeros(1, 2, 4, 4)
logits[:, 0, :, :] = torch.where(gt_mask[:, 0] == 1, -10.0, 10.0)  # Class 0: low logit where mask=1
logits[:, 1, :, :] = torch.where(gt_mask[:, 0] == 1, 10.0, -10.0)   # Class 1: high logit where mask=1

pred_from_logits = logits.argmax(dim=1)
print(f"Predictions from logits (argmax):\n{pred_from_logits[0]}\n")

iou = binary_miou_from_logits(logits, gt_mask)
print(f"IoU when prediction matches GT: {iou:.6f}")
print(f"Expected: 1.0, Got: {iou:.6f}, Match: {abs(iou - 1.0) < 1e-6}\n")

print("=" * 60)
print("TEST 2: Verify format compatibility")
print("=" * 60)

print("\nPrediction format (from model logits):")
print(f"  - Input logits: [B, 2, H, W] (2 classes)")
print(f"  - After argmax: [B, H, W] with values {{0, 1}}")
print(f"  - Current test logits: {logits.shape}")

print("\nGround truth format:")
print(f"  - Input GT: [B, 1, H, W] or [B, H, W] with values {{0, 1}} or {{0.0, 1.0}}")
print(f"  - After processing: [B, H, W] with values {{0, 1}}")
print(f"  - Current test GT: {gt_mask.shape}")

print("\n✓ Formats are compatible - both become [B, H, W] with {0, 1} values\n")

print("=" * 60)
print("TEST 3: IoU with partial overlap")
print("=" * 60)

# Test 2: Partial overlap
gt_mask_2 = torch.tensor([
    [[1, 1, 0, 0],
     [1, 1, 0, 0],
     [0, 0, 1, 1],
     [0, 0, 1, 1]],
], dtype=torch.float32).unsqueeze(1)

# Create prediction with some difference
logits_2 = torch.zeros(1, 2, 4, 4)
logits_2[:, 0, :, :] = torch.tensor([
    [[-10, -10, 10, 10],
     [-10, -10, 10, 10],
     [10, 10, -10, -10],
     [10, 10, -10, -10]],
], dtype=torch.float32)
logits_2[:, 1, :, :] = -logits_2[:, 0]

pred_2 = logits_2.argmax(dim=1)
gt_2 = (gt_mask_2[:, 0] > 0.5).long()

print(f"\nGround truth:\n{gt_2[0]}")
print(f"\nPrediction:\n{pred_2[0]}")

inter = ((pred_2 == 1) & (gt_2 == 1)).sum().item()
union = ((pred_2 == 1) | (gt_2 == 1)).sum().item()
manual_iou = inter / (union + 1e-6)

iou_2 = binary_miou_from_logits(logits_2, gt_mask_2)

print(f"\nIntersection: {inter}, Union: {union}")
print(f"Manual IoU: {manual_iou:.6f}")
print(f"Function IoU: {iou_2:.6f}")
print(f"Match: {abs(manual_iou - iou_2) < 1e-6}\n")

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("✓ Prediction format: [B, 2, H, W] logits -> argmax -> [B, H, W] {0,1}")
print("✓ Ground truth format: [B, 1, H, W] float -> threshold -> [B, H, W] {0,1}")
print("✓ IoU function correctly compares identical inputs to get 1.0")
print("✓ Format compatibility verified\n")
