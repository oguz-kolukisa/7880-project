"""Visualize training metrics from 1shot and 5shot JSON files."""
import json
import matplotlib.pyplot as plt
import numpy as np

# Read data
with open('1shot.json', 'r') as f:
    data_1shot = json.load(f)

with open('5shot.json', 'r') as f:
    data_5shot = json.load(f)

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training Loss
ax1.plot(data_1shot['epochs'], data_1shot['train_losses'], 'b-', linewidth=2, label='1-shot')
ax1.plot(data_5shot['epochs'], data_5shot['train_losses'], 'r-', linewidth=2, label='5-shot')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Loss', fontsize=12)
ax1.set_title('Training Loss Progression (COCO-20^i, ResNet-50)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([1, 70])

# Plot 2: Validation IoU
ax2.plot(data_1shot['epochs'], data_1shot['val_ious'], 'b-', linewidth=2, label='1-shot')
ax2.plot(data_5shot['epochs'], data_5shot['val_ious'], 'r-', linewidth=2, label='5-shot')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Validation mIoU', fontsize=12)
ax2.set_title('Validation mIoU Progression (COCO-20^i, ResNet-50)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([1, 70])
ax2.set_ylim([0, 0.25])

plt.tight_layout()
plt.savefig('images/training_curves.png', dpi=300, bbox_inches='tight')
print("Saved: images/training_curves.png")

# Create individual plots as well
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data_1shot['epochs'], data_1shot['train_losses'], 'b-', linewidth=2, label='1-shot', marker='o', markersize=3, markevery=5)
ax.plot(data_5shot['epochs'], data_5shot['train_losses'], 'r-', linewidth=2, label='5-shot', marker='s', markersize=3, markevery=5)
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Training Loss', fontsize=13)
ax.set_title('Training Loss Progression (COCO-20^i, ResNet-50)', fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim([1, 70])
plt.tight_layout()
plt.savefig('images/train_loss_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: images/train_loss_comparison.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data_1shot['epochs'], data_1shot['val_ious'], 'b-', linewidth=2, label='1-shot', marker='o', markersize=3, markevery=5)
ax.plot(data_5shot['epochs'], data_5shot['val_ious'], 'r-', linewidth=2, label='5-shot', marker='s', markersize=3, markevery=5)
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Validation mIoU', fontsize=13)
ax.set_title('Validation mIoU Progression (COCO-20^i, ResNet-50)', fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([1, 70])
ax.set_ylim([0, 0.25])
plt.tight_layout()
plt.savefig('images/val_iou_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: images/val_iou_comparison.png")

# Print final metrics
print("\n=== Final Metrics ===")
print(f"1-shot - Final Loss: {data_1shot['train_losses'][-1]:.4f}")
print(f"1-shot - Final Val mIoU: {data_1shot['val_ious'][-1]:.4f}")
print(f"1-shot - Best Val mIoU: {max(data_1shot['val_ious']):.4f} (Epoch {data_1shot['epochs'][data_1shot['val_ious'].index(max(data_1shot['val_ious']))]})")

print(f"\n5-shot - Final Loss: {data_5shot['train_losses'][-1]:.4f}")
print(f"5-shot - Final Val mIoU: {data_5shot['val_ious'][-1]:.4f}")
print(f"5-shot - Best Val mIoU: {max(data_5shot['val_ious']):.4f} (Epoch {data_5shot['epochs'][data_5shot['val_ious'].index(max(data_5shot['val_ious']))]})")
