"""Generate visual comparison examples of ground truth vs predicted segmentation masks."""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import random

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.abcb import ABCB
from src.coco20i import Coco20iDataset

def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    model = ABCB(backbone_name='resnet50', pretrained_backbone=True, freeze_backbone=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def denormalize_image(img_tensor):
    """Denormalize image from ImageNet normalization and convert to proper RGB format."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # Reverse ImageNet normalization
    img = img_tensor * std + mean
    # Ensure values are in valid range
    img = torch.clamp(img, 0, 1)
    # Convert from CHW to HWC for matplotlib
    return img

def predict_mask(model, support_img, support_mask, query_img, device='cuda'):
    """Generate prediction using the model."""
    with torch.no_grad():
        # Add batch and shot dimensions
        support_img = support_img.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 3, H, W]
        support_mask = support_mask.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
        query_img = query_img.unsqueeze(0).to(device)  # [1, 3, H, W]
        
        # Forward pass with correct argument order
        out = model(
            query_img=query_img,
            support_img=support_img,
            support_mask=support_mask,
            return_all=False
        )
        
        # Get the final prediction logits
        logits = out["logits"]  # [1, 2, H, W]
        pred_mask = (logits[:, 1:2] > logits[:, 0:1]).float()  # Compare foreground vs background
        
    return pred_mask.squeeze(0).squeeze(0).cpu()

def create_comparison_figure(examples, model_name, output_path):
    """Create a comparison figure with 4 examples."""
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    for idx, example in enumerate(examples):
        # Column 0: Support image with mask overlay
        ax = axes[idx, 0]
        support_img = example['support_img'].permute(1, 2, 0).numpy()
        support_img = np.clip(support_img, 0, 1)  # Ensure proper range
        ax.imshow(support_img)
        support_mask_np = example['support_mask'].squeeze().numpy()  # Remove extra dims
        masked = np.ma.masked_where(support_mask_np == 0, support_mask_np)
        ax.imshow(masked, alpha=0.5, cmap='jet')
        ax.set_title('Support Image', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Column 1: Query image
        ax = axes[idx, 1]
        query_img = example['query_img'].permute(1, 2, 0).numpy()
        query_img = np.clip(query_img, 0, 1)  # Ensure proper range, remove white filter effect
        ax.imshow(query_img)
        ax.set_title('Query Image', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Column 2: Ground Truth Mask
        ax = axes[idx, 2]
        ax.imshow(query_img)
        gt_mask = example['gt_mask'].squeeze().numpy()  # Remove extra dims
        masked_gt = np.ma.masked_where(gt_mask == 0, gt_mask)
        ax.imshow(masked_gt, alpha=0.6, cmap='Greens', vmin=0, vmax=1)
        ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Column 3: Predicted Mask
        ax = axes[idx, 3]
        ax.imshow(query_img)
        pred_mask = example['pred_mask'].squeeze().numpy()  # Remove extra dims
        masked_pred = np.ma.masked_where(pred_mask == 0, pred_mask)
        ax.imshow(masked_pred, alpha=0.6, cmap='Blues', vmin=0, vmax=1)
        ax.set_title('Predicted', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Column 4: Overlay comparison
        ax = axes[idx, 4]
        ax.imshow(query_img)
        # Green for GT, Red for prediction, Yellow for overlap
        overlay = np.zeros((*gt_mask.shape, 3))
        overlay[gt_mask > 0] = [0, 1, 0]  # Green for GT
        overlay[pred_mask > 0] += [1, 0, 0]  # Red for prediction (overlap becomes yellow)
        ax.imshow(overlay, alpha=0.5)
        ax.set_title('Overlay (GT=Green, Pred=Red)', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add IoU score as text
        intersection = (gt_mask * pred_mask).sum()
        union = ((gt_mask + pred_mask) > 0).sum()
        iou = intersection / (union + 1e-8)
        fig.text(0.02, 0.98 - idx * 0.24, f'Example {idx+1} - IoU: {iou:.3f}', 
                fontsize=11, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Segmentation Results - {model_name}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Paths
    data_root = "./data/coco"
    model_1shot_path = "./output/output/coco20i_resnet50_1shot/fold0/model.pt"
    model_5shot_path = "./output/output/coco20i_resnet50_5shot/fold0/model.pt"
    
    # Load models
    print("Loading 1-shot model...")
    model_1shot = load_model(model_1shot_path, device)
    print("Loading 5-shot model...")
    model_5shot = load_model(model_5shot_path, device)
    
    # Create validation dataset
    print("Loading validation dataset...")
    val_dataset = Coco20iDataset(
        root=data_root,
        fold=0,
        train=False,
        shots=1,
        episodes=2000,
        seed=42  # Fixed seed for reproducibility
    )
    
    # Select 4 random examples
    random.seed(None)  # Use random seed for different examples each time
    example_indices = random.sample(range(len(val_dataset)), 4)
    
    # Generate examples for 1-shot model
    print("\nGenerating examples for 1-shot model...")
    examples_1shot = []
    for i, idx in enumerate(example_indices):
        print(f"  Processing example {i+1}/4...")
        episode = val_dataset[idx]
        
        support_img = denormalize_image(episode['support_images'][0])
        support_mask = episode['support_masks'][0]
        query_img = denormalize_image(episode['query_images'])
        query_mask = episode['query_masks']
        
        pred_mask = predict_mask(model_1shot, episode['support_images'][0], 
                                episode['support_masks'][0], 
                                episode['query_images'], device)
        
        examples_1shot.append({
            'support_img': support_img,
            'support_mask': support_mask,
            'query_img': query_img,
            'gt_mask': query_mask,
            'pred_mask': pred_mask
        })
    
    # Generate examples for 5-shot model
    print("\nGenerating examples for 5-shot model...")
    val_dataset_5shot = Coco20iDataset(
        root=data_root,
        fold=0,
        train=False,
        shots=5,
        episodes=2000,
        seed=42
    )
    
    examples_5shot = []
    for i, idx in enumerate(example_indices):
        print(f"  Processing example {i+1}/4...")
        episode = val_dataset_5shot[idx]
        
        # Use first support image for visualization
        support_img = denormalize_image(episode['support_images'][0])
        support_mask = episode['support_masks'][0]
        query_img = denormalize_image(episode['query_images'])
        query_mask = episode['query_masks']
        
        pred_mask = predict_mask(model_5shot, episode['support_images'][0], 
                                episode['support_masks'][0], 
                                episode['query_images'], device)
        
        examples_5shot.append({
            'support_img': support_img,
            'support_mask': support_mask,
            'query_img': query_img,
            'gt_mask': query_mask,
            'pred_mask': pred_mask
        })
    
    # Create comparison figures
    print("\nCreating visualization figures...")
    output_dir = Path("./images")
    output_dir.mkdir(exist_ok=True)
    
    create_comparison_figure(examples_1shot, "COCO-20i ResNet-50 1-shot", 
                            output_dir / "examples_1shot.png")
    create_comparison_figure(examples_5shot, "COCO-20i ResNet-50 5-shot", 
                            output_dir / "examples_5shot.png")
    
    print("\n✓ Done! Generated 4 example comparisons for both models.")

if __name__ == "__main__":
    main()
