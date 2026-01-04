"""Debug training - check if model learns anything during training."""
import torch
import torch.nn.functional as F
import logging
from pathlib import Path
from src.abcb import ABCB
from src.pascal5i import Pascal5iDataset
from torch.utils.data import DataLoader
from src.train_abcb import unpack_episode, random_scale_flip_and_crop, abcb_loss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold = 0
    
    # Build dataset
    logging.info(f"Building Pascal5i dataset for fold {fold}")
    train_ds = Pascal5iDataset(
        root="/mnt/j/Workspace/7880-project/data",
        fold=fold,
        train=True,
        shots=1,
        queries=1,
        episodes=10,  # Small for debugging
        seed=123,
    )
    
    # Create model
    logging.info("Creating ABCB model")
    model = ABCB(
        backbone_name="resnet50",
        T=3,
        use_correlation=True,
        max_support_tokens=1024,
        max_fg_tokens=512,
    ).to(device)
    
    model.train()
    
    # Create dataloader
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0)
    
    # Setup optimizer
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-3,
        momentum=0.9,
        weight_decay=1e-4,
    )
    
    # Train for a few batches
    logging.info("Starting training loop...")
    for step, batch in enumerate(train_loader):
        if step >= 5:  # Only 5 batches
            break
            
        support_img, support_mask, query_img, query_mask = unpack_episode(batch)
        
        support_img = support_img.to(device)
        support_mask = support_mask.to(device)
        query_img = query_img.to(device)
        query_mask = query_mask.to(device)
        
        logging.info(f"\n=== Batch {step} BEFORE augmentation ===")
        logging.info(f"support_mask unique: {torch.unique(support_mask).tolist()}, fg_ratio: {(support_mask > 0.5).float().mean().item():.4f}")
        logging.info(f"query_mask unique: {torch.unique(query_mask).tolist()}, fg_ratio: {(query_mask > 0.5).float().mean().item():.4f}")
        
        # Apply augmentation (like in training)
        B, K = support_img.shape[:2]
        supp_imgs_aug, supp_masks_aug = [], []
        for k in range(K):
            si, sm = random_scale_flip_and_crop(
                support_img[:, k], support_mask[:, k], crop_size=473
            )
            supp_imgs_aug.append(si)
            supp_masks_aug.append(sm)
        support_img = torch.stack(supp_imgs_aug, dim=1)
        support_mask = torch.stack(supp_masks_aug, dim=1)
        query_img, query_mask = random_scale_flip_and_crop(
            query_img, query_mask, crop_size=473
        )
        
        logging.info(f"=== Batch {step} AFTER augmentation ===")
        logging.info(f"support_mask unique: {torch.unique(support_mask).tolist()}, fg_ratio: {(support_mask > 0.5).float().mean().item():.4f}")
        logging.info(f"query_mask unique: {torch.unique(query_mask).tolist()}, fg_ratio: {(query_mask > 0.5).float().mean().item():.4f}")
        
        # Forward pass
        optimizer.zero_grad()
        out = model(
            query_img=query_img,
            support_img=support_img,
            support_mask=support_mask,
            return_all=True,
        )
        
        # Check predictions
        logits = out["P_list"][-1]
        pred = logits.argmax(dim=1)
        
        logging.info(f"=== Batch {step} Predictions ===")
        logging.info(f"Logits shape: {logits.shape}, range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        logging.info(f"Logits[:, 0] (bg) mean: {logits[:, 0].mean().item():.3f}, Logits[:, 1] (fg) mean: {logits[:, 1].mean().item():.3f}")
        logging.info(f"Pred foreground ratio: {(pred == 1).float().mean().item():.4f}")
        
        # Compute loss
        loss = abcb_loss(out["P_list"], out["Phat_list"], query_mask, lam=0.2)
        logging.info(f"Loss: {loss.item():.4f}")
        
        # Check gradients
        loss.backward()
        
        # Check gradient norms
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        logging.info(f"Gradient norm: {total_norm:.4f}")
        
        optimizer.step()

if __name__ == "__main__":
    main()
