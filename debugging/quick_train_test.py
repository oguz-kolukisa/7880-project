"""Quick training test to verify model learns with fixed dataset."""
import torch
import logging
from src.abcb import ABCB
from src.pascal5i import Pascal5iDataset
from src.train_abcb import train_abcb

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold = 0
    
    # Build datasets with small episodes for quick testing
    logging.info("Building datasets...")
    train_ds = Pascal5iDataset(
        root="/mnt/j/Workspace/7880-project/data",
        fold=fold,
        train=True,
        shots=1,
        queries=1,
        episodes=50,  # Small for quick test
        seed=123,
    )
    
    val_ds = Pascal5iDataset(
        root="/mnt/j/Workspace/7880-project/data",
        fold=fold,
        train=False,
        shots=1,
        queries=1,
        episodes=20,
        seed=321,
    )
    
    # Create model
    logging.info("Creating model...")
    model = ABCB(
        backbone_name="resnet50",
        T=3,
        use_correlation=True,
        max_support_tokens=1024,
        max_fg_tokens=512,
    )
    
    # Train for more epochs to see if model learns
    logging.info("Starting training...")
    metrics = train_abcb(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        device=device,
        epochs=10,  # More epochs
        batch_size=2,
        base_lr=2e-3,
        num_workers=0,
        save_path=None,
    )
    
    logging.info("\n=== Training Results ===")
    logging.info(f"Epochs: {metrics['epochs']}")
    logging.info(f"Train losses: {[f'{x:.4f}' for x in metrics['train_losses']]}")
    logging.info(f"Val IoUs: {[f'{x:.4f}' for x in metrics['val_ious']]}")
    
    if all(iou == 0.0 for iou in metrics['val_ious']):
        logging.error("FAILED: All validation IoUs are still 0!")
    else:
        logging.info("SUCCESS: Model achieved non-zero validation IoU!")

if __name__ == "__main__":
    main()
