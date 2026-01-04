"""Debug validation IoU computation with fixed dataset."""
import torch
import torch.nn.functional as F
import logging
from src.abcb import ABCB
from src.pascal5i import Pascal5iDataset
from torch.utils.data import DataLoader
from src.train_abcb import unpack_episode, binary_miou_from_logits

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold = 0
    
    # Build validation dataset
    val_ds = Pascal5iDataset(
        root="/mnt/j/Workspace/7880-project/data",
        fold=fold,
        train=False,
        shots=1,
        queries=1,
        episodes=5,
        seed=321,
    )
    
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    
    # Create model (random initialization is fine for this test)
    model = ABCB(
        backbone_name="resnet50",
        T=3,
        use_correlation=True,
        max_support_tokens=1024,
        max_fg_tokens=512,
    ).to(device)
    model.eval()
    
    # Process one batch
    batch = next(iter(val_loader))
    support_img, support_mask, query_img, query_mask = unpack_episode(batch)
    
    support_img = support_img.to(device)
    support_mask = support_mask.to(device)
    query_img = query_img.to(device)
    query_mask = query_mask.to(device)
    
    logging.info(f"Batch shapes:")
    logging.info(f"  support_img: {support_img.shape}")
    logging.info(f"  support_mask: {support_mask.shape}, unique={torch.unique(support_mask).tolist()}, fg={support_mask.float().mean():.3f}")
    logging.info(f"  query_img: {query_img.shape}")
    logging.info(f"  query_mask: {query_mask.shape}, unique={torch.unique(query_mask).tolist()}, fg={query_mask.float().mean():.3f}")
    
    with torch.no_grad():
        out = model(
            query_img=query_img,
            support_img=support_img,
            support_mask=support_mask,
            return_all=False,
        )
    
    logits = out["logits"]
    logging.info(f"\nLogits: shape={logits.shape}, dtype={logits.dtype}")
    logging.info(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    logging.info(f"Logits[:, 0] mean: {logits[:, 0].mean():.3f}")
    logging.info(f"Logits[:, 1] mean: {logits[:, 1].mean():.3f}")
    
    pred = logits.argmax(dim=1)
    logging.info(f"Pred: unique={torch.unique(pred).tolist()}, fg={pred.float().mean():.3f}")
    
    # Compute IoU using the training function
    iou = binary_miou_from_logits(logits, query_mask)
    logging.info(f"\nIoU from binary_miou_from_logits: {iou:.4f}")
    
    # Manual IoU computation
    gt = query_mask[:, 0] if query_mask.dim() == 4 else query_mask
    if gt.dtype in [torch.float32, torch.float64]:
        gt = (gt > 0.5).long()
    else:
        gt = gt.long()
    
    logging.info(f"\nGT: unique={torch.unique(gt).tolist()}, fg={gt.float().mean():.3f}")
    
    for b in range(pred.shape[0]):
        inter = ((pred[b] == 1) & (gt[b] == 1)).sum().item()
        union = ((pred[b] == 1) | (gt[b] == 1)).sum().item()
        iou_b = inter / (union + 1e-6)
        
        pred_fg = (pred[b] == 1).sum().item()
        gt_fg = (gt[b] == 1).sum().item()
        
        logging.info(f"Sample {b}: pred_fg={pred_fg}, gt_fg={gt_fg}, inter={inter}, union={union}, iou={iou_b:.4f}")

if __name__ == "__main__":
    main()
