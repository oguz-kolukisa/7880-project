"""Debug validation IoU issue - check predictions during validation."""
import torch
import torch.nn.functional as F
import logging
from pathlib import Path
from src.abcb import ABCB
from src.pascal5i import Pascal5iDataset
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def unpack_episode(batch):
    if isinstance(batch, dict):
        support_img = batch.get("support_img", batch.get("support_images", batch.get("I_s")))
        support_mask = batch.get("support_mask", batch.get("support_masks", batch.get("G_s")))
        query_img = batch.get("query_img", batch.get("query_image", batch.get("query_images", batch.get("I_q"))))
        query_mask = batch.get("query_mask", batch.get("query_masks", batch.get("query_gt", batch.get("G_q"))))
    else:
        support_img, support_mask, query_img, query_mask = batch

    if support_img.dim() == 4:
        support_img = support_img.unsqueeze(1)
    if support_mask.dim() == 4:
        support_mask = support_mask.unsqueeze(1)

    return support_img, support_mask, query_img, query_mask

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold = 0
    
    # Build dataset
    logging.info(f"Building Pascal5i dataset for fold {fold}")
    val_ds = Pascal5iDataset(
        root="/mnt/j/Workspace/7880-project/data",
        fold=fold,
        train=False,
        shots=1,
        queries=1,
        episodes=2,
        seed=321,
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
    
    # Load checkpoint
    ckpt_path = f"/mnt/j/Workspace/7880-project/output/pascal5i_resnet50_1shot/fold{fold}/model.pt"
    if Path(ckpt_path).exists():
        logging.info(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        logging.warning(f"No checkpoint found at {ckpt_path}, using random initialization")
    
    model.eval()
    
    # Create dataloader
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    
    # Get one batch
    logging.info("Processing validation batch...")
    batch = next(iter(val_loader))
    support_img, support_mask, query_img, query_mask = unpack_episode(batch)
    
    support_img = support_img.to(device)
    support_mask = support_mask.to(device)
    query_img = query_img.to(device)
    query_mask = query_mask.to(device)
    
    logging.info(f"Batch shapes:")
    logging.info(f"  support_img: {support_img.shape}")
    logging.info(f"  support_mask: {support_mask.shape}, unique={torch.unique(support_mask).tolist()}")
    logging.info(f"  query_img: {query_img.shape}")
    logging.info(f"  query_mask: {query_mask.shape}, unique={torch.unique(query_mask).tolist()}")
    
    with torch.no_grad():
        # Test with FP32
        logging.info("\n=== Testing with FP32 ===")
        out_fp32 = model(
            query_img=query_img,
            support_img=support_img,
            support_mask=support_mask,
            return_all=False,
        )
        logits_fp32 = out_fp32["logits"]
        pred_fp32 = logits_fp32.argmax(dim=1)
        
        logging.info(f"FP32 logits shape: {logits_fp32.shape}, dtype: {logits_fp32.dtype}")
        logging.info(f"FP32 logits range: min={logits_fp32.min().item():.4f}, max={logits_fp32.max().item():.4f}")
        logging.info(f"FP32 logits[:, 0] (bg) mean: {logits_fp32[:, 0].mean().item():.4f}")
        logging.info(f"FP32 logits[:, 1] (fg) mean: {logits_fp32[:, 1].mean().item():.4f}")
        logging.info(f"FP32 pred unique: {torch.unique(pred_fp32).tolist()}")
        logging.info(f"FP32 pred foreground ratio: {(pred_fp32 == 1).float().mean().item():.4f}")
        
        # Test with FP16 (mixed precision)
        logging.info("\n=== Testing with FP16 (autocast) ===")
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out_fp16 = model(
                query_img=query_img,
                support_img=support_img,
                support_mask=support_mask,
                return_all=False,
            )
        logits_fp16 = out_fp16["logits"]
        pred_fp16 = logits_fp16.argmax(dim=1)
        
        logging.info(f"FP16 logits shape: {logits_fp16.shape}, dtype: {logits_fp16.dtype}")
        logging.info(f"FP16 logits range: min={logits_fp16.min().item():.4f}, max={logits_fp16.max().item():.4f}")
        logging.info(f"FP16 logits[:, 0] (bg) mean: {logits_fp16[:, 0].mean().item():.4f}")
        logging.info(f"FP16 logits[:, 1] (fg) mean: {logits_fp16[:, 1].mean().item():.4f}")
        logging.info(f"FP16 pred unique: {torch.unique(pred_fp16).tolist()}")
        logging.info(f"FP16 pred foreground ratio: {(pred_fp16 == 1).float().mean().item():.4f}")
        
        # Compute IoU for both
        gt = query_mask[:, 0] if query_mask.dim() == 4 else query_mask
        if gt.dtype in [torch.float32, torch.float64]:
            gt = (gt > 0.5).long()
        else:
            gt = gt.long()
        
        logging.info(f"\nGround truth unique: {torch.unique(gt).tolist()}")
        logging.info(f"Ground truth foreground ratio: {(gt == 1).float().mean().item():.4f}")
        
        # IoU for FP32
        def compute_iou(pred, gt):
            ious = []
            for b in range(pred.shape[0]):
                inter = ((pred[b] == 1) & (gt[b] == 1)).sum().item()
                union = ((pred[b] == 1) | (gt[b] == 1)).sum().item()
                iou = inter / (union + 1e-6)
                ious.append(iou)
            return ious
        
        iou_fp32 = compute_iou(pred_fp32, gt)
        iou_fp16 = compute_iou(pred_fp16, gt)
        
        logging.info(f"\nFP32 IoU scores: {[f'{x:.4f}' for x in iou_fp32]}, avg={sum(iou_fp32)/len(iou_fp32):.4f}")
        logging.info(f"FP16 IoU scores: {[f'{x:.4f}' for x in iou_fp16]}, avg={sum(iou_fp16)/len(iou_fp16):.4f}")
        
        # Check support mask statistics
        logging.info(f"\n=== Support mask statistics ===")
        for b in range(support_mask.shape[0]):
            for k in range(support_mask.shape[1]):
                sm = support_mask[b, k, 0]
                fg_pixels = (sm > 0.5).sum().item()
                total_pixels = sm.numel()
                logging.info(f"Sample {b}, shot {k}: {fg_pixels}/{total_pixels} pixels ({fg_pixels/total_pixels*100:.2f}%)")

if __name__ == "__main__":
    main()
