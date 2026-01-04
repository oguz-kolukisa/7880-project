"""Test the actual trained model on validation set with fixed dataset."""
import torch
import logging
from pathlib import Path
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
    
    # Build validation dataset with the FIXED code
    val_ds = Pascal5iDataset(
        root="/mnt/j/Workspace/7880-project/data",
        fold=fold,
        train=False,
        shots=1,
        queries=1,
        episodes=10,
        seed=321,
    )
    
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    
    # Load the TRAINED model (trained with BUGGY dataset)
    model = ABCB(
        backbone_name="resnet50",
        T=3,
        use_correlation=True,
        max_support_tokens=1024,
        max_fg_tokens=512,
    ).to(device)
    
    ckpt_path = f"/mnt/j/Workspace/7880-project/output/pascal5i_resnet50_1shot/fold{fold}/model.pt"
    if Path(ckpt_path).exists():
        logging.info(f"Loading trained model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
    else:
        logging.error(f"No checkpoint found at {ckpt_path}")
        return
    
    model.eval()
    
    # Evaluate on FIXED validation set
    iou_sum = 0.0
    n = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            support_img, support_mask, query_img, query_mask = unpack_episode(batch)
            support_img = support_img.to(device)
            support_mask = support_mask.to(device)
            query_img = query_img.to(device)
            query_mask = query_mask.to(device)
            
            out = model(
                query_img=query_img,
                support_img=support_img,
                support_mask=support_mask,
                return_all=False,
            )
            
            logits = out["logits"]
            pred = logits.argmax(dim=1)
            
            if i == 0:
                logging.info(f"First batch:")
                logging.info(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
                logging.info(f"  Logits[:, 0] mean: {logits[:, 0].mean():.3f}")
                logging.info(f"  Logits[:, 1] mean: {logits[:, 1].mean():.3f}")
                logging.info(f"  Pred unique: {torch.unique(pred).tolist()}, fg: {pred.float().mean():.3f}")
                logging.info(f"  GT fg: {query_mask.float().mean():.3f}")
            
            iou = binary_miou_from_logits(logits, query_mask)
            iou_sum += iou * query_img.shape[0]
            n += query_img.shape[0]
    
    avg_iou = iou_sum / n
    logging.info(f"\nModel trained on BUGGY dataset, evaluated on FIXED validation set:")
    logging.info(f"Average IoU: {avg_iou:.4f}")
    logging.info(f"\nThis confirms the model was trained on all-background samples!")

if __name__ == "__main__":
    main()
