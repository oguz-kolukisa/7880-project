"""Training utilities for the ABCB model (matches temp.ipynb and paper settings)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


def unpack_episode(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        support_img = batch.get("support_img", batch.get("support_images", batch.get("I_s")))
        support_mask = batch.get("support_mask", batch.get("support_masks", batch.get("G_s")))
        query_img = batch.get("query_img", batch.get("query_image", batch.get("query_images", batch.get("I_q"))))
        query_mask = batch.get("query_mask", batch.get("query_masks", batch.get("query_gt", batch.get("G_q"))))
        if support_img is None:
            raise KeyError(f"Can't find support/query keys in batch dict: {list(batch.keys())}")
    else:
        support_img, support_mask, query_img, query_mask = batch

    if support_img.dim() == 4:
        support_img = support_img.unsqueeze(1)
    if support_mask.dim() == 4:
        support_mask = support_mask.unsqueeze(1)

    return support_img, support_mask, query_img, query_mask


def random_scale_flip_and_crop(
    img: torch.Tensor,
    mask: torch.Tensor,
    crop_size: int = 473,
    scale_range: Tuple[float, float] = (0.5, 2.0),
    p_flip: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, _, H, W = img.shape
    device = img.device

    s = torch.empty(B, device=device).uniform_(scale_range[0], scale_range[1])
    new_hw = [(max(1, int(H * float(si))), max(1, int(W * float(si)))) for si in s]

    resized_imgs, resized_masks = [], []
    max_h, max_w = 0, 0
    for b in range(B):
        nh, nw = new_hw[b]
        resized = F.interpolate(img[b : b + 1], size=(nh, nw), mode="bilinear", align_corners=False)
        resized_mask = F.interpolate(mask[b : b + 1].float(), size=(nh, nw), mode="nearest")
        resized_imgs.append(resized)
        resized_masks.append(resized_mask)
        max_h = max(max_h, nh)
        max_w = max(max_w, nw)

    imgs, masks = [], []
    for im, ma in zip(resized_imgs, resized_masks):
        pad = (0, max_w - im.shape[-1], 0, max_h - im.shape[-2])
        if pad[1] or pad[3]:
            imgs.append(F.pad(im, pad, mode="constant", value=0.0))
            masks.append(F.pad(ma, pad, mode="constant", value=0.0))
        else:
            imgs.append(im)
            masks.append(ma)
    img = torch.cat(imgs, dim=0)
    mask = torch.cat(masks, dim=0)

    flip = torch.rand(B, device=device) < p_flip
    if flip.any():
        img[flip] = torch.flip(img[flip], dims=[3])
        mask[flip] = torch.flip(mask[flip], dims=[3])

    _, _, H2, W2 = img.shape
    pad_h = max(0, crop_size - H2)
    pad_w = max(0, crop_size - W2)
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        _, _, H2, W2 = img.shape

    top = torch.randint(0, H2 - crop_size + 1, (B,), device=device)
    left = torch.randint(0, W2 - crop_size + 1, (B,), device=device)

    crops_img, crops_mask = [], []
    for b in range(B):
        t, l = int(top[b]), int(left[b])
        crops_img.append(img[b : b + 1, :, t : t + crop_size, l : l + crop_size])
        crops_mask.append(mask[b : b + 1, :, t : t + crop_size, l : l + crop_size])
    return torch.cat(crops_img, dim=0), torch.cat(crops_mask, dim=0)


def poly_lr(base_lr: float, cur_iter: int, max_iter: int, power: float = 0.9) -> float:
    return base_lr * ((1.0 - float(cur_iter) / float(max_iter)) ** power)


def abcb_loss(
    P_list: List[torch.Tensor],
    Phat_list: List[torch.Tensor],
    G_q: torch.Tensor,
    lam: float = 0.2,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Compute ABCB loss with proper shape handling.
    
    Args:
        P_list: list of prediction logits [B, 2, H, W] or similar
        Phat_list: list of auxiliary predictions
        G_q: ground truth [B, 1, H, W] or [B, H, W]
        lam: weight for auxiliary loss
        ignore_index: index to ignore in cross-entropy
    """
    if G_q.dim() == 4:
        G = G_q[:, 0]
    else:
        G = G_q
    
    # Threshold GT: handle both float [0,1] and int {0,1}
    if G.dtype == torch.float32 or G.dtype == torch.float64:
        G = (G > 0.5).long()
    else:
        G = G.long()
    
    logging.debug(f"Loss input: P_list[0].shape={P_list[0].shape}, G.shape={G.shape}, G dtype={G.dtype}, unique values={torch.unique(G)}")

    loss = 0.0
    for i, Pt in enumerate(P_list):
        # Ensure spatial dimensions match
        if Pt.shape[-2:] != G.shape[-2:]:
            Pt_up = F.interpolate(Pt, size=G.shape[-2:], mode="bilinear", align_corners=False)
            logging.debug(f"Upsampled P_list[{i}] from {Pt.shape} to {Pt_up.shape}")
        else:
            Pt_up = Pt
        loss = loss + F.cross_entropy(Pt_up, G, ignore_index=ignore_index)

    for i, Phat in enumerate(Phat_list):
        # Ensure spatial dimensions match
        if Phat.shape[-2:] != G.shape[-2:]:
            Phat_up = F.interpolate(Phat, size=G.shape[-2:], mode="bilinear", align_corners=False)
            logging.debug(f"Upsampled Phat_list[{i}] from {Phat.shape} to {Phat_up.shape}")
        else:
            Phat_up = Phat
        loss = loss + lam * F.cross_entropy(Phat_up, G, ignore_index=ignore_index)

    logging.debug(f"Total loss={loss.item():.4f}")
    return loss


@torch.no_grad()
def binary_miou_from_logits(logits: torch.Tensor, G_q: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute binary IoU from logits and ground truth mask.
    
    Args:
        logits: [B, 2, H, W] or [B, H, W] tensor
        G_q: [B, 1, H, W] or [B, H, W] ground truth mask (values in [0,1] or {0,1})
        eps: small value to avoid division by zero
    
    Returns:
        IoU score (float)
    """
    # Handle logits shape: if [B, 2, H, W], take argmax; if [B, H, W], assume raw predictions
    if logits.dim() == 4 and logits.shape[1] == 2:
        pred = logits.argmax(dim=1)  # [B, H, W]
    elif logits.dim() == 4:
        pred = logits[:, 0]  # Take first channel if single-channel output
    else:
        pred = logits  # Already [B, H, W]
    
    # Handle GT shape and threshold
    if G_q.dim() == 4:
        G_q = G_q[:, 0]  # [B, 1, H, W] -> [B, H, W]
    
    # Ensure logits and GT have same spatial dimensions
    if pred.shape[-2:] != G_q.shape[-2:]:
        pred = F.interpolate(pred.unsqueeze(1).float(), size=G_q.shape[-2:], mode="nearest").squeeze(1).long()
        logging.debug(f"Upsampled pred from {logits.shape} to {pred.shape}")
    
    # Threshold GT: handle both [0,1] float and {0,1} int
    if G_q.dtype == torch.float32 or G_q.dtype == torch.float64:
        gt = (G_q > 0.5).long()
    else:
        gt = G_q.long()
    
    # Compute binary IoU (foreground class = 1)
    inter = ((pred == 1) & (gt == 1)).sum().item()
    union = ((pred == 1) | (gt == 1)).sum().item()
    iou = float(inter) / float(union + eps)
    
    logging.debug(f"IoU: pred shape={pred.shape}, gt shape={gt.shape}, inter={inter}, union={union}, iou={iou:.4f}")
    return iou


def train_abcb(
    model: torch.nn.Module,
    train_ds,
    val_ds,
    device: str = "cuda",
    epochs: int = 25,
    batch_size: int = 16,
    base_lr: float = 2e-3,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    lam: float = 0.2,
    crop_size: int = 473,
    num_workers: int = 4,
    save_path: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, List[float]]:
    model = model.to(device)
    model = torch.compile(model, mode="reduce-overhead")  # Optimize with torch.compile

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scaler = GradScaler()  # For mixed precision (float16) training
    max_iter = epochs * len(train_loader)
    cur_iter = 0

    epochs_log: List[int] = []
    train_losses_log: List[float] = []
    val_ious_log: List[float] = []

    epoch_pbar = tqdm(range(epochs), desc="Epochs", dynamic_ncols=True, leave=False)
    total_steps = 0
    training_done = False
    for epoch in epoch_pbar:
        if training_done:
            break
        model.train()
        epoch_loss_sum = 0.0

        step_pbar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Train {epoch + 1}/{epochs}",
            leave=False,
            dynamic_ncols=True,
            position=1,
        )
        for batch in step_pbar:
            if max_steps is not None and total_steps >= max_steps:
                training_done = True
                break
            support_img, support_mask, query_img, query_mask = unpack_episode(batch)

            support_img = support_img.to(device, non_blocking=True)
            support_mask = support_mask.to(device, non_blocking=True)
            query_img = query_img.to(device, non_blocking=True)
            query_mask = query_mask.to(device, non_blocking=True)

            B, K = support_img.shape[:2]
            supp_imgs_aug, supp_masks_aug = [], []
            for k in range(K):
                si, sm = random_scale_flip_and_crop(
                    support_img[:, k], support_mask[:, k], crop_size=crop_size
                )
                supp_imgs_aug.append(si)
                supp_masks_aug.append(sm)
            support_img = torch.stack(supp_imgs_aug, dim=1)
            support_mask = torch.stack(supp_masks_aug, dim=1)
            query_img, query_mask = random_scale_flip_and_crop(
                query_img, query_mask, crop_size=crop_size
            )

            lr = poly_lr(base_lr, cur_iter, max_iter, power=0.9)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            # Float16 mixed precision training
            with autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
                out = model(
                    query_img=query_img,
                    support_img=support_img,
                    support_mask=support_mask,
                    return_all=True,
                )

                loss = abcb_loss(out["P_list"], out["Phat_list"], query_mask, lam=lam)

            # Backprop with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss_sum += loss.item()
            cur_iter += 1
            total_steps += 1

        step_pbar.close()
        if training_done:
            break
        avg_train_loss = epoch_loss_sum / max(1, len(train_loader))

        model.eval()
        iou_sum, n = 0.0, 0
        val_pbar = tqdm(
            total=len(val_loader),
            desc=f"Val {epoch + 1}/{epochs}",
            leave=False,
            dynamic_ncols=True,
            position=1,
        )
        with torch.no_grad():
            for batch in val_loader:
                support_img, support_mask, query_img, query_mask = unpack_episode(batch)
                support_img = support_img.to(device, non_blocking=True)
                support_mask = support_mask.to(device, non_blocking=True)
                query_img = query_img.to(device, non_blocking=True)
                query_mask = query_mask.to(device, non_blocking=True)

                out = model(
                    query_img=query_img,
                    support_img=support_img,
                    support_mask=support_mask,
                    return_all=False,
                )
                logits = out["logits"]

                iou = binary_miou_from_logits(logits, query_mask)
                iou_sum += iou * query_img.shape[0]
                n += query_img.shape[0]
                val_pbar.set_postfix(iou=f"{iou:.4f}")
                val_pbar.update(1)

        val_pbar.close()
        val_iou = iou_sum / max(1, n)

        epochs_log.append(epoch + 1)
        train_losses_log.append(avg_train_loss)
        val_ious_log.append(val_iou)

        epoch_pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_iou=f"{val_iou:.4f}")

    metrics = {"epochs": epochs_log, "train_losses": train_losses_log, "val_ious": val_ious_log}
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.debug(f"Saved training metrics to {save_path}")
    return metrics


__all__ = [
    "unpack_episode",
    "random_scale_flip_and_crop",
    "poly_lr",
    "abcb_loss",
    "binary_miou_from_logits",
    "train_abcb",
]
