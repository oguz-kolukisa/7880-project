"""Evaluate ABCB across all folds for dataset replication."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.abcb import ABCB
from src.coco20i import Coco20iDataset
from src.pascal5i import Pascal5iDataset
from src.train_abcb import binary_miou_from_logits, unpack_episode


def build_dataset(
    dataset_name: str,
    root: str,
    fold: int,
    episodes: int,
    seed: int,
) -> torch.utils.data.Dataset:
    dataset_name = dataset_name.lower()
    if dataset_name == "pascal5i":
        return Pascal5iDataset(root=root, fold=fold, train=False, episodes=episodes, seed=seed)
    if dataset_name == "coco20i":
        return Coco20iDataset(root=root, fold=fold, train=False, episodes=episodes, seed=seed)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_checkpoint(model: torch.nn.Module, path: Path, device: str) -> None:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def evaluate_fold(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    max_steps: Optional[int] = None,
) -> float:
    model.eval()
    iou_sum, n = 0.0, 0
    eval_pbar = tqdm(dataloader, desc="Eval", leave=False, dynamic_ncols=True)
    steps = 0
    first_batch = True
    for batch in eval_pbar:
        if max_steps is not None and steps >= max_steps:
            break
        support_img, support_mask, query_img, query_mask = unpack_episode(batch)
        
        # Debug first batch shapes
        if first_batch:
            logging.debug(f"Eval batch shapes: support_img={support_img.shape}, query_img={query_img.shape}, query_mask={query_mask.shape}")
            first_batch = False
        
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
        
        # Debug model output
        if steps == 0:
            logging.debug(f"Model output logits shape: {out['logits'].shape}, dtype={out['logits'].dtype}")
            logging.debug(f"Logits value range: min={out['logits'].min():.4f}, max={out['logits'].max():.4f}")
            logging.debug(f"Query mask shape: {query_mask.shape}, dtype={query_mask.dtype}, unique values={torch.unique(query_mask)}")
        
        iou = binary_miou_from_logits(out["logits"], query_mask)
        iou_sum += iou * query_img.shape[0]
        n += query_img.shape[0]
        eval_pbar.set_postfix(iou=f"{iou:.4f}")
        steps += 1
    return iou_sum / max(1, n)


def evaluate_all_folds(
    dataset_name: str,
    data_root: str,
    checkpoint_pattern: str,
    backbone_name: str,
    device: str,
    episodes: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_steps: Optional[int] = None,
) -> Dict[int, float]:
    fold_scores: Dict[int, float] = {}
    fold_pbar = tqdm(range(4), desc="Folds", dynamic_ncols=True, leave=False)
    for fold in fold_pbar:
        dataset = build_dataset(dataset_name, data_root, fold, episodes, seed)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        model = ABCB(backbone_name=backbone_name, pretrained_backbone=False)
        ckpt_path = Path(checkpoint_pattern.format(fold=fold))
        load_checkpoint(model, ckpt_path, device)
        model = model.to(device)

        fold_scores[fold] = evaluate_fold(model, dataloader, device, max_steps)
        fold_pbar.set_postfix(fold=fold, miou=f"{fold_scores[fold]:.4f}")
    return fold_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ABCB across folds.")
    parser.add_argument("--dataset", choices=["pascal5i", "coco20i"], default="pascal5i")
    parser.add_argument("--data-root", required=True, help="Dataset root directory.")
    parser.add_argument(
        "--checkpoint-pattern",
        required=True,
        help="Checkpoint path pattern with '{fold}' placeholder.",
    )
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet101"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None, help="Limit evaluation to max steps for debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = evaluate_all_folds(
        dataset_name=args.dataset,
        data_root=args.data_root,
        checkpoint_pattern=args.checkpoint_pattern,
        backbone_name=args.backbone,
        device=args.device,
        episodes=args.episodes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    mean_score = sum(scores.values()) / max(1, len(scores))
    for fold, score in scores.items():
        print(f"Fold {fold}: mIoU={score:.4f}")
    print(f"Mean mIoU: {mean_score:.4f}")

    if args.output:
        output_path = Path(args.output)
        payload = {
            "dataset": args.dataset,
            "backbone": args.backbone,
            "episodes": args.episodes,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "fold_scores": scores,
            "mean_miou": mean_score,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
