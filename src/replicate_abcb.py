"""End-to-end replication script: download, train, and evaluate across folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from src.abcb import ABCB
from src.coco20i import Coco20iDataset, download_coco2017
from src.eval_abcb import evaluate_fold
from src.pascal5i import Pascal5iDataset, download_pascal5i
from src.train_abcb import train_abcb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replicate ABCB results across folds.")
    parser.add_argument("--data-root", required=True, help="Dataset root directory.")
    parser.add_argument("--output-dir", required=True, help="Directory to save checkpoints/results.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download", action="store_true", help="Download datasets with torchvision.")
    return parser.parse_args()


def build_datasets(
    dataset: str,
    root: str,
    fold: int,
    shots: int,
    train_episodes: int,
    val_episodes: int,
    seed: int,
    download: bool,
) -> Dict[str, torch.utils.data.Dataset]:
    if dataset == "pascal5i":
        if download:
            download_pascal5i(root=root)
        train_ds = Pascal5iDataset(
            root=root,
            fold=fold,
            train=True,
            shots=shots,
            episodes=train_episodes,
            seed=seed,
        )
        val_ds = Pascal5iDataset(
            root=root,
            fold=fold,
            train=False,
            shots=shots,
            episodes=val_episodes,
            seed=seed,
        )
        return {"train": train_ds, "val": val_ds}

    if dataset == "coco20i":
        if download:
            download_coco2017(root=root)
        mask_dir = Path(root) / "masks"
        if not mask_dir.exists():
            raise FileNotFoundError(
                "COCO-20i masks not found. Expected masks/ directory with PNG masks."
            )
        train_ds = Coco20iDataset(
            root=root,
            fold=fold,
            train=True,
            shots=shots,
            episodes=train_episodes,
            seed=seed,
        )
        val_ds = Coco20iDataset(
            root=root,
            fold=fold,
            train=False,
            shots=shots,
            episodes=val_episodes,
            seed=seed,
        )
        return {"train": train_ds, "val": val_ds}

    raise ValueError(f"Unsupported dataset: {dataset}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_hparams = {
        "pascal5i": {
            "train_episodes": 20000,
            "val_episodes": 2000,
            "epochs": 250,
            "batch_size": 16,
            "base_lr": 0.002,
            "crop_size": 473,
            "num_workers": 4,
            "seed": 0,
        },
        "coco20i": {
            "train_episodes": 20000,
            "val_episodes": 2000,
            "epochs": 70,
            "batch_size": 8,
            "base_lr": 0.005,
            "crop_size": 641,
            "num_workers": 4,
            "seed": 0,
        },
    }

    datasets = ["pascal5i", "coco20i"]
    backbones = ["resnet50", "resnet101"]
    shots_list = [1, 5]
    folds = [0, 1, 2, 3]

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset in datasets:
        hparams = paper_hparams[dataset]
        for backbone in backbones:
            for shots in shots_list:
                key = f"{dataset}_{backbone}_{shots}shot"
                results[key] = {}
                for fold in folds:
                    fold_dir = output_dir / key / f"fold{fold}"
                    fold_dir.mkdir(parents=True, exist_ok=True)

                    fold_datasets = build_datasets(
                        dataset=dataset,
                        root=args.data_root,
                        fold=fold,
                        shots=shots,
                        train_episodes=hparams["train_episodes"],
                        val_episodes=hparams["val_episodes"],
                        seed=hparams["seed"],
                        download=args.download,
                    )

                    model = ABCB(backbone_name=backbone, pretrained_backbone=True)
                    train_abcb(
                        model=model,
                        train_ds=fold_datasets["train"],
                        val_ds=fold_datasets["val"],
                        device=args.device,
                        epochs=hparams["epochs"],
                        batch_size=hparams["batch_size"],
                        base_lr=hparams["base_lr"],
                        crop_size=hparams["crop_size"],
                        num_workers=hparams["num_workers"],
                    )

                    ckpt_path = fold_dir / "model.pt"
                    torch.save(model.state_dict(), ckpt_path)

                    score = evaluate_fold(
                        model=model,
                        dataloader=torch.utils.data.DataLoader(
                            fold_datasets["val"],
                            batch_size=hparams["batch_size"],
                            shuffle=False,
                            num_workers=hparams["num_workers"],
                            pin_memory=True,
                        ),
                        device=args.device,
                    )
                    results[key][str(fold)] = {"miou": score}
                    print(f"{key} fold {fold}: mIoU={score:.4f}")

                mean_score = sum(v["miou"] for v in results[key].values()) / max(
                    1, len(results[key])
                )
                results[key]["mean"] = {"miou": mean_score}

    payload = {
        "paper_hparams": paper_hparams,
        "results": results,
    }
    (output_dir / "replication_results.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
