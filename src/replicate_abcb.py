"""End-to-end replication script: download, train, and evaluate across folds."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch

# Allow running the script via `python src/replicate_abcb.py` without installing the package.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

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
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with limited steps (max_steps=2).")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare datasets/masks and exit.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        help="Logging level (default: INFO)")
    parser.add_argument("--push-to-hub", action="store_true", help="After each evaluation, upload output directory to Hugging Face Hub.")
    parser.add_argument("--hf-repo", default="okolukisa1/7880-project", help="Hugging Face repo ID for uploads.")
    parser.add_argument("--hf-branch", default="main", help="Target branch for Hugging Face uploads.")
    return parser.parse_args()


def upload_output_dir(
    output_dir: Path,
    repo_id: str,
    branch: str,
) -> None:
    """Upload the entire output directory to Hugging Face Hub.

    Uses huggingface_hub.HfApi.upload_folder. Requires `huggingface-cli login`
    or HF_TOKEN env var. Best-effort; logs warnings on failure.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logging.warning("huggingface_hub not installed; skipping push to hub")
        return

    api = HfApi()
    try:
        logging.info(f"Uploading {output_dir} to {repo_id}@{branch}")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(output_dir),
            path_in_repo="output",
            commit_message="Update output artifacts",
            revision=branch,
        )
        logging.info("Upload to Hugging Face completed")
    except Exception as exc:  # pragma: no cover - network/auth dependent
        logging.warning(f"Push to Hugging Face failed: {exc}")


def build_datasets(
    dataset: str,
    root: str,
    fold: int,
    shots: int,
    train_episodes: int,
    val_episodes: int,
    seed: int,
) -> Dict[str, torch.utils.data.Dataset] | None:
    if dataset == "pascal5i":
        logging.info(f"Building Pascal5i train dataset: fold={fold}, shots={shots}, episodes={train_episodes}")
        train_ds = Pascal5iDataset(
            root=root,
            fold=fold,
            train=True,
            shots=shots,
            episodes=train_episodes,
            seed=seed,
        )
        logging.info(f"Building Pascal5i val dataset: fold={fold}, shots={shots}, episodes={val_episodes}")
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
        coco_root = str(Path(root) / "coco")
        logging.info(f"Building COCO20i train dataset: fold={fold}, shots={shots}, episodes={train_episodes}")
        train_ds = Coco20iDataset(
            root=coco_root,
            fold=fold,
            train=True,
            shots=shots,
            episodes=train_episodes,
            seed=seed,
        )
        logging.info(f"Building COCO20i val dataset: fold={fold}, shots={shots}, episodes={val_episodes}")
        val_ds = Coco20iDataset(
            root=coco_root,
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
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_hparams = {
        "pascal5i": {
            "train_episodes": 20000,
            "val_episodes": 2000,
            "epochs": 30,
            "batch_size": 32,  # Doubled from 16
            "base_lr": 0.004,  # Scaled proportionally with batch size (was 0.002)
            "crop_size": 473,
            "num_workers": 4,
            "seed": 0,
        },
        "coco20i": {
            "train_episodes": 20000,
            "val_episodes": 2000,
            "epochs": 15,
            "batch_size": 16,  # Doubled from 8
            "base_lr": 0.010,  # Scaled proportionally with batch size (was 0.005)
            "crop_size": 641,
            "num_workers": 4,
            "seed": 0,
        },
    }

    # Debug-mode overrides: reduce work and disable workers for fast iteration
    if args.debug:
        logging.info("Debug mode: applying lightweight hyperparameter overrides")
        for d in paper_hparams:
            paper_hparams[d]["train_episodes"] = 4
            paper_hparams[d]["val_episodes"] = 2
            paper_hparams[d]["epochs"] = 1
            paper_hparams[d]["batch_size"] = 2
            paper_hparams[d]["num_workers"] = 0

    datasets = ["pascal5i", "coco20i"]
    backbones = ["resnet50", "resnet101"]
    shots_list = [1, 5]
    folds = [0, 1, 2, 3]

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    results_path = output_dir / "replication_results.json"
    if results_path.exists():
        logging.info("Loading existing results from replication_results.json")
        with open(results_path, 'r') as f:
            data = json.load(f)
            results = data.get("results", {})
    else:
        logging.info("No existing results found, starting fresh")
    if args.download:
        logging.info("Starting dataset downloads")
        download_pascal5i(root=args.data_root)
        coco_root = str(Path(args.data_root) / "coco")
        download_coco2017(root=coco_root, image_set="train")
        download_coco2017(root=coco_root, image_set="val")
        logging.info("Downloads completed")

    if args.prepare_only:
        logging.info("Prepare-only: building datasets and caches, then exiting.")
        for dataset in datasets:
            hparams = paper_hparams[dataset]
            for backbone in backbones:
                for shots in shots_list:
                    key = f"{dataset}_{backbone}_{shots}shot"
                    for fold in folds:
                        try:
                            fold_datasets = build_datasets(
                                dataset=dataset,
                                root=args.data_root,
                                fold=fold,
                                shots=shots,
                                train_episodes=hparams["train_episodes"],
                                val_episodes=hparams["val_episodes"],
                                seed=hparams["seed"],
                            )
                            if fold_datasets is not None:
                                logging.debug(f"Prepared dataset for {dataset} fold {fold} ({key})")
                        except Exception as e:
                            logging.warning(f"Failed preparing {dataset} fold {fold}: {e}")
        logging.info("Preparation complete. Exiting due to --prepare-only.")
        return

    for dataset in datasets:
        hparams = paper_hparams[dataset]
        for backbone in backbones:
            for shots in shots_list:
                key = f"{dataset}_{backbone}_{shots}shot"
                if key not in results:
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
                    )
                    if fold_datasets is None:
                        logging.warning(f"Skipping {dataset} fold {fold}: dataset not available.")
                        continue
                    logging.debug(f"Created dataset for {dataset} fold {fold}")

                    ckpt_path = fold_dir / "model.pt"
                    if str(fold) in results[key]:
                        logging.info(f"Results exist for {key} fold {fold}, skipping evaluation and model loading.")
                        score = results[key][str(fold)]["miou"]
                    else:
                        model = ABCB(
                            backbone_name=backbone, 
                            pretrained_backbone=True,
                            freeze_backbone=False,  # Make backbone trainable
                        )
                        max_steps = 2 if args.debug else None
                        if ckpt_path.exists():
                            logging.info(f"Checkpoint exists for {key} fold {fold}, loading and skipping training.")
                            model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
                            model = model.to(args.device)
                        else:
                            logging.info(f"Training {backbone} on {dataset} with {shots} shots, fold {fold}")
                            metrics_path = fold_dir / "training_metrics.json"
                            train_metrics = train_abcb(
                                model=model,
                                train_ds=fold_datasets["train"],
                                val_ds=fold_datasets["val"],
                                device=args.device,
                                epochs=hparams["epochs"],
                                batch_size=hparams["batch_size"],
                                base_lr=hparams["base_lr"],
                                crop_size=hparams["crop_size"],
                                num_workers=hparams["num_workers"],
                                save_path=str(metrics_path),
                                max_steps=max_steps,
                            )
                            logging.info(f"Finished training {backbone} on {dataset} with {shots} shots, fold {fold}")
                            torch.save(model.state_dict(), ckpt_path)

                        logging.info(f"Starting evaluation for {key} fold {fold}")
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
                            max_steps=max_steps,
                        )
                        results[key][str(fold)] = {"miou": score}
                        logging.info(f"Finished evaluation for {key} fold {fold}")
                        # Save results after each new evaluation
                        payload = {
                            "paper_hparams": paper_hparams,
                            "results": results,
                        }
                        (output_dir / "replication_results.json").write_text(json.dumps(payload, indent=2))
                        if args.push_to_hub:
                            upload_output_dir(
                                output_dir=output_dir,
                                repo_id=args.hf_repo,
                                branch=args.hf_branch,
                            )
                    logging.info(f"{key} fold {fold}: mIoU={score:.4f}")

                if "mean" not in results[key]:
                    logging.debug(f"Calculating mean for {key}")
                    mean_score = sum(v["miou"] for v in results[key].values() if isinstance(v, dict) and "miou" in v) / max(
                        1, len([v for v in results[key].values() if isinstance(v, dict) and "miou" in v])
                    )
                    results[key]["mean"] = {"miou": mean_score}
                    logging.info(f"Mean mIoU for {key}: {mean_score:.4f}")

    payload = {
        "paper_hparams": paper_hparams,
        "results": results,
    }
    (output_dir / "replication_results.json").write_text(json.dumps(payload, indent=2))
    logging.info("Final results saved to replication_results.json")


if __name__ == "__main__":
    main()
