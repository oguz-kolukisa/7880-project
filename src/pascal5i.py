"""Pascal-5i episodic dataset utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

import torchvision

from src.third_party.pascal_5i.pascal5i_reader import Pascal5iReader


def download_pascal5i(root: str, image_set: str = "train") -> None:
    """Download SBD and VOC2012 data required for PASCAL-5i."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    sbd_path = root_path / "sbd"
    sbd_path.mkdir(parents=True, exist_ok=True)
    sbd_ready = (sbd_path / "cls").exists() and (sbd_path / "img").exists()
    if not sbd_ready:
        torchvision.datasets.SBDataset(
            str(sbd_path),
            image_set=image_set,
            mode="segmentation",
            download=True,
        )
    voc_root = root_path / "VOCdevkit" / "VOC2012"
    if not voc_root.exists():
        torchvision.datasets.VOCSegmentation(
            str(root_path),
            image_set="trainval",
            download=True,
        )


class Pascal5iDataset(Dataset):
    """Few-shot episodic dataset for PASCAL-5i.

    Args:
        root: Root directory containing the PASCAL-5i data (SBD + VOC2012).
        fold: Fold index in [0, 3].
        train: Use train split if True, otherwise validation split.
        shots: Number of support images per episode.
        queries: Number of query images per episode.
        episodes: Number of episodes (length of the dataset).
        seed: Base seed for deterministic episode sampling.
        download: If True, download SBD and VOC2012 using torchvision.
    """

    def __init__(
        self,
        root: str,
        fold: int,
        train: bool = True,
        shots: int = 1,
        queries: int = 1,
        episodes: int = 1000,
        seed: int = 0,
        download: bool = False,
    ) -> None:
        super().__init__()
        if shots < 1:
            raise ValueError("shots must be >= 1")
        if queries < 1:
            raise ValueError("queries must be >= 1")
        if queries != 1:
            raise ValueError("queries must be 1 to match the training pipeline")
        if episodes < 1:
            raise ValueError("episodes must be >= 1")

        if download:
            download_pascal5i(root=root)
        self.reader = Pascal5iReader(root=root, fold=fold, train=train)
        self.shots = shots
        self.queries = queries
        self.episodes = episodes
        self.seed = seed
        self.class_ids = list(self.reader.label_set)

    def __len__(self) -> int:
        return self.episodes

    def _sample_indices(
        self, rng: random.Random, class_id: int
    ) -> Tuple[List[int], List[int]]:
        candidates = list(self.reader.get_img_containing_class(class_id))
        total_needed = self.shots + self.queries
        if len(candidates) >= total_needed:
            sampled = rng.sample(candidates, total_needed)
        else:
            sampled = [rng.choice(candidates) for _ in range(total_needed)]
        support_indices = sampled[: self.shots]
        query_indices = sampled[self.shots :]
        return support_indices, query_indices

    def _build_binary_mask(self, target: torch.Tensor, class_id: int) -> torch.Tensor:
        return (target == class_id).long().unsqueeze(0)

    def _load_items(self, indices: Sequence[int], class_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        images: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        for idx in indices:
            image, target = self.reader[idx]
            images.append(image)
            masks.append(self._build_binary_mask(target, class_id))
        return torch.stack(images, dim=0), torch.stack(masks, dim=0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed + idx)
        class_id = rng.choice(self.class_ids)

        support_indices, query_indices = self._sample_indices(rng, class_id)
        support_images, support_masks = self._load_items(support_indices, class_id)
        query_images, query_masks = self._load_items(query_indices, class_id)

        return {
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "support_images": support_images,
            "support_masks": support_masks,
            "query_images": query_images.squeeze(0),
            "query_masks": query_masks.squeeze(0),
        }


__all__ = ["Pascal5iDataset", "download_pascal5i"]
