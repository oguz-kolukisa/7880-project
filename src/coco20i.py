"""COCO-20i episodic dataset utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision


def download_coco2017(root: str, image_set: str = "train") -> None:
    """Download COCO images/annotations via torchvision (masks must be prepared separately)."""
    ann_file = f"{root}/annotations/instances_{image_set}2017.json"
    img_dir = f"{root}/{image_set}2017"
    torchvision.datasets.CocoDetection(
        root=img_dir,
        annFile=ann_file,
        download=True,
    )


def _list_images(image_dir: Path) -> List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png")
    images: List[Path] = []
    for ext in exts:
        images.extend(sorted(image_dir.glob(ext)))
    return images


class Coco20iReader:
    """Reader for COCO-style semantic masks stored as PNGs with class IDs 1..80."""

    def __init__(self, root: str, fold: int, train: bool = True) -> None:
        if fold < 0 or fold > 3:
            raise ValueError("fold must be in [0, 3]")
        self.root = Path(root)
        self.train = train

        self.images, self.targets = self._collect_pairs()
        self.val_label_set = list(range(fold * 20 + 1, fold * 20 + 21))
        self.train_label_set = [i for i in range(1, 81) if i not in self.val_label_set]
        self.label_set = self.train_label_set if train else self.val_label_set
        self.to_tensor = torchvision.transforms.ToTensor()

        self.class_img_map: Dict[int, List[int]] = {cid: [] for cid in self.label_set}
        self.img_class_map: Dict[int, List[int]] = {}
        self._build_class_maps(fold)

    def _collect_pairs(self) -> Tuple[List[Path], List[Path]]:
        image_dir = self.root / "images"
        mask_dir = self.root / "masks"
        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(
                "Expected COCO-20i layout with 'images/' and 'masks/' directories."
            )

        images = _list_images(image_dir)
        pairs: List[Tuple[Path, Path]] = []
        for image_path in images:
            mask_path = mask_dir / f"{image_path.stem}.png"
            if mask_path.exists():
                pairs.append((image_path, mask_path))
        image_paths = [pair[0] for pair in pairs]
        mask_paths = [pair[1] for pair in pairs]
        return image_paths, mask_paths

    def _load_mask(self, file_path: Path) -> np.ndarray:
        target = Image.open(file_path)
        target_np = np.array(target, dtype=np.int64)
        target_np[target_np > 80] = 0
        return target_np

    def _build_class_maps(self, fold: int) -> None:
        cache_path = self.root / f"coco20i_fold{fold}_{'train' if self.train else 'val'}.pt"
        if cache_path.exists():
            cached = torch.load(cache_path)
            self.class_img_map = cached["class_img_map"]
            self.img_class_map = cached["img_class_map"]
            self.images = cached["images"]
            self.targets = cached["targets"]
            return

        filtered_images: List[Path] = []
        filtered_targets: List[Path] = []
        for idx, mask_path in enumerate(self.targets):
            mask = self._load_mask(mask_path)
            appended = False
            for class_id in self.label_set:
                if class_id in mask:
                    if not appended:
                        filtered_images.append(self.images[idx])
                        filtered_targets.append(mask_path)
                        appended = True
                    image_index = len(filtered_images) - 1
                    self.class_img_map[class_id].append(image_index)
                    self.img_class_map.setdefault(image_index, []).append(class_id)

        self.images = filtered_images
        self.targets = filtered_targets
        torch.save(
            {
                "class_img_map": self.class_img_map,
                "img_class_map": self.img_class_map,
                "images": self.images,
                "targets": self.targets,
            },
            cache_path,
        )

    def __len__(self) -> int:
        return len(self.images)

    def get_img_containing_class(self, class_id: int) -> List[int]:
        return self.class_img_map[class_id]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[idx]).convert("RGB")
        mask_np = self._load_mask(self.targets[idx])
        image_tensor = self.to_tensor(image)
        mask_tensor = torch.as_tensor(mask_np, dtype=torch.long)
        return image_tensor, mask_tensor


class Coco20iDataset(Dataset):
    """Few-shot episodic dataset for COCO-20i.

    Args:
        root: Root directory containing the COCO-20i data.
        fold: Fold index in [0, 3].
        train: Use train split if True, otherwise validation split.
        shots: Number of support images per episode.
        queries: Number of query images per episode.
        episodes: Number of episodes (length of the dataset).
        seed: Base seed for deterministic episode sampling.
        download: If True, download COCO images/annotations via torchvision.
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
            download_coco2017(root=root)
        self.reader = Coco20iReader(root=root, fold=fold, train=train)
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


__all__ = ["Coco20iDataset", "Coco20iReader", "download_coco2017"]
