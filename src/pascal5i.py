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
    import tarfile
    import urllib.request
    
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    
    # Download SBD
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
    
    # Download VOC2012 from mirror
    voc_root = root_path / "VOCdevkit" / "VOC2012"
    if not voc_root.exists():
        voc_url = "https://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar"
        voc_tar = root_path / "VOCtrainval_11-May-2012.tar"
        
        if not voc_tar.exists():
            print(f"Downloading VOC2012 from {voc_url}...")
            urllib.request.urlretrieve(voc_url, voc_tar)
            print(f"Downloaded to {voc_tar}")
        
        print(f"Extracting {voc_tar}...")
        with tarfile.open(voc_tar, "r") as tar:
            tar.extractall(root_path)
        print(f"Extracted VOC2012 to {root_path / 'VOCdevkit'}")
        
        # Clean up tar file
        voc_tar.unlink()
        print("Removed tar file")


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
        """Build binary mask for the given class_id.
        
        For validation: target is remapped to {0, 1, 2, 3, 4, 5} where 0 is background
        and 1-5 correspond to the 5 validation classes. We need to map the original
        class_id to its position in val_label_set.
        
        For training: target has val classes set to 0, and classes > max(val_label_set)
        are shifted down by 5. So original class 6 becomes 1, class 7 becomes 2, etc.
        We need to apply the same transformation to class_id before comparing.
        """
        if not self.reader.train:
            # Validation: remap class_id to its index in val_label_set (1-5)
            if class_id in self.reader.val_label_set:
                remapped_id = self.reader.val_label_set.index(class_id) + 1
                return (target == remapped_id).long().unsqueeze(0)
            else:
                # Class not in validation set, return all zeros
                return torch.zeros_like(target).unsqueeze(0)
        else:
            # Training: apply the same offset that set_bg_pixel applies
            # Classes 1-5 (val_label_set) are set to 0
            # Classes > 5 are shifted down by 5
            max_val_label = max(self.reader.val_label_set)
            if class_id <= max_val_label:
                # This is a validation class, which was set to 0
                return torch.zeros_like(target).unsqueeze(0)
            else:
                # Apply the same shift: class_id - 5
                remapped_id = class_id - 5
                return (target == remapped_id).long().unsqueeze(0)

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
