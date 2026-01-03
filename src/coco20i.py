"""COCO-20i episodic dataset utilities."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as F
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm


def generate_coco_masks(root: str, num_workers: int = 1) -> None:
    """Generate semantic masks for COCO-20i from annotations."""
    import multiprocessing
    import numpy as np
    from PIL import Image
    try:
        from pycocotools.coco import COCO
        from pycocotools import mask as mask_utils
    except ImportError:
        raise ImportError("pycocotools is required to generate COCO masks. Install with: pip install pycocotools")

    def _process_image(args):
        ann_file, img_dir, mask_dir, img_id = args
        coco = COCO(str(ann_file))
        img_info = coco.loadImgs(img_id)[0]
        img_path = img_dir / img_info['file_name']
        if not img_path.exists():
            return

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Create semantic mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            if 'segmentation' in ann:
                rle = mask_utils.frPyObjects(ann['segmentation'], img_info['height'], img_info['width'])
                m = mask_utils.decode(rle)
                if len(m.shape) == 3:
                    m = np.sum(m, axis=2) > 0
                mask[m > 0] = ann['category_id']

        # Save mask
        mask_img = Image.fromarray(mask)
        mask_path = mask_dir / f"{img_path.stem}.png"
        mask_img.save(mask_path)

    root_path = Path(root)
    mask_dir = root_path / "masks"
    mask_dir.mkdir(exist_ok=True)

    logging.info(f"Generating COCO masks in {mask_dir}")
    for image_set in ["train2017", "val2017"]:
        ann_file = root_path / "annotations" / f"instances_{image_set}.json"
        img_dir = root_path / "images" / image_set

        if not ann_file.exists() or not img_dir.exists():
            logging.warning(f"Skipping {image_set}: annotations or images not found")
            continue

        coco = COCO(str(ann_file))
        img_ids = coco.getImgIds()
        logging.info(f"Processing {len(img_ids)} images for {image_set}")

        if num_workers > 1:
            with multiprocessing.Pool(num_workers) as pool:
                args_list = [(ann_file, img_dir, mask_dir, img_id) for img_id in img_ids]
                pool.map(_process_image, args_list)
        else:
            from tqdm import tqdm
            for img_id in tqdm(img_ids, desc=f"Generating masks for {image_set}", leave=False):
                _process_image((ann_file, img_dir, mask_dir, img_id))
    logging.info("COCO mask generation completed")


def download_coco2017(root: str, image_set: str = "train") -> None:
    """Download COCO images/annotations via official archives (masks must be prepared separately)."""
    valid_splits = {"train", "val"}
    if image_set not in valid_splits:
        raise ValueError(f"image_set must be one of {valid_splits}")

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    # Download shared annotations zip once.
    ann_file = root_path / "annotations" / f"instances_{image_set}2017.json"
    if not ann_file.exists():
        logging.info(f"Downloading COCO annotations to {root_path}")
        download_and_extract_archive(
            url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            download_root=str(root_path),
            extract_root=str(root_path),
            filename="annotations_trainval2017.zip",
        )
        logging.info("Annotations download completed")

    # Download requested split images.
    img_dir = root_path / "images" / f"{image_set}2017"
    if not img_dir.exists():
        logging.info(f"Downloading COCO {image_set} images to {root_path / 'images'}")
        download_and_extract_archive(
            url=f"http://images.cocodataset.org/zips/{image_set}2017.zip",
            download_root=str(root_path),
            extract_root=str(root_path / "images"),
            filename=f"{image_set}2017.zip",
        )
        logging.info(f"{image_set} images download completed")


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
        image_set = "train2017" if self.train else "val2017"
        image_dir = self.root / "images" / image_set
        mask_dir = self.root / "masks"
        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(
                "Expected COCO-20i layout with 'images/{train2017,val2017}/' and 'masks/' directories."
            )

        # List images and masks once, then match by stem to avoid many stat() calls
        images = _list_images(image_dir)
        mask_files = _list_images(mask_dir)
        mask_stems = {p.stem for p in mask_files}

        pairs: List[Tuple[Path, Path]] = []
        for image_path in tqdm(images, desc="Collecting image-mask pairs", leave=False):
            stem = image_path.stem
            if stem in mask_stems:
                pairs.append((image_path, mask_dir / f"{stem}.png"))

        image_paths = [pair[0] for pair in pairs]
        mask_paths = [pair[1] for pair in pairs]
        return image_paths, mask_paths

    def _load_mask(self, file_path: Path) -> np.ndarray:
        # Open via a binary file object to avoid Path.resolve() overhead
        try:
            with file_path.open("rb") as fh:
                target = Image.open(fh)
                target = target.convert("L")
                target_np = np.array(target, dtype=np.int64)
        except Exception:
            # If a file can't be opened, return an empty mask of zeros
            return np.zeros((0, 0), dtype=np.int64)
        target_np[target_np > 80] = 0
        return target_np

    def _build_class_maps(self, fold: int) -> None:
        cache_path = self.root / f"coco20i_fold{fold}_{'train' if self.train else 'val'}.pt"
        # Debug: report cache presence and try to load safely
        logging.debug(f"Checking cache path: {cache_path} (exists={cache_path.exists()})")
        if cache_path.exists():
            try:
                cached = torch.load(cache_path)
                self.class_img_map = cached["class_img_map"]
                self.img_class_map = cached["img_class_map"]
                self.images = cached["images"]
                self.targets = cached["targets"]
                logging.debug(f"Loaded cache from {cache_path}")
                return
            except Exception as e:
                logging.warning(f"Failed to load cache {cache_path}: {e}. Rebuilding class maps.")

        filtered_images: List[Path] = []
        filtered_targets: List[Path] = []
            # Use tqdm with a known total to show progress correctly
        for idx, mask_path in enumerate(tqdm(self.targets, desc="Building class maps", total=len(self.targets), leave=False)):
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
        generate_masks: bool = False,
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
        mask_dir = Path(root) / "masks"
        if not mask_dir.exists():
            generate_coco_masks(root=root)
        self.reader = Coco20iReader(root=root, fold=fold, train=train)
        self.shots = shots
        self.queries = queries
        self.episodes = episodes
        self.seed = seed
        self.crop_size = 641
        self.class_ids = [cid for cid in self.reader.label_set if len(self.reader.get_img_containing_class(cid)) > 0]

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
            image = F.resize(image, (self.crop_size, self.crop_size))
            target = F.resize(target.unsqueeze(0).float(), (self.crop_size, self.crop_size), interpolation=F.InterpolationMode.NEAREST).squeeze(0).long()
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
