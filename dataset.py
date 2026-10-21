"""Dataset loading, validation, patient-level splitting, preprocessing and augmentation.

Nothing here touches the disk until a manifest is actually loaded, so models and tests
can run without any dataset.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DatasetError(Exception):
    """Raised with a readable message when the dataset is missing or invalid."""


# --------------------------------------------------------------------------- config helpers
def resolve_path(cfg: dict, value: str | Path) -> Path:
    """Resolve a config path relative to the folder containing config.yaml."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(cfg["_base_dir"]) / path


# --------------------------------------------------------------------------- image handling
def to_float_image(array: np.ndarray, in_channels: int) -> np.ndarray:
    """Convert any 8/16-bit grayscale or colour array to float32 HxWxC scaled to [0, 1]."""
    img = np.asarray(array, dtype=np.float32)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    if img.ndim == 3:
        img = img[..., :3]
        if in_channels == 1:
            img = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if img.ndim == 2:
        img = img[..., None]
        if in_channels == 3:
            img = np.repeat(img, 3, axis=2)
    if img.ndim != 3 or img.shape[2] != in_channels:
        raise DatasetError(f"Unsupported image shape {np.asarray(array).shape} for in_channels={in_channels}")
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)  # constant image -> zeros
    return img.astype(np.float32)


def read_image(path: Path, in_channels: int) -> np.ndarray:
    try:
        with Image.open(path) as im:
            array = np.array(im)
    except Exception as exc:  # unreadable / corrupt file
        raise DatasetError(f"Cannot read image {path}: {exc}") from exc
    return to_float_image(array, in_channels)


def mask_to_class_ids(array: np.ndarray, num_classes: int, threshold: int) -> np.ndarray:
    """Binary masks are thresholded; multi-class masks must already contain class IDs."""
    mask = np.asarray(array)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if num_classes == 2:
        return (mask > threshold).astype(np.uint8)
    if mask.min() < 0 or mask.max() >= num_classes:
        raise DatasetError(f"Mask values {mask.min()}..{mask.max()} are outside 0..{num_classes - 1}")
    return mask.astype(np.uint8)


def read_mask(path: Path, num_classes: int, threshold: int) -> np.ndarray:
    try:
        with Image.open(path) as im:
            array = np.array(im)
    except Exception as exc:
        raise DatasetError(f"Cannot read mask {path}: {exc}") from exc
    return mask_to_class_ids(array, num_classes, threshold)


def build_transform(cfg: dict, train: bool) -> A.Compose:
    """Resize (+ synchronized spatial augmentation for training). Masks use nearest-neighbour."""
    size = cfg["image"]["size"]
    steps = [A.Resize(size, size)]
    aug = cfg["augmentation"]
    if train and aug.get("enabled", False):
        steps += [
            A.HorizontalFlip(p=aug["horizontal_flip"]),
            A.Affine(
                scale=(1 - aug["scale_limit"], 1 + aug["scale_limit"]),
                translate_percent=(-aug["shift_limit"], aug["shift_limit"]),
                rotate=(-aug["rotate_limit"], aug["rotate_limit"]),
                border_mode=0,  # constant padding
                fill=0,
                fill_mask=0,
                p=aug["probability"],
            ),
        ]
    return A.Compose(steps)


def normalize(image: np.ndarray, cfg: dict) -> torch.Tensor:
    """[0, 1] HxWxC float -> standardized CxHxW tensor."""
    mean, std = cfg["image"]["normalize_mean"], cfg["image"]["normalize_std"]
    return torch.from_numpy(((image - mean) / std).transpose(2, 0, 1).copy()).float()


def apply_lung_mask(image: torch.Tensor, mask: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Keep lung pixels (mask > 0) and set the rest to the normalized value of black."""
    background = (0.0 - cfg["image"]["normalize_mean"]) / cfg["image"]["normalize_std"]
    keep = (mask > 0).unsqueeze(-3).to(image.dtype)
    return image * keep + background * (1 - keep)


def preprocess_for_inference(path: str | Path, cfg: dict) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load one image for prediction. Returns (1xCxHxW tensor, original (H, W))."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Input image not found: {path}")
    image = read_image(path, cfg["image"]["in_channels"])
    resized = build_transform(cfg, train=False)(image=image)["image"]
    return normalize(resized, cfg).unsqueeze(0), image.shape[:2]


# --------------------------------------------------------------------------- manifest
def label_to_index(value: str, class_names: list[str]) -> int:
    value = str(value).strip()
    if value in class_names:
        return class_names.index(value)
    try:
        index = int(float(value))
    except ValueError:
        raise DatasetError(f"Invalid label '{value}'. Use one of {class_names} or 0..{len(class_names) - 1}.")
    if not 0 <= index < len(class_names) or float(value) != index:
        raise DatasetError(f"Label {value} is outside 0..{len(class_names) - 1}.")
    return index


def read_manifest(cfg: dict) -> list[dict]:
    manifest = resolve_path(cfg, cfg["data"]["manifest"])
    if not manifest.is_file():
        raise DatasetError(
            f"Dataset manifest not found: {manifest}\n"
            "Add your dataset and create a CSV with columns such as "
            "image_path,mask_path,label,patient_id,split (see README.md), "
            "or update data.manifest in config.yaml."
        )
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [{k.strip(): (v or "").strip() for k, v in row.items() if k} for row in reader]
        columns = [c.strip() for c in (reader.fieldnames or [])]
    if not rows:
        raise DatasetError(f"Manifest {manifest} has no rows.")
    image_col = cfg["data"]["image_column"]
    if image_col not in columns:
        raise DatasetError(f"Manifest is missing the image column '{image_col}'. Found: {columns}")
    return rows


def split_rows(rows: list[dict], cfg: dict) -> dict[str, list[dict]]:
    """Use the split column if present, else a seeded patient-level split. Rejects patient leakage."""
    d = cfg["data"]
    split_col, patient_col = d["split_column"], d["patient_column"]
    has_patients = all(r.get(patient_col) for r in rows)

    if all(r.get(split_col) for r in rows):
        splits = {s: [] for s in SPLITS}
        for i, row in enumerate(rows, start=2):
            name = row[split_col].lower()
            if name in ("valid", "validation"):
                name = "val"
            if name not in splits:
                raise DatasetError(f"Manifest line {i}: invalid split '{row[split_col]}' (use train/val/test).")
            splits[name].append(row)
    else:
        if not has_patients:
            print(f"Warning: '{patient_col}' is missing for some rows; splitting by image. "
                  "Images from the same patient may leak between splits.")
        groups = sorted({r[patient_col] if has_patients else str(i) for i, r in enumerate(rows)})
        random.Random(cfg["seed"]).shuffle(groups)
        n_test = round(len(groups) * d["test_fraction"])
        n_val = round(len(groups) * d["val_fraction"])
        assign = {g: "test" for g in groups[:n_test]}
        assign.update({g: "val" for g in groups[n_test:n_test + n_val]})
        splits = {s: [] for s in SPLITS}
        for i, row in enumerate(rows):
            splits[assign.get(row[patient_col] if has_patients else str(i), "train")].append(row)

    if has_patients:
        owners: dict[str, set[str]] = {}
        for name, items in splits.items():
            for row in items:
                owners.setdefault(row[patient_col], set()).add(name)
        leaked = sorted(p for p, s in owners.items() if len(s) > 1)
        if leaked:
            raise DatasetError(f"Patient-level leakage: {len(leaked)} patient(s) appear in several splits, "
                               f"e.g. {leaked[:5]}. Assign each patient to exactly one split.")
    return splits


def validate_rows(rows: list[dict], cfg: dict, task: str) -> list[dict]:
    """Keep rows that have the fields the task needs and check files/labels. Returns cleaned rows."""
    d = cfg["data"]
    needs_mask = task == "segmentation" or cfg["classification"]["masked_input"]
    needed = [d["image_column"]] + ([d["mask_column"]] if needs_mask else [])
    if task == "classification":
        needed.append(d["label_column"])
    usable = [r for r in rows if all(r.get(c) for c in needed)]
    if len(usable) < len(rows):
        print(f"Note: {len(rows) - len(usable)} row(s) skipped for {task} (missing {', '.join(needed)}).")

    root = resolve_path(cfg, d["root"])
    errors: list[str] = []
    cleaned = []
    for row in usable:
        item = {"image": root / row[d["image_column"]], "patient": row.get(d["patient_column"], "")}
        if needs_mask:
            item["mask"] = root / row[d["mask_column"]]
        for key in ("image", "mask"):
            if key in item:
                if not item[key].is_file():
                    errors.append(f"missing {key}: {item[key]}")
                elif item[key].suffix.lower() not in IMAGE_SUFFIXES:
                    errors.append(f"unsupported {key} format: {item[key]}")
        if task == "classification":
            try:
                item["label"] = label_to_index(row[d["label_column"]], cfg["classification"]["class_names"])
            except DatasetError as exc:
                errors.append(str(exc))
        cleaned.append(item)
    if errors:
        shown = "\n  ".join(errors[:10])
        raise DatasetError(f"{len(errors)} dataset problem(s):\n  {shown}" + ("\n  ..." if len(errors) > 10 else ""))
    return cleaned


def load_splits(cfg: dict, task: str) -> dict[str, list[dict]]:
    """Read, split and validate the manifest for 'segmentation' or 'classification'."""
    splits = split_rows(read_manifest(cfg), cfg)
    result = {name: validate_rows(rows, cfg, task) for name, rows in splits.items()}
    counts = ", ".join(f"{k}={len(v)}" for k, v in result.items())
    print(f"{task.capitalize()} data: {counts}")
    if task == "classification" and result["train"]:
        dist = Counter(item["label"] for item in result["train"])
        names = cfg["classification"]["class_names"]
        print("Train label counts: " + ", ".join(f"{names[k]}={v}" for k, v in sorted(dist.items())))
    return result


# --------------------------------------------------------------------------- torch dataset
class LungDataset(Dataset):
    """Returns dicts with 'image' (CxHxW float), plus 'mask' (HxW long) and/or 'label' (long)."""

    def __init__(self, items: list[dict], cfg: dict, task: str, train: bool):
        self.items, self.cfg, self.task = items, cfg, task
        self.transform = build_transform(cfg, train)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item, cfg = self.items[index], self.cfg
        image = read_image(item["image"], cfg["image"]["in_channels"])
        sample: dict = {}
        if "mask" in item:
            mask = read_mask(item["mask"], cfg["segmentation"]["num_classes"], cfg["data"]["mask_threshold"])
            if mask.shape != image.shape[:2]:
                raise DatasetError(f"Shape mismatch: image {item['image']} is {image.shape[:2]}, "
                                   f"mask {item['mask']} is {mask.shape}")
            out = self.transform(image=image, mask=mask)
            tensor, mask_t = normalize(out["image"], cfg), torch.from_numpy(out["mask"].copy()).long()
            if self.task == "segmentation":
                sample["mask"] = mask_t
            else:  # classification with masked_input
                tensor = apply_lung_mask(tensor, mask_t, cfg)
        else:
            tensor = normalize(self.transform(image=image)["image"], cfg)
        sample["image"] = tensor
        if "label" in item:
            sample["label"] = torch.tensor(item["label"], dtype=torch.long)
        return sample


def make_loader(items: list[dict], cfg: dict, task: str, train: bool, batch_size: int) -> DataLoader:
    generator = torch.Generator().manual_seed(cfg["seed"])

    def seed_worker(worker_id: int) -> None:
        seed = (torch.initial_seed() + worker_id) % 2**32
        np.random.seed(seed)
        random.seed(seed)

    return DataLoader(
        LungDataset(items, cfg, task, train),
        batch_size=batch_size,
        shuffle=train,
        num_workers=cfg["data"]["num_workers"],
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        drop_last=train and len(items) > batch_size,  # avoid 1-sample BatchNorm batches
    )
