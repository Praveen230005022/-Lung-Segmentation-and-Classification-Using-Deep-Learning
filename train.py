"""Training loops, losses, class weighting, seeding and checkpoint handling."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import DatasetError, LungDataset, load_splits, make_loader, resolve_path
from evaluate import evaluate_classification, evaluate_segmentation
from model import LungClassifier, UNet, build_classifier, build_unet, count_parameters


# --------------------------------------------------------------------------- setup
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(setting: str = "auto") -> torch.device:
    setting = str(setting).lower()
    if setting == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available; using CPU.")
        return torch.device("cpu")
    if setting in ("cpu", "cuda"):
        return torch.device(setting)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- losses
def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """1 - soft Dice over foreground classes (computed over the whole batch)."""
    probs = logits.softmax(dim=1)[:, 1:]
    one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()[:, 1:]
    inter = (probs * one_hot).sum(dim=(0, 2, 3))
    total = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    return 1 - ((2 * inter + eps) / (total + eps)).mean()


class SegmentationLoss(nn.Module):
    """Class-weighted cross-entropy, optionally plus a weighted soft Dice loss."""

    def __init__(self, class_weights: torch.Tensor | None, num_classes: int, dice_weight: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.num_classes, self.dice_weight = num_classes, dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.ce(logits, target)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * soft_dice_loss(logits, target, self.num_classes)
        return loss


def inverse_frequency(counts: np.ndarray) -> np.ndarray:
    """Weights proportional to 1 / frequency, normalized to mean 1. Absent classes get weight 1."""
    counts = counts.astype(np.float64)
    weights = np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 1.0)
    return weights / weights.mean()


def resolve_class_weights(setting, num_classes: int, auto_counts) -> torch.Tensor:
    """`setting` is 'auto', 'none'/null, or a list of floats."""
    if setting is None or str(setting).lower() == "none":
        return torch.ones(num_classes)
    if str(setting).lower() == "auto":
        counts = auto_counts()
        weights = inverse_frequency(counts)
        print("Class counts: " + ", ".join(str(int(c)) for c in counts)
              + " -> weights: " + ", ".join(f"{w:.3f}" for w in weights))
        return torch.tensor(weights, dtype=torch.float32)
    weights = [float(w) for w in setting]
    if len(weights) != num_classes or min(weights) <= 0:
        raise ValueError(f"class_weights must be 'auto' or {num_classes} positive numbers, got {setting}")
    return torch.tensor(weights, dtype=torch.float32)


def pixel_counts(items: list[dict], cfg: dict) -> np.ndarray:
    """Pixel count per segmentation class on resized (non-augmented) training masks."""
    k = cfg["segmentation"]["num_classes"]
    counts = np.zeros(k, dtype=np.int64)
    dataset = LungDataset(items, cfg, "segmentation", train=False)
    for i in tqdm(range(len(dataset)), desc="class weights", leave=False):
        counts += np.bincount(dataset[i]["mask"].numpy().ravel(), minlength=k)[:k]
    return counts


def make_optimizer(model: nn.Module, section: dict) -> torch.optim.Optimizer:
    name = str(section.get("optimizer", "adam")).lower()
    args = dict(lr=float(section["learning_rate"]), weight_decay=float(section.get("weight_decay", 0.0)))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **args)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **args)
    raise ValueError(f"Unsupported optimizer '{name}' (use adam or adamw).")


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(path: Path, model: nn.Module, optimizer, epoch: int, best: float, history: dict,
                    cfg: dict, task: str) -> None:
    """Atomically write model + optimizer state and the settings needed for inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_metric": best,
        "history": history,
        "in_channels": model.in_channels,
        "num_classes": model.num_classes,
        "base_channels": model.base_channels,
        "image": dict(cfg["image"]),
        "class_names": list(cfg["classification"]["class_names"]) if task == "classification" else None,
        "masked_input": bool(cfg["classification"]["masked_input"]) if task == "classification" else False,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=True)


def load_model(path: str | Path, task: str, device: torch.device) -> tuple[nn.Module, dict]:
    """Rebuild the network from a checkpoint and load its weights (eval mode)."""
    ckpt = load_checkpoint(path, device)
    if ckpt.get("task") != task:
        raise ValueError(f"{path} is a '{ckpt.get('task')}' checkpoint, expected '{task}'.")
    cls = UNet if task == "segmentation" else LungClassifier
    model = cls(ckpt["in_channels"], ckpt["num_classes"], ckpt["base_channels"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


# --------------------------------------------------------------------------- generic loop
def train_one_epoch(model, loader, loss_fn, optimizer, device, target_key: str, desc: str) -> float:
    model.train()
    total, count = 0.0, 0
    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        images, targets = batch["image"].to(device), batch[target_key].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Loss became NaN/inf; lower the learning rate or check the data.")
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
        count += images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(count, 1)


def _prepare(cfg: dict, task: str, device: torch.device, resume: bool, build):
    """Load data (fails clearly if missing), build model/optimizer and optionally resume."""
    splits = load_splits(cfg, task)
    if not splits["train"]:
        raise DatasetError(f"No training samples for {task}.")
    if not splits["val"]:
        raise DatasetError(f"No validation samples for {task}; add 'val' rows or set data.val_fraction > 0.")
    section = cfg[task]
    model = build(cfg).to(device)
    optimizer = make_optimizer(model, section)
    ckpt_path = resolve_path(cfg, section["checkpoint"])
    start, best, history = 1, -math.inf, {"train_loss": [], "val_loss": [], "val_metric": []}
    if resume:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"--resume given but no checkpoint at {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path, device)
        model.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start, best, history = ckpt["epoch"] + 1, ckpt["best_metric"], ckpt["history"]
        print(f"Resumed from {ckpt_path} (best epoch {ckpt['epoch']}, metric {best:.4f}).")
        if start > int(section["epochs"]):
            print(f"Already trained for {ckpt['epoch']} of {section['epochs']} epochs; "
                  "increase the epochs in config.yaml to continue.")
    print(f"Model parameters: {count_parameters(model):,} | device: {device}")
    return splits, model, optimizer, ckpt_path, start, best, history


# --------------------------------------------------------------------------- segmentation
def train_segmentation(cfg: dict, device: torch.device, resume: bool = False) -> dict:
    """Train the U-Net; keep the checkpoint with the best validation Dice."""
    set_seed(cfg["seed"])
    section, k = cfg["segmentation"], cfg["segmentation"]["num_classes"]
    splits, model, optimizer, ckpt_path, start, best, history = _prepare(cfg, "segmentation", device, resume, build_unet)

    weights = resolve_class_weights(section["class_weights"], k, lambda: pixel_counts(splits["train"], cfg))
    loss_fn = SegmentationLoss(weights.to(device), k, float(section["dice_loss_weight"]))
    train_loader = make_loader(splits["train"], cfg, "segmentation", True, section["batch_size"])
    val_loader = make_loader(splits["val"], cfg, "segmentation", False, section["batch_size"])

    epochs = int(section["epochs"])
    for epoch in range(start, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, "mask", f"epoch {epoch}/{epochs}")
        val = evaluate_segmentation(model, val_loader, device, k, loss_fn, desc="validate")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_metric"].append(val["dice"])
        improved = val["dice"] > best
        if improved:
            best = val["dice"]
            save_checkpoint(ckpt_path, model, optimizer, epoch, best, history, cfg, "segmentation")
        print(f"Epoch {epoch:3d}/{epochs} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
              f"val Dice {val['dice']:.4f} | IoU {val['iou']:.4f}" + ("  * saved" if improved else ""))

    print(f"\nBest validation Dice: {best:.4f} (target benchmark {section['target_dice']:.2f}). "
          f"Checkpoint: {ckpt_path}")
    return history


# --------------------------------------------------------------------------- classification
def train_classification(cfg: dict, device: torch.device, resume: bool = False) -> dict:
    """Train the CNN classifier; keep the checkpoint with the best validation macro F1."""
    set_seed(cfg["seed"])
    section = cfg["classification"]
    k = len(section["class_names"])
    if k < 2:
        raise ValueError("classification.class_names needs at least two classes.")
    splits, model, optimizer, ckpt_path, start, best, history = _prepare(
        cfg, "classification", device, resume, build_classifier)

    def label_counts() -> np.ndarray:
        return np.bincount([item["label"] for item in splits["train"]], minlength=k)

    weights = resolve_class_weights(section["class_weights"], k, label_counts)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    train_loader = make_loader(splits["train"], cfg, "classification", True, section["batch_size"])
    val_loader = make_loader(splits["val"], cfg, "classification", False, section["batch_size"])

    epochs = int(section["epochs"])
    for epoch in range(start, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, "label", f"epoch {epoch}/{epochs}")
        val = evaluate_classification(model, val_loader, device, k, loss_fn, desc="validate")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_metric"].append(val["f1_macro"])
        improved = val["f1_macro"] > best
        if improved:
            best = val["f1_macro"]
            save_checkpoint(ckpt_path, model, optimizer, epoch, best, history, cfg, "classification")
        print(f"Epoch {epoch:3d}/{epochs} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
              f"val acc {val['accuracy']:.4f} | macro F1 {val['f1_macro']:.4f}" + ("  * saved" if improved else ""))

    print(f"\nBest validation macro F1: {best:.4f}. Checkpoint: {ckpt_path}")
    return history
