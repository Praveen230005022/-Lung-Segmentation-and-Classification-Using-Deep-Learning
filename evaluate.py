"""Segmentation and classification metrics, evaluation loops and the `evaluate` command."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm


def safe_div(numerator: float, denominator: float, empty: float = 1.0) -> float:
    """Ratio that returns `empty` when the denominator is 0 (e.g. empty prediction and target)."""
    return float(numerator) / float(denominator) if denominator > 0 else empty


# --------------------------------------------------------------------------- segmentation
def dice_per_image(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Dice per image, averaged over foreground classes (1..K-1).

    pred/target: (N, H, W) class IDs. If a class is absent from both prediction and target
    the score for that class is 1.0 (nothing to find, nothing predicted).
    """
    scores = []
    for c in range(1, num_classes):
        p, t = (pred == c), (target == c)
        inter = (p & t).flatten(1).sum(1).float()
        total = p.flatten(1).sum(1).float() + t.flatten(1).sum(1).float()
        scores.append(torch.where(total > 0, 2 * inter / total.clamp(min=1), torch.ones_like(total)))
    return torch.stack(scores, dim=1).mean(dim=1)


class SegmentationMeter:
    """Accumulates a pixel confusion matrix and per-image Dice over a dataset."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        self.dice_scores: list[float] = []

    def update(self, logits_or_pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits_or_pred.argmax(1) if logits_or_pred.dim() == 4 else logits_or_pred
        pred, target = pred.detach().cpu().long(), target.detach().cpu().long()
        k = self.num_classes
        self.confusion += torch.bincount(target.flatten() * k + pred.flatten(), minlength=k * k).reshape(k, k)
        self.dice_scores += dice_per_image(pred, target, k).tolist()

    def compute(self) -> dict[str, float]:
        cm = self.confusion.double()
        total = cm.sum().item()
        tp = cm.diag()
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        tn = total - tp - fp - fn
        fg = range(1, self.num_classes)
        mean = lambda values: float(np.mean(values)) if values else 0.0  # noqa: E731
        return {
            "dice": mean(self.dice_scores),
            "dice_global": mean([safe_div(2 * tp[c], 2 * tp[c] + fp[c] + fn[c]) for c in fg]),
            "iou": mean([safe_div(tp[c], tp[c] + fp[c] + fn[c]) for c in fg]),
            "pixel_accuracy": safe_div(tp.sum(), total, 0.0),
            "precision": mean([safe_div(tp[c], tp[c] + fp[c], 1.0 if tp[c] + fn[c] == 0 else 0.0) for c in fg]),
            "recall": mean([safe_div(tp[c], tp[c] + fn[c]) for c in fg]),
            "specificity": mean([safe_div(tn[c], tn[c] + fp[c]) for c in fg]),
        }


@torch.no_grad()
def evaluate_segmentation(model, loader, device, num_classes: int, loss_fn=None, desc: str = "eval") -> dict:
    model.eval()
    meter, losses = SegmentationMeter(num_classes), []
    for batch in tqdm(loader, desc=desc, leave=False):
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        logits = model(images)
        if loss_fn is not None:
            losses.append(loss_fn(logits, masks).item())
        meter.update(logits, masks)
    metrics = meter.compute()
    if losses:
        metrics["loss"] = float(np.mean(losses))
    return metrics


# --------------------------------------------------------------------------- classification
def classification_metrics(y_true, y_pred, num_classes: int) -> dict:
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


@torch.no_grad()
def evaluate_classification(model, loader, device, num_classes: int, loss_fn=None, desc: str = "eval") -> dict:
    model.eval()
    y_true, y_pred, losses = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        images, labels = batch["image"].to(device), batch["label"].to(device)
        logits = model(images)
        if loss_fn is not None:
            losses.append(loss_fn(logits, labels).item())
        y_true += labels.cpu().tolist()
        y_pred += logits.argmax(1).cpu().tolist()
    metrics = classification_metrics(y_true, y_pred, num_classes)
    if losses:
        metrics["loss"] = float(np.mean(losses))
    return metrics


# --------------------------------------------------------------------------- reporting
def format_metrics(metrics: dict, class_names: list[str] | None = None) -> str:
    lines = []
    for key, value in metrics.items():
        if key == "confusion_matrix":
            names = class_names or [str(i) for i in range(len(value))]
            width = max(len(n) for n in names) + 2
            lines.append("confusion_matrix (rows = true, cols = predicted):")
            lines.append(" " * width + "".join(n.rjust(width) for n in names))
            lines += [names[i].ljust(width) + "".join(str(v).rjust(width) for v in row) for i, row in enumerate(value)]
        else:
            lines.append(f"{key:>15}: {value:.4f}")
    return "\n".join(lines)


def run_evaluation(cfg: dict, device: torch.device, split: str = "test") -> None:
    """Evaluate every trained model (segmentation and/or classification) on a dataset split."""
    from dataset import DatasetError, load_splits, make_loader, resolve_path
    from train import load_model

    seg_path = resolve_path(cfg, cfg["segmentation"]["checkpoint"])
    cls_path = resolve_path(cfg, cfg["classification"]["checkpoint"])
    if not seg_path.is_file() and not cls_path.is_file():
        raise FileNotFoundError(
            f"No trained checkpoints found ({seg_path}, {cls_path}). "
            "Run `python main.py train-segmentation` and/or `python main.py train-classification` first."
        )

    for task, path in (("segmentation", seg_path), ("classification", cls_path)):
        if not path.is_file():
            print(f"\nSkipping {task}: no checkpoint at {path}")
            continue
        model, ckpt = load_model(path, task, device)
        # Preprocess exactly as during training (settings stored in the checkpoint).
        eval_cfg = {**cfg, "image": ckpt["image"],
                    "classification": {**cfg["classification"], "masked_input": ckpt["masked_input"],
                                       "class_names": ckpt["class_names"] or cfg["classification"]["class_names"]}}
        items = load_splits(eval_cfg, task)[split]
        if not items:
            raise DatasetError(f"The '{split}' split has no {task} samples to evaluate.")
        loader = make_loader(items, eval_cfg, task, train=False, batch_size=cfg[task]["batch_size"])
        print(f"\n{task.capitalize()} results on '{split}' ({len(items)} images, checkpoint epoch {ckpt['epoch']}):")
        if task == "segmentation":
            metrics = evaluate_segmentation(model, loader, device, ckpt["num_classes"])
            print(format_metrics(metrics))
            target = cfg["segmentation"]["target_dice"]
            status = "meets" if metrics["dice"] >= target else "is below"
            print(f"Mean Dice {metrics['dice']:.4f} {status} the target benchmark of {target:.2f}.")
        else:
            metrics = evaluate_classification(model, loader, device, ckpt["num_classes"])
            print(format_metrics(metrics, ckpt["class_names"]))
