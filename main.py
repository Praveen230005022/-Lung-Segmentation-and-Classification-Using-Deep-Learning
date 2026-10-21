"""Command-line entry point.

    python main.py test                      # dataset-free functional checks
    python main.py train-segmentation [--resume]
    python main.py train-classification [--resume]
    python main.py evaluate [--split test]
    python main.py predict --input path/to/image.png [--output mask.png] [--show]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset import (DatasetError, apply_lung_mask, build_transform, mask_to_class_ids, normalize,
                     preprocess_for_inference, read_manifest, resolve_path, split_rows, to_float_image)
from evaluate import SegmentationMeter, classification_metrics, run_evaluation
from model import LungClassifier, UNet, build_classifier, build_unet, count_parameters
from train import (SegmentationLoss, get_device, inverse_frequency, load_model, make_optimizer,
                   save_checkpoint, set_seed, train_classification, train_segmentation)

REQUIRED_SECTIONS = ("data", "image", "augmentation", "segmentation", "classification")


def load_config(path: str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"config.yaml is missing section(s): {missing}")
    cfg.setdefault("seed", 42)
    cfg.setdefault("device", "auto")
    cfg["_base_dir"] = str(path.resolve().parent)
    return cfg


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def segment(model: UNet, image: torch.Tensor) -> torch.Tensor:
    """(1, C, H, W) normalized image -> (1, H, W) predicted class IDs."""
    model.eval()
    return model(image).argmax(dim=1)


@torch.no_grad()
def classify(model: LungClassifier, image: torch.Tensor) -> torch.Tensor:
    """(N, C, H, W) image -> (N, K) class probabilities."""
    model.eval()
    return model(image).softmax(dim=1)


def predict(cfg: dict, device: torch.device, input_path: str, output: str | None, show: bool) -> None:
    seg_path = resolve_path(cfg, cfg["segmentation"]["checkpoint"])
    if not seg_path.is_file():
        raise FileNotFoundError(f"No trained segmentation model at {seg_path}. "
                                "Run `python main.py train-segmentation` with a real dataset first.")
    unet, seg_ckpt = load_model(seg_path, "segmentation", device)
    image, original_size = preprocess_for_inference(input_path, {**cfg, "image": seg_ckpt["image"]})
    mask = segment(unet, image.to(device))
    lung_fraction = (mask > 0).float().mean().item()
    print(f"Segmentation: lung pixels = {lung_fraction:.1%} of the image "
          f"({seg_ckpt['num_classes']} classes, model epoch {seg_ckpt['epoch']}).")
    if lung_fraction == 0:
        print("Warning: no lung region was detected.")

    cls_path = resolve_path(cfg, cfg["classification"]["checkpoint"])
    if not cls_path.is_file():
        print(f"Classification cannot be performed: no trained classifier at {cls_path}. "
              "The segmentation mask alone does not provide a diagnosis label.")
    else:
        classifier, cls_ckpt = load_model(cls_path, "classification", device)
        cls_cfg = {**cfg, "image": cls_ckpt["image"]}
        cls_image, _ = preprocess_for_inference(input_path, cls_cfg)
        cls_image = cls_image.to(device)
        if cls_ckpt["masked_input"]:
            if cls_image.shape[-2:] != mask.shape[-2:]:
                mask_for_cls = torch.nn.functional.interpolate(mask[None].float(), size=cls_image.shape[-2:],
                                                               mode="nearest")[0].long()
            else:
                mask_for_cls = mask
            cls_image = apply_lung_mask(cls_image, mask_for_cls, cls_cfg)
        probs = classify(classifier, cls_image)[0].cpu()
        names = cls_ckpt["class_names"]
        best = int(probs.argmax())
        print(f"Classification: {names[best]} ({probs[best]:.1%})  |  "
              + ", ".join(f"{n}: {p:.1%}" for n, p in zip(names, probs.tolist())))
        print("Note: model output for research use only, not a medical diagnosis.")

    if output or show:
        from PIL import Image

        full_mask = Image.fromarray(mask[0].cpu().numpy().astype(np.uint8)).resize(
            (original_size[1], original_size[0]), Image.NEAREST)
        if output:
            scale = 255 // max(seg_ckpt["num_classes"] - 1, 1)
            Image.fromarray(np.asarray(full_mask) * scale).save(output)
            print(f"Saved mask to {output}")
        if show:
            import matplotlib.pyplot as plt

            with Image.open(input_path) as im:
                base = np.asarray(im.convert("L"))
            plt.imshow(base, cmap="gray")
            plt.imshow(np.ma.masked_equal(np.asarray(full_mask), 0), alpha=0.4, cmap="autumn")
            plt.title("Predicted lung mask")
            plt.axis("off")
            plt.show()


# --------------------------------------------------------------------------- self-test
def self_test(cfg: dict, device: torch.device) -> None:
    """Functional checks on random in-memory tensors. Numbers here are NOT model performance."""
    set_seed(cfg["seed"])
    size, c_in = cfg["image"]["size"], cfg["image"]["in_channels"]
    k_seg, k_cls = cfg["segmentation"]["num_classes"], len(cfg["classification"]["class_names"])
    checks = 0

    def ok(message: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  [ok] {message}")

    print(f"Running functional self-test on {device} (random tensors, no dataset, nothing saved)...")

    # 1. U-Net forward pass with the configured size and an odd size.
    unet = build_unet(cfg).to(device)
    out = unet(torch.randn(2, c_in, size, size, device=device))
    assert out.shape == (2, k_seg, size, size), out.shape
    ok(f"U-Net forward {tuple(out.shape)}, {count_parameters(unet):,} parameters")
    small = UNet(c_in, k_seg, base_channels=8).to(device)
    odd = small(torch.randn(1, c_in, 250, 197, device=device))
    assert odd.shape == (1, k_seg, 250, 197), odd.shape
    ok("U-Net handles sizes not divisible by 16 (250x197)")

    # 2. Classifier forward pass.
    classifier = build_classifier(cfg).to(device)
    logits = classifier(torch.randn(3, c_in, size, size, device=device))
    assert logits.shape == (3, k_cls), logits.shape
    ok(f"Classifier forward {tuple(logits.shape)}, {count_parameters(classifier):,} parameters")

    # 3. Losses and backpropagation (weighted CE + Dice for segmentation, CE for classification).
    images = torch.randn(2, c_in, 64, 64, device=device)
    masks = torch.randint(0, k_seg, (2, 64, 64), device=device)
    weights = torch.tensor(inverse_frequency(np.bincount(masks.cpu().numpy().ravel(), minlength=k_seg)),
                           dtype=torch.float32, device=device)
    seg_loss = SegmentationLoss(weights, k_seg, dice_weight=0.5)
    optimizer = make_optimizer(small, cfg["segmentation"])
    before = [p.detach().clone() for p in small.parameters()]
    loss = seg_loss(small(images), masks)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in small.parameters())
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, small.parameters()))
    ok(f"Segmentation loss {loss.item():.4f} backpropagates and Adam updates weights")
    labels = torch.randint(0, k_cls, (3,), device=device)
    cls_loss = torch.nn.CrossEntropyLoss()(classifier(torch.randn(3, c_in, 64, 64, device=device)), labels)
    cls_loss.backward()
    assert all(p.grad is not None for p in classifier.parameters() if p.requires_grad)
    ok("Classification loss backpropagates")

    # 4. Metrics on hand-made cases with known answers.
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    target[0, :4] = 1
    meter = SegmentationMeter(2)
    meter.update(target.clone(), target)
    m = meter.compute()
    assert abs(m["dice"] - 1) < 1e-6 and abs(m["iou"] - 1) < 1e-6 and m["pixel_accuracy"] == 1.0
    ok("Perfect prediction gives Dice = IoU = accuracy = 1 (second image: empty mask counted as 1)")
    pred = torch.zeros_like(target)
    pred[0, :2] = 1  # predicts half of the lung in image 0
    meter = SegmentationMeter(2)
    meter.update(pred, target)
    m = meter.compute()
    assert abs(m["recall"] - 0.5) < 1e-6 and m["precision"] == 1.0 and abs(m["specificity"] - 1) < 1e-6
    assert abs(m["dice"] - (2 / 3 + 1) / 2) < 1e-6 and abs(m["iou"] - 0.5) < 1e-6
    ok("Partial prediction gives the expected Dice, IoU, precision, recall and specificity")
    empty = SegmentationMeter(2)
    empty.update(torch.zeros(1, 4, 4, dtype=torch.long), torch.ones(1, 4, 4, dtype=torch.long))
    assert empty.compute()["dice"] == 0.0 and empty.compute()["precision"] == 0.0
    ok("Empty prediction on a non-empty mask gives Dice 0 without division errors")
    cm = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert cm["accuracy"] == 0.75 and cm["confusion_matrix"] == [[1, 1], [0, 2]]
    ok("Classification accuracy, macro F1 and confusion matrix")

    # 5. Preprocessing: normalization, mask conversion, synchronized augmentation.
    raw = np.random.default_rng(0).integers(0, 65535, (300, 280), dtype=np.uint16)
    scaled = to_float_image(raw, c_in)
    assert scaled.shape == (300, 280, c_in) and 0.0 <= scaled.min() and scaled.max() <= 1.0
    assert set(np.unique(mask_to_class_ids(np.array([[0, 255], [100, 200]]), 2, 127))) == {0, 1}
    pattern = np.zeros((300, 280), dtype=np.uint8)
    pattern[60:240, 40:130] = 1
    aug = build_transform({**cfg, "augmentation": {**cfg["augmentation"], "enabled": True, "probability": 1.0,
                                                   "horizontal_flip": 1.0}}, train=True)
    out = aug(image=pattern[..., None].astype(np.float32).repeat(c_in, 2), mask=pattern)
    agreement = ((out["image"][..., 0] > 0.5) == (out["mask"] > 0)).mean()
    assert out["mask"].shape == (size, size) and agreement > 0.97, agreement
    assert normalize(out["image"], cfg).shape == (c_in, size, size)
    ok(f"Normalization, mask conversion and image-mask synchronized augmentation ({agreement:.1%} aligned)")

    # 6. Patient-level splitting and leakage detection (in-memory rows).
    d = cfg["data"]
    rows = [{d["image_column"]: f"{i}.png", d["patient_column"]: f"p{i // 3}"} for i in range(60)]
    splits = split_rows(rows, cfg)
    owners = {}
    for name, items in splits.items():
        for r in items:
            owners.setdefault(r[d["patient_column"]], set()).add(name)
    assert all(len(s) == 1 for s in owners.values()) and all(splits.values())
    leaky = [{**rows[0], d["split_column"]: "train"}, {**rows[1], d["split_column"]: "test"}]
    try:
        split_rows(leaky, cfg)
        raise AssertionError("leakage not detected")
    except DatasetError:
        pass
    ok("Patient-level split keeps each patient in one split; leakage is rejected")

    # 7. Checkpoint save/load in a temporary folder (deleted afterwards) and inference.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "unet.pt"
        save_checkpoint(path, small, optimizer, 1, 0.0, {"train_loss": [], "val_loss": [], "val_metric": []},
                        cfg, "segmentation")
        loaded, ckpt = load_model(path, "segmentation", device)
        x = torch.randn(1, c_in, 64, 64, device=device)
        small.eval()
        assert torch.allclose(loaded(x), small(x), atol=1e-5)
        pred_mask = segment(loaded, x)
        assert pred_mask.shape == (1, 64, 64) and int(pred_mask.max()) < k_seg
        cpath = Path(tmp) / "classifier.pt"
        save_checkpoint(cpath, classifier, None, 1, 0.0, {}, cfg, "classification")
        cls_model, _ = load_model(cpath, "classification", device)
        probs = classify(cls_model, apply_lung_mask(x, pred_mask, cfg))
        assert probs.shape == (1, k_cls) and torch.allclose(probs.sum(1), torch.ones(1, device=device))
    assert not Path(tmp).exists()
    ok("Checkpoint save/load round-trip, segmentation -> classification inference; temp files removed")

    # 8. Missing dataset gives a clear error.
    try:
        read_manifest({**cfg, "data": {**d, "manifest": "__no_such_dir__/manifest.csv"}})
        raise AssertionError("missing manifest not reported")
    except DatasetError:
        pass
    ok("Missing dataset manifest raises a clear DatasetError")

    print(f"\nAll {checks} checks passed. These checks verify the code only; they say nothing about "
          "model accuracy. Train on a real dataset to measure Dice.")


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lung segmentation (U-Net) and classification")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("test", help="run dataset-free functional checks")
    for name in ("train-segmentation", "train-classification"):
        p = sub.add_parser(name)
        p.add_argument("--resume", action="store_true", help="continue from the saved (best) checkpoint")
    p = sub.add_parser("evaluate")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p = sub.add_parser("predict")
    p.add_argument("--input", required=True, help="path to an image")
    p.add_argument("--output", help="optional path to save the predicted mask (PNG)")
    p.add_argument("--show", action="store_true", help="display an overlay with matplotlib")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        device = get_device(cfg["device"])
        if args.command == "test":
            self_test(cfg, device)
        elif args.command == "train-segmentation":
            train_segmentation(cfg, device, resume=args.resume)
        elif args.command == "train-classification":
            train_classification(cfg, device, resume=args.resume)
        elif args.command == "evaluate":
            run_evaluation(cfg, device, split=args.split)
        elif args.command == "predict":
            predict(cfg, device, args.input, args.output, args.show)
    except (DatasetError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
