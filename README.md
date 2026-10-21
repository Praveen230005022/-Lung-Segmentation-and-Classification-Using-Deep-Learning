# Lung Segmentation and Classification Using Deep Learning

A minimal PyTorch pipeline for lung segmentation and image classification. It has:

- a U-Net built from scratch, trained with Adam and class-weighted cross-entropy (Dice loss optional)
- a lightweight CNN classifier
- an optional pipeline that segments the lungs and then classifies the image

Segmentation is evaluated with Dice, IoU, pixel accuracy, precision, recall and specificity, against a **target** Dice of 0.91. Classification is evaluated with accuracy, precision, recall, macro F1 and a confusion matrix.

> **No dataset is included, and no performance has been measured yet.** The 0.91 Dice is a target, not a result.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py test                  # checks the code without any dataset
```

## Dataset format

Put your images and masks under `data/` and create `data/manifest.csv`:

```csv
image_path,mask_path,label,patient_id,split
images/p001_a.png,masks/p001_a.png,normal,p001,train
images/p002_a.png,masks/p002_a.png,abnormal,p002,val
images/p003_a.png,,abnormal,p003,test
```

- **Paths:** relative to `data.root`. Supported formats are PNG, JPG, BMP and TIFF, in 8- or 16-bit grayscale (RGB is converted to grayscale).
- **`mask_path`:**
  - A mask must have the same height and width as its image.
  - With 2 classes, pixels above `mask_threshold` are lung.
  - With more classes, the mask must store class IDs directly (0 to N−1).
- **`label`:** a name from `classification.class_names` or an integer ID. Leave it empty for rows used only for segmentation, and leave `mask_path` empty for rows used only for classification.
- **`split`:** optional. If it's missing, the code makes a seeded split by `patient_id`. Either way, a patient who appears in more than one split is rejected as data leakage.

## Configuration

All settings are in `config.yaml`: paths and column names, image size and channels, augmentation, class counts, epochs, batch size, learning rate, optimizer, class weights (`auto` or a list), `dice_loss_weight`, seed, device (`auto`, `cpu` or `cuda`) and checkpoint paths.

## Commands

```bash
python main.py test                          # dataset-free functional checks (random tensors)
python main.py train-segmentation            # trains U-Net, saves the best checkpoint by val Dice
python main.py train-classification          # trains classifier, saves the best by val macro F1
python main.py train-segmentation --resume   # continue from the saved checkpoint
python main.py evaluate                      # test-split metrics for every trained model
python main.py evaluate --split val
python main.py predict --input path/to/image.png
python main.py predict --input img.png --output mask.png --show
```

Use `--config other.yaml` to choose a different config file.

## Important limitations

- `python main.py test` checks only that the code works. Its numbers are not model performance.
- Training, evaluation and prediction require a real dataset and trained checkpoints. The commands exit with a clear error when either is missing.
- Only the best checkpoint is saved, so `--resume` continues from the best epoch, not the last one.
- `predict` reports a class only if a trained classifier exists. A segmentation mask on its own does not provide a diagnosis.
- With `masked_input: true`, the classifier sees ground-truth masks during training and U-Net masks during prediction, so segmentation errors affect classification.
- This is research code and not a medical device. Don't use it for clinical decisions.
