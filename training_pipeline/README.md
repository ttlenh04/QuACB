# Training Pipeline

This directory contains the training and reproducibility pipeline for **QuCAB-UNet**, a U-Net-based crack segmentation model with a **Quantum-Inspired Channel Attention Bottleneck (QCAB)**. It supports dataset preparation, preprocessing, model training, validation and test evaluation, bottleneck variant comparison, and reproducible experimentation. :contentReference[oaicite:0]{index=0}

---

## Overview

The training pipeline is designed for **binary crack segmentation** under a controlled experimental setting. All compared bottleneck variants are trained using the same backbone, preprocessing steps, optimization settings, and stopping criteria to ensure a fair comparison. :contentReference[oaicite:1]{index=1}

The main compared variants are:

- **none** — plain bottleneck
- **SE** — squeeze-and-excitation bottleneck
- **QCAB** — proposed quantum-inspired channel attention bottleneck

---

## Dataset

The project uses a manually annotated crack segmentation dataset derived from the **Concrete Crack Images for Classification** source dataset. :contentReference[oaicite:2]{index=2}

### Dataset Summary

- **Total images:** 1,000
- **Training images:** 800
- **Validation images:** 100
- **Test images:** 100
- **Split ratio:** 80 / 10 / 10
- **Random seed:** 42

### Annotation Format

The dataset is stored in **COCO-style format** and converted into binary segmentation masks for model training and evaluation.

### Preprocessing

The pipeline applies the following preprocessing steps:

- resize all images to **384 × 384**
- convert annotations into **binary masks**
- normalize image pixel values to **[0, 1]**
- resize masks using **nearest-neighbor interpolation** to preserve label integrity

---

## Files in This Module

Typical files used in this training pipeline include:

- `qcab_crack_segmentation_reproducible.ipynb` — main notebook for the full training workflow
- `requirements.txt` — Python dependencies
- `crack_segmentation_dataset_coco.zip` — dataset archive

If your local filenames are slightly different, update the paths in the notebook accordingly. :contentReference[oaicite:3]{index=3}

---

## Recommended Environment

This pipeline is recommended to run in:

- **Google Colab** for the easiest setup
- or a local Python environment with GPU support

---

## Installation

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

### Required Packages

- `tensorflow==2.19.0`
- `numpy==2.0.2`
- `pandas==2.2.2`
- `matplotlib==3.10.0`
- `opencv-python==4.13.0.92`
- `pycocotools==2.0.11`

---

## Suggested Directory Structure

A simple working structure may look like this:

```text
training_pipeline/
├── qcab_crack_segmentation_reproducible.ipynb
├── requirements.txt
├── crack_segmentation_dataset_coco.zip

```

You can adapt this structure based on your actual notebook paths.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Place the dataset archive in the working directory:

```text
crack_segmentation_dataset_coco.zip
```

Then extract it before training.

Example:

```bash
unzip crack_segmentation_dataset_coco.zip -d data/
```

On Windows, you can also extract the archive manually and update the notebook paths if needed.

### 3. Open the training notebook

Open:

```text
qcab_crack_segmentation_reproducible.ipynb
```

Run the notebook cells in order.

### 4. Configure dataset paths

Before training, make sure the notebook paths correctly point to:

- the image directory
- the annotation file or mask directory
- the output directory for checkpoints and predictions

### 5. Train the model variants

The pipeline is designed to compare the following bottleneck variants:

- plain bottleneck
- SE bottleneck
- QCAB bottleneck

### 6. Select the best checkpoint

Model selection is based on **minimum validation loss**.

### 7. Tune the prediction threshold

After training, threshold selection is performed on the validation set only by sweeping thresholds from:

- **0.20 to 0.75**
- with a **step size of 0.05**

The threshold with the best validation IoU is then fixed for final test evaluation.

### 8. Evaluate on the test set

After checkpoint selection and threshold tuning, the chosen model is evaluated once on the test set. :contentReference[oaicite:4]{index=4}

---

## Training Configuration

The default controlled training setting is:

- **Task:** binary crack segmentation
- **Input resolution:** 384 × 384
- **Output activation:** sigmoid
- **Loss function:** Binary Cross-Entropy + Dice Loss
- **Optimizer:** Adam
- **Initial learning rate:** 1e-3
- **Batch size:** 4
- **Maximum epochs:** 10
- **Validation criterion:** validation loss
- **Checkpoint rule:** save best model by minimum validation loss
- **Learning-rate scheduler:** ReduceLROnPlateau
  - factor: 0.3
  - patience: 4
  - minimum learning rate: 1e-6
- **Early stopping:**
  - patience: 4
  - restore best weights: enabled

### Default QCAB Configuration

- `q_dim = 8`
- `reduction = 16`
- `dropout_rate = 0.05`
- `residual_scale = 0.35`
- max pooling: enabled
- trigonometric interaction: enabled

---

## Evaluation Metrics

The primary segmentation metrics are:

- **IoU (Intersection over Union)**
- **Dice coefficient**

These metrics are used for comparing the bottleneck variants under the same training setup.

---

## Main Experimental Result

Under the same training protocol, the reported performance is: :contentReference[oaicite:5]{index=5}

| Variant | Test IoU | Test Dice | Best Threshold |
|---|---:|---:|---:|
| Plain bottleneck | 0.6162 | 0.7625 | 0.20 |
| SE bottleneck | 0.6144 | 0.7611 | 0.30 |
| QCAB bottleneck | 0.6798 | 0.8094 | 0.55 |

This indicates that **QCAB achieved the best overall segmentation performance** in the controlled comparison. :contentReference[oaicite:6]{index=6}

---

## Reproducibility Notes

To keep results reproducible:

- use the same train/validation/test split
- keep the preprocessing pipeline unchanged
- train all variants with the same optimization budget
- use the same validation-based checkpointing strategy
- perform threshold selection on the validation set only
- avoid tuning hyperparameters on the test set

---

## Expected Outputs

Typical outputs from this pipeline may include:

- trained model checkpoints
- validation and test metrics
- predicted segmentation masks
- threshold selection results
- training curves for loss and Dice
- comparison results across bottleneck variants

---

## Notes on COCO-Based Data

The dataset starts from COCO-style annotations and is converted into binary masks for segmentation training. Make sure:

- annotations match the correct images
- masks are generated correctly
- resized masks preserve foreground labels
- image and mask filenames remain aligned after preprocessing

---

## Suggested Workflow

A practical workflow is:

1. install dependencies
2. extract the dataset
3. generate or verify binary masks
4. open the notebook
5. set dataset and output paths
6. train the compared bottleneck variants
7. select the best checkpoint using validation loss
8. tune the threshold on validation predictions
9. evaluate once on the test set
10. save metrics, plots, and exported artifacts

---

## Troubleshooting

### ModuleNotFoundError

Reinstall all packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Dataset path not found

Check the extraction path and update file paths in the notebook.

### `pycocotools` installation issues on Windows

If `pycocotools` fails on Windows, use:

- **Google Colab**
- or a compatible **conda / wheel-based installation**

### Out-of-memory issues

Try:

- reducing the batch size
- running on GPU in Colab
- or training one bottleneck variant at a time

### Results do not match expected values

Verify that:

- image size is **384 × 384**
- the split remains **80 / 10 / 10**
- the random seed is unchanged
- threshold tuning is done only on validation data
- all variants use the same training configuration

---

## Related Modules

This training pipeline is part of a larger project repository. Other project components may include:

- the main project `README`
- dataset packaging
- `raspberry_pi_deployment/` for ONNX export and edge inference benchmarking
