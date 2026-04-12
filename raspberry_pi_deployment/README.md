# Training Pipeline

This directory contains the training and reproducibility pipeline for **QuCAB-UNet**, a U-Net-based crack segmentation model with a **Quantum-Inspired Channel Attention Bottleneck (QCAB)**.

The pipeline supports:
- dataset preparation
- preprocessing
- model training
- validation and test evaluation
- bottleneck variant comparison
- reproducible experimentation

## Overview

This training pipeline is designed for **binary crack segmentation** under a controlled experimental setting. All compared bottleneck variants are trained with the same backbone, preprocessing steps, optimization settings, and stopping criteria to ensure a fair comparison.

The main compared variants are:
- `none` — plain bottleneck
- `SE` — squeeze-and-excitation bottleneck
- `QCAB` — proposed quantum-inspired channel attention bottleneck

## Dataset

The project uses a manually annotated crack segmentation dataset derived from the **Concrete Crack Images for Classification** source dataset.

### Dataset Summary
- Total images: **1,000**
- Training images: **800**
- Validation images: **100**
- Test images: **100**
- Split ratio: **80 / 10 / 10**
- Random seed: **42**

### Annotation Format
The dataset is stored in **COCO-style format** and converted into binary segmentation masks for model training and evaluation.

### Preprocessing
The pipeline applies the following preprocessing steps:
- resize all images to **384 × 384**
- convert annotations into **binary masks**
- normalize image pixel values to **[0, 1]**
- resize masks using **nearest-neighbor interpolation** to preserve label integrity

## Files in This Module

Typical files used in this training pipeline include:
- `qcab_crack_segmentation_reproducible.ipynb` — main notebook for the full training workflow
- `requirements.txt` — Python dependencies
- `crack_segmentation_dataset_coco.zip` — dataset archive

If your local filenames are slightly different, update the paths in the notebook accordingly.

## Recommended Environment

This pipeline is recommended to run in:
- **Google Colab** for the easiest setup
- or a local Python environment with GPU support

## Installation

Install the required dependencies with:

```bash
pip install -r requirements.txt
