# QuCAB-UNet Training and Reproducibility Pipeline

This directory contains the training and reproducibility pipeline for **QuCAB-UNet**, a U-Net-based crack segmentation model with a **Quantum-Inspired Channel Attention Bottleneck (QCAB)**. It supports dataset preparation, preprocessing, model training, validation and test evaluation, bottleneck variant comparison, and reproducible experimentation.

## Overview

Crack segmentation remains challenging because crack patterns are often thin, fragmented, and visually weak, making them easy to confuse with background textures, shadows, and illumination variations. This training pipeline is designed to support controlled and reproducible experiments for **QuCAB-UNet**, with a particular focus on bottleneck refinement in a standard encoder-decoder segmentation framework.

The proposed **QCAB** module is inserted at the bottleneck stage to improve semantic feature refinement before decoding, while preserving the simplicity and reproducibility of the overall U-Net architecture.

## Main Idea

The **QCAB (Quantum-Inspired Channel Attention Bottleneck)** module refines bottleneck features through the following steps:

- constructing a compact global channel descriptor,
- projecting it into two angle-like latent branches,
- mixing them through trigonometric interaction,
- generating channel-wise gates,
- and applying residual scaling for bottleneck recalibration.

This design is intended to better preserve weak, thin, and discontinuous crack structures during decoder reconstruction.

## What This Pipeline Supports

This training pipeline includes support for:

- dataset preparation and preprocessing,
- binary mask generation and resizing,
- model training under a fixed controlled protocol,
- validation-based checkpoint selection,
- threshold-based test evaluation,
- comparison of bottleneck variants,
- and reproducible crack segmentation experiments.

## Dataset
The experiments are based on a manually annotated crack segmentation dataset derived from the Concrete Crack Images for Classification source.

A curated subset of 1,000 images was annotated at pixel level and divided into:

- 800 training images  
- 100 validation images  
- 100 test images  

All images and masks are resized to 384 × 384 for training and evaluation.

The dataset used in these experiments is provided in the `crack_segmentation_dataset_coco` section/directory of this repository.

## Preprocessing

The preprocessing pipeline includes:

- resizing all input images to `384 × 384`,
- converting annotations into binary crack masks,
- normalizing image intensities to the `[0, 1]` range,
- and resizing masks with nearest-neighbor interpolation to preserve label integrity.

The same preprocessing pipeline is applied consistently across all compared bottleneck variants.

## Training Setup

The default controlled training setting is:

- **Input size:** `384 × 384`
- **Task:** Binary crack segmentation
- **Output activation:** `Sigmoid`
- **Loss:** `Binary Cross-Entropy + Dice Loss`
- **Optimizer:** `Adam`
- **Initial learning rate:** `1e-3`
- **Batch size:** `4`
- **Maximum epochs:** `10`
- **Learning-rate schedule:** `ReduceLROnPlateau`
- **Checkpoint selection:** Best model based on minimum validation loss
- **Early stopping:** Validation-based early stopping with best-weight restoration

## Compared Bottleneck Variants

This pipeline supports controlled comparison of the following bottleneck settings:

- **Plain bottleneck**
- **SE bottleneck**
- **QCAB bottleneck**

All variants are evaluated under the same backbone, preprocessing, optimization budget, and stopping strategy to ensure fair comparison.

## Main Results

Under the same backbone and training protocol, the **QCAB bottleneck** achieved the best overall performance:

- **Test IoU:** `0.6798`
- **Test Dice:** `0.8094`

Compared with:

- **Plain bottleneck:** IoU `0.6162`, Dice `0.7625`
- **SE bottleneck:** IoU `0.6144`, Dice `0.7611`

These results suggest that structured bottleneck refinement with QCAB improves crack segmentation performance over both the plain bottleneck and a representative conventional channel-attention baseline.

## Reproducibility

This directory is intended to support reproducible experimentation and should include the components required to rerun the training and evaluation process, such as:

- training scripts or notebooks,
- model implementation files,
- preprocessing logic,
- validation and test evaluation code,
- threshold selection procedure,
- dependency specification,
- and experiment configuration details.

To ensure reproducibility, all experiments should use the same fixed data split and controlled training configuration.

## Requirements

Typical dependencies include:

- Python 3.10+
- TensorFlow or PyTorch
- NumPy
- OpenCV
- Matplotlib
- scikit-learn
- ONNX
- ONNX Runtime

Please refer to `requirements.txt` for the full dependency list used in the implementation.
