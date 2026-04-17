# QuCAB-UNet for Crack Segmentation

This repository contains the implementation of **QuCAB-UNet**, a U-Net-based model for binary crack segmentation. The project focuses on improving bottleneck feature refinement through **QCAB (Quantum-Inspired Channel Attention Bottleneck)**, designed to better preserve thin, weak, and fragmented crack patterns during decoding.

## Overview

Crack segmentation is challenging because cracks are often visually weak, thin, and easily confused with background textures or lighting variations. QuCAB-UNet addresses this by inserting a lightweight channel-attention bottleneck module into a standard U-Net pipeline, while keeping the overall encoder-decoder structure simple and reproducible.

## Main Idea

The proposed **QCAB** module refines bottleneck features before decoding by:
- building a compact global channel descriptor,
- projecting it into two angle-like latent branches,
- mixing them through trigonometric interaction,
- generating channel-wise gates,
- and applying residual scaling for bottleneck recalibration.

## Dataset

The project uses a manually annotated crack segmentation dataset derived from the **Concrete Crack Images for Classification** source. A curated subset of **1,000 images** was annotated at pixel level and split into:
- **800** training images
- **100** validation images
- **100** test images

All images and masks are resized to **384×384** for training and evaluation.

## Training Setup

The controlled training setting includes:
- input size: `384x384`
- loss: `Binary Cross-Entropy + Dice Loss`
- optimizer: `Adam`
- learning rate: `1e-3`
- batch size: `4`
- max epochs: `10`
- early stopping and validation-based checkpoint selection

## Main Results

Under the same backbone and training protocol, **QCAB bottleneck** achieved the best performance:
- **Test IoU:** `0.6798`
- **Test Dice:** `0.8094`

Compared with:
- Plain bottleneck: IoU `0.6162`, Dice `0.7625`
- SE bottleneck: IoU `0.6144`, Dice `0.7611`

## Repository Contents

This repository currently includes:
- the main training and reproducibility notebook
- dataset files for crack segmentation
- dependency specification
- a **`raspberry_pi_deployment/`** folder for ONNX-based edge inference and benchmarking on Raspberry Pi

## Raspberry Pi Deployment

Besides model training, the project also includes a deployment-oriented component for **Raspberry Pi 4**. Trained models are exported to **ONNX** and benchmarked with ONNX Runtime on CPU.

## Purpose

This repository is intended for:
- reproducible crack segmentation experiments,
- comparison of bottleneck variants,
- QCAB-based model research,
- and lightweight edge deployment testing.


## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{pham2026qucabunet,
  title   = {QuCAB-UNet: A Quantum-Inspired Channel Attention Bottleneck U-Net for Crack Segmentation},
  author  = {Pham, Lenh Phan Cong},
  year    = {under review}
}
