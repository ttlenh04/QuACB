# Raspberry Pi Deployment

This directory contains the deployment and benchmarking pipeline for running **QuCAB-UNet** and its bottleneck variants on **Raspberry Pi 4** using **ONNX Runtime**.

It is intended for lightweight edge inference experiments after training. The module includes exported ONNX models and a benchmarking script for measuring inference latency on CPU.

## Overview

The deployment pipeline is designed to evaluate whether the trained crack segmentation models can run on resource-constrained hardware. It focuses on ONNX-based inference and latency benchmarking on Raspberry Pi 4.

The compared model variants are:
- `none` — plain bottleneck
- `SE` — squeeze-and-excitation bottleneck
- `QCAB` — quantum-inspired channel attention bottleneck

## Directory Structure

A typical structure for this module is:

```text
raspberry_pi_deployment/
├── benchmark/
│   └── benchmark_onnx_pi.py
├── model_onnx/
│   ├── none/
│   ├── SE/
│   └── QACB/

The model_onnx/ directory stores exported ONNX models and metadata files for each bottleneck variant.

Contents
benchmark/

Contains the benchmarking script used to evaluate ONNX model latency on Raspberry Pi.

Main file:

benchmark_onnx_pi.py
model_onnx/

Contains ONNX export artifacts for the compared model variants:

none/
SE/
QACB/

Each model folder may include:

an ONNX model file
a metadata JSON file

Example:

qcab_b1_384x384_raw.onnx
qcab_b1_384x384_meta.json
Hardware Target

The deployment experiments are intended for Raspberry Pi 4 Model B.

The benchmark setting reported in the project uses:

ONNX Runtime
CPUExecutionProvider
batch size = 1
input resolution = 384 × 384
Benchmark Script

The provided script benchmarks one model at a time and includes:

warm-up runs
measured runs
latency tracking
thermal monitoring
throttling checks
CSV export for summary and thermal logs

The script supports the following model options:

plain
se
qcab
Expected Model Files

The benchmark script expects ONNX and metadata files corresponding to:

plain bottleneck
SE bottleneck
QCAB bottleneck

Example expected names:

none_b1_384x384.onnx
se_b1_384x384.onnx
qcab_b1_384x384.onnx
matching *_meta.json files

If your file names or locations are different, update the paths inside benchmark_onnx_pi.py before running.

Benchmark Settings

The benchmark script is configured with:

warm-up runs: 5
measured runs: 20
random seed: 42
sleep between runs: 0.2 seconds

Thermal thresholds used in the script:

safe: 70°C
warning: 75°C
danger: 80°C

If the device reaches the danger threshold, benchmarking is aborted automatically.

Installation

Install the required runtime packages first.

A minimal environment typically includes:

numpy
onnxruntime

Example:

pip install numpy onnxruntime

If you also need utilities for model export or preprocessing, install the full project dependencies separately.

How to Run
1. Copy the files to Raspberry Pi

Transfer this directory and the ONNX model files to your Raspberry Pi.

2. Check the benchmark paths

Open benchmark/benchmark_onnx_pi.py and verify the BASE_DIR and model file paths match your local deployment directory.

3. Run a single benchmark

Benchmark the plain model:

python benchmark/benchmark_onnx_pi.py --model plain

Benchmark the SE model:

python benchmark/benchmark_onnx_pi.py --model se

Benchmark the QCAB model:

python benchmark/benchmark_onnx_pi.py --model qcab
What the Script Does

For each selected model, the script:

loads the ONNX model with ONNX Runtime
resolves the input tensor shape
generates a dummy input tensor
performs warm-up inference runs
performs measured inference runs
checks temperature and throttling status before and after runs
computes mean latency, standard deviation, and FPS
saves benchmark results to CSV files
Output Files

The benchmark script saves:

a latency summary CSV
a thermal log CSV

Example outputs:

benchmark_latency_plain.csv
benchmark_latency_se.csv
benchmark_latency_qcab.csv
benchmark_thermal_log_plain.csv
benchmark_thermal_log_se.csv
benchmark_thermal_log_qcab.csv
Reported Benchmark Setting

The project benchmark on Raspberry Pi 4 was conducted with:

ONNX Runtime
CPU execution
batch size 1
input resolution 384 × 384
5 warm-up runs
20 measured runs

This setup was used to compare the deployment efficiency of the three bottleneck variants under the same conditions.

Notes
This module is for inference and benchmarking only, not for model training.
Make sure the exported ONNX files are compatible with the installed ONNX Runtime version.
Thermal conditions can affect latency, so benchmarking should be performed under stable device conditions.
If you rename folders such as QACB to QCAB, update the README and paths consistently.
Suggested Workflow

A practical deployment workflow is:

train the models in training_pipeline/
export trained models to ONNX
organize ONNX files inside model_onnx/
copy the deployment folder to Raspberry Pi
verify benchmark paths in benchmark_onnx_pi.py
run latency benchmarks for plain, se, and qcab
collect CSV summaries and compare results
Troubleshooting
Model file not found

Check:

the model path
the file name
the BASE_DIR value inside the benchmark script
ONNX Runtime error

Make sure:

onnxruntime is installed
the ONNX model is not corrupted
the model was exported correctly
Temperature too high

If the script aborts because of temperature:

let the Raspberry Pi cool down
improve airflow or add a heatsink/fan
reduce repeated benchmark frequency
Unexpected input shape

If the model input shape is different from 384 × 384, verify the exported model and update your deployment assumptions accordingly.

Related Modules

This deployment module is part of the larger project repository. Other related components include:

the main project README
training_pipeline/ for training and reproducibility
dataset packaging for crack segmentation experiments
