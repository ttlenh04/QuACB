# Raspberry Pi Deployment

This module provides the deployment and benchmarking setup for comparing three model bottleneck variants on **Raspberry Pi 4 Model B** using **ONNX Runtime**.

## Compared Model Variants

The compared model variants are:

- `plain` — plain bottleneck
- `se` — squeeze-and-excitation bottleneck
- `qcab` — quantum-inspired channel attention bottleneck

## Directory Structure

A typical structure for this module is:

```text
raspberry_pi_deployment/
├── README.md
├── benchmark/
│   └── benchmark_onnx_pi.py
└── model_onnx/
    ├── plain/
    ├── se/
    └── qcab/
```

The `model_onnx/` directory stores exported ONNX models and metadata files for each bottleneck variant.

### `benchmark/`

Contains the benchmarking script used to evaluate ONNX model latency on Raspberry Pi.

**Main file:**

- `benchmark_onnx_pi.py`

### `model_onnx/`

Contains ONNX export artifacts for the compared model variants:

- `plain/`
- `se/`
- `qcab/`

Each model folder may include:

- an ONNX model file
- a metadata JSON file

**Example files:**

- `qcab_b1_384x384_raw.onnx`
- `qcab_b1_384x384_meta.json`

## Hardware Target

The deployment experiments are intended for **Raspberry Pi 4 Model B**.

The benchmark setting reported in the project uses:

- **ONNX Runtime** `CPUExecutionProvider`
- **batch size = 1**
- **input resolution = 384 × 384**

## Benchmark Script

The provided script benchmarks **one model at a time** and includes:

- warm-up runs
- measured runs
- latency tracking
- thermal monitoring
- throttling checks
- CSV export for summary and thermal logs

The script supports the following model options:

- `plain`
- `se`
- `qcab`

## Expected Model Files

The benchmark script expects ONNX and metadata files corresponding to:

- plain bottleneck
- SE bottleneck
- QCAB bottleneck

**Example expected names:**

- `plain_b1_384x384.onnx`
- `se_b1_384x384.onnx`
- `qcab_b1_384x384.onnx`

with matching `*_meta.json` files.

If your file names or locations are different, update the paths inside `benchmark_onnx_pi.py` before running.

## Benchmark Settings

The benchmark script is configured with:

- **warm-up runs:** 5
- **measured runs:** 20
- **random seed:** 42
- **sleep between runs:** 0.2 seconds

### Thermal thresholds used in the script

- **safe:** 70°C
- **warning:** 75°C
- **danger:** 80°C

If the device reaches the danger threshold, benchmarking is aborted automatically.

## Installation

Install the required runtime packages first.

A minimal environment typically includes:

- `numpy`
- `onnxruntime`

### Example

```bash
pip install numpy onnxruntime
```

If you also need utilities for model export or preprocessing, install the full project dependencies separately.

## How to Run

### 1. Copy the files to Raspberry Pi

Transfer this directory and the ONNX model files to your Raspberry Pi.

### 2. Check the benchmark paths

Open `benchmark/benchmark_onnx_pi.py` and verify that the base directory and model file paths match your local deployment directory.

### 3. Run a single benchmark

Benchmark the plain model:

```bash
python benchmark/benchmark_onnx_pi.py --model plain
```

Benchmark the SE model:

```bash
python benchmark/benchmark_onnx_pi.py --model se
```

Benchmark the QCAB model:

```bash
python benchmark/benchmark_onnx_pi.py --model qcab
```

## What the Script Does

For each selected model, the script:

1. loads the ONNX model with ONNX Runtime
2. resolves the input tensor shape
3. generates a dummy input tensor
4. performs warm-up inference runs
5. performs measured inference runs
6. checks temperature and throttling status before and after runs
7. computes mean latency, standard deviation, and FPS
8. saves benchmark results to CSV files

## Output Files

The benchmark script saves:

- a latency summary CSV
- a thermal log CSV

**Example outputs:**

- `benchmark_latency_plain.csv`
- `benchmark_latency_se.csv`
- `benchmark_latency_qcab.csv`
- `benchmark_thermal_log_plain.csv`
- `benchmark_thermal_log_se.csv`
- `benchmark_thermal_log_qcab.csv`

## Reported Benchmark Setting

The project benchmark on Raspberry Pi 4 was conducted with:

- ONNX Runtime CPU execution
- batch size 1
- input resolution 384 × 384
- 5 warm-up runs
- 20 measured runs

This setup was used to compare the deployment efficiency of the three bottleneck variants under the same conditions.

## Notes

- This module is for **inference and benchmarking only**, not for model training.
- Make sure the exported ONNX files are compatible with the installed ONNX Runtime version.
- Thermal conditions can affect latency, so benchmarking should be performed under stable device conditions.
- If folder names or file names differ from the defaults, update the paths in the benchmarking script accordingly.

## Suggested Workflow

A practical deployment workflow is:

1. train the models in `training_pipeline/`
2. export trained models to ONNX
3. organize ONNX files inside `model_onnx/`
4. copy the deployment folder to Raspberry Pi
5. verify benchmark paths in `benchmark_onnx_pi.py`
6. run latency benchmarks for `plain`, `se`, and `qcab`
7. collect CSV summaries and compare results

## Troubleshooting

### Model file not found

Check:

- the model path
- the file name
- the base directory value inside the benchmark script

### ONNX Runtime error

Make sure:

- `onnxruntime` is installed
- the ONNX model is not corrupted
- the model was exported correctly

### Temperature too high

If the script aborts because of temperature:

- let the Raspberry Pi cool down
- improve airflow or add a heatsink/fan
- reduce repeated benchmark frequency

### Unexpected input shape

If the model input shape is different from `384 × 384`, verify the exported model and update your deployment assumptions accordingly.

## Related Modules

This deployment module is part of the larger project repository.

Other related components include:

- the main project `README.md`
- `training_pipeline/` for training and reproducibility
- `crack_segmentation_dataset_coco/` for the dataset used in the experiments
