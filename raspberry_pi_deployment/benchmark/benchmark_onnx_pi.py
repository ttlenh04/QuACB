import os
import time
import json
import csv
import argparse
import subprocess
import numpy as np
import onnxruntime as ort

BASE_DIR = "/home/lenhta/onnx_export_e1e2"

MODEL_FILES = {
    "plain": os.path.join(BASE_DIR, "none_b1_384x384.onnx"),
    "se": os.path.join(BASE_DIR, "se_b1_384x384.onnx"),
    "qcab": os.path.join(BASE_DIR, "qcab_b1_384x384.onnx"),
}

META_FILES = {
    "plain": os.path.join(BASE_DIR, "none_b1_384x384_meta.json"),
    "se": os.path.join(BASE_DIR, "se_b1_384x384_meta.json"),
    "qcab": os.path.join(BASE_DIR, "qcab_b1_384x384_meta.json"),
}

DISPLAY_NAMES = {
    "plain": "Plain",
    "se": "SE",
    "qcab": "QCAB",
}

WARMUP_RUNS = 5
MEASURED_RUNS = 20
SEED = 42

SAFE_TEMP_C = 70.0
WARNING_TEMP_C = 75.0
DANGER_TEMP_C = 80.0

SLEEP_BETWEEN_RUNS_SEC = 0.2


def print_header(title):
    print("\n" + "=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)


def set_seed(seed=42):
    np.random.seed(seed)


def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return out
    except Exception:
        return None


def get_temp_c():
    out = run_cmd(["vcgencmd", "measure_temp"])
    if out is None:
        return None
    try:
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        return None


def get_arm_clock_hz():
    out = run_cmd(["vcgencmd", "measure_clock", "arm"])
    if out is None:
        return None
    try:
        return int(out.split("=")[1])
    except Exception:
        return None


def get_throttled():
    out = run_cmd(["vcgencmd", "get_throttled"])
    if out is None:
        return None
    try:
        return out.split("=")[1]
    except Exception:
        return None


def thermal_status_string(temp_c):
    if temp_c is None:
        return "unknown"
    if temp_c >= DANGER_TEMP_C:
        return "danger"
    if temp_c >= WARNING_TEMP_C:
        return "warning"
    if temp_c >= SAFE_TEMP_C:
        return "warm"
    return "safe"


def get_thermal_snapshot():
    temp_c = get_temp_c()
    throttled = get_throttled()
    arm_clock_hz = get_arm_clock_hz()
    status = thermal_status_string(temp_c)
    return {
        "temp_c": temp_c,
        "throttled": throttled,
        "arm_clock_hz": arm_clock_hz,
        "status": status,
    }


def assert_safe_or_abort(stage_name, run_idx=None):
    snap = get_thermal_snapshot()

    idx_text = f" | run={run_idx}" if run_idx is not None else ""
    print(
        f"[THERMAL] stage={stage_name}{idx_text} | "
        f"temp={snap['temp_c']} C | status={snap['status']} | "
        f"clock={snap['arm_clock_hz']} Hz | throttled={snap['throttled']}",
        flush=True
    )

    if snap["temp_c"] is not None and snap["temp_c"] >= DANGER_TEMP_C:
        raise RuntimeError(
            f"Aborted immediately: dangerous temperature reached at stage='{stage_name}'"
            f"{idx_text} ({snap['temp_c']:.2f} C >= {DANGER_TEMP_C:.1f} C)."
        )

    return snap


def get_params_from_meta(meta_path):
    if not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None

    for k in ["num_params", "params", "total_params", "parameter_count", "trainable_params"]:
        if k in meta:
            v = safe_int(meta[k], None)
            if v is not None:
                return v
    return None


def format_params_million(num_params):
    if num_params is None:
        return "N/A"
    return round(num_params / 1e6, 4)


def resolve_input_shape(inp):
    raw_shape = inp.shape
    if len(raw_shape) != 4:
        raise ValueError(f"Expected 4D input, got shape={raw_shape}")

    default_nhwc = [1, 384, 384, 3]
    default_nchw = [1, 3, 384, 384]

    second_int = safe_int(raw_shape[1], None)
    last_int = safe_int(raw_shape[3], None)

    if second_int == 3:
        guess = default_nchw
    elif last_int == 3:
        guess = default_nhwc
    else:
        guess = default_nhwc

    shape = []
    for i, dim in enumerate(raw_shape):
        dim_int = safe_int(dim, None)
        if dim_int is None or dim_int <= 0:
            shape.append(guess[i])
        else:
            shape.append(dim_int)

    return tuple(shape)


def make_dummy_input(shape):
    return np.random.rand(*shape).astype(np.float32)


def benchmark_single_model(model_key):
    model_name = DISPLAY_NAMES[model_key]
    model_path = MODEL_FILES[model_key]
    meta_path = META_FILES[model_key]

    print_header(f"[MODEL] {model_name}")
    print(f"Path: {model_path}", flush=True)

    before_snap = assert_safe_or_abort("before_benchmark")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )

    inp = session.get_inputs()[0]
    outputs = session.get_outputs()
    input_name = inp.name
    input_shape = resolve_input_shape(inp)

    print(f"Input name : {input_name}", flush=True)
    print(f"Input shape: {inp.shape} -> resolved to {input_shape}", flush=True)
    for i, out in enumerate(outputs):
        print(f"Output[{i}] name={out.name}, shape={out.shape}", flush=True)

    dummy = make_dummy_input(input_shape)

    warmup_times_ms = []
    measured_times_ms = []
    thermal_log = []
    aborted = False
    abort_reason = ""

    print(f"Warm-up: {WARMUP_RUNS} runs", flush=True)
    for i in range(WARMUP_RUNS):
        snap_pre = assert_safe_or_abort("warmup_pre", i + 1)

        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy})
        end = time.perf_counter()

        t = (end - start) * 1000.0
        warmup_times_ms.append(t)

        snap_post = assert_safe_or_abort("warmup_post", i + 1)

        thermal_log.append({
            "phase": "warmup",
            "run_index": i + 1,
            "latency_ms": round(t, 3),
            "temp_pre_c": snap_pre["temp_c"],
            "temp_post_c": snap_post["temp_c"],
            "status_pre": snap_pre["status"],
            "status_post": snap_post["status"],
            "clock_pre_hz": snap_pre["arm_clock_hz"],
            "clock_post_hz": snap_post["arm_clock_hz"],
            "throttled_pre": snap_pre["throttled"],
            "throttled_post": snap_post["throttled"],
        })

        print(
            f"  warm-up {i+1:02d}/{WARMUP_RUNS}: {t:.2f} ms | "
            f"temp pre/post: {snap_pre['temp_c']} -> {snap_post['temp_c']} C",
            flush=True
        )

        if SLEEP_BETWEEN_RUNS_SEC > 0:
            time.sleep(SLEEP_BETWEEN_RUNS_SEC)

    print(f"Measured: {MEASURED_RUNS} runs", flush=True)
    for i in range(MEASURED_RUNS):
        try:
            snap_pre = assert_safe_or_abort("measured_pre", i + 1)

            start = time.perf_counter()
            _ = session.run(None, {input_name: dummy})
            end = time.perf_counter()

            t = (end - start) * 1000.0
            measured_times_ms.append(t)

            snap_post = assert_safe_or_abort("measured_post", i + 1)

            thermal_log.append({
                "phase": "measured",
                "run_index": i + 1,
                "latency_ms": round(t, 3),
                "temp_pre_c": snap_pre["temp_c"],
                "temp_post_c": snap_post["temp_c"],
                "status_pre": snap_pre["status"],
                "status_post": snap_post["status"],
                "clock_pre_hz": snap_pre["arm_clock_hz"],
                "clock_post_hz": snap_post["arm_clock_hz"],
                "throttled_pre": snap_pre["throttled"],
                "throttled_post": snap_post["throttled"],
            })

            running_mean = float(np.mean(measured_times_ms))
            print(
                f"  run {i+1:02d}/{MEASURED_RUNS}: {t:.2f} ms | "
                f"running mean: {running_mean:.2f} ms | "
                f"temp pre/post: {snap_pre['temp_c']} -> {snap_post['temp_c']} C",
                flush=True
            )

            if SLEEP_BETWEEN_RUNS_SEC > 0:
                time.sleep(SLEEP_BETWEEN_RUNS_SEC)

        except RuntimeError as e:
            aborted = True
            abort_reason = str(e)
            print(f"[ABORT] {abort_reason}", flush=True)
            break

    after_snap = get_thermal_snapshot()

    num_params = get_params_from_meta(meta_path)
    params_m = format_params_million(num_params)

    if len(measured_times_ms) > 0:
        mean_latency = float(np.mean(measured_times_ms))
        std_latency = float(np.std(measured_times_ms, ddof=1)) if len(measured_times_ms) > 1 else 0.0
        fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0
    else:
        mean_latency = None
        std_latency = None
        fps = None

    print("-" * 80, flush=True)
    print(f"Params (M)                : {params_m}", flush=True)
    print(f"Completed measured runs   : {len(measured_times_ms)}/{MEASURED_RUNS}", flush=True)
    print(f"Aborted                   : {aborted}", flush=True)
    if aborted:
        print(f"Abort reason              : {abort_reason}", flush=True)
    print(f"Mean latency (ms)         : {mean_latency}", flush=True)
    print(f"Std latency (ms)          : {std_latency}", flush=True)
    print(f"FPS                       : {fps}", flush=True)
    print(f"Final temp (C)            : {after_snap['temp_c']}", flush=True)
    print(f"Final throttled           : {after_snap['throttled']}", flush=True)
    print(f"Final ARM clock (Hz)      : {after_snap['arm_clock_hz']}", flush=True)

    summary = {
        "Model": model_name,
        "Params (M)": params_m,
        "Warm-up Runs": WARMUP_RUNS,
        "Measured Runs Target": MEASURED_RUNS,
        "Measured Runs Completed": len(measured_times_ms),
        "Aborted": aborted,
        "Abort Reason": abort_reason,
        "Mean Latency (ms)": round(mean_latency, 3) if mean_latency is not None else "",
        "Std Latency (ms)": round(std_latency, 3) if std_latency is not None else "",
        "FPS": round(fps, 3) if fps is not None else "",
        "Input Shape": str(input_shape),
        "Temp Before (C)": before_snap["temp_c"],
        "Temp Final (C)": after_snap["temp_c"],
        "Throttled Before": before_snap["throttled"],
        "Throttled Final": after_snap["throttled"],
        "ARM Clock Before (Hz)": before_snap["arm_clock_hz"],
        "ARM Clock Final (Hz)": after_snap["arm_clock_hz"],
        "Danger Temp Threshold (C)": DANGER_TEMP_C,
        "Sleep Between Runs (s)": SLEEP_BETWEEN_RUNS_SEC,
    }

    summary_csv = os.path.join(BASE_DIR, f"benchmark_latency_{model_key}.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    thermal_csv = os.path.join(BASE_DIR, f"benchmark_thermal_log_{model_key}.csv")
    if len(thermal_log) > 0:
        with open(thermal_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(thermal_log[0].keys()))
            writer.writeheader()
            writer.writerows(thermal_log)

    print(f"Saved summary CSV : {summary_csv}", flush=True)
    print(f"Saved thermal CSV : {thermal_csv}", flush=True)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Thermally safe ONNX benchmark on Raspberry Pi 4")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["plain", "se", "qcab"],
        help="Which single model to benchmark"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    print_header("Safe ONNX Raspberry Pi Benchmark")
    print(f"Base dir                  : {BASE_DIR}", flush=True)
    print(f"Selected model            : {args.model} ({DISPLAY_NAMES[args.model]})", flush=True)
    print(f"Warm-up runs              : {WARMUP_RUNS}", flush=True)
    print(f"Measured runs             : {MEASURED_RUNS}", flush=True)
    print(f"Safe temp threshold (C)   : {SAFE_TEMP_C}", flush=True)
    print(f"Warning temp threshold (C): {WARNING_TEMP_C}", flush=True)
    print(f"Danger temp threshold (C) : {DANGER_TEMP_C}", flush=True)
    print(f"Sleep between runs (s)    : {SLEEP_BETWEEN_RUNS_SEC}", flush=True)
    print(f"Execution provider        : CPUExecutionProvider", flush=True)

    summary = benchmark_single_model(args.model)

    print_header("Benchmark Completed")
    print(summary, flush=True)


if __name__ == "__main__":
    main()