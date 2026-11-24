#!/usr/bin/env python3
"""
Raspberry Pi 5 benchmark for the ONNX models in pi_bundle.zip.

Setup on the Pi (64-bit Raspberry Pi OS):
    python3 -m venv ~/qenv && source ~/qenv/bin/activate
    pip install onnxruntime numpy
    unzip pi_bundle.zip -d pi_bundle
    python3 pi_benchmark.py pi_bundle            # extra ONNX files can be dropped into pi_bundle/onnx/

For every ONNX file it measures, each in a fresh process:
  - accuracy on the 380-image test subset, compared with the accuracy Kaggle measured on the same images
  - single-image latency (median / p95) with 1 and 4 threads
  - peak memory (RSS) of the process
  - CPU temperature and throttling flags before and after
Results: pi_results.csv (and printed as a table).
"""
import argparse, csv, glob, json, os, platform, resource, subprocess, sys, time

import numpy as np


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def temp_c():
    t = sh("vcgencmd measure_temp")  # temp=48.3'C
    if t.startswith("temp="):
        return float(t[5:].split("'")[0])
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except Exception:
        return float("nan")


def wait_cool(limit=65.0, target=55.0, max_wait=300):
    t = temp_c()
    if not np.isfinite(t) or t < limit:
        return
    print(f"  cooling down from {temp_c():.1f} C ...", flush=True)
    t0 = time.time()
    while temp_c() > target and time.time() - t0 < max_wait:
        time.sleep(5)


def bench_one(bundle, onnx_path, threads, warmup, runs):
    """Runs in a child process so peak RSS belongs to one model only."""
    import onnxruntime as ort
    meta = json.load(open(os.path.join(bundle, "pi_meta.json")))
    data = np.load(os.path.join(bundle, "pi_test_subset.npz"))
    name = os.path.basename(onnx_path)
    model = "mobilevit_xxs" if "mobilevit" in name else "mobilenetv3_small_100"
    mean = np.array(meta["normalisation"][model]["mean"], np.float32).reshape(1, 3, 1, 1)
    std = np.array(meta["normalisation"][model]["std"], np.float32).reshape(1, 3, 1, 1)
    x = ((data["images_u8"].astype(np.float32) / 255.0).transpose(0, 3, 1, 2) - mean) / std
    y = data["labels"]

    res = {"file": name, "threads": threads}
    for th in threads:
        so = ort.SessionOptions()
        so.intra_op_num_threads = th
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        if th == max(threads):  # accuracy once, with the most threads
            preds = np.concatenate([sess.run(None, {"input": x[i:i + 32]})[0].argmax(1) for i in range(0, len(x), 32)])
            res["subset_acc_pi"] = float((preds == y).mean())
        one = x[:1]
        for _ in range(warmup):
            sess.run(None, {"input": one})
        t = []
        for i in range(runs):
            xi = x[i % len(x)][None]
            t0 = time.perf_counter()
            sess.run(None, {"input": xi})
            t.append((time.perf_counter() - t0) * 1000)
        res[f"median_ms_{th}t"] = float(np.median(t))
        res[f"p95_ms_{th}t"] = float(np.percentile(t, 95))
    res["peak_rss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print("RESULT " + json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--threads", default="1,4")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--single", help=argparse.SUPPRESS)
    a = ap.parse_args()
    threads = [int(t) for t in a.threads.split(",")]
    if a.single:
        return bench_one(a.bundle, a.single, threads, a.warmup, a.runs)

    import onnxruntime as ort
    meta = json.load(open(os.path.join(a.bundle, "pi_meta.json")))
    expected = meta.get("expected_subset_acc_kaggle", {})
    print("Device:", sh("cat /proc/device-tree/model").replace("\x00", "") or platform.node())
    print("Arch:", platform.machine(), "| Python", platform.python_version(), "| onnxruntime", ort.__version__)
    print("RAM:", sh("free -m | awk '/Mem:/ {print $2\" MB\"}'"), "| CPU:", sh("lscpu | grep 'Model name' | head -1"))
    print("Throttled flags at start:", sh("vcgencmd get_throttled") or "n/a", "| temp", temp_c(), "C\n")
    if platform.machine() not in ("aarch64", "arm64"):
        print("WARNING: not a 64-bit ARM system - results will not represent the Pi 5.\n")

    rows = []
    for f in sorted(glob.glob(os.path.join(a.bundle, "onnx", "*.onnx"))):
        wait_cool()
        t_before = temp_c()
        out = subprocess.run([sys.executable, __file__, a.bundle, "--single", f, "--threads", a.threads,
                              "--warmup", str(a.warmup), "--runs", str(a.runs)], capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(f"{os.path.basename(f)}: FAILED\n{out.stderr[-500:]}")
            rows.append({"file": os.path.basename(f), "error": out.stderr[-200:]})
            continue
        r = json.loads(line[0][7:])
        r.pop("threads")
        r.update({"subset_acc_kaggle": expected.get(r["file"], float("nan")), "temp_before_C": t_before,
                  "temp_after_C": temp_c(), "throttled_after": sh("vcgencmd get_throttled")})
        rows.append(r)
        lat = "  ".join(f"{t}t {r[f'median_ms_{t}t']:7.2f} ms" for t in threads)
        print(f"{r['file']:48s} acc Pi {r['subset_acc_pi']:.4f} / Kaggle {r['subset_acc_kaggle']:.4f}  {lat}  "
              f"RSS {r['peak_rss_MB']:6.1f} MB  {r['temp_after_C']:.1f} C", flush=True)

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "file", k))
    with open("pi_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\nSaved pi_results.csv | throttled flags at end:", sh("vcgencmd get_throttled") or "n/a")


if __name__ == "__main__":
    main()
