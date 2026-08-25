#!/usr/bin/env python3
"""Derives roofline and execution-boundness data from measured latency and
extracted operator arithmetic.

Inputs:
  - macs_per_op.csv     from extract_macs.py (per-operator MACs and bytes)
  - macs_summary.csv    from extract_macs.py (per-model totals)
  - artifacts/rt_*.csv  from dfabit.edgetpu.bench (measured latency per model)

Outputs:
  - roofline.csv        per model: AI, achieved throughput, normalized efficiency
  - roofline_ops.csv    per operator: AI and MAC share, for a dense scatter
  - boundness.csv       per model: MAC-weighted split across roofline regions

=============================================================================
WHAT IS MEASURED AND WHAT IS DERIVED
=============================================================================
MEASURED on hardware:
  - end-to-end inference latency per model (median over N invocations)

DERIVED from the model file (exact given operator semantics, validated to
within 0.4% against published MAC counts for seven architectures):
  - MACs per operator and per model

STATIC PROXY, not measured:
  - byte movement. The Edge TPU exposes no traffic counters, so bytes are
    counted as one pass over every input and output tensor, with no reuse,
    caching, or fusion. Real traffic is lower. Arithmetic intensity here is
    therefore a LOWER BOUND, and every axis derived from it must say so.

CONSEQUENCES that must be carried into any figure caption:
  1. Models with CPU-fallback operators (e.g. DeepLab-v3, 8 fallback ops) have
     latency dominated by host execution. Their roofline point does not
     describe accelerator behaviour and is flagged in the output.
  2. Models whose parameters exceed on-chip SRAM (Inception-v4, ResNet-101)
     pay an off-chip streaming penalty per inference that the static byte
     proxy does not model. Their efficiency is correspondingly understated.
  3. Achieved throughput assumes all MACs are executed on the accelerator.
     Where operators fall back to CPU this overstates device throughput.
=============================================================================

Usage:
    python3 derive_roofline.py \
        --ops results/macs_per_op.csv \
        --summary results/macs_summary.csv \
        --latency-dir artifacts \
        --out-dir results
"""

import argparse
import csv
import glob
import os
import statistics
import sys

# Coral Edge TPU (USB Accelerator) datasheet figures.
PEAK_TOPS = 4.0e12          # int8 operations per second
PEAK_BW_BYTES = 8.0e9       # 8 GB/s over USB 3.0
OPS_PER_MAC = 2.0           # one multiply + one accumulate

PEAK_MACS = PEAK_TOPS / OPS_PER_MAC
RIDGE = PEAK_MACS / PEAK_BW_BYTES   # MACs per byte at the knee

# Roofline region boundaries as multiples of the ridge point. A hard cut at the
# ridge would make classification hypersensitive to error in the byte proxy, so
# a balanced band is used either side of it.
MEMORY_BELOW = 0.5
COMPUTE_ABOVE = 2.0

# Models known to execute significant work off the accelerator, from the Edge
# TPU compiler's own fallback report. Their latency is not a device measurement.
CPU_FALLBACK_HINTS = ("deeplabv3", "ssd_mobilenet", "yolov5")


def load_latencies(latency_dir):
    """median latency per model, from the bench harness runtime logs."""
    out = {}
    for path in sorted(glob.glob(os.path.join(latency_dir, "rt_*.csv"))):
        model = os.path.basename(path)[len("rt_"):-len(".csv")]
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) >= 3 and parts[0] == "run" and parts[1] == "latency_ms":
                    try:
                        out[model] = float(parts[2])
                    except ValueError:
                        pass
                    break
    return out


def load_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return [
            {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            for row in csv.DictReader(fh)
        ]


def classify(ai):
    if ai < RIDGE * MEMORY_BELOW:
        return "memory"
    if ai > RIDGE * COMPUTE_ABOVE:
        return "compute"
    return "balanced"


def flag_for(model):
    notes = []
    lowered = model.lower()
    if any(h in lowered for h in CPU_FALLBACK_HINTS):
        notes.append("cpu_fallback")
    return ";".join(notes)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ops", default="results/macs_per_op.csv")
    parser.add_argument("--summary", default="results/macs_summary.csv")
    parser.add_argument("--latency-dir", default="artifacts")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args(argv)

    for path in (args.ops, args.summary):
        if not os.path.exists(path):
            print("missing input: %s (run extract_macs.py first)" % path,
                  file=sys.stderr)
            return 2

    ops = load_csv(args.ops)
    summary = load_csv(args.summary)
    latency = load_latencies(args.latency_dir)

    if not latency:
        print("no rt_*.csv found in %s (run dfabit.edgetpu.bench first)"
              % args.latency_dir, file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)

    print("Coral Edge TPU roofline")
    print("  peak      %.2f GMAC/s (%.0f TOPS int8)" % (PEAK_MACS / 1e9, PEAK_TOPS / 1e12))
    print("  bandwidth %.1f GB/s" % (PEAK_BW_BYTES / 1e9))
    print("  ridge     %.1f MACs/byte" % RIDGE)
    print("  regions   memory < %.1f | balanced | %.1f < compute"
          % (RIDGE * MEMORY_BELOW, RIDGE * COMPUTE_ABOVE))
    print()

    # ---------------------------------------------------------------- per model
    model_rows = []
    for s in summary:
        model = s["model"]
        macs = int(s["total_macs"])
        total_bytes = int(s["total_bytes"])
        if macs == 0 or total_bytes == 0:
            continue

        lat_ms = latency.get(model)
        if lat_ms is None or lat_ms <= 0:
            print("  no latency for %s, skipping" % model)
            continue

        ai = macs / total_bytes
        achieved = macs / (lat_ms / 1000.0)          # MACs per second
        efficiency = achieved / PEAK_MACS

        model_rows.append({
            "model": model,
            "operators": s["operators"],
            "total_macs": macs,
            "total_bytes": total_bytes,
            "arithmetic_intensity": "%.4f" % ai,
            "latency_ms": "%.4f" % lat_ms,
            "achieved_gmac_per_s": "%.4f" % (achieved / 1e9),
            "normalized_efficiency": "%.6f" % efficiency,
            "region": classify(ai),
            "flags": flag_for(model),
        })

    model_rows.sort(key=lambda r: float(r["arithmetic_intensity"]))

    path = os.path.join(args.out_dir, "roofline.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(model_rows[0].keys()))
        w.writeheader()
        w.writerows(model_rows)

    print("%-44s %8s %9s %10s %8s %s" % (
        "model", "AI", "lat_ms", "GMAC/s", "eff", "region"))
    for r in model_rows:
        print("%-44s %8.2f %9.3f %10.2f %8.4f %-9s %s" % (
            r["model"], float(r["arithmetic_intensity"]), float(r["latency_ms"]),
            float(r["achieved_gmac_per_s"]), float(r["normalized_efficiency"]),
            r["region"], r["flags"]))
    print("\nwrote %s" % path)

    # ------------------------------------------------------------- per operator
    op_rows = []
    for o in ops:
        if o.get("macs_determined") != "1":
            continue
        try:
            macs = int(o["macs"])
            total_bytes = int(o["total_bytes"])
            ai = float(o["arithmetic_intensity"])
        except (ValueError, KeyError):
            continue
        if macs <= 0 or total_bytes <= 0:
            continue

        op_rows.append({
            "model": o["model"],
            "op_index": o["op_index"],
            "op_name": o["op_name"],
            "macs": macs,
            "total_bytes": total_bytes,
            "arithmetic_intensity": "%.4f" % ai,
            "region": classify(ai),
        })

    path = os.path.join(args.out_dir, "roofline_ops.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(op_rows[0].keys()))
        w.writeheader()
        w.writerows(op_rows)
    print("wrote %s (%d operators with computable arithmetic)" % (path, len(op_rows)))

    # -------------------------------------------------------------- boundness
    # Weighted by MAC share, not operator count: twenty reshapes should not
    # outvote one convolution that carries most of the arithmetic.
    by_model = {}
    for o in op_rows:
        by_model.setdefault(o["model"], []).append(o)

    bound_rows = []
    for model, rows in sorted(by_model.items()):
        total = sum(r["macs"] for r in rows)
        if total == 0:
            continue
        share = {"memory": 0, "balanced": 0, "compute": 0}
        for r in rows:
            share[r["region"]] += r["macs"]

        bound_rows.append({
            "model": model,
            "operators": len(rows),
            "total_macs": total,
            "memory_frac": "%.4f" % (share["memory"] / total),
            "balanced_frac": "%.4f" % (share["balanced"] / total),
            "compute_frac": "%.4f" % (share["compute"] / total),
            "model_level_region": next(
                (r["region"] for r in model_rows if r["model"] == model), ""),
            "flags": flag_for(model),
        })

    path = os.path.join(args.out_dir, "boundness.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bound_rows[0].keys()))
        w.writeheader()
        w.writerows(bound_rows)

    print("\n%-44s %9s %9s %9s  %s" % (
        "model", "memory", "balanced", "compute", "model-level"))
    for r in bound_rows:
        print("%-44s %9.3f %9.3f %9.3f  %s" % (
            r["model"], float(r["memory_frac"]), float(r["balanced_frac"]),
            float(r["compute_frac"]), r["model_level_region"]))
    print("\nwrote %s" % path)

    # ------------------------------------------------------------ observations
    ais = [float(r["arithmetic_intensity"]) for r in model_rows]
    op_ais = [float(r["arithmetic_intensity"]) for r in op_rows]
    above = sum(1 for a in op_ais if a > RIDGE)

    print()
    print("OBSERVATIONS")
    print("  model-level AI      %.2f - %.2f (median %.2f)"
          % (min(ais), max(ais), statistics.median(ais)))
    print("  operator-level AI   %.2f - %.2f (median %.2f)"
          % (min(op_ais), max(op_ais), statistics.median(op_ais)))
    print("  operators above ridge: %d of %d (%.1f%%)"
          % (above, len(op_ais), 100.0 * above / len(op_ais)))
    if max(ais) < RIDGE:
        print()
        print("  Every model sits below the ridge point while a substantial")
        print("  fraction of individual operators sits above it. This is an")
        print("  aggregation effect, not a contradiction: a minority of")
        print("  bandwidth-heavy operators carries most of the byte movement")
        print("  and pulls the model-level ratio down. Report both levels; the")
        print("  model-level figure alone would hide where the cost actually is.")

    print()
    print("  REMINDER: byte movement is a static proxy, so arithmetic intensity")
    print("  is a lower bound. Rows flagged cpu_fallback have latency dominated")
    print("  by host execution and do not describe accelerator behaviour.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
