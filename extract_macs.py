#!/usr/bin/env python3
"""Extracts per-operator MACs and byte movement from a TFLite model.

Figures 7 (roofline) and 8 (execution-time boundness) both need arithmetic
intensity, which is MACs divided by bytes moved. Neither is reported by the Edge
TPU compiler, but both are derivable from the model file: operator kinds and
tensor shapes fully determine MAC counts for the operators that dominate these
networks, and tensor footprints give byte movement.

What is measured versus derived, stated plainly because the distinction matters
for how these numbers may be reported:

  DERIVED FROM THE MODEL FILE (exact, given the operator semantics):
    - MACs per operator, from input/output tensor shapes
    - Tensor footprints in bytes, from shape and dtype

  NOT MEASURED ANYWHERE:
    - Actual bytes transferred at runtime. The Edge TPU exposes no traffic
      counters. Byte movement here is a static upper bound assuming every
      input and output tensor crosses the memory interface once, with no
      reuse, caching, or fusion. Real traffic is lower, often much lower.

So arithmetic intensity computed here is a LOWER bound on the true value, and
must be labelled a static proxy wherever it is plotted. It is directly
comparable between models -- the same assumption applies to all of them -- which
is what the roofline and boundness figures actually require.

Operators whose MAC count is not shape-determined (CUSTOM, DELEGATE, and the
Edge TPU's fused subgraph op) are reported with macs=0 and flagged, rather than
guessed at.

Usage:
    python3 extract_macs.py --model models/mobilenet_v2_1.0_224_quant.tflite
    python3 extract_macs.py --model-dir models --out results/macs.csv
"""

import argparse
import csv
import glob
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Bytes per element, by TFLite dtype name.
_DTYPE_BYTES = {
    "int8": 1, "uint8": 1, "bool": 1,
    "int16": 2, "uint16": 2, "float16": 2, "bfloat16": 2,
    "int32": 4, "uint32": 4, "float32": 4,
    "int64": 8, "uint64": 8, "float64": 8,
}

# Operators whose arithmetic is fully determined by tensor shapes.
_SHAPE_DETERMINED = {
    "CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV",
    "FULLY_CONNECTED", "BATCH_MATMUL",
}

# Operators that move data but perform no multiply-accumulates.
_ZERO_MAC = {
    "RESHAPE", "QUANTIZE", "DEQUANTIZE", "CONCATENATION", "PAD", "PADV2",
    "TRANSPOSE", "SLICE", "STRIDED_SLICE", "SPLIT", "PACK", "UNPACK",
    "GATHER", "TILE", "CAST", "SHAPE", "SQUEEZE", "EXPAND_DIMS",
    "RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR", "MAX_POOL_2D",
    "AVERAGE_POOL_2D", "MEAN", "REDUCE_MAX", "ARG_MAX", "SOFTMAX",
    "LOGISTIC", "RELU", "RELU6", "TANH", "ADD", "SUB", "MUL", "DIV",
    "MAXIMUM", "MINIMUM", "SUM", "EXP", "SQRT", "SQUARE",
}

# Operators we cannot reason about: fused, custom, or delegated subgraphs.
_OPAQUE = {"CUSTOM", "DELEGATE", "edgetpu-custom-op"}


def _numel(shape):
    n = 1
    for d in shape:
        d = int(d)
        if d <= 0:
            return 0
        n *= d
    return n


def _tensor_bytes(detail):
    dtype = detail.get("dtype")
    name = getattr(dtype, "__name__", str(dtype)).replace("<class 'numpy.", "").rstrip("'>")
    width = _DTYPE_BYTES.get(name, 1)
    return _numel(detail.get("shape", [])) * width


def compute_macs(op_name, in_shapes, out_shapes):
    """MACs for one operator, from its tensor shapes.

    Returns (macs, determined) where `determined` is False when the operator's
    arithmetic cannot be inferred from shapes alone.
    """
    if op_name in _OPAQUE:
        return 0, False
    if op_name in _ZERO_MAC:
        return 0, True
    if op_name not in _SHAPE_DETERMINED:
        # Unknown operator: report zero and flag, rather than inventing a model.
        return 0, False

    if not out_shapes:
        return 0, False
    out = out_shapes[0]

    if op_name == "FULLY_CONNECTED":
        # out [batch, units]; weights [units, in_features]
        if len(in_shapes) >= 2 and len(in_shapes[1]) >= 2:
            units = int(in_shapes[1][0])
            in_features = int(in_shapes[1][1])
            batch = _numel(out) // units if units else 0
            return batch * units * in_features, True
        return 0, False

    if op_name == "BATCH_MATMUL":
        if len(in_shapes) >= 2 and len(in_shapes[0]) >= 2 and len(in_shapes[1]) >= 2:
            k = int(in_shapes[0][-1])
            return _numel(out) * k, True
        return 0, False

    if op_name in ("CONV_2D", "TRANSPOSE_CONV"):
        # weights [out_ch, kh, kw, in_ch]; one MAC per output element per tap.
        if len(in_shapes) >= 2 and len(in_shapes[1]) == 4:
            _, kh, kw, in_ch = (int(x) for x in in_shapes[1])
            return _numel(out) * kh * kw * in_ch, True
        return 0, False

    if op_name == "DEPTHWISE_CONV_2D":
        # weights [1, kh, kw, channels]; each output element sees kh*kw taps.
        if len(in_shapes) >= 2 and len(in_shapes[1]) == 4:
            _, kh, kw, _ = (int(x) for x in in_shapes[1])
            return _numel(out) * kh * kw, True
        return 0, False

    return 0, False


def analyze(model_path):
    from tensorflow.lite.python import interpreter as tfl

    interp = tfl.Interpreter(model_path=model_path)
    interp.allocate_tensors()

    details = {d["index"]: d for d in interp.get_tensor_details()}

    try:
        ops = interp._get_ops_details()
    except AttributeError:
        raise RuntimeError(
            "this TFLite build does not expose _get_ops_details(); "
            "per-operator MACs cannot be extracted"
        )

    rows = []
    for op in ops:
        name = op.get("op_name", "UNKNOWN")

        in_shapes, out_shapes = [], []
        in_bytes = out_bytes = 0

        for idx in op.get("inputs", []):
            if idx < 0 or idx not in details:
                continue
            d = details[idx]
            in_shapes.append(list(d.get("shape", [])))
            in_bytes += _tensor_bytes(d)

        for idx in op.get("outputs", []):
            if idx < 0 or idx not in details:
                continue
            d = details[idx]
            out_shapes.append(list(d.get("shape", [])))
            out_bytes += _tensor_bytes(d)

        macs, determined = compute_macs(name, in_shapes, out_shapes)
        total_bytes = in_bytes + out_bytes

        rows.append({
            "op_index": op.get("index", -1),
            "op_name": name,
            "macs": macs,
            "macs_determined": int(determined),
            "input_bytes": in_bytes,
            "output_bytes": out_bytes,
            "total_bytes": total_bytes,
            # Static proxy: assumes one pass over every tensor, no reuse.
            "arithmetic_intensity": (macs / total_bytes) if total_bytes else 0.0,
        })

    return rows


def summarize(model, rows):
    total_macs = sum(r["macs"] for r in rows)
    total_bytes = sum(r["total_bytes"] for r in rows)
    undetermined = sum(1 for r in rows if not r["macs_determined"])

    return {
        "model": model,
        "operators": len(rows),
        "operators_undetermined": undetermined,
        "total_macs": total_macs,
        "total_bytes": total_bytes,
        "arithmetic_intensity": (total_macs / total_bytes) if total_bytes else 0.0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="", help="single .tflite file")
    parser.add_argument("--model-dir", default="", help="directory of .tflite files")
    parser.add_argument("--out", default="", help="per-operator CSV output")
    parser.add_argument("--summary-out", default="", help="per-model summary CSV")
    args = parser.parse_args(argv)

    targets = []
    if args.model:
        targets.append(args.model)
    if args.model_dir:
        for path in sorted(glob.glob(os.path.join(args.model_dir, "*.tflite"))):
            # Skip compiled artifacts: their graph is one opaque fused op.
            if path.endswith("_edgetpu.tflite"):
                continue
            targets.append(path)

    if not targets:
        print("nothing to analyze; pass --model or --model-dir", file=sys.stderr)
        return 2

    all_rows, summaries = [], []

    for path in targets:
        base = os.path.basename(path)[:-len(".tflite")]
        try:
            rows = analyze(path)
        except Exception as exc:
            print("  FAILED %s: %s" % (base, exc), file=sys.stderr)
            continue

        for r in rows:
            r["model"] = base
        all_rows.extend(rows)

        s = summarize(base, rows)
        summaries.append(s)
        print("%-44s %4d ops  %14d MACs  %12d B  AI %8.3f%s" % (
            base, s["operators"], s["total_macs"], s["total_bytes"],
            s["arithmetic_intensity"],
            "  (%d undetermined)" % s["operators_undetermined"]
            if s["operators_undetermined"] else ""))

    if args.out and all_rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "model", "op_index", "op_name", "macs", "macs_determined",
                "input_bytes", "output_bytes", "total_bytes",
                "arithmetic_intensity"])
            w.writeheader()
            w.writerows(all_rows)
        print("\nwrote %s (%d operator rows)" % (args.out, len(all_rows)))

    if args.summary_out and summaries:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_out)), exist_ok=True)
        with open(args.summary_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "model", "operators", "operators_undetermined",
                "total_macs", "total_bytes", "arithmetic_intensity"])
            w.writeheader()
            w.writerows(summaries)
        print("wrote %s (%d models)" % (args.summary_out, len(summaries)))

    print()
    print("NOTE: byte movement is a STATIC UPPER BOUND (every tensor counted")
    print("once, no reuse or fusion). Arithmetic intensity is therefore a lower")
    print("bound and must be labelled a static proxy wherever plotted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
