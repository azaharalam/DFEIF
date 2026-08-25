#!/usr/bin/env python3
"""Edge TPU benchmark harness with selectable instrumentation depth.

This is the workload DFIT measures. It runs a compiled *_edgetpu.tflite model
for N iterations at one of four instrumentation levels:

  none   - bare invoke() loop, no DFIT bookkeeping at all (the baseline arm)
  ids    - attach a stable id per invocation and record it
  lite   - record program begin/end boundaries plus per-invocation timing
  full   - per-invocation timing, tensor metadata, and a JSONL event trace

The point of the levels is that they cost different amounts. Running `none` and
`full` as the two arms of dfabitctl's A/B measurement gives a real, measured
instrumentation overhead on real hardware -- not an estimate.
"""

import argparse
import json
import os
import statistics
import sys
import time

# FNV-1a, matching src/metadata/stable_id.cc so ids agree with the C++ side.
_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211
_MASK = 0xFFFFFFFFFFFFFFFF


def _fnv1a64(text):
    h = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * _FNV_PRIME) & _MASK
    return h


def stable_id(scope, key):
    return _fnv1a64("{}:{}".format(scope, key))


def make_interpreter(model_path, use_tpu=True):
    """Builds an interpreter, on the Edge TPU or on CPU."""
    if use_tpu:
        try:
            from pycoral.utils.edgetpu import make_interpreter as _mk
            return _mk(model_path)
        except ImportError:
            pass
        from tflite_runtime.interpreter import Interpreter, load_delegate
        return Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate("libedgetpu.so.1")],
        )

    from tflite_runtime.interpreter import Interpreter
    return Interpreter(model_path=model_path)


def tensor_details(interpreter):
    """Op/tensor metadata the Edge TPU actually exposes at runtime."""
    out = []
    for detail in interpreter.get_tensor_details():
        name = detail.get("name", "")
        shape = [int(d) for d in detail.get("shape", [])]
        dtype = detail.get("dtype")
        dtype_name = getattr(dtype, "__name__", str(dtype))
        out.append({
            "name": name,
            "shape": shape,
            "dtype": dtype_name,
            "index": int(detail.get("index", -1)),
            "stable_id": stable_id("edgetpu_tensor", name),
        })
    return out


def run(args):
    import numpy as np

    interpreter = make_interpreter(args.model, use_tpu=not args.cpu)
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    shape = input_detail["shape"]
    dtype = input_detail["dtype"]

    # Deterministic input: the same bytes every run, so the two arms of an A/B
    # measurement differ only in instrumentation, never in data.
    rng = np.random.RandomState(args.seed)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        sample = rng.randint(info.min, info.max + 1, size=shape).astype(dtype)
    else:
        sample = rng.random_sample(shape).astype(dtype)

    level = args.instrument
    trace_file = None
    if level == "full" and args.trace:
        os.makedirs(os.path.dirname(os.path.abspath(args.trace)), exist_ok=True)
        trace_file = open(args.trace, "w", encoding="utf-8")

    model_id = stable_id("edgetpu_model", os.path.basename(args.model))
    details = tensor_details(interpreter) if level == "full" else []

    def emit(event, payload):
        if trace_file is None:
            return
        trace_file.write(json.dumps({"event": event, **payload}) + "\n")

    if level in ("lite", "full"):
        emit("program_begin", {
            "model": os.path.basename(args.model),
            "stable_id": model_id,
            "ts_ns": time.perf_counter_ns(),
            "tensor_count": len(details),
        })

    interpreter.set_tensor(input_detail["index"], sample)

    # Warmup: never counted. Covers first-invoke setup and USB negotiation.
    for _ in range(args.warmup):
        interpreter.invoke()

    latencies = []
    for i in range(args.iters):
        if level == "none":
            # The tight loop. Nothing but the call itself.
            t0 = time.perf_counter_ns()
            interpreter.invoke()
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1e6)
            continue

        invocation_id = stable_id("edgetpu_invocation", "{}#{}".format(args.model, i))

        if level == "ids":
            t0 = time.perf_counter_ns()
            interpreter.invoke()
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1e6)
            continue

        t0 = time.perf_counter_ns()
        interpreter.invoke()
        t1 = time.perf_counter_ns()
        elapsed_ms = (t1 - t0) / 1e6
        latencies.append(elapsed_ms)

        if level == "full":
            emit("invoke", {
                "iteration": i,
                "stable_id": invocation_id,
                "model_stable_id": model_id,
                "begin_ns": t0,
                "end_ns": t1,
                "latency_ms": elapsed_ms,
            })

    if level in ("lite", "full"):
        emit("program_end", {
            "stable_id": model_id,
            "ts_ns": time.perf_counter_ns(),
            "iterations": args.iters,
        })

    if level == "full":
        for detail in details:
            emit("tensor", detail)

    if trace_file is not None:
        trace_file.close()

    return latencies, details, model_id


def write_runtime_log(path, latencies, details, model_id):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    ordered = sorted(latencies)
    median = statistics.median(ordered)
    mean = statistics.fmean(ordered)
    stdev = statistics.stdev(ordered) if len(ordered) > 1 else 0.0
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]

    rows = [
        ("run", "latency_ms", median, "ms", model_id),
        ("run", "latency_mean_ms", mean, "ms", model_id),
        ("run", "latency_stddev_ms", stdev, "ms", model_id),
        ("run", "latency_p95_ms", p95, "ms", model_id),
        ("run", "latency_min_ms", ordered[0], "ms", model_id),
        ("run", "latency_max_ms", ordered[-1], "ms", model_id),
        ("run", "iterations", len(ordered), "count", 0),
        ("run", "throughput", 1000.0 / median if median > 0 else 0.0,
         "inferences_per_s", model_id),
    ]

    # Static byte proxy from tensor shapes. This is derived from metadata, not
    # measured on the device -- the Edge TPU exposes no traffic counters -- so it
    # is named to say so.
    if details:
        total_bytes = 0
        width = {"int8": 1, "uint8": 1, "int16": 2, "float16": 2,
                 "int32": 4, "float32": 4, "int64": 8, "float64": 8}
        for d in details:
            if not d["shape"]:
                continue
            n = 1
            for dim in d["shape"]:
                if dim <= 0:
                    n = 0
                    break
                n *= dim
            total_bytes += n * width.get(d["dtype"], 1)
        rows.append(("run", "tensor_bytes_static_proxy", total_bytes, "B", 0))
        rows.append(("run", "tensor_count", len(details), "count", 0))

    with open(path, "w", encoding="utf-8") as fh:
        for phase, name, value, unit, sid in rows:
            fh.write("{},{},{},{},{}\n".format(phase, name, value, unit, sid))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="*_edgetpu.tflite path")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--instrument", default="full",
                        choices=["none", "ids", "lite", "full"])
    parser.add_argument("--runtime-log", default="", help="DFIT runtime log output")
    parser.add_argument("--trace", default="", help="JSONL event trace (full mode)")
    parser.add_argument("--cpu", action="store_true",
                        help="run on CPU instead of the Edge TPU")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not os.path.exists(args.model):
        print("model not found: {}".format(args.model), file=sys.stderr)
        return 2
    if args.iters < 1:
        print("--iters must be >= 1", file=sys.stderr)
        return 2

    latencies, details, model_id = run(args)

    if args.runtime_log:
        write_runtime_log(args.runtime_log, latencies, details, model_id)

    if not args.quiet:
        ordered = sorted(latencies)
        print("instrument={} device={} iters={} median={:.4f} ms mean={:.4f} ms".format(
            args.instrument,
            "cpu" if args.cpu else "edgetpu",
            len(latencies),
            statistics.median(ordered),
            statistics.fmean(ordered),
        ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())