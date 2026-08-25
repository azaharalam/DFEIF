#!/usr/bin/env python3
"""Generates benchmark models that are not available in the Coral model zoo.

Three of the benchmarks in the evaluation have no ready-made INT8 TFLite file:

  GEMM        a single dense matrix multiply, 1024x1024
  SpMM        the same shape with a mostly-zero weight matrix
  ResNet-101  a second, deeper ResNet alongside the trained ResNet-50

Rather than substitute an unrelated model and label it with the intended name,
this script builds each one to the stated specification and quantizes it to
INT8, which is what the Edge TPU compiler requires.

Two honesty constraints are baked in, and both are printed at the end of a run
so they cannot be quietly forgotten:

  1. SpMM executes DENSELY. TFLite has no sparse kernel for the Edge TPU, so the
     zeros reduce weight entropy but not arithmetic. It is a sparse *workload*,
     not sparse *execution*. Compile-stage metrics (operator mapping, memory
     residency, parameter footprint) are genuinely comparable against GEMM; a
     runtime speed-up claim would not be.

  2. ResNet-101 is generated with RANDOM WEIGHTS (weights=None), because the
     pretrained checkpoints are not redistributable here. Architecture fully
     determines operator count, TPU mapping, and memory residency, so every
     metric this framework collects is unaffected. Accuracy is meaningless and
     must never be reported for this model.

Usage:
    python3 make_benchmarks.py --out-dir models
    python3 make_benchmarks.py --out-dir models --only gemm,spmm
    python3 make_benchmarks.py --out-dir models --size 512 --sparsity 0.95
"""

import argparse
import os
import sys

# Silence TensorFlow's startup chatter before it is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def convert_int8(model, sample_shape, seed=0, samples=32):
    """Full INT8 quantization. The Edge TPU compiler rejects anything else.

    `sample_shape` is the input shape WITHOUT the batch dimension.
    """
    import numpy as np
    import tensorflow as tf

    rng = np.random.RandomState(seed + 1)

    def representative_dataset():
        for _ in range(samples):
            yield [rng.random_sample((1,) + tuple(sample_shape)).astype("float32")]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def measured_zero_fraction(tflite_bytes):
    """Actual zero fraction of the largest 2-D weight tensor in the file.

    Worth checking rather than trusting the requested sparsity: quantization can
    move near-zero values off zero, and the compiler sees what is in the file.
    """
    try:
        import numpy as np
        from tensorflow.lite.python import interpreter as tfl
    except ImportError:
        return None

    interp = tfl.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()

    best = None
    for detail in interp.get_tensor_details():
        shape = detail.get("shape")
        if shape is None or len(shape) != 2:
            continue
        try:
            tensor = interp.get_tensor(detail["index"])
        except (ValueError, RuntimeError):
            continue  # activation tensors are unreadable before invoke
        if best is None or tensor.size > best.size:
            best = tensor

    if best is None:
        return None
    return float(np.count_nonzero(best == 0)) / float(best.size)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_matmul(size, sparsity, seed=0):
    """A one-operator graph: [1, size] x [size, size] as FULLY_CONNECTED."""
    import numpy as np
    import tensorflow as tf

    rng = np.random.RandomState(seed)
    weights = rng.standard_normal((size, size)).astype("float32")

    if sparsity > 0.0:
        # Deterministic given the seed, so the generated model is reproducible.
        mask = rng.random_sample((size, size)) >= sparsity
        weights = weights * mask.astype("float32")

    bias = np.zeros((size,), dtype="float32")

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(size,), batch_size=1),
        tf.keras.layers.Dense(size, use_bias=True),
    ])
    model.layers[0].set_weights([weights, bias])
    return model, (size,)


def build_resnet101(resolution=224):
    """A genuine ResNet-101 architecture, randomly initialized."""
    import tensorflow as tf

    model = tf.keras.applications.ResNet101(
        weights=None,
        input_shape=(resolution, resolution, 3),
        classes=1000,
    )
    return model, (resolution, resolution, 3)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def generate(name, builder, out_dir, note=None):
    path = os.path.join(out_dir, name + ".tflite")

    if os.path.exists(path):
        print("  %s already exists, skipping" % path)
        return True

    print("building %s" % name)
    try:
        model, sample_shape = builder()
        blob = convert_int8(model, sample_shape)
    except Exception as exc:  # a failed model must not kill the whole run
        print("  FAILED: %s (%s)" % (name, exc), file=sys.stderr)
        return False

    with open(path, "wb") as fh:
        fh.write(blob)

    size_mib = len(blob) / (1024.0 * 1024.0)
    print("  wrote %s (%.2f MiB)" % (path, size_mib))

    if note == "sparsity":
        zf = measured_zero_fraction(blob)
        if zf is not None:
            print("  measured weight zero-fraction: %.3f" % zf)

    if size_mib > 8.0:
        # The Edge TPU has ~8 MB of on-chip SRAM. Larger models still compile,
        # but parameters stream off-chip every inference, which the compiler
        # will report and which dominates latency.
        print("  note: exceeds on-chip SRAM; expect off-chip streaming")

    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--size", type=int, default=1024,
                        help="GEMM/SpMM matrix dimension (default 1024)")
    parser.add_argument("--sparsity", type=float, default=0.9,
                        help="fraction of SpMM weights forced to zero")
    parser.add_argument("--resolution", type=int, default=224,
                        help="ResNet-101 input resolution")
    parser.add_argument("--only", default="",
                        help="comma-separated subset: gemm,spmm,resnet101")
    args = parser.parse_args(argv)

    if not 0.0 <= args.sparsity < 1.0:
        print("--sparsity must be in [0, 1)", file=sys.stderr)
        return 2

    wanted = {t.strip().lower() for t in args.only.split(",") if t.strip()}
    if not wanted:
        wanted = {"gemm", "spmm", "resnet101"}

    os.makedirs(args.out_dir, exist_ok=True)

    targets = []
    if "gemm" in wanted:
        targets.append((
            "GEMM_%d_quant" % args.size,
            lambda: build_matmul(args.size, 0.0),
            None,
        ))
    if "spmm" in wanted:
        targets.append((
            "SpMM_%d_quant" % args.size,
            lambda: build_matmul(args.size, args.sparsity),
            "sparsity",
        ))
    if "resnet101" in wanted:
        targets.append((
            "resnet101_%d_quant" % args.resolution,
            lambda: build_resnet101(args.resolution),
            None,
        ))

    ok = 0
    for name, builder, note in targets:
        if generate(name, builder, args.out_dir, note):
            ok += 1

    print()
    print("generated %d/%d models into %s" % (ok, len(targets), args.out_dir))
    print()
    print("REPORTING CONSTRAINTS -- carry these into the paper:")
    if "spmm" in wanted:
        print("  * SpMM executes densely. The Edge TPU has no sparse kernel, so")
        print("    its zeros change weight statistics, not arithmetic. Compare")
        print("    it against GEMM on compile-stage metrics only.")
    if "resnet101" in wanted:
        print("  * ResNet-101 is randomly initialized. Operator count, mapping")
        print("    and memory residency are architecture-determined and valid;")
        print("    accuracy is meaningless and must not be reported.")
    print()
    print("next: compile each, capturing the summary that carries the memory")
    print("block (the auto-generated *_edgetpu.log has only the operator table):")
    print("  cd %s" % args.out_dir)
    print("  for f in *.tflite; do b=${f%%.tflite}; [[ $b == *_edgetpu ]] && continue; \\")
    print("    edgetpu_compiler -s \"$f\" > \"${b}_summary.log\" 2>&1; done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())