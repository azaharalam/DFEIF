import os
import numpy as np

def _require_native():
    try:
        import dfabit_native  # built by CMake if pybind11 is available
        return dfabit_native
    except Exception as e:
        raise RuntimeError(
            "dfabit_native not found. Build with pybind11 installed (DFABIT_BUILD_PYTHON=ON)."
        ) from e

def _make_interpreter(model_path: str):
    # Manifest extraction uses tflite_runtime (pycoral optional)
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception as e:
        raise RuntimeError("tflite_runtime is required for manifest extraction.") from e
    itp = Interpreter(model_path=model_path)
    itp.allocate_tensors()
    return itp

def _dtype_nbytes(dtype):
    return int(np.dtype(dtype).itemsize)

def _tensor_nbytes(shape, dtype):
    if shape is None:
        return 0
    n = 1
    for d in shape:
        if int(d) < 0:
            return 0
        n *= int(d)
    return int(n) * _dtype_nbytes(dtype)

def _estimate_macs(op_name: str, in_shapes, out_shapes):
    name = (op_name or "").lower()
    # Heuristics; you can refine per-model later
    if "fullyconnected" in name or name == "fully_connected":
        if len(in_shapes) >= 2 and len(out_shapes) >= 1:
            out = out_shapes[0]
            w = in_shapes[1]
            if len(out) >= 2 and len(w) >= 2:
                batch = int(out[0]); out_dim = int(out[-1]); in_dim = int(w[-1])
                if batch > 0 and out_dim > 0 and in_dim > 0:
                    return batch * out_dim * in_dim
    if "conv2d" in name or "conv_2d" in name or "convolution" in name:
        if len(in_shapes) >= 2 and len(out_shapes) >= 1:
            x = in_shapes[0]; w = in_shapes[1]; y = out_shapes[0]
            if len(y) == 4 and len(w) >= 3:
                out_h, out_w, out_c = int(y[1]), int(y[2]), int(y[3])
                k_h = int(w[1]); k_w = int(w[2])
                if "depthwise" in name:
                    in_c = int(x[3]) if len(x) == 4 else out_c
                    return max(0, out_h*out_w*in_c*k_h*k_w)
                else:
                    in_c = int(x[3]) if len(x) == 4 else 0
                    return max(0, out_h*out_w*out_c*in_c*k_h*k_w)
    if "matmul" in name:
        if len(in_shapes) >= 2 and len(out_shapes) >= 1:
            a = in_shapes[0]; y = out_shapes[0]
            if len(a) >= 2 and len(y) >= 2:
                m = int(y[-2]); n = int(y[-1]); k = int(a[-1])
                return max(0, m*n*k)
    return 0

def generate_program_analysis(model_path: str, out_csv: str):
    dfabit_native = _require_native()

    itp = _make_interpreter(model_path)
    tensors = {td["index"]: td for td in itp.get_tensor_details()}

    get_ops = getattr(itp, "_get_ops_details", None)
    if not callable(get_ops):
        raise RuntimeError("tflite interpreter does not expose _get_ops_details() in this environment.")
    ops = list(get_ops())

    base = os.path.basename(model_path).replace(".tflite", "").replace("_edgetpu", "").lower()
    op_records = []
    for i, op in enumerate(ops):
        op_name = op.get("op_name", "UNKNOWN")
        inputs = [int(x) for x in op.get("inputs", []) if int(x) >= 0]
        outputs = [int(x) for x in op.get("outputs", []) if int(x) >= 0]

        in_shapes = [list(tensors[idx].get("shape", [])) for idx in inputs if idx in tensors]
        out_shapes = [list(tensors[idx].get("shape", [])) for idx in outputs if idx in tensors]

        in_bytes = sum(_tensor_nbytes(tensors[idx].get("shape", []), tensors[idx].get("dtype", np.int8))
                      for idx in inputs if idx in tensors)
        out_bytes = sum(_tensor_nbytes(tensors[idx].get("shape", []), tensors[idx].get("dtype", np.int8))
                       for idx in outputs if idx in tensors)
        total_bytes = int(in_bytes + out_bytes)
        macs = int(_estimate_macs(op_name, in_shapes, out_shapes))

        rec = dfabit_native.OpRecord()
        rec.op_id = f"0:{i}"
        rec.op_name = op_name
        rec.layer = f"{op_name} {tuple(out_shapes[0]) if out_shapes else ()}"
        rec.mac_ops = macs
        rec.total_bytes = total_bytes
        rec.measured_latency_ms = 0.0
        rec.kernel_id = f"coral.{base}.{op_name.lower()}{i}"
        op_records.append(rec)

    ctx = dfabit_native.Context()
    cfg = dfabit_native.ProgramAnalysisConfig()
    cfg.model_name = base
    cfg.out_csv_path = out_csv
    ctx.set_program_analysis_config(cfg)
    ctx.set_ops(op_records)

    tool = dfabit_native.ProgramAnalyzer()
    tool.run(ctx)
