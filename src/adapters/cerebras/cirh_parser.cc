#include "dfabit/adapters/cerebras/cirh_parser.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <utility>

#include "dfabit/metadata/stable_id.h"

namespace dfabit::adapters::cerebras {

namespace {

std::size_t DtypeBytes(const std::string& d) {
  if (d == "f64" || d == "i64" || d == "ui64") return 8;
  if (d == "f32" || d == "i32" || d == "ui32") return 4;
  if (d == "f16" || d == "bf16" || d == "cbfloat16" || d == "i16" || d == "ui16") return 2;
  if (d == "i8" || d == "ui8" || d == "i1") return 1;
  return 4;
}

struct Tensor {
  std::vector<long long> dims;
  std::string dtype;
};

double Numel(const std::vector<long long>& dims) {
  if (dims.empty()) return 1.0;
  double n = 1.0;
  for (const auto d : dims) n *= static_cast<double>(d);
  return n;
}

// "2850x197x768xf32" -> dims [2850,197,768], dtype f32. A non-numeric segment
// means a dynamic or unsupported shape, which yields empty dims rather than a
// guess.
Tensor ParseTensorBody(const std::string& body) {
  Tensor t;
  std::vector<std::string> parts;
  std::string cur;
  for (const char c : body) {
    if (c == 'x') { parts.push_back(cur); cur.clear(); }
    else { cur += c; }
  }
  parts.push_back(cur);
  if (parts.empty()) return t;

  t.dtype = parts.back();
  for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
    try {
      t.dims.push_back(std::stoll(parts[i]));
    } catch (...) {
      t.dims.clear();
      return t;
    }
  }
  return t;
}

std::vector<Tensor> ExtractTensors(const std::string& text) {
  static const std::regex re(R"(tensor<([^>]*)>)");
  std::vector<Tensor> out;
  for (auto it = std::sregex_iterator(text.begin(), text.end(), re);
       it != std::sregex_iterator(); ++it) {
    out.push_back(ParseTensorBody((*it)[1].str()));
  }
  return out;
}

bool AttrIsTrue(const std::string& attrs, const std::string& key) {
  const std::regex re(key + R"(\s*=\s*(true|false))");
  std::smatch m;
  if (std::regex_search(attrs, m, re)) return m[1].str() == "true";
  return false;
}

std::string AttrString(const std::string& attrs, const std::string& key) {
  const std::regex re(key + R"RX(\s*=\s*"([^"]*)")RX");
  std::smatch m;
  if (std::regex_search(attrs, m, re)) return m[1].str();
  return std::string();
}

// Multiply-accumulate count where the operator's shapes determine it.
double ComputeMacs(
    const std::string& op,
    const std::vector<Tensor>& ins,
    const std::vector<Tensor>& outs,
    const std::string& attrs,
    bool* determined) {
  *determined = false;
  if (outs.empty()) return 0.0;
  const double out_elems = Numel(outs[0].dims);

  if (op == "MatMul" || op == "SparseActMatMul") {
    // SparseActMatMul is the mixture-of-experts matmul: its sparsity is in
    // which experts activate, not in the arithmetic each performs, so this is
    // an upper bound on work actually executed.
    if (ins.size() >= 2 && ins[0].dims.size() >= 2 && ins[1].dims.size() >= 2) {
      const auto& a = ins[0].dims;
      const double k = AttrIsTrue(attrs, "transpose_a")
                           ? static_cast<double>(a[a.size() - 2])
                           : static_cast<double>(a.back());
      *determined = true;
      return out_elems * k;
    }
    return 0.0;
  }

  if (op == "DotGeneral") {
    if (!ins.empty() && ins[0].dims.size() >= 2) {
      *determined = true;
      return out_elems * static_cast<double>(ins[0].dims.back());
    }
    return 0.0;
  }

  if (op == "Convolution") {
    if (ins.size() >= 2 && ins[1].dims.size() == 4) {
      const auto dir = AttrString(attrs, "conv_pass_direction");
      if (dir.rfind("BWD_KERNEL", 0) == 0) {
        // The result is the weight gradient; the contraction runs over the
        // batch and spatial extent of the two operands.
        if (ins[0].dims.size() == 4) {
          *determined = true;
          return out_elems * static_cast<double>(ins[0].dims[0]) *
                 static_cast<double>(ins[1].dims[2]) *
                 static_cast<double>(ins[1].dims[3]);
        }
        return 0.0;
      }
      *determined = true;
      return out_elems * static_cast<double>(ins[1].dims[1]) *
             static_cast<double>(ins[1].dims[2]) *
             static_cast<double>(ins[1].dims[3]);
    }
    return 0.0;
  }

  // Operators that do real arithmetic but no multiply-accumulates, and pure
  // data movement. Both are zero-MAC by definition rather than by omission.
  static const std::vector<std::string> kZeroMac = {
      "LayerNorm", "LayerNormGrad", "Softmax", "Gelu", "GeluGrad", "L2Norm",
      "Reduce", "SparseSoftmaxCrossEntropyWithLogits", "Mul", "Add", "Sub",
      "Div", "Sqrt", "Select", "Cast", "Compare", "Clamp", "Reciprocal",
      "Cos", "Sin", "Exp", "Log", "BiasAdd", "BiasAddGrad", "RandomUniform",
      "Arange", "Const", "Sigmoid", "Tanh", "Abs", "Rsqrt", "Neg", "Sign",
      "Reshape", "Transpose", "BroadcastInDim", "Slice", "Concat", "Gather",
      "Scatter", "ScatterNd", "Pad", "StridedSlice", "Squeeze", "ExpandDims",
      "Split", "Stack", "Tile", "Reverse", "Where", "Max", "Min", "Pow", "Erf"};
  if (std::find(kZeroMac.begin(), kZeroMac.end(), op) != kZeroMac.end()) {
    *determined = true;
    return 0.0;
  }

  return 0.0;
}

std::string PhaseOf(
    const std::string& module_path,
    const std::string& aten_op,
    const std::string& attrs) {
  auto lower = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
  };
  const auto p = lower(module_path);
  const auto a = lower(aten_op);

  if (AttrString(attrs, "conv_pass_direction").rfind("BWD", 0) == 0) return "backward";
  if (p.find(".bwd") != std::string::npos ||
      a.find("backward") != std::string::npos) return "backward";
  if (p.rfind("grad_scaler", 0) == 0 ||
      p.find("optimizer") != std::string::npos ||
      p.find("scheduler") != std::string::npos) return "optimizer";
  return "forward";
}

// Frames belonging to the framework rather than to the model being compiled.
// An operator's callsite chain runs from the innermost dispatch frame out to
// the CLI entry point, so neither end names the code a developer wrote: the
// innermost is PyTorch's function-mode dispatch, shared by nearly every
// operator, and the outermost is the launcher. The first frame that is not
// infrastructure is the model source.
//
// This list is specific to the Cerebras PyTorch stack.
const char* const kFrameworkFrames[] = {
    "torch/",
    "cerebras/pytorch/",
    "cerebras/modelzoo/trainer/",
    "cerebras/modelzoo/cli/",
    "cszoo",
};

bool IsFrameworkFrame(const std::string& path) {
  for (const auto* prefix : kFrameworkFrames) {
    if (path.find(prefix) != std::string::npos) return true;
  }
  return false;
}

struct LocInfo {
  std::string module_path;
  std::string aten_op;
  std::string src_file;
  std::string src_line;
};

std::map<std::string, LocInfo> ParseLocations(const std::string& text) {
  std::map<std::string, LocInfo> out;
  static const std::regex loc_def(R"(^#(loc\d+)\s*=\s*loc\((.*)\)\s*$)");
  static const std::regex loc_name(R"RX(^"([^"]*)")RX");
  static const std::regex callsite(R"RX(callsite\("([^"]*)"\("([^"]*)":(\d+):)RX");

  std::istringstream is(text);
  std::string line;
  while (std::getline(is, line)) {
    if (line.rfind("#loc", 0) != 0) continue;
    std::smatch m;
    if (!std::regex_match(line, m, loc_def)) continue;

    LocInfo info;
    const auto body = m[2].str();

    std::smatch nm;
    std::string name;
    if (std::regex_search(body, nm, loc_name)) name = nm[1].str();

    const auto slash = name.rfind('/');
    if (slash != std::string::npos) {
      info.module_path = name.substr(0, slash);
      info.aten_op = name.substr(slash + 1);
    } else {
      info.module_path = name;
    }

    for (auto it = std::sregex_iterator(body.begin(), body.end(), callsite);
         it != std::sregex_iterator(); ++it) {
      const auto path = (*it)[2].str();
      if (IsFrameworkFrame(path)) continue;
      info.src_file = path;
      info.src_line = (*it)[3].str();
      break;
    }

    out[m[1].str()] = std::move(info);
  }
  return out;
}

}  // namespace

dfabit::core::Status CirhParser::ParseFile(
    const std::string& path,
    std::vector<CirhOp>* ops) const {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return {dfabit::core::StatusCode::kNotFound, "failed to open cirh.mlir: " + path};
  }
  std::stringstream buf;
  buf << ifs.rdbuf();
  return ParseText(buf.str(), ops);
}

dfabit::core::Status CirhParser::ParseText(
    const std::string& text,
    std::vector<CirhOp>* ops) const {
  if (!ops) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ops is null"};
  }
  ops->clear();

  const auto locs = ParseLocations(text);
  const dfabit::metadata::StableIdAssigner assigner;

  // %681 = cirh.MatMul %679, %649 {attrs} : (t, t) -> t loc(#loc95)
  // The operand list must match greedily up to the last ")" before "->".
  // A non-greedy group stops at the first ")" inside a nested tensor type or
  // attribute dictionary, which silently drops every variadic elementwise
  // operator while still matching the fixed-arity ones -- an under-count that
  // leaves arithmetic totals correct and operator counts wrong.
  // Matching the whole line with one expression is fragile: parentheses appear
  // in the operand list, in nested attribute dictionaries, and in the trailing
  // loc(...) marker, so any pattern anchored on them mis-binds for some
  // operator arities. "->" occurs exactly once, so the line is split there and
  // each half matched separately.
  static const std::regex head_re(
      R"(^\s*(%[\w#]+)\s*=\s*cirh\.(\w+)(.*)$)");
  static const std::regex loc_re(R"(loc\(#(loc\d+)\)\s*$)");

  std::istringstream is(text);
  std::string line;
  while (std::getline(is, line)) {
    if (line.find("cirh.") == std::string::npos) continue;
    std::smatch m;
    if (!std::regex_match(line, m, head_re)) continue;

    CirhOp op;
    op.ssa = m[1].str();
    op.op_name = m[2].str();
    const auto rest = m[3].str();

    std::string loc_id;
    std::smatch lm;
    if (std::regex_search(rest, lm, loc_re)) loc_id = lm[1].str();

    // Split on the single "->". Everything before it holds the operands and
    // any attribute block; everything after is the result type.
    const auto arrow = rest.rfind("->");
    std::string lhs = arrow == std::string::npos ? rest : rest.substr(0, arrow);
    std::string rhs = arrow == std::string::npos ? std::string() : rest.substr(arrow + 2);

    const auto attrs = lhs;

    // The operand TYPES are the parenthesised list immediately before "->",
    // not the operand values that precede the attribute dictionary. An
    // attribute block may itself contain tensor-typed values -- Convolution
    // carries stride, dilation and padding as "dense<1> : tensor<2xi64>" --
    // so scanning the whole left-hand side picks those up and shifts every
    // operand index. Taking the last parenthesised group avoids both that and
    // the opposite error of truncating the types away entirely.
    std::string operand_text;
    {
      const auto close = lhs.rfind(')');
      if (close != std::string::npos) {
        int depth = 0;
        std::size_t open = std::string::npos;
        for (std::size_t i = close + 1; i-- > 0;) {
          if (lhs[i] == ')') ++depth;
          else if (lhs[i] == '(') {
            if (--depth == 0) { open = i; break; }
          }
          if (i == 0) break;
        }
        if (open != std::string::npos) {
          operand_text = lhs.substr(open + 1, close - open - 1);
        }
      }
    }
    // Operand SSA names come from the value list, which precedes the attribute
    // dictionary. Attribute values are never SSA references, so scanning the
    // text before the first "{" is exact.
    {
      static const std::regex ssa_re(R"(%[\w#]+)");
      std::string value_text = lhs;
      const auto brace = value_text.find('{');
      if (brace != std::string::npos) value_text = value_text.substr(0, brace);
      const auto colon = value_text.find(':');
      if (colon != std::string::npos) value_text = value_text.substr(0, colon);
      for (auto it = std::sregex_iterator(value_text.begin(), value_text.end(), ssa_re);
           it != std::sregex_iterator(); ++it) {
        op.operand_ssa.push_back(it->str());
      }
    }

    const auto ins = ExtractTensors(operand_text);
    auto outs = ExtractTensors(rhs);
    if (outs.empty()) {
      // Operators written without an arrow, such as cirh.Const, state their
      // result type directly.
      outs = ExtractTensors(lhs);
      if (!outs.empty()) outs = {outs.back()};
    }

    const auto it = locs.find(loc_id);
    if (it != locs.end()) {
      op.module_path = it->second.module_path;
      op.aten_op = it->second.aten_op;
      op.src_file = it->second.src_file;
      op.src_line = it->second.src_line;
    }

    op.phase = PhaseOf(op.module_path, op.aten_op, attrs);

    if (!outs.empty()) {
      std::string shape;
      for (std::size_t i = 0; i < outs[0].dims.size(); ++i) {
        if (i) shape += "x";
        shape += std::to_string(outs[0].dims[i]);
      }
      op.shape = shape;
      op.dtype = outs[0].dtype;
    }

    bool determined = false;
    op.macs = ComputeMacs(op.op_name, ins, outs, attrs, &determined);
    op.macs_determined = determined;

    double bytes = 0.0;
    for (const auto& t : ins) bytes += Numel(t.dims) * static_cast<double>(DtypeBytes(t.dtype));
    for (const auto& t : outs) bytes += Numel(t.dims) * static_cast<double>(DtypeBytes(t.dtype));
    op.total_bytes = bytes;

    if (!outs.empty()) {
      op.result_bytes =
          Numel(outs[0].dims) * static_cast<double>(DtypeBytes(outs[0].dtype));
    }

    const auto symbol = (op.module_path.empty() ? std::string("anon") : op.module_path) +
                        "#" + op.ssa.substr(1);
    op.stable_id = assigner.Assign(
        "hidden_ir", symbol + "|" + op.op_name + "|compile");

    ops->push_back(std::move(op));
  }

  return dfabit::core::Status::Ok();
}

dfabit::metadata::ModelDesc CirhParser::ToModel(
    const std::vector<CirhOp>& ops,
    const std::string& graph_name) {
  dfabit::metadata::ModelDesc model;
  model.backend_name = "cerebras";
  model.graph_name = graph_name;
  model.model_name = graph_name;

  for (const auto& op : ops) {
    dfabit::metadata::OpDesc desc;
    desc.op_name = op.op_name;
    desc.dialect = "cirh";
    desc.stage_tag = op.phase;
    desc.stable_id = op.stable_id;
    // one multiply plus one accumulate
    desc.estimated_flops = static_cast<std::int64_t>(op.macs * 2.0);
    desc.estimated_bytes = static_cast<std::int64_t>(op.total_bytes);
    desc.attributes["module_path"] = op.module_path;
    desc.attributes["aten_op"] = op.aten_op;
    desc.attributes["shape"] = op.shape;
    desc.attributes["dtype"] = op.dtype;
    desc.attributes["phase"] = op.phase;
    desc.attributes["macs"] = std::to_string(static_cast<long long>(op.macs));
    desc.attributes["macs_determined"] = op.macs_determined ? "1" : "0";
    desc.attributes["ssa"] = op.ssa;
    desc.attributes["result_bytes"] = std::to_string(
        static_cast<long long>(op.result_bytes));

    // Operands and result are published as TensorDesc entries keyed by SSA
    // name. A tool can then reconstruct the def-use chain from OpDesc alone,
    // without knowing which backend produced it.
    for (const auto& operand : op.operand_ssa) {
      dfabit::metadata::TensorDesc t;
      t.name = operand;
      desc.inputs.push_back(std::move(t));
    }
    {
      dfabit::metadata::TensorDesc t;
      t.name = op.ssa;
      t.dtype = op.dtype;
      desc.outputs.push_back(std::move(t));
    }
    if (!op.src_file.empty()) {
      desc.attributes["src"] = op.src_file + ":" + op.src_line;
    }
    model.ops.push_back(std::move(desc));
  }

  return model;
}

std::vector<MetricSample> CirhParser::ToMetricSamples(
    const std::vector<CirhOp>& ops) {
  std::vector<MetricSample> out;
  if (ops.empty()) return out;

  std::map<std::string, double> macs_by_phase;
  std::map<std::string, double> bytes_by_phase;
  std::map<std::string, double> count_by_phase;
  double undetermined = 0.0;
  double with_lineage = 0.0;

  for (const auto& op : ops) {
    macs_by_phase[op.phase] += op.macs;
    bytes_by_phase[op.phase] += op.total_bytes;
    count_by_phase[op.phase] += 1.0;
    if (!op.macs_determined) undetermined += 1.0;
    if (!op.module_path.empty()) with_lineage += 1.0;
  }

  const auto push = [&out](const std::string& name, double v,
                           const std::string& unit, const std::string& phase) {
    MetricSample m;
    m.name = name;
    m.value = v;
    m.unit = unit;
    m.stage = "compile";
    m.attributes["source"] = "cirh_mlir";
    if (!phase.empty()) m.attributes["phase"] = phase;
    out.push_back(std::move(m));
  };

  push("cirh_operators", static_cast<double>(ops.size()), "count", "");
  push("cirh_operators_undetermined", undetermined, "count", "");
  push("cirh_operators_with_lineage", with_lineage, "count", "");

  for (const auto& kv : count_by_phase) {
    push("cirh_phase_operators", kv.second, "count", kv.first);
    push("cirh_phase_macs", macs_by_phase[kv.first], "mac", kv.first);
    push("cirh_phase_bytes", bytes_by_phase[kv.first], "B", kv.first);
  }

  double total_macs = 0.0;
  double total_bytes = 0.0;
  for (const auto& kv : macs_by_phase) total_macs += kv.second;
  for (const auto& kv : bytes_by_phase) total_bytes += kv.second;
  push("cirh_total_macs", total_macs, "mac", "");
  push("cirh_total_bytes", total_bytes, "B", "");
  if (total_bytes > 0.0) {
    // Static proxy: byte counts are a footprint, not measured traffic, so this
    // intensity is a lower bound.
    push("cirh_arithmetic_intensity", total_macs / total_bytes, "mac_per_byte", "");
  }

  return out;
}

std::vector<std::string> CirhParser::FindAllCirh(const std::string& model_dir) {
  std::vector<std::string> out;
  if (model_dir.empty()) return out;

  std::error_code ec;
  if (!std::filesystem::is_directory(model_dir, ec)) return out;

  for (auto it = std::filesystem::recursive_directory_iterator(
           model_dir, std::filesystem::directory_options::skip_permission_denied, ec);
       it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
    if (ec) break;
    if (!it->is_regular_file(ec)) continue;
    if (it->path().filename() != "cirh.mlir") continue;
    out.push_back(it->path().string());
  }

  std::sort(out.begin(), out.end());
  return out;
}

std::string CirhParser::FindNewestCirh(const std::string& model_dir) {
  if (model_dir.empty()) return std::string();
  std::error_code ec;
  if (!std::filesystem::is_directory(model_dir, ec)) return std::string();

  std::string best;
  std::filesystem::file_time_type best_time{};
  bool have = false;

  for (auto it = std::filesystem::recursive_directory_iterator(
           model_dir, std::filesystem::directory_options::skip_permission_denied, ec);
       it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
    if (ec) break;
    if (!it->is_regular_file(ec)) continue;
    if (it->path().filename() != "cirh.mlir") continue;
    const auto mtime = it->last_write_time(ec);
    if (ec) continue;
    if (!have || mtime > best_time) {
      best = it->path().string();
      best_time = mtime;
      have = true;
    }
  }
  return best;
}

}  // namespace dfabit::adapters::cerebras
