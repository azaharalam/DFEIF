#include "dfabit/runtime/runtime_correlation.h"

#include <utility>

namespace dfabit::runtime {

void RuntimeCorrelator::Reset() {
  symbol_to_id_.clear();
  id_to_name_.clear();
}

void RuntimeCorrelator::IndexModel(const dfabit::metadata::ModelDesc& model) {
  Reset();
  for (const auto& op : model.ops) {
    symbol_to_id_[op.op_name] = op.stable_id;
    id_to_name_[op.stable_id] = op.op_name;

    const auto attr_it = op.attributes.find("symbol");
    if (attr_it != op.attributes.end() && !attr_it->second.empty()) {
      symbol_to_id_[attr_it->second] = op.stable_id;
    }
  }
}

dfabit::core::Status RuntimeCorrelator::Correlate(
    const std::vector<RuntimeRecord>& runtime_records,
    std::vector<CorrelatedRuntimeRecord>* correlated) const {
  if (!correlated) {
    return {dfabit::core::StatusCode::kInvalidArgument, "correlated is null"};
  }

  correlated->clear();
  correlated->reserve(runtime_records.size());

  for (const auto& rec : runtime_records) {
    CorrelatedRuntimeRecord out;
    const auto it = symbol_to_id_.find(rec.symbol);
    if (it != symbol_to_id_.end()) {
      out.stable_id = it->second;
      const auto name_it = id_to_name_.find(out.stable_id);
      if (name_it != id_to_name_.end()) {
        out.op_name = name_it->second;
      }
    }
    out.stage = rec.stage;
    out.metric_name = rec.metric_name;
    out.metric_value = rec.metric_value;
    out.unit = rec.unit;
    correlated->push_back(std::move(out));
  }

  return dfabit::core::Status::Ok();
}

std::vector<dfabit::adapters::MetricSample> RuntimeCorrelator::ToMetricSamples(
    const std::vector<CorrelatedRuntimeRecord>& correlated) const {
  std::vector<dfabit::adapters::MetricSample> out;
  out.reserve(correlated.size());

  for (const auto& rec : correlated) {
    dfabit::adapters::MetricSample sample;
    sample.name = rec.metric_name;
    sample.value = rec.metric_value;
    sample.unit = rec.unit;
    sample.stage = rec.stage;
    sample.stable_id = rec.stable_id;
    if (!rec.op_name.empty()) {
      sample.attributes["op_name"] = rec.op_name;
    }
    out.push_back(std::move(sample));
  }

  return out;
}

}  // namespace dfabit::runtime