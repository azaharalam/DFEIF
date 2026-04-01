#include "dfabit/analysis/report.h"
#include <fstream>
#include <iomanip>

namespace dfabit::analysis {

static void WriteCsvEscaped(std::ostream& os, const std::string& s) {
  // Minimal CSV escaping: wrap if it contains commas/quotes/newlines
  bool needs = false;
  for (char c : s) {
    if (c == ',' || c == '"' || c == '\n' || c == '\r') { needs = true; break; }
  }
  if (!needs) { os << s; return; }
  os << '"';
  for (char c : s) {
    if (c == '"') os << "\"\"";
    else os << c;
  }
  os << '"';
}

dfabit::core::Status WriteProgramAnalysisCsv(
    const std::string& out_path,
    const std::vector<OpRecord>& ops) {

  std::ofstream ofs(out_path, std::ios::out);
  if (!ofs.is_open()) {
    return {dfabit::core::StatusCode::kInternal, "failed to open output csv: " + out_path};
  }

  // EXACT column order to match Program_Analysis_ds.csv
  ofs << "kernel_id,layer,mac_ops,total_bytes,measured_latency_ms,"
         "bytes_per_mac,throughput_MACs_per_s,efficiency_vs_peak\n";

  ofs << std::fixed << std::setprecision(6);

  for (const auto& op : ops) {
    const double bytes_per_mac = (op.mac_ops > 0) ? (static_cast<double>(op.total_bytes) / static_cast<double>(op.mac_ops)) : 0.0;
    const double throughput = (op.mac_ops > 0 && op.measured_latency_ms > 0.0)
        ? (static_cast<double>(op.mac_ops) / (op.measured_latency_ms / 1000.0))
        : 0.0;
    const double efficiency = 0.0;  // v1: keep 0 unless you later provide peak

    WriteCsvEscaped(ofs, op.kernel_id); ofs << ",";
    WriteCsvEscaped(ofs, op.layer); ofs << ",";
    ofs << op.mac_ops << ",";
    ofs << op.total_bytes << ",";
    ofs << op.measured_latency_ms << ",";
    ofs << bytes_per_mac << ",";
    ofs << throughput << ",";
    ofs << efficiency << "\n";
  }

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::analysis
