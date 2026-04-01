#pragma once

namespace dfabit::trace {

// v1 event names
inline constexpr const char* kSessionStart = "SESSION_START";
inline constexpr const char* kSessionEnd   = "SESSION_END";
inline constexpr const char* kModelLoad    = "MODEL_LOAD";
inline constexpr const char* kInvokeBegin  = "INVOKE_BEGIN";
inline constexpr const char* kInvokeEnd    = "INVOKE_END";
inline constexpr const char* kGraphManifest= "GRAPH_MANIFEST";

}  // namespace dfabit::trace
