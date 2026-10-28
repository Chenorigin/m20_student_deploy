#pragma once
#include "json.hpp"
#include <stdexcept>
#include <string>

namespace m20_cts {
// Runtime choice, separate from the immutable training/model contract.
inline std::string ExecutionMode(const nlohmann::json& configuration) {
  const std::string mode=configuration.at("aos").value("cts_execution_mode",std::string("landing"));
  if(mode!="landing"&&mode!="nominal_pd")
    throw std::runtime_error("cts_execution_mode must be landing or nominal_pd");
  return mode;
}
}
