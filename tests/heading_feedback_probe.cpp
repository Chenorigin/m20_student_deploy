#include "student_heading_feedback.hpp"
#include <iostream>
#include <limits>

using nlohmann::json;

int main() {
  try {
    json requests;
    std::cin >> requests;
    json responses = json::array();
    for (auto request : requests) {
      try {
        auto metadata = request.at("metadata");
        if (request.contains("metadata_injection")) {
          const auto& injection = request.at("metadata_injection");
          const std::string key = injection.at("key");
          const std::string value = injection.at("value");
          metadata["tuning"][key] = value == "nan"
              ? std::numeric_limits<double>::quiet_NaN()
              : std::numeric_limits<double>::infinity();
        }
        StudentHeadingFeedback feedback(metadata);
        json results = json::array();
        for (const auto& operation : request.at("operations")) {
          if (operation.value("reset", false)) {
            feedback.Reset();
            results.push_back({{"mode", feedback.Mode()}, {"target", feedback.Target()}});
            continue;
          }
          try {
            auto raw = operation.at("raw").get<std::array<float, 3>>();
            float yaw = operation.at("yaw");
            float rate = operation.at("rate");
            auto obs = operation.at("obs").get<std::vector<float>>();
            const auto bad = operation.value("invalid", std::string{});
            if (bad == "raw_nan") raw[0] = std::numeric_limits<float>::quiet_NaN();
            if (bad == "yaw_inf") yaw = std::numeric_limits<float>::infinity();
            if (bad == "rate_nan") rate = std::numeric_limits<float>::quiet_NaN();
            if (bad == "obs_nan") obs[57] = std::numeric_limits<float>::quiet_NaN();
            if (bad == "obs_inf") obs[244] = std::numeric_limits<float>::infinity();
            if (bad == "obs_short") obs.pop_back();
            const auto result = feedback.Apply(raw, yaw, rate, obs,
                                                operation.at("step").get<std::uint64_t>());
            results.push_back({{"command", result}, {"mode", feedback.Mode()},
                               {"target", feedback.Target()}, {"stairs", feedback.Stairs()}});
          } catch (const std::exception& e) {
            results.push_back({{"error", e.what()}, {"mode", feedback.Mode()},
                               {"target", feedback.Target()}});
          }
        }
        responses.push_back({{"results", results}});
      } catch (const std::exception& e) {
        responses.push_back({{"error", e.what()}});
      }
    }
    std::cout << responses.dump() << '\n';
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
