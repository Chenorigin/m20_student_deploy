#pragma once

#include "json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Single-environment CPU equivalent of m20_student_v1.v3_5_feedback.
// The argument is provenance.command_feedback, not the whole provenance.
// Feed the unmodified delivered 789D observation before inserting its command.
class StudentHeadingFeedback {
 public:
  explicit StudentHeadingFeedback(const nlohmann::json& command_feedback) {
    if (!command_feedback.is_object() ||
        command_feedback.value("version", std::string{}) != "legacy_v1_v3_5" ||
        !command_feedback.contains("tuning") ||
        !command_feedback.at("tuning").is_object())
      throw std::runtime_error("unsupported student command_feedback metadata");
    const auto& t = command_feedback.at("tuning");
    translation_threshold_ = Read(t, "translation_threshold");
    yaw_threshold_ = Read(t, "yaw_threshold");
    flat_ = {Read(t, "heading_hold_kp"), Read(t, "heading_hold_kd"),
             Read(t, "heading_hold_max_yaw_rate"), Read(t, "heading_hold_deadband")};
    stairs_ = {Read(t, "stair_heading_hold_kp"), Read(t, "stair_heading_hold_kd"),
               Read(t, "stair_heading_hold_max_yaw_rate"),
               Read(t, "stair_heading_hold_deadband")};
    rate_enter_ = Read(t, "heading_hold_rate_enter");
  }

  void Reset() {
    mode_ = 0;
    target_ = 0.f;
    // Python reset clears the stored pre-step state as well, but preserves
    // the watermark so a retry of the same step still starts from zero.
    before_mode_ = 0;
    before_target_ = 0.f;
  }

  std::array<float, 3> Apply(const std::array<float, 3>& raw, float yaw,
                           float body_yaw_rate, const std::vector<float>& obs,
                           std::uint64_t step) {
    if (obs.size() != kObservationDim || !std::isfinite(yaw) ||
        !std::isfinite(body_yaw_rate) ||
        !std::all_of(raw.begin(), raw.end(), Finite) ||
        !std::all_of(obs.begin(), obs.end(), Finite))
      throw std::runtime_error("invalid student heading feedback input");
    const bool stairs = ScanStairs(obs);

    int previous_mode = mode_;
    float target = target_;
    int next_before_mode = before_mode_;
    float next_before_target = before_target_;
    if (has_step_ && step < step_) {
      previous_mode = 0;
      target = 0.f;
      next_before_mode = 0;
      next_before_target = 0.f;
    } else if (has_step_ && step == step_ && has_before_) {
      previous_mode = before_mode_;
      target = before_target_;
    }
    if (!has_step_ || step != step_) {
      next_before_mode = previous_mode;
      next_before_target = target;
    }

    const bool straight = std::abs(raw[0]) > translation_threshold_ &&
                          std::abs(raw[1]) <= translation_threshold_ &&
                          std::abs(raw[2]) <= yaw_threshold_;
    const bool zero = std::sqrt(raw[0] * raw[0] + raw[1] * raw[1]) <=
                          translation_threshold_ &&
                      std::abs(raw[2]) <= yaw_threshold_;
    const int mode = straight ? 1 : (zero ? 2 : 0);
    if (mode != previous_mode && mode != 0) target = yaw;
    const float delta = target - yaw;
    const float error = std::atan2(std::sin(delta), std::cos(delta));
    const auto& gains = stairs ? stairs_ : flat_;
    const bool active = std::abs(error) > gains[3] ||
                        std::abs(body_yaw_rate) > rate_enter_;
    const float correction =
        std::clamp(gains[0] * error - gains[1] * body_yaw_rate,
                   -gains[2], gains[2]);
    auto result = raw;
    if (mode != 0) result[2] = active ? correction : 0.f;
    if (!std::isfinite(result[2]))
      throw std::runtime_error("nonfinite student heading feedback result");

    mode_ = mode;
    target_ = target;
    before_mode_ = next_before_mode;
    before_target_ = next_before_target;
    has_before_ = true;
    step_ = step;
    has_step_ = true;
    last_stairs_ = stairs;
    return result;
  }

  int Mode() const { return mode_; }
  float Target() const { return target_; }
  bool Stairs() const { return last_stairs_; }

 private:
  static constexpr std::size_t kObservationDim = 789;
  static bool Finite(float v) { return std::isfinite(v); }

  static float Read(const nlohmann::json& t, const char* key) {
    if (!t.contains(key) || !t.at(key).is_number())
      throw std::runtime_error(std::string("missing/invalid heading tuning: ") + key);
    const double value = t.at(key).get<double>();
    const float narrowed = static_cast<float>(value);
    if (!std::isfinite(value) || value < 0. || !std::isfinite(narrowed))
      throw std::runtime_error(std::string("missing/invalid heading tuning: ") + key);
    return narrowed;
  }

  static bool ScanStairs(const std::vector<float>& obs) {
    std::array<float, 17> profile{};
    std::array<bool, 17> columns{};
    for (int col = 0; col < 17; ++col) {
      std::array<float, 11> values{};
      std::size_t count = 0;
      for (int row = 0; row < 11; ++row) {
        const std::size_t i = row * 17 + col;
        const float z = -obs[57 + i];
        const float y = (row - 5) * .1f;
        if (obs[244 + i] > .5f && obs[431 + i] > 0.f &&
            std::isfinite(z) && std::abs(z) < 1.e6f && std::abs(y) <= .35f)
          values[count++] = z;
      }
      columns[col] = count >= 2;
      if (count) {
        // torch.nanmedian selects the lower middle value for even counts.
        std::sort(values.begin(), values.begin() + count);
        profile[col] = values[(count - 1) / 2];
      }
    }
    for (int col = 0; col < 16; ++col) {
      // The fixed 11x17 training grid selects these exact edge indices.
      // Computing two grid products in one expression permits ARM FMA to
      // round the inclusive -0.45m edge outside its own boundary.
      if (col >= 3 && col <= 14 && columns[col] &&
          columns[col + 1] && std::abs(profile[col + 1] - profile[col]) > .06f)
        return true;
    }
    return false;
  }

  float translation_threshold_ = 0.f, yaw_threshold_ = 0.f, rate_enter_ = 0.f;
  std::array<float, 4> flat_{}, stairs_{};
  int mode_ = 0, before_mode_ = 0;
  float target_ = 0.f, before_target_ = 0.f;
  std::uint64_t step_ = 0;
  bool has_step_ = false, has_before_ = false, last_stairs_ = false;
};
