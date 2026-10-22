#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace m20::v11 {

struct Point {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
};

using Feature = std::array<float, 6>;
enum class FeatureMode { PositiveRisers, SignedEdges, SignedEdgesRearAxle };

inline constexpr const char* kPositiveFeatureContract = "v11_positive_risers_v1";
inline constexpr const char* kSignedFeatureContract = "v11_signed_edges_v2";
inline constexpr const char* kSignedRearAxleFeatureContract = "v11_signed_edges_rear_axle_v3";

inline bool IsSignedMode(FeatureMode mode) {
    return mode == FeatureMode::SignedEdges || mode == FeatureMode::SignedEdgesRearAxle;
}

inline const char* FeatureContract(FeatureMode mode) {
    if (mode == FeatureMode::SignedEdgesRearAxle) {
        return kSignedRearAxleFeatureContract;
    }
    return mode == FeatureMode::SignedEdges ? kSignedFeatureContract : kPositiveFeatureContract;
}

inline const char* FeatureLayoutLabel(FeatureMode mode) {
    return IsSignedMode(mode)
        ? "B1,signedH1,D1,B2,signedH2,D2"
        : "B1,H1,D1,B2,H2,D2";
}

inline FeatureMode FeatureModeFromEnvironment() {
    const char* value = std::getenv("M20_V11_FEATURE_CONTRACT");
    if (value == nullptr || std::string(value).empty() ||
        std::string(value) == kPositiveFeatureContract) {
        return FeatureMode::PositiveRisers;
    }
    if (std::string(value) == kSignedFeatureContract) {
        return FeatureMode::SignedEdges;
    }
    if (std::string(value) == kSignedRearAxleFeatureContract) {
        return FeatureMode::SignedEdgesRearAxle;
    }
    throw std::runtime_error(
        "M20_V11_FEATURE_CONTRACT must be v11_positive_risers_v1, "
        "v11_signed_edges_v2, or v11_signed_edges_rear_axle_v3");
}

inline bool IsValidFeature(const Feature& feature, FeatureMode mode = FeatureMode::PositiveRisers) {
    for (int slot = 0; slot < 2; ++slot) {
        const std::size_t offset = static_cast<std::size_t>(slot * 3);
        const float bit = feature[offset];
        const float signed_height = feature[offset + 1];
        const float distance = feature[offset + 2];
        if (!std::isfinite(bit) || !std::isfinite(signed_height) ||
            !std::isfinite(distance) || bit < 0.0F || bit > 1.0F ||
            std::abs(signed_height) > 0.232F || distance < 0.0F) {
            return false;
        }
        if (!IsSignedMode(mode) && signed_height < 0.0F) {
            return false;
        }
    }
    return true;
}

struct ExtractionResult {
    Feature feature{};
    std::size_t accepted_points = 0;
    std::size_t populated_cells = 0;
    bool geometry_ready = false;
};

class RiserExtractor {
public:
    static constexpr int kRows = 21;
    static constexpr int kColumns = 15;
    static constexpr float kXMinimum = -0.75F;
    static constexpr float kXMaximum = 1.25F;
    static constexpr float kYMinimum = -0.70F;
    static constexpr float kYMaximum = 0.70F;
    static constexpr float kGridSpacing = 0.10F;
    static constexpr float kMinimumStepHeight = 0.048F;
    static constexpr float kMaximumStepHeight = 0.232F;
    static constexpr float kCorridorHalfWidth = 0.20F;
    static constexpr float kPassedDistance = -0.30F;
    static constexpr float kMinimumRiserSeparation = 0.15F;
    static constexpr float kAnticipationMinimum = 0.02F;
    static constexpr float kAnticipationMaximum = 1.20F;
    static constexpr float kRearAxleDistanceOffset = 0.3141F;

    using HeightGrid = std::array<float, kRows * kColumns>;

    explicit RiserExtractor(FeatureMode mode = FeatureMode::PositiveRisers) : mode_(mode) {}

    ExtractionResult Extract(const std::vector<Point>& points) const {
        using Cell = std::vector<float>;
        std::array<Cell, kRows * kColumns> cells;
        ExtractionResult result;

        for (const Point& point : points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
                !std::isfinite(point.z) || point.x < kXMinimum - 0.05F ||
                point.x > kXMaximum + 0.05F || point.y < kYMinimum - 0.05F ||
                point.y > kYMaximum + 0.05F) {
                continue;
            }
            const int row = static_cast<int>(std::lround(
                (point.x - kXMinimum) / kGridSpacing));
            const int column = static_cast<int>(std::lround(
                (point.y - kYMinimum) / kGridSpacing));
            if (row < 0 || row >= kRows || column < 0 || column >= kColumns) {
                continue;
            }
            cells[static_cast<std::size_t>(row * kColumns + column)].push_back(point.z);
            ++result.accepted_points;
        }

        HeightGrid heights;
        heights.fill(std::numeric_limits<float>::quiet_NaN());
        for (std::size_t index = 0; index < cells.size(); ++index) {
            auto& samples = cells[index];
            if (samples.size() < kMinimumSamplesPerCell) {
                continue;
            }
            // A downward training ray observes the tread surface.  The median
            // rejects sparse vertical-riser pixels and RealSense flying points.
            const std::size_t middle = samples.size() / 2;
            std::nth_element(samples.begin(), samples.begin() + middle, samples.end());
            heights[index] = samples[middle];
            ++result.populated_cells;
        }

        result.geometry_ready = result.accepted_points >= kMinimumAcceptedPoints &&
            result.populated_cells >= kMinimumPopulatedCells;
        if (!result.geometry_ready) {
            return result;
        }

        result.feature = ExtractFeature(heights, mode_);
        return result;
    }

    ExtractionResult ExtractHeightGrid(
        const HeightGrid& source, bool x_descending, bool y_descending) const {
        HeightGrid heights;
        heights.fill(std::numeric_limits<float>::quiet_NaN());
        ExtractionResult result;
        for (int source_row = 0; source_row < kRows; ++source_row) {
            const int row = x_descending ? kRows - 1 - source_row : source_row;
            for (int source_column = 0; source_column < kColumns; ++source_column) {
                const int column = y_descending
                    ? kColumns - 1 - source_column : source_column;
                const float height = source[static_cast<std::size_t>(
                    source_row * kColumns + source_column)];
                if (!std::isfinite(height)) {
                    continue;
                }
                heights[static_cast<std::size_t>(row * kColumns + column)] = height;
                ++result.accepted_points;
                ++result.populated_cells;
            }
        }
        result.geometry_ready = result.populated_cells >= kMinimumPopulatedCells;
        if (!result.geometry_ready) {
            return result;
        }
        result.feature = ExtractFeature(heights, mode_);
        return result;
    }

    static Feature ExtractFeature(
        const HeightGrid& heights, FeatureMode mode = FeatureMode::PositiveRisers) {
        std::vector<std::pair<float, float>> risers;
        for (int row = 0; row + 1 < kRows; ++row) {
            float best_height = 0.0F;
            float best_magnitude = -std::numeric_limits<float>::infinity();
            bool found = false;
            for (int column = 0; column < kColumns; ++column) {
                const float y = kYMinimum + static_cast<float>(column) * kGridSpacing;
                if (std::abs(y) > kCorridorHalfWidth + 1.0e-5F) {
                    continue;
                }
                const float low = heights[static_cast<std::size_t>(row * kColumns + column)];
                const float upper = heights[static_cast<std::size_t>((row + 1) * kColumns + column)];
                if (!std::isfinite(low) || !std::isfinite(upper)) {
                    continue;
                }
                const float height = upper - low;
                const float magnitude = std::abs(height);
                const bool sign_allowed = IsSignedMode(mode) || height > 0.0F;
                if (sign_allowed && magnitude >= kMinimumStepHeight && magnitude <= kMaximumStepHeight &&
                    magnitude > best_magnitude) {
                    best_magnitude = magnitude;
                    best_height = height;
                    found = true;
                }
            }
            const float x = kXMinimum + (static_cast<float>(row) + 0.5F) * kGridSpacing;
            if (found && x > kPassedDistance) {
                risers.emplace_back(x, best_height);
            }
        }
        return SelectTwoRisers(risers, mode);
    }

    static float ActorDistance(float base_distance, float height, FeatureMode mode) {
        if (mode == FeatureMode::SignedEdgesRearAxle && height < 0.0F) {
            return base_distance + kRearAxleDistanceOffset;
        }
        return base_distance;
    }

    static Feature SelectTwoRisers(
        const std::vector<std::pair<float, float>>& risers,
        FeatureMode mode = FeatureMode::PositiveRisers) {
        Feature feature{};
        if (risers.empty()) {
            return feature;
        }
        const auto first = std::min_element(
            risers.begin(), risers.end(), [mode](const auto& left, const auto& right) {
                return std::abs(ActorDistance(left.first, left.second, mode)) <
                    std::abs(ActorDistance(right.first, right.second, mode));
            });
        const float first_x = first->first;
        const float first_height = first->second;
        std::vector<std::pair<float, float>> later;
        for (const auto& riser : risers) {
            const bool same_sign = (riser.second > 0.0F) == (first_height > 0.0F);
            if (same_sign && riser.first > first_x + kMinimumRiserSeparation) {
                later.push_back(riser);
            }
        }
        std::sort(later.begin(), later.end());

        const float first_distance = ActorDistance(first_x, first_height, mode);
        if (first_distance >= kAnticipationMinimum && first_distance <= kAnticipationMaximum) {
            feature = {1.0F, first_height, first_distance, 0.0F, 0.0F, 0.0F};
            if (!later.empty()) {
                const float later_distance = ActorDistance(
                    later.front().first, later.front().second, mode);
                if (mode != FeatureMode::SignedEdgesRearAxle ||
                    (later_distance >= kAnticipationMinimum &&
                     later_distance <= kAnticipationMaximum)) {
                    feature[3] = 1.0F;
                    feature[4] = later.front().second;
                    feature[5] = later_distance;
                }
            }
            return feature;
        }
        if (!later.empty()) {
            const float later_distance = ActorDistance(later.front().first, later.front().second, mode);
            if (later_distance >= kAnticipationMinimum &&
                later_distance <= kAnticipationMaximum) {
                feature = {1.0F, later.front().second, later_distance,
                           0.0F, 0.0F, 0.0F};
            }
        }
        return feature;
    }

private:
    static constexpr std::size_t kMinimumSamplesPerCell = 3;
    static constexpr std::size_t kMinimumAcceptedPoints = 120;
    static constexpr std::size_t kMinimumPopulatedCells = 18;
    FeatureMode mode_;
};

class FeatureTracker {
public:
    FeatureTracker() : delayed_(kDelaySteps + 1, Feature{}) {}

    Feature Step(const Feature& detected, bool new_sensor_frame) {
        if (new_sensor_frame) {
            for (int slot = 0; slot < 2; ++slot) {
                const int offset = slot * 3;
                const bool detected_bit = detected[static_cast<std::size_t>(offset)] > 0.5F;
                const bool prior_bit = tracked_[static_cast<std::size_t>(offset)] > 0.5F;
                if (detected_bit) {
                    ++confirm_count_[static_cast<std::size_t>(slot)];
                    invalid_count_[static_cast<std::size_t>(slot)] = 0;
                    tracked_[static_cast<std::size_t>(offset + 1)] =
                        detected[static_cast<std::size_t>(offset + 1)];
                    tracked_[static_cast<std::size_t>(offset + 2)] =
                        detected[static_cast<std::size_t>(offset + 2)];
                } else {
                    confirm_count_[static_cast<std::size_t>(slot)] = 0;
                    ++invalid_count_[static_cast<std::size_t>(slot)];
                }
                const bool actor_bit =
                    (detected_bit && confirm_count_[static_cast<std::size_t>(slot)] >=
                        kConfirmationFrames) ||
                    (prior_bit && invalid_count_[static_cast<std::size_t>(slot)] <=
                        kHysteresisFrames);
                tracked_[static_cast<std::size_t>(offset)] = actor_bit ? 1.0F : 0.0F;
                if (!actor_bit) {
                    tracked_[static_cast<std::size_t>(offset + 1)] = 0.0F;
                    tracked_[static_cast<std::size_t>(offset + 2)] = 0.0F;
                }
            }
        }
        delayed_.pop_front();
        delayed_.push_back(tracked_);
        Feature output = delayed_.front();
        output[1] *= output[0];
        output[2] *= output[0];
        output[4] *= output[3];
        output[5] *= output[3];
        return output;
    }

    void Reset() {
        tracked_.fill(0.0F);
        confirm_count_.fill(0);
        invalid_count_.fill(0);
        delayed_.assign(kDelaySteps + 1, Feature{});
    }

private:
    static constexpr int kConfirmationFrames = 2;
    static constexpr int kHysteresisFrames = 2;
    static constexpr int kDelaySteps = 2;
    Feature tracked_{};
    std::array<int, 2> confirm_count_{};
    std::array<int, 2> invalid_count_{};
    std::deque<Feature> delayed_;
};

}  // namespace m20::v11
