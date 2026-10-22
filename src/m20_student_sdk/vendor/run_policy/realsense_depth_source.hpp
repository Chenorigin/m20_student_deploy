#pragma once

#ifdef M20_HAS_REALSENSE2

#include <librealsense2/rs.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

// Process-local D435i source for the depth student.  This class deliberately
// has no ROS dependency: camera frames never leave the rl_deploy process.
class RealSenseDepthSource {
public:
    static constexpr int kWidth = 64;
    static constexpr int kHeight = 48;
    static constexpr int kFrameDim = kWidth * kHeight;
    static constexpr int kHistoryLength = 3;
    static constexpr int kTensorDim = kHistoryLength * kFrameDim;
    static constexpr int kObservationDim = kTensorDim + 2;
    // The student was trained with a 64x48 pinhole camera whose configured
    // aperture/focal length correspond to approximately 87 x 58 degrees.
    // D435i 848x480 is the matching depth profile (640x480 is only ~79 x 64).
    static constexpr int kCaptureWidth = 848;
    static constexpr int kCaptureHeight = 480;
    static constexpr int kCaptureFps = 30;
    static constexpr float kTrainingHorizontalFovDegrees = 87.0F;
    static constexpr float kTrainingVerticalFovDegrees = 58.0F;
    static constexpr float kFovToleranceDegrees = 4.0F;

    struct Snapshot {
        std::array<float, kObservationDim> observation{};
        bool ready = false;
        float sample_age_seconds = std::numeric_limits<float>::infinity();
        float sample_rate_hz = 0.0F;
        float max_sample_gap_seconds = std::numeric_limits<float>::infinity();
        uint64_t sample_count = 0;
        int source_width = 0;
        int source_height = 0;
        float horizontal_fov_degrees = 0.0F;
        float vertical_fov_degrees = 0.0F;
        float depth_scale = 0.0F;
        std::string error;
    };

    RealSenseDepthSource() {
        running_.store(true, std::memory_order_release);
        worker_ = std::thread(&RealSenseDepthSource::AcquisitionLoop, this);
    }

    ~RealSenseDepthSource() {
        running_.store(false, std::memory_order_release);
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    RealSenseDepthSource(const RealSenseDepthSource&) = delete;
    RealSenseDepthSource& operator=(const RealSenseDepthSource&) = delete;

    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot.sample_count = sample_count_;
        snapshot.sample_rate_hz = sample_rate_hz_;
        snapshot.max_sample_gap_seconds = max_sample_gap_seconds_;
        snapshot.source_width = source_width_;
        snapshot.source_height = source_height_;
        snapshot.horizontal_fov_degrees = horizontal_fov_degrees_;
        snapshot.vertical_fov_degrees = vertical_fov_degrees_;
        snapshot.depth_scale = depth_scale_;
        snapshot.error = last_error_;
        if (history_.size() != kHistoryLength) {
            return snapshot;
        }

        size_t offset = 0;
        for (const auto& frame : history_) {
            std::copy(frame.begin(), frame.end(), snapshot.observation.begin() + offset);
            offset += frame.size();
        }
        const float sample_age = Seconds(now - last_sample_time_);
        const float fresh_age = Seconds(now - last_fresh_time_);
        snapshot.sample_age_seconds = sample_age;
        snapshot.observation[kTensorDim] = sample_age <= kInputTimeoutSeconds ?
            last_valid_fraction_ : 0.0F;
        snapshot.observation[kTensorDim + 1] = std::clamp(
            fresh_age / kStatusMaxAgeSeconds, 0.0F, 1.0F);
        snapshot.ready = sample_age <= kInputTimeoutSeconds &&
            snapshot.observation[kTensorDim + 1] < kPerceptiveStopAge;
        return snapshot;
    }

private:
    using DepthFrame = std::array<float, kFrameDim>;
    static constexpr float kMinDistanceMetres = 0.001F;
    static constexpr float kMaxDistanceMetres = 4.0F;
    static constexpr float kMinimumValidAreaFraction = 0.50F;
    static constexpr float kFreshValidFraction = 0.50F;
    static constexpr float kInputTimeoutSeconds = 0.25F;
    static constexpr float kStatusMaxAgeSeconds = 0.50F;
    static constexpr float kPerceptiveStopAge = 0.50F;
    static constexpr int kMaskRows = 6;
    static constexpr int kMaskTopWidth = 16;
    static constexpr int kMaskBottomWidth = 32;

    template <class Rep, class Period>
    static float Seconds(const std::chrono::duration<Rep, Period>& duration) {
        return std::chrono::duration<float>(duration).count();
    }

    static bool IsMaskedPixel(int row, int column) {
        const int first_mask_row = kHeight - kMaskRows;
        if (row < first_mask_row) {
            return false;
        }
        const int row_offset = row - first_mask_row;
        const float alpha = static_cast<float>(row_offset) /
            static_cast<float>(std::max(kMaskRows - 1, 1));
        const int mask_width = static_cast<int>(std::lround(
            kMaskTopWidth + alpha * (kMaskBottomWidth - kMaskTopWidth)));
        const int left = (kWidth - mask_width) / 2;
        return column >= left && column < left + mask_width;
    }

    static bool MakeFrame(
        const rs2::depth_frame& input, float depth_scale,
        DepthFrame& output, float& valid_fraction) {
        const int source_width = input.get_width();
        const int source_height = input.get_height();
        const int stride_bytes = input.get_stride_in_bytes();
        const auto* source_bytes = static_cast<const uint8_t*>(input.get_data());
        if (source_width <= 0 || source_height <= 0 || source_bytes == nullptr ||
            stride_bytes < source_width * static_cast<int>(sizeof(uint16_t)) ||
            !std::isfinite(depth_scale) || depth_scale <= 0.0F) {
            return false;
        }

        int valid_output_pixels = 0;
        for (int output_row = 0; output_row < kHeight; ++output_row) {
            const int source_row_begin = output_row * source_height / kHeight;
            const int source_row_end = (output_row + 1) * source_height / kHeight;
            for (int output_column = 0; output_column < kWidth; ++output_column) {
                const int output_index = output_row * kWidth + output_column;
                output[output_index] = 1.0F;
                if (IsMaskedPixel(output_row, output_column)) {
                    continue;
                }

                const int source_column_begin = output_column * source_width / kWidth;
                const int source_column_end = (output_column + 1) * source_width / kWidth;
                double depth_sum = 0.0;
                int valid_count = 0;
                int total_count = 0;
                for (int source_row = source_row_begin; source_row < source_row_end; ++source_row) {
                    const auto* row = reinterpret_cast<const uint16_t*>(
                        source_bytes + static_cast<size_t>(source_row) * stride_bytes);
                    for (int source_column = source_column_begin;
                         source_column < source_column_end; ++source_column) {
                        ++total_count;
                        const float depth = static_cast<float>(row[source_column]) * depth_scale;
                        if (std::isfinite(depth) && depth > kMinDistanceMetres &&
                            depth <= kMaxDistanceMetres) {
                            depth_sum += depth;
                            ++valid_count;
                        }
                    }
                }
                if (total_count > 0 &&
                    static_cast<float>(valid_count) / static_cast<float>(total_count) >=
                        kMinimumValidAreaFraction) {
                    const float depth = static_cast<float>(depth_sum / valid_count);
                    output[output_index] = std::clamp(
                        depth / kMaxDistanceMetres * 2.0F - 1.0F, -1.0F, 1.0F);
                    ++valid_output_pixels;
                }
            }
        }
        valid_fraction = static_cast<float>(valid_output_pixels) /
            static_cast<float>(kFrameDim);
        return true;
    }

    void StoreFrame(
        const DepthFrame& frame, float valid_fraction,
        int source_width, int source_height, float horizontal_fov_degrees,
        float vertical_fov_degrees, float depth_scale,
        std::chrono::steady_clock::time_point now) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (history_.empty()) {
            if (valid_fraction < kFreshValidFraction) {
                last_error_ = "waiting for a depth frame with at least 50% valid coverage";
                return;
            }
            for (int index = 0; index < kHistoryLength; ++index) {
                history_.push_back(frame);
            }
            last_fresh_time_ = now;
        } else {
            history_.push_back(frame);
            while (history_.size() > kHistoryLength) {
                history_.pop_front();
            }
            if (valid_fraction >= kFreshValidFraction) {
                last_fresh_time_ = now;
            }
        }

        if (sample_count_ > 0) {
            const float interval = Seconds(now - last_sample_time_);
            sample_intervals_.push_back(interval);
            while (sample_intervals_.size() > 100) {
                sample_intervals_.pop_front();
            }
            float interval_sum = 0.0F;
            max_sample_gap_seconds_ = 0.0F;
            for (const float value : sample_intervals_) {
                interval_sum += value;
                max_sample_gap_seconds_ = std::max(max_sample_gap_seconds_, value);
            }
            if (interval_sum > 0.0F) {
                sample_rate_hz_ = static_cast<float>(sample_intervals_.size()) / interval_sum;
            }
        }
        last_sample_time_ = now;
        last_valid_fraction_ = valid_fraction;
        source_width_ = source_width;
        source_height_ = source_height;
        horizontal_fov_degrees_ = horizontal_fov_degrees;
        vertical_fov_degrees_ = vertical_fov_degrees;
        depth_scale_ = depth_scale;
        ++sample_count_;
        last_error_.clear();
    }

    void SetError(const std::string& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = error;
    }

    void AcquisitionLoop() {
        while (running_.load(std::memory_order_acquire)) {
            try {
                rs2::pipeline pipeline;
                rs2::config config;
                config.enable_stream(
                    RS2_STREAM_DEPTH, kCaptureWidth, kCaptureHeight,
                    RS2_FORMAT_Z16, kCaptureFps);
                if (const char* serial = std::getenv("M20_REALSENSE_SERIAL")) {
                    if (serial[0] != '\0') {
                        config.enable_device(serial);
                    }
                }
                const rs2::pipeline_profile profile = pipeline.start(config);
                const auto sensor = profile.get_device().first<rs2::depth_sensor>();
                const float depth_scale = sensor.get_depth_scale();
                const auto depth_profile = profile.get_stream(RS2_STREAM_DEPTH)
                    .as<rs2::video_stream_profile>();
                const auto intrinsics = depth_profile.get_intrinsics();
                if (intrinsics.width != kCaptureWidth ||
                    intrinsics.height != kCaptureHeight ||
                    depth_profile.fps() != kCaptureFps) {
                    throw std::runtime_error("RealSense selected an unexpected depth profile");
                }
                constexpr float kRadiansToDegrees = 57.2957795131F;
                const float horizontal_fov_degrees = 2.0F * std::atan(
                    static_cast<float>(intrinsics.width) / (2.0F * intrinsics.fx)) *
                    kRadiansToDegrees;
                const float vertical_fov_degrees = 2.0F * std::atan(
                    static_cast<float>(intrinsics.height) / (2.0F * intrinsics.fy)) *
                    kRadiansToDegrees;
                if (!std::isfinite(horizontal_fov_degrees) ||
                    !std::isfinite(vertical_fov_degrees) ||
                    std::abs(horizontal_fov_degrees - kTrainingHorizontalFovDegrees) >
                        kFovToleranceDegrees ||
                    std::abs(vertical_fov_degrees - kTrainingVerticalFovDegrees) >
                        kFovToleranceDegrees) {
                    throw std::runtime_error(
                        "RealSense depth FOV does not match the student training camera");
                }
                const auto device = profile.get_device();
                std::cout << "Direct RealSense depth source started: "
                          << device.get_info(RS2_CAMERA_INFO_NAME) << " #"
                          << device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)
                          << ", " << intrinsics.width << "x" << intrinsics.height
                          << " Z16 at " << depth_profile.fps() << " Hz, FOV="
                          << horizontal_fov_degrees << "x" << vertical_fov_degrees
                          << " deg, scale=" << depth_scale
                          << " m/unit; no depth ROS topic is used." << std::endl;

                auto next_sample_time = std::chrono::steady_clock::now();
                int consecutive_errors = 0;
                while (running_.load(std::memory_order_acquire)) {
                    rs2::frameset frames;
                    try {
                        frames = pipeline.wait_for_frames(1000);
                        consecutive_errors = 0;
                    } catch (const rs2::error& error) {
                        SetError(error.what());
                        if (++consecutive_errors >= 3) {
                            throw;
                        }
                        continue;
                    }
                    const rs2::depth_frame depth = frames.get_depth_frame();
                    if (!depth) {
                        SetError("RealSense frameset has no depth frame");
                        continue;
                    }
                    const auto now = std::chrono::steady_clock::now();
                    if (now < next_sample_time) {
                        continue;
                    }
                    if (now - next_sample_time > std::chrono::milliseconds(250)) {
                        next_sample_time = now;
                    }
                    next_sample_time += std::chrono::milliseconds(100);

                    DepthFrame frame{};
                    float valid_fraction = 0.0F;
                    if (!MakeFrame(depth, depth_scale, frame, valid_fraction)) {
                        SetError("invalid RealSense Z16 depth frame");
                        continue;
                    }
                    StoreFrame(
                        frame, valid_fraction, depth.get_width(), depth.get_height(),
                        horizontal_fov_degrees, vertical_fov_degrees,
                        depth_scale, now);
                }
                pipeline.stop();
            } catch (const std::exception& error) {
                SetError(error.what());
                if (running_.load(std::memory_order_acquire)) {
                    std::cerr << "Direct RealSense depth source error: " << error.what()
                              << "; retrying in 1 s." << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            }
        }
    }

    mutable std::mutex mutex_;
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::deque<DepthFrame> history_;
    std::deque<float> sample_intervals_;
    std::chrono::steady_clock::time_point last_sample_time_{};
    std::chrono::steady_clock::time_point last_fresh_time_{};
    float last_valid_fraction_ = 0.0F;
    float sample_rate_hz_ = 0.0F;
    float max_sample_gap_seconds_ = std::numeric_limits<float>::infinity();
    float depth_scale_ = 0.0F;
    uint64_t sample_count_ = 0;
    int source_width_ = 0;
    int source_height_ = 0;
    float horizontal_fov_degrees_ = 0.0F;
    float vertical_fov_degrees_ = 0.0F;
    std::string last_error_{"camera has not produced a depth frame"};
};

#endif  // M20_HAS_REALSENSE2
