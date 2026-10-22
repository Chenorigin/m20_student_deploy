#pragma once

#include <cstdint>

namespace m20::imu_loopback {

inline constexpr uint32_t kMagic = 0x4d323049U;  // "M20I"
inline constexpr uint32_t kVersion = 1U;
inline constexpr uint16_t kPort = 15012U;

struct Packet {
    uint32_t magic = kMagic;
    uint32_t version = kVersion;
    uint64_t sequence = 0;
    int64_t stamp_nanoseconds = 0;
    float roll = 0.0F;
    float pitch = 0.0F;
    float yaw = 0.0F;
    float omega_x = 0.0F;
    float omega_y = 0.0F;
    float omega_z = 0.0F;
    float acc_x = 0.0F;
    float acc_y = 0.0F;
    float acc_z = 0.0F;
};

static_assert(sizeof(Packet) == 64, "unexpected IMU loopback packet layout");

}  // namespace m20::imu_loopback
