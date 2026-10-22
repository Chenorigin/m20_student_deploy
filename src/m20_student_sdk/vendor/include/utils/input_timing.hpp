#pragma once
#include <chrono>
namespace m20 {
// One receipt deadline is shared by queued key validation, steering expiry,
// and the receive-only readiness probe. Reprocessing cannot renew it.
inline constexpr auto kInputPacketLifetime=std::chrono::milliseconds(200);
}
