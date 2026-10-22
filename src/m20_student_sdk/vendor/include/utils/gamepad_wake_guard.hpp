#pragma once

#include <chrono>

namespace m20 {

// A reconnect must not turn a queued or repeated L1 into a stand request.
// The operator explicitly rearms with R2 after heartbeat traffic settles.
class GamepadWakeGuard {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    bool ObservePacket(TimePoint now, bool heartbeat) {
        const bool woke = !seen_packet_ || now - last_packet_ > kDisconnectGap ||
                          (heartbeat_seen_ && now - last_heartbeat_ > kDisconnectGap);
        if (woke) {
            armed_ = false;
            l2_allowed_ = false;
            wake_time_ = now;
            heartbeat_seen_ = false;
        }
        seen_packet_ = true;
        last_packet_ = now;
        if (heartbeat) { last_heartbeat_ = now; heartbeat_seen_ = true; }
        return woke;
    }

    void ObserveR2(TimePoint now, bool fresh = true) {
        armed_ = false;
        l2_allowed_ = false;
        if (fresh && heartbeat_seen_ &&
            now - last_heartbeat_ <= kHeartbeatFreshness &&
            now - wake_time_ >= kSettleTime) {
            armed_ = true;
            armed_until_ = now + kStandArmWindow;
        }
    }

    bool ConsumeStand(TimePoint now) {
        if (!armed_ || now > armed_until_ || !heartbeat_seen_ ||
            now - last_heartbeat_ > kHeartbeatFreshness) return false;
        armed_ = false;
        l2_allowed_ = true;
        l2_until_ = now + kL2Window;
        return true;
    }

    bool AllowL2(TimePoint now) const {
        return l2_allowed_ && now <= l2_until_ && heartbeat_seen_ &&
            now - last_heartbeat_ <= kHeartbeatFreshness;
    }
    void ObserveLieDown() { armed_ = false; l2_allowed_ = false; }
    bool StandArmed() const { return armed_; }

private:
    static constexpr auto kDisconnectGap = std::chrono::milliseconds(2500);
    static constexpr auto kSettleTime = std::chrono::milliseconds(1000);
    static constexpr auto kHeartbeatFreshness = std::chrono::milliseconds(500);
    static constexpr auto kStandArmWindow = std::chrono::seconds(5);
    static constexpr auto kL2Window = std::chrono::seconds(30);
    bool seen_packet_ = false;
    bool heartbeat_seen_ = false;
    bool armed_ = false;
    bool l2_allowed_ = false;
    TimePoint last_packet_{};
    TimePoint wake_time_{};
    TimePoint last_heartbeat_{};
    TimePoint armed_until_{};
    TimePoint l2_until_{};
};

}  // namespace m20
