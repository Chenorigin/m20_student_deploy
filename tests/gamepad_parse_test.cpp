#include "custom_types.h"

using namespace types;

#include "user_command_interface.h"
using namespace interface;
#include "udp_server.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

std::vector<char> MakePacket(const std::string& body) {
    std::vector<char> packet(16 + body.size(), 0);
    packet[0] = static_cast<char>(0xeb);
    packet[1] = static_cast<char>(0x91);
    packet[2] = static_cast<char>(0xeb);
    packet[3] = static_cast<char>(0x90);
    packet[4] = static_cast<char>(body.size() & 0xff);
    packet[5] = static_cast<char>((body.size() >> 8) & 0xff);
    packet[8] = 1;
    std::copy(body.begin(), body.end(), packet.begin() + 16);
    return packet;
}

int main() {
    auto packet = MakePacket(
        R"({"PatrolDevice":{"Type":100,"Command":100,"Items":{}}})");
    GamepadInterface gamepad(RobotName::M20);
    MotionStateFeedback feedback;
    gamepad.SetMotionStateFeedback(&feedback);
    int message_id = -1;
    FromType from_type{};
    if (!gamepad.ParseHeader(packet.data(), static_cast<int>(packet.size()),
                             message_id, nullptr, from_type)) {
        std::cerr << "valid APDU header rejected\n";
        return 1;
    }
    DataInfo data{};
    data.ip = "127.0.0.1";
    data.port = 30000;
    data.fromType = from_type;
    data.buffer = packet.data() + 16;
    data.len = static_cast<int>(packet.size()) - 16;
    data.command_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::string response;
    if (!gamepad.ParseData(data, response) ||
        !std::isfinite(gamepad.GetUserCommand()->time_stamp) ||
        gamepad.GetUserCommand()->time_stamp <= 0.0 ||
        response.find("\"ErrorCode\":0") == std::string::npos) {
        std::cerr << "valid heartbeat did not update command timestamp\n";
        return 1;
    }
    auto send_key = [&](const std::string& key) {
        packet = MakePacket(std::string(
            R"({"PatrolDevice":{"Type":1101,"Command":12,"Items":{"KeyList":[")") +
            key + R"("]}}})");
        data.buffer = packet.data() + 16;
        data.len = static_cast<int>(packet.size()) - 16;
        return gamepad.ParseData(data, response);
    };
    feedback.UpdateCurrentState(RobotMotionState::WaitingForStand);
    if (!send_key("G20_KEY_L1") ||
        gamepad.GetUserCommand()->target_mode == uint8_t(RobotMotionState::StandingUp)) {
        std::cerr << "wakeup L1 started standing without rearm\n";
        return 1;
    }
    feedback.UpdateCurrentState(RobotMotionState::StandingUp);
    if (!send_key("G20_KEY_L2") ||
        gamepad.GetUserCommand()->target_mode == uint8_t(RobotMotionState::RLControlMode)) {
        std::cerr << "wakeup L2 bypassed damping lockout\n";
        return 1;
    }
    feedback.UpdateCurrentState(RobotMotionState::WaitingForStand);
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    packet = MakePacket(R"({"PatrolDevice":{"Type":100,"Command":100,"Items":{}}})");
    data.buffer = packet.data() + 16;
    data.len = static_cast<int>(packet.size()) - 16;
    if (!gamepad.ParseData(data, response) || !send_key("G20_KEY_R2") ||
        !send_key("G20_KEY_L1") ||
        gamepad.GetUserCommand()->target_mode != uint8_t(RobotMotionState::StandingUp)) {
        std::cerr << "settled heartbeat, R2, L1 did not arm standing\n";
        return 1;
    }
    data.ip = "127.0.0.2";
    if (send_key("G20_KEY_L1") ||
        gamepad.GetUserCommand()->target_mode != uint8_t(RobotMotionState::StandingUp)) {
        std::cerr << "unbound source changed a motion command\n";
        return 1;
    }
    data.ip = "127.0.0.1";
    feedback.UpdateCurrentState(RobotMotionState::LieDown);
    if (!send_key("G20_KEY_R2")) return 1;
    packet = MakePacket(
        R"({"PatrolDevice":{"Type":1101,"Command":12,"Items":{"KeyList":["G20_KEY_L1","G20_KEY_R1"]}}})");
    data.buffer = packet.data() + 16;
    data.len = static_cast<int>(packet.size()) - 16;
    if (!gamepad.ParseData(data, response) ||
        gamepad.GetUserCommand()->target_mode != uint8_t(RobotMotionState::LieDown)) {
        std::cerr << "R1 did not override same-packet L1\n";
        return 1;
    }
    m20::GamepadWakeGuard guard;
    const auto start = m20::GamepadWakeGuard::TimePoint{};
    guard.ObservePacket(start, true);
    guard.ObserveR2(start);
    if (guard.StandArmed()) return 1;
    guard.ObservePacket(start + std::chrono::milliseconds(1100), true);
    guard.ObserveR2(start + std::chrono::milliseconds(1100));
    if (!guard.StandArmed() ||
        guard.ConsumeStand(start + std::chrono::milliseconds(1701))) return 1;
    guard.ObservePacket(start + std::chrono::milliseconds(1800), true);
    if (!guard.ConsumeStand(start + std::chrono::milliseconds(1900)) ||
        guard.ConsumeStand(start + std::chrono::milliseconds(2000)) ||
        !guard.AllowL2(start + std::chrono::milliseconds(2000)) ||
        !guard.ObservePacket(start + std::chrono::milliseconds(4500), false) ||
        guard.StandArmed() ||
        guard.AllowL2(start + std::chrono::milliseconds(4500))) {
        std::cerr << "wake gap failed to revoke stand rearm\n";
        return 1;
    }
    // Exercise the same APDU/JSON path as UDP 30000 while disconnected from
    // hardware. R1 and R2 must remain distinct in RL control mode.
    feedback.UpdateCurrentState(RobotMotionState::RLControlMode);
    for (const auto& item : {
             std::pair<const char*, RobotMotionState>{"G20_KEY_R1", RobotMotionState::LieDown},
             {"G20_KEY_R2", RobotMotionState::JointDamping}}) {
        packet = MakePacket(std::string(
            R"({"PatrolDevice":{"Type":1101,"Command":12,"Items":{"KeyList":[")") +
            item.first + R"("]}}})");
        if (!gamepad.ParseHeader(packet.data(), static_cast<int>(packet.size()),
                                 message_id, nullptr, from_type)) {
            std::cerr << "valid key APDU header rejected\n";
            return 1;
        }
        data.buffer = packet.data() + 16;
        data.len = static_cast<int>(packet.size()) - 16;
        if (!gamepad.ParseData(data, response) ||
            gamepad.GetUserCommand()->target_mode != static_cast<uint8_t>(item.second)) {
            std::cerr << "gamepad key mapped to the wrong motion state: " << item.first << '\n';
            return 1;
        }
    }
    data.command_deadline = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
    if (send_key("G20_KEY_L1") ||
        gamepad.GetUserCommand()->target_mode != uint8_t(RobotMotionState::JointDamping)) {
        std::cerr << "expired key was accepted after receive timestamp check\n";
        return 1;
    }
    if (!send_key("G20_KEY_R2") ||
        gamepad.GetUserCommand()->target_mode != uint8_t(RobotMotionState::JointDamping)) {
        std::cerr << "expired R2 failed to request damping\n";
        return 1;
    }
    feedback.UpdateCurrentState(RobotMotionState::WaitingForStand);
    data.command_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    if (!send_key("G20_KEY_L1") ||
        gamepad.GetUserCommand()->target_mode == uint8_t(RobotMotionState::StandingUp)) {
        std::cerr << "expired R2 incorrectly armed L1\n";
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2600));
    data.command_deadline = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
    if (!send_key("G20_KEY_R2") ||
        gamepad.GetUserCommand()->target_mode != uint8_t(RobotMotionState::JointDamping)) {
        std::cerr << "R2 failed after gamepad heartbeat timeout\n";
        return 1;
    }
    std::cout << "gamepad_heartbeat_parse=PASS\n"
              << "gamepad_wake_requires_r2_rearm=PASS\n"
              << "gamepad_reverse_key_priority=PASS\n"
              << "gamepad_r1_liedown_r2_damping=PASS\n"
              << "gamepad_expired_command_gate=PASS\n";
    return 0;
}
