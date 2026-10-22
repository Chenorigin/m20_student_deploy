#pragma once

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace m20_gamepad {

constexpr uint16_t kGamepadPort = 30000;

struct CommandPacket {
    int type = -1;
    int command = -1;
    std::string body;
};

inline bool ExtractJsonInteger(
    std::string_view body, std::string_view key, int& output) {
    const std::string quoted_key = "\"" + std::string(key) + "\"";
    size_t position = body.find(quoted_key);
    if (position == std::string_view::npos) {
        return false;
    }
    position = body.find(':', position + quoted_key.size());
    if (position == std::string_view::npos) {
        return false;
    }
    ++position;
    while (position < body.size() &&
           std::isspace(static_cast<unsigned char>(body[position]))) {
        ++position;
    }

    bool negative = false;
    if (position < body.size() && body[position] == '-') {
        negative = true;
        ++position;
    }
    if (position >= body.size() ||
        !std::isdigit(static_cast<unsigned char>(body[position]))) {
        return false;
    }

    int64_t value = 0;
    while (position < body.size() &&
           std::isdigit(static_cast<unsigned char>(body[position]))) {
        value = value * 10 + (body[position] - '0');
        if (value > static_cast<int64_t>(std::numeric_limits<int>::max()) + 1) {
            return false;
        }
        ++position;
    }
    if (negative) {
        value = -value;
    }
    if (value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        return false;
    }
    output = static_cast<int>(value);
    return true;
}

inline bool ParseCommandPacket(
    const uint8_t* payload, size_t payload_size, CommandPacket& output) {
    constexpr size_t kHeaderSize = 16;
    if (payload == nullptr || payload_size < kHeaderSize ||
        payload[0] != 0xeb || (payload[1] != 0x90 && payload[1] != 0x91) ||
        payload[2] != 0xeb || payload[3] != 0x90 ||
        (payload[8] & 0x03) != 0x01) {
        return false;
    }

    const size_t body_size = static_cast<size_t>(payload[4]) |
        (static_cast<size_t>(payload[5]) << 8);
    if (body_size != payload_size - kHeaderSize) {
        return false;
    }

    output.body.assign(
        reinterpret_cast<const char*>(payload + kHeaderSize), body_size);
    return ExtractJsonInteger(output.body, "Type", output.type) &&
        ExtractJsonInteger(output.body, "Command", output.command);
}

inline bool IsForwardedCommand(const CommandPacket& packet) {
    return (packet.type == 100 && packet.command == 100) ||
        (packet.type == 2 && packet.command == 21) ||
        (packet.type == 1101 && packet.command == 12);
}

inline const char* CommandName(const CommandPacket& packet) {
    if (packet.type == 100 && packet.command == 100) {
        return "heartbeat";
    }
    if (packet.type == 2 && packet.command == 21) {
        return "steer";
    }
    if (packet.type == 1101 && packet.command == 12) {
        return "key";
    }
    return "other";
}

inline std::string ExtractKeyName(const CommandPacket& packet) {
    if (packet.type != 1101 || packet.command != 12) {
        return {};
    }
    constexpr std::string_view kPrefix = "G20_KEY_";
    const size_t begin = packet.body.find(kPrefix);
    if (begin == std::string::npos) {
        return {};
    }
    size_t end = begin;
    while (end < packet.body.size()) {
        const unsigned char value = static_cast<unsigned char>(packet.body[end]);
        if (!(std::isalnum(value) || value == '_')) {
            break;
        }
        ++end;
    }
    return packet.body.substr(begin, end - begin);
}

}  // namespace m20_gamepad
