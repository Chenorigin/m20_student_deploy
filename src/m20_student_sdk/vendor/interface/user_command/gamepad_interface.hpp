/**
 * @file gamepad_interface.hpp
 * @brief gamepad
 * @author DeepRobotics
 * @version 1.0
 * @date 2026-01-07
 *
 * @copyright Copyright (c) 2025  DeepRobotics
 *
 */

#pragma once

#include <arpa/inet.h>
#include <unistd.h>

#include <atomic>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common_types.h"
#include "custom_types.h"
#include "heartbeat.hpp"
#include "gamepad_wake_guard.hpp"
#include "input_timing.hpp"
#include "json.hpp"
#include "student_trace.hpp"
#include "user_command_interface.h"

using json = nlohmann::json;

class GamepadInterface : public UserCommandInterface {
private:
    static constexpr float kMaxForward = 1.0f;
    static constexpr float kMaxSide = 0.5f;
    static constexpr float kMaxYaw = 0.6f;
    std::array<std::array<float,2>,3> command_ranges_{{{-kMaxForward,kMaxForward},{-kMaxSide,kMaxSide},{-kMaxYaw,kMaxYaw}}};

    static void ClipNumber(float& num, float low, float high) {
        if (num < low) num = low;
        if (num > high) num = high;
    }

public:
    GamepadInterface(RobotName robot_name) : UserCommandInterface(robot_name) {
        if(const char* path=std::getenv("M20_ROBOT_CONFIG")){
            std::ifstream stream(path);if(!stream)throw std::runtime_error("cannot read gamepad robot config");
            json config;stream>>config;
            if(config.contains("command_operator_ranges")){
                const char* keys[]={"lin_vel_x","lin_vel_y","ang_vel_z"};
                for(int axis=0;axis<3;++axis){
                    const auto& bounds=config.at("command_operator_ranges").at(keys[axis]);
                    if(!bounds.is_array()||bounds.size()!=2||!bounds[0].is_number()||!bounds[1].is_number())throw std::runtime_error("invalid operator command range");
                    const float lo=bounds[0].get<float>(),hi=bounds[1].get<float>();
                    if(!std::isfinite(lo)||!std::isfinite(hi)||lo>=0||hi<=0)throw std::runtime_error("invalid finite operator command range");
                    const auto& training=config.at("command_training_ranges").at(keys[axis]);
                    if(lo<training[0].get<float>()||hi>training[1].get<float>())throw std::runtime_error("operator command range exceeds policy contract");
                    command_ranges_[axis]={lo,hi};
                }
            }
        }
        mHeartbeat_ = std::make_unique<Heartbeat>();

        mHandleMap_[100][100] = std::bind(&GamepadInterface::Handle_Heartbeat, this, std::placeholders::_1,
                                          std::placeholders::_2, std::placeholders::_3);
        mHandleMap_[2][21] = std::bind(&GamepadInterface::Handle_Steer, this, std::placeholders::_1,
                                       std::placeholders::_2, std::placeholders::_3);
        mHandleMap_[1101][12] = std::bind(&GamepadInterface::Handle_Key, this, std::placeholders::_1,
                                          std::placeholders::_2, std::placeholders::_3);
    }

    ~GamepadInterface() {}

    void Start() override {
        std::cout << "\n╔════════════════════════════════════════════════╗\n"
                  << "║                GAMEPAD TELEOP                  ║\n"
                  << "╚════════════════════════════════════════════════╝\n"
                  << "  Movement:  Left joystick\n"
                  << "  Rotation:  Right joystick\n"
                  << "  Mode:      R2 (damping)  L1 (stand)  L2 (control)\n"
                  << "\n";
    }

    void Stop() override {}
    UserCommand* GetUserCommand() override {
        std::lock_guard<std::mutex> lock(packet_mutex_);
        // A live handset heartbeat does not renew an old joystick target.
        if (std::chrono::steady_clock::now() >= steer_deadline_) {
            usr_cmd_->forward_vel_scale = 0.0f;
            usr_cmd_->side_vel_scale = 0.0f;
            usr_cmd_->turnning_vel_scale = 0.0f;
        }
        return usr_cmd_;
    }
    int ParseHeader(const char* buffer, int len) {
        const int headerSize = 16;
        if (len < headerSize) return -1;

        if ((uint8_t)buffer[0] == 0xeb && (uint8_t)buffer[2] == 0xeb && (uint8_t)buffer[3] == 0x90) {
            if ((uint8_t)buffer[1] != 0x90 && (uint8_t)buffer[1] != 0x91) {
                std::cout<<"Header format error"<<std::endl;
                return -1;
            }
            int bodyLen = ((uint16_t)buffer[5] << 8) | (uint8_t)buffer[4];
            return bodyLen + headerSize;
        }
        std::cout<<"Header magic error"<<std::endl;
        return -1;
    }

    bool ParseHeader(const char* buffer, int len, int& data_id, char* body, FromType& fromType) {
        const int headerSize = 16;
        if (len < headerSize) return false;

        if ((uint8_t)buffer[0] == 0xeb && (uint8_t)buffer[2] == 0xeb && (uint8_t)buffer[3] == 0x90) {
            if ((uint8_t)buffer[1] == 0x90) {
                fromType = FromType::inside;
            } else if ((uint8_t)buffer[1] == 0x91) {
                fromType = FromType::user;
            } else {
                return false;
            }

            int bodyLen = ((uint16_t)buffer[5] << 8) | (uint8_t)buffer[4];
            data_id = ((uint16_t)buffer[7] << 8) | (uint8_t)buffer[6];

            if (bodyLen != len - headerSize) {
                std::cout << "recv lost! bodyLen=" << bodyLen << " real=" << len - headerSize << std::endl;
                return false;
            }

            uint8_t dateFlag = buffer[8] & 0x03;
            if (dateFlag != 0x01) {
                std::cout << "Data format error" << std::endl;
                return false;
            }

            if (body != nullptr) {
                memcpy(body, buffer + headerSize, bodyLen);
                body[bodyLen] = '\0';
            }
        } else {
            std::cout << "Header sync error" << std::endl;
            return false;
        }
        return true;
    }

    bool ParseData(const DataInfo& dataInfo, std::string& response) {
        std::lock_guard<std::mutex> packet_lock(packet_mutex_);
        int type = 0, command = 0;
        bool ret = false;

        json document;
        json items;

        ret = ParseJson(dataInfo.buffer, dataInfo.len, document, type, command, items);
        if (!ret) {
            std::cout<<"JSON parse failed"<<std::endl;
            response = "JSON format parsing failed";
            return false;
        }
        const auto now = std::chrono::steady_clock::now();
        const bool is_heartbeat = type == 100 && command == 100;
        const bool known_command = is_heartbeat || (type == 2 && command == 21) ||
                                   (type == 1101 && command == 12);
        bool is_damping_key = false;
        if (type == 1101 && command == 12 && items.contains("KeyList") &&
            items["KeyList"].is_array()) {
            for (const auto& key : items["KeyList"]) {
                is_damping_key = is_damping_key ||
                    (key.is_string() && key.get<std::string>() == "G20_KEY_R2");
            }
        }
        if (!known_command) {
            response = "Unknown gamepad command";
            return false;
        }
        if (!is_damping_key && std::chrono::steady_clock::now() >= dataInfo.command_deadline) {
            response = "Expired gamepad command";
            return false;
        }
        const bool source_timed_out = !source_bound_ ||
            now - last_source_packet_ > std::chrono::milliseconds(2500);
        if (source_timed_out) {
            usr_cmd_->target_mode = uint8_t(RobotMotionState::JointDamping);
            usr_cmd_->forward_vel_scale = 0.0f;
            usr_cmd_->side_vel_scale = 0.0f;
            usr_cmd_->turnning_vel_scale = 0.0f;
        }
        if (source_timed_out && !is_heartbeat && !is_damping_key) {
            response = "Gamepad heartbeat required before commands";
            return false;
        }
        if (source_timed_out && is_heartbeat) {
            source_ip_ = dataInfo.ip;
            source_from_ = dataInfo.fromType;
            source_bound_ = true;
        }
        if (!source_bound_ || dataInfo.ip != source_ip_ ||
            dataInfo.fromType != source_from_) {
            response = "Gamepad source mismatch";
            return false;
        }
        last_source_packet_ = now;
        ignore_motion_keys_this_packet_ = wake_guard_.ObservePacket(now, is_heartbeat);
        if (ignore_motion_keys_this_packet_) {
            usr_cmd_->target_mode = uint8_t(RobotMotionState::JointDamping);
            usr_cmd_->forward_vel_scale = 0.0f;
            usr_cmd_->side_vel_scale = 0.0f;
            usr_cmd_->turnning_vel_scale = 0.0f;
            std::cout << "[SAFETY] Gamepad wake/reconnect: damping and stand lockout. "
                         "Wait for heartbeat, then press R2 before L1."
                      << std::endl;
        }
        if (is_heartbeat) {
            mHeartbeat_->RecordHeartbeat(now);
            usr_cmd_->time_stamp = std::chrono::duration<double>(
                now.time_since_epoch()).count();
        }

        CmdInfo cmd_info;
        cmd_info.ip = dataInfo.ip;
        cmd_info.port = dataInfo.port;
        cmd_info.fromType = dataInfo.fromType;
        cmd_info.type = type;
        cmd_info.command = command;
        cmd_info.command_deadline = dataInfo.command_deadline;

        if (!is_damping_key && std::chrono::steady_clock::now() >= dataInfo.command_deadline) {
            response = "Expired gamepad command";
            return false;
        }
        if (mHandleMap_[type][command]) {
            mHandleMap_[type][command](items, response, cmd_info);
        } else {
            response = buildResponse(cmd_info, 1, "Unknown command");
        }
        return true;
    }

    void AddDataHeader(const std::string& in, std::string& out, int data_id = 0, NetInfo perNetInfo = {0}) {
        out.resize(16);

        out[0] = 0xeb;
        if (perNetInfo.fromType == FromType::inside) {
            out[1] = 0x90;
        } else {
            out[1] = 0x91;
        }
        out[2] = 0xeb;
        out[3] = 0x90;

        auto size = in.size();

        out[4] = (uint8_t)(size & 0xFF);
        out[5] = (uint8_t)((size >> 8) & 0xFF);

        out[6] = (uint8_t)(data_id & 0xFF);
        out[7] = (uint8_t)((data_id >> 8) & 0xFF);

        out[8] = 1;

        for (int i = 9; i < 16; i++) out[i] = 0x00;

        out += in;
    }

private:
    std::unique_ptr<Heartbeat> mHeartbeat_;
    std::mutex packet_mutex_;
    m20::GamepadWakeGuard wake_guard_;
    bool source_bound_ = false;
    bool ignore_motion_keys_this_packet_ = false;
    std::string source_ip_;
    FromType source_from_ = FromType::user;
    std::chrono::steady_clock::time_point last_source_packet_{};
    std::chrono::steady_clock::time_point steer_deadline_{};

    static bool CommandFresh(const CmdInfo& command) {
        return std::chrono::steady_clock::now() < command.command_deadline;
    }

    using HANDLE_FUNC = std::function<void(json& item, std::string& res, const CmdInfo& cmd_info)>;

    std::unordered_map<int, std::unordered_map<int, HANDLE_FUNC>> mHandleMap_;

    bool ParseJson(const char* buf, int len, json& document, int& type, int& command, json& items) {
        if (!buf || len == 0) return false;

        try {
            document = json::parse(buf, buf + len);

            if (!document.is_object() || !document.contains("PatrolDevice"))
                throw std::runtime_error("No PatrolDevice");

            json& root = document["PatrolDevice"];
            if (!root.is_object()) throw std::runtime_error("PatrolDevice not object");

            if (!root.contains("Items")) throw std::runtime_error("No Items");

            items = root["Items"];

            if (!root.contains("Type") || !root.contains("Command")) throw std::runtime_error("No Type/Command");

            if (!root["Type"].is_number_integer() || !root["Command"].is_number_integer())
                throw std::runtime_error("Type/Command not Int");

            type = root["Type"].get<int>();
            command = root["Command"].get<int>();

        } catch (const json::parse_error& e) {
            std::cerr << "JSON Parse Error: " << e.what() << std::endl;
            return false;
        } catch (const std::exception& e) {
            std::cerr << "JSON Logic Error: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    void JSONRootToBroker(const int& type, const int& command, json& root, bool isNew = false) {
        if (isNew) {
            root = json::object();
            root["PatrolDevice"] = {
                {"Items", json::object()}, {"Type", type}, {"Command", command}, {"Time", getCurrentDateTime()}};
        } else {
            root["PatrolDevice"]["Type"] = type;
            root["PatrolDevice"]["Command"] = command;
            root["PatrolDevice"]["Time"] = getCurrentDateTime();
        }
    }

    std::string buildResponse(const CmdInfo& cmd_info, const int& errorCode, std::string errorMsg = "") {
        std::string res;
        json document;

        JSONRootToBroker(cmd_info.type, cmd_info.command, document, true);

        document["PatrolDevice"]["Items"]["ErrorCode"] = errorCode;

        if (!errorMsg.empty()) {
            document["PatrolDevice"]["Items"]["ErrorMessage"] = errorMsg;
        }

        res = document.dump();
        return res;
    }

    void Handle_Heartbeat(json& items, std::string& res, const CmdInfo& cmd_info) {
        NetInfo netInfo;
        netInfo.ip = cmd_info.ip;
        netInfo.port = cmd_info.port;
        netInfo.fromType = cmd_info.fromType;
        mHeartbeat_->wakeUp(netInfo);

        res = buildResponse(cmd_info, 0);
    }

    static void TraceSteer(const json& items,const char* result,const json& received,
                           const std::array<float,3>* clipped=nullptr) noexcept {
        try {
            json row={{"event","steering_receive"},{"result",result},{"handler_receive",received},
                {"theoretical_time_basis","asynchronous_input_no_scheduled_deadline"},
                {"theoretical_monotonic_ns",nullptr},{"theoretical_wall_ns",nullptr},{"raw_items",items}};
            if(clipped)row["accepted_command"]=*clipped;
            StudentTrace::Instance().Push(std::move(row));
        }catch(...){try{StudentTrace::Instance().RecordConstructionFailure();}catch(...){}}
    }
    void Handle_Steer(json& items, std::string& res, const CmdInfo& cmd_info) {
        json received;try{received=StudentTrace::Stamp();}catch(...){}
        if (!items.contains("X") || !items.contains("Y") ||
            !items["X"].is_number() || !items["Y"].is_number() ||
            (items.contains("Yaw") && !items["Yaw"].is_number())) {
            res = buildResponse(cmd_info, 2, "Missing fields");
            TraceSteer(items,"rejected_missing_or_invalid_fields",received);
            return;
        }

        try {
            float forward = items["X"].get<float>();
            float side = items["Y"].get<float>();
            float yaw = items.contains("Yaw") ? items["Yaw"].get<float>() : 0.0f;
            if(!std::isfinite(forward)||!std::isfinite(side)||!std::isfinite(yaw)){
                res=buildResponse(cmd_info,3,"Nonfinite steering command");
                TraceSteer(items,"rejected_nonfinite",received);return;
            }

            ClipNumber(forward, command_ranges_[0][0], command_ranges_[0][1]);
            ClipNumber(side, command_ranges_[1][0], command_ranges_[1][1]);
            ClipNumber(yaw, command_ranges_[2][0], command_ranges_[2][1]);

            if (!CommandFresh(cmd_info)) {
                res = buildResponse(cmd_info, 4, "Expired steering command");
                TraceSteer(items,"rejected_expired",received);
                return;
            }
            usr_cmd_->forward_vel_scale = forward;
            usr_cmd_->side_vel_scale = side;
            usr_cmd_->turnning_vel_scale = yaw;
            steer_deadline_ = std::min(cmd_info.command_deadline,
                std::chrono::steady_clock::now()+m20::kInputPacketLifetime);
            const std::array<float,3> accepted{forward,side,yaw};
            TraceSteer(items,"accepted",received,&accepted);
        } catch (const std::exception& e) {
            res = buildResponse(cmd_info, 3, "Data type error");
            TraceSteer(items,"rejected_data_type",received);
            return;
        }

    }

    void Handle_Key(json& items, std::string& res, const CmdInfo& cmd_info) {
        if(!msfb_){res=buildResponse(cmd_info,4,"Motion feedback is not initialized");return;}
        static std::unordered_map<std::string, KeyCode> keyMap = {{"G20_KEY_L1", KeyCode::L1},
                                                                  {"G20_KEY_L2", KeyCode::L2},
                                                                  {"G20_KEY_R1", KeyCode::R1},
                                                                  {"G20_KEY_R2", KeyCode::R2}};

        if (items.contains("KeyList") && items["KeyList"].is_array()) {
            // Damping wins even if a packet contains both R2 and L1.
            for (const auto& key : items["KeyList"]) {
                if (key.is_string() && key.get<std::string>() == "G20_KEY_R2") {
                    usr_cmd_->target_mode = uint8_t(RobotMotionState::JointDamping);
                    wake_guard_.ObserveR2(std::chrono::steady_clock::now(),
                                          CommandFresh(cmd_info));
                    std::cout << "[MODE] Joint Damping\n";
                    res = buildResponse(cmd_info, 0);
                    return;
                }
            }
            for (const auto& key : items["KeyList"]) {
                if (key.is_string() && key.get<std::string>() == "G20_KEY_R1") {
                    if (!CommandFresh(cmd_info)) {
                        res = buildResponse(cmd_info, 4, "Expired key command");
                        return;
                    }
                    wake_guard_.ObserveLieDown();
                    usr_cmd_->target_mode = uint8_t(RobotMotionState::LieDown);
                    std::cout << "[MODE] Lie Down\n";
                    res = buildResponse(cmd_info, 0);
                    return;
                }
            }
            for (const auto& key : items["KeyList"]) {
                if (key.is_string()) {
                    std::string keyStr = key.get<std::string>();

                    if (keyMap.count(keyStr)) {
                        KeyCode code = keyMap[keyStr];
                        switch (code) {
                            case KeyCode::L1:
                                if (ignore_motion_keys_this_packet_ ||
                                    !wake_guard_.StandArmed()) {
                                    std::cout << "[SAFETY] Ignoring L1 until a settled heartbeat "
                                                 "and a separate R2 press rearm standing."
                                              << std::endl;
                                    break;
                                }
                                if (msfb_->GetCurrentState() == RobotMotionState::WaitingForStand 
                                    || msfb_->GetCurrentState() == RobotMotionState::LieDown) {
                                    if (!wake_guard_.ConsumeStand(std::chrono::steady_clock::now())) break;
                                    if (!CommandFresh(cmd_info)) {
                                        res = buildResponse(cmd_info, 4, "Expired key command");
                                        return;
                                    }
                                    usr_cmd_->target_mode = uint8_t(RobotMotionState::StandingUp);
                                    std::cout << "[MODE] Standing Up\n";
                                }
                                break;
                            case KeyCode::L2:
                                if (!ignore_motion_keys_this_packet_ &&
                                    wake_guard_.AllowL2(std::chrono::steady_clock::now()) &&
                                    msfb_->GetCurrentState() == RobotMotionState::StandingUp) {
                                    if (!CommandFresh(cmd_info)) {
                                        res = buildResponse(cmd_info, 4, "Expired key command");
                                        return;
                                    }
                                    usr_cmd_->target_mode = uint8_t(RobotMotionState::RLControlMode);
                                    std::cout << "[MODE] RL Control\n";
                                }
                                break;
                            case KeyCode::R1:
                                // Handled above, before any motion key.
                                break;
                            case KeyCode::R2:
                                // Handled above, before any motion key.
                                break;
                            default:
                                break;
                        }
                    }
                }
            }
        }
        res = buildResponse(cmd_info, 0);
    }

    std::string getCurrentDateTime() {
        auto now = std::chrono::system_clock::now();
        std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
        std::tm* localTime = std::localtime(&currentTime);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        std::stringstream ss;
        ss << std::put_time(localTime, "%Y-%m-%d %H:%M:%S") << "." << std::setw(3) << std::setfill('0') << ms.count();
        return ss.str();
    }
};
