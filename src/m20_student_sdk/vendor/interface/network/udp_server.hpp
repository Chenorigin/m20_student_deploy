/**
 * @file udp_server.hpp
 * @brief udp server
 * @author DeepRobotics
 * @version 1.0
 * @date 2026-01-07
 *
 * @copyright Copyright (c) 2025  DeepRobotics
 *
 */

#pragma once

#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <time.h>

#include "common_types.h"
#include "gamepad_interface.hpp"
#include "gamepad_packet_tap.hpp"
#include <cstdlib>

#define UDP_PORT 30000
#define IP "0.0.0.0"
#define UDP_BUFFER_SIZE 1024 * 32

class GamepadInterface;

class UdpServer {
public:
    UdpServer(GamepadInterface* gp) : mGamepadInterface_(gp) {
        if (!Init()) {
            throw std::runtime_error("[UdpServer] failed to bind UDP/30000");
        }

        mIsRunning_ = true;
        mServerThread_ = std::thread(&UdpServer::RecvLoop, this);
        std::cout << (tap_ ? "Gamepad passive capture; vendor owns UDP/30000" : "UDP Server started on port 30000") << std::endl;
    }

    ~UdpServer() {
        mIsRunning_ = false;

        // Receive has a bounded timeout. Join before closing to avoid fd reuse.
        if (mServerThread_.joinable()) mServerThread_.join();
        if (!tap_ && mFd >= 0) close(mFd);
        mFd = -1;

    }

private:
    bool Init() {
        const char* mode = std::getenv("M20_GAMEPAD_TRANSPORT");
        if (mode && std::string(mode) == "passive") {
            const char* interface = std::getenv("M20_GAMEPAD_INTERFACE");
            if (!interface || !*interface)
                throw std::runtime_error("passive gamepad requires M20_GAMEPAD_INTERFACE");
            tap_ = std::make_unique<GamepadPacketTap>(interface);
            mFd = tap_->fd();
            return true;
        }
        if (mode && std::string(mode) != "udp")
            throw std::runtime_error("unknown M20_GAMEPAD_TRANSPORT");
        mFd = socket(AF_INET, SOCK_DGRAM, 0);
        if (mFd < 0) {
            perror("Create socket failed");
            return false;
        }

        // Keep the kernel backlog bounded; old motion keys must not be replayed
        // after the controller thread has been delayed.
        int recvBufSize = 64 * 1024;
        setsockopt(mFd, SOL_SOCKET, SO_RCVBUF, &recvBufSize, sizeof(recvBufSize));
        int bufSize = 1024 * 1024;
        setsockopt(mFd, SOL_SOCKET, SO_SNDBUF, &bufSize, sizeof(bufSize));
        int timestampOn = 1;
        if (setsockopt(mFd, SOL_SOCKET, SO_TIMESTAMPNS,
                       &timestampOn, sizeof(timestampOn)) < 0) {
            perror("Enable receive timestamps failed");
            close(mFd);
            return false;
        }

        struct timeval timeout;
        timeout.tv_sec = 5;
        timeout.tv_usec = 0;
        if (setsockopt(mFd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
            perror("Set timeout failed");
            close(mFd);
            return false;
        }

        int opt = 1;
        setsockopt(mFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in serverAddr{};
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = inet_addr(IP);
        serverAddr.sin_port = htons(UDP_PORT);

        if (bind(mFd, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
            perror("Bind failed");
            close(mFd);
            return false;
        }

        return true;
    }

    void RecvLoop() {
        sockaddr_in client_addr{};

        std::vector<char> buffer(UDP_BUFFER_SIZE);

        while (mIsRunning_) {
            alignas(cmsghdr) char control[CMSG_SPACE(sizeof(timespec))]{};
            iovec payload{buffer.data(), buffer.size()};
            msghdr message{};
            sockaddr_ll packet_address{};
            message.msg_name = tap_ ? static_cast<void*>(&packet_address) : static_cast<void*>(&client_addr);
            message.msg_namelen = tap_ ? sizeof(packet_address) : sizeof(client_addr);
            message.msg_iov = &payload;
            message.msg_iovlen = 1;
            message.msg_control = control;
            message.msg_controllen = sizeof(control);
            ssize_t recvLen = recvmsg(mFd, &message, 0);

            if (recvLen > 0) {
                if (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) continue;
                size_t body_offset = 0, body_length = recvLen;
                if (tap_ && (packet_address.sll_pkttype != PACKET_HOST ||
                    !GamepadPacketTap::Decode(buffer.data(), recvLen, tap_->local(),
                                              client_addr, body_offset, body_length))) continue;
                timespec received_at{};
                bool timestamp_found = false;
                for (cmsghdr* item = CMSG_FIRSTHDR(&message); item != nullptr;
                     item = CMSG_NXTHDR(&message, item)) {
                    if (item->cmsg_level == SOL_SOCKET && item->cmsg_type == SCM_TIMESTAMPNS &&
                        item->cmsg_len >= CMSG_LEN(sizeof(timespec))) {
                        std::memcpy(&received_at, CMSG_DATA(item), sizeof(received_at));
                        timestamp_found = true;
                        break;
                    }
                }
                timespec now{};
                clock_gettime(CLOCK_REALTIME, &now);
                const double age_seconds = double(now.tv_sec - received_at.tv_sec) +
                    double(now.tv_nsec - received_at.tv_nsec) * 1.0e-9;
                // Old packets remain parseable only so a trusted R2 damping
                // request can still stop the robot. Positive commands get an
                // already-expired deadline and fail at their final write.
                const double remaining_seconds =
                    timestamp_found && age_seconds >= 0.0 && age_seconds <= std::chrono::duration<double>(m20::kInputPacketLifetime).count()
                        ? std::chrono::duration<double>(m20::kInputPacketLifetime).count() - age_seconds : -1.0;
                const auto command_deadline = std::chrono::steady_clock::now() +
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(remaining_seconds));
                auto data = std::make_shared<std::vector<char>>(buffer.begin() + body_offset, buffer.begin() + body_offset + body_length);

                auto headerLen = mGamepadInterface_->ParseHeader(data->data(), body_length);
                if (headerLen > 0 && headerLen == body_length) {
                    dataHandle(mFd, data, headerLen, client_addr, command_deadline);
                } else {
                    std::cout<<"Invalid header"<<std::endl;
                }
            }
        }
    }

    void dataHandle(int socket, std::shared_ptr<std::vector<char>> data, int dataLen,
                    const sockaddr_in& clientAddr,
                    std::chrono::steady_clock::time_point command_deadline) {
            FromType from_type;
            bool ret = false;
            int dataId = -1;
            std::string res;
            DataInfo dataInfo;

            if (mGamepadInterface_->ParseHeader(data->data(), dataLen, dataId, nullptr, from_type)) {
                std::string ip = inet_ntoa(clientAddr.sin_addr);
                int port = ntohs(clientAddr.sin_port);

                dataInfo.ip = ip;
                dataInfo.port = port;
                dataInfo.fromType = from_type;
                dataInfo.buffer = data->data() + 16;
                dataInfo.len = dataLen - 16;
                dataInfo.command_deadline = command_deadline;

                ret = mGamepadInterface_->ParseData(dataInfo, res);
            } else {
                res = "Protocol header error,error code:0x2";
            }

            if (res.length() > 0) {
                std::string Response;
                mGamepadInterface_->AddDataHeader(res, Response, dataId, dataInfo);
                if (!tap_) {
                    sendData(socket, Response.c_str(), Response.length(), clientAddr);
                } else if (ret) {
                    // Preserve vendor heartbeat/status ownership. SDK key ACK only.
                    const auto request = json::parse(dataInfo.buffer, dataInfo.buffer + dataInfo.len, nullptr, false);
                    if (!request.is_discarded() && request.contains("PatrolDevice")) {
                        const auto& body = request["PatrolDevice"];
                        if (body.contains("Type") && body.contains("Command") &&
                            body["Type"] == 1101 && body["Command"] == 12 &&
                            !tap_->Reply(Response, clientAddr))
                            std::cerr << "Gamepad private ACK failed" << std::endl;
                    }
                }
            }
    }

    void sendData(int socket, const char* data, int dataLen, const sockaddr_in& clientAddr) {
        if (dataLen > 0) {
            ssize_t sentBytes = 0;
            while (sentBytes < dataLen) {
                ssize_t sendNum = sendto(socket, data + sentBytes, dataLen - sentBytes, 0,
                                         (struct sockaddr*)&clientAddr, sizeof(clientAddr));
                if (sendNum < 0) {
                    std::cout << "Send data failure: " << strerror(errno) << std::endl;
                    return;
                }
                sentBytes += sendNum;
            }
        }
    }

private:
    std::unique_ptr<GamepadPacketTap> tap_;
    int mFd = -1;
    std::atomic<bool> mIsRunning_{false};
    std::thread mServerThread_;

    GamepadInterface* mGamepadInterface_ = nullptr;
};
