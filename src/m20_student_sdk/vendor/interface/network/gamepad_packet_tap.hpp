#pragma once

// AOS: observe copies of incoming gamepad packets without owning vendor UDP.
// SOCK_DGRAM/AF_PACKET delivers the IPv4 header (no Ethernet header).
#include <arpa/inet.h>
#include <linux/filter.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

class GamepadPacketTap {
public:
    static unsigned Sum(const unsigned char* p, size_t n, unsigned sum = 0) {
        while (n >= 2) { sum += (unsigned(p[0]) << 8) | p[1]; p += 2; n -= 2; }
        if (n) sum += unsigned(p[0]) << 8;
        while (sum >> 16) sum = (sum & 0xffff) + (sum >> 16);
        return sum;
    }
    static bool Decode(const char* data, size_t size, in_addr local,
                       sockaddr_in& peer, size_t& offset, size_t& length) {
        const auto* p = reinterpret_cast<const unsigned char*>(data);
        if (size < 28 || p[0] >> 4 != 4 || p[9] != IPPROTO_UDP) return false;
        const size_t ihl = (p[0] & 15) * 4;
        const size_t total = (size_t(p[2]) << 8) | p[3];
        if (ihl < 20 || total > size || total < ihl + 8 ||
            (p[6] & 0x3f) || p[7] || std::memcmp(p + 16, &local, 4)) return false;
        const size_t udp_length = (size_t(p[ihl + 4]) << 8) | p[ihl + 5];
        if (p[ihl + 2] != (30000 >> 8) || p[ihl + 3] != (30000 & 255) ||
            udp_length < 8 || ihl + udp_length != total) return false;
        // AF_PACKET runs before the IP/UDP receive stack. Match its corruption
        // checks ourselves. Zero UDP checksum is legal for IPv4. Never trust
        // an unfinished checksum from TX/loopback offload: reject it unless
        // the bytes themselves validate; physical inbound frames are complete.
        if (Sum(p, ihl) != 0xffff) return false;
        if ((p[ihl + 6] || p[ihl + 7]) &&
            Sum(p + ihl, udp_length, Sum(p + 12, 8, IPPROTO_UDP + udp_length)) != 0xffff)
            return false;
        peer = {};
        peer.sin_family = AF_INET;
        std::memcpy(&peer.sin_addr, p + 12, 4);
        std::memcpy(&peer.sin_port, p + ihl, 2);
        offset = ihl + 8;
        length = udp_length - 8;
        return true;
    }

    explicit GamepadPacketTap(const std::string& interface) {
        try {
            if (interface.empty() || interface.size() >= IFNAMSIZ)
                throw std::runtime_error("invalid gamepad capture interface");
            int query = ::socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
            if (query < 0) throw std::runtime_error("interface query socket failed");
            ifreq request{};
            std::strncpy(request.ifr_name, interface.c_str(), IFNAMSIZ - 1);
            const int result = ioctl(query, SIOCGIFADDR, &request);
            ::close(query);
            if (result < 0) throw std::runtime_error("gamepad interface has no IPv4");
            local_ = reinterpret_cast<sockaddr_in*>(&request.ifr_addr)->sin_addr;
            // Protocol zero: receive nothing until the filter is installed.
            fd_ = ::socket(AF_PACKET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
            if (fd_ < 0) throw std::runtime_error("gamepad packet tap requires CAP_NET_RAW");
            sock_filter code[] = {
                BPF_STMT(BPF_LD | BPF_B | BPF_ABS, 9),
                BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, IPPROTO_UDP, 0, 8),
                BPF_STMT(BPF_LD | BPF_W | BPF_ABS, 16),
                BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, ntohl(local_.s_addr), 0, 6),
                BPF_STMT(BPF_LD | BPF_H | BPF_ABS, 6),
                BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, 0x3fff, 4, 0),
                BPF_STMT(BPF_LDX | BPF_B | BPF_MSH, 0),
                BPF_STMT(BPF_LD | BPF_H | BPF_IND, 2),
                BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 30000, 0, 1),
                BPF_STMT(BPF_RET | BPF_K, 65535),
                BPF_STMT(BPF_RET | BPF_K, 0),
            };
            sock_fprog program{static_cast<unsigned short>(sizeof(code) / sizeof(code[0])), code};
            int enabled = 1, buffer = 64 * 1024;
            timeval timeout{0, 200000};
            if (setsockopt(fd_, SOL_SOCKET, SO_ATTACH_FILTER, &program, sizeof(program)) ||
                setsockopt(fd_, SOL_SOCKET, SO_TIMESTAMPNS, &enabled, sizeof(enabled)) ||
                setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &buffer, sizeof(buffer)) ||
                setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)))
                throw std::runtime_error("gamepad tap filter/timestamp setup failed");
            sockaddr_ll address{};
            address.sll_family = AF_PACKET;
            address.sll_protocol = htons(ETH_P_IP);
            address.sll_ifindex = if_nametoindex(interface.c_str());
            if (!address.sll_ifindex || ::bind(fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)))
                throw std::runtime_error("gamepad tap interface bind failed");
        } catch (...) { if (fd_ >= 0) ::close(fd_); fd_ = -1; throw; }
    }
    ~GamepadPacketTap() { if (fd_ >= 0) ::close(fd_); }
    GamepadPacketTap(const GamepadPacketTap&) = delete;
    GamepadPacketTap& operator=(const GamepadPacketTap&) = delete;
    int fd() const { return fd_; }
    in_addr local() const { return local_; }

    // Only call for the SDK private key acknowledgement. Vendor owns all
    // heartbeat/status responses. No UDP bind, no duplicate heartbeat sender.
    bool Reply(const std::string& payload, const sockaddr_in& peer) const {
        if (payload.size() > 65507) return false;
        int output = ::socket(AF_INET, SOCK_RAW | SOCK_CLOEXEC, IPPROTO_UDP);
        if (output < 0) return false;
        sockaddr_in source{};
        source.sin_family = AF_INET;
        source.sin_addr = local_;
        bool ok = ::bind(output, reinterpret_cast<sockaddr*>(&source), sizeof(source)) == 0;
        std::vector<unsigned char> packet(8 + payload.size(), 0);
        uint16_t port = htons(30000), length = htons(packet.size());
        std::memcpy(packet.data(), &port, 2);
        std::memcpy(packet.data() + 2, &peer.sin_port, 2);
        std::memcpy(packet.data() + 4, &length, 2);
        std::memcpy(packet.data() + 8, payload.data(), payload.size());
        sockaddr_in target = peer;
        target.sin_port = 0; // RAW socket uses IPPROTO_UDP; UDP port is in header.
        if (ok) ok = sendto(output, packet.data(), packet.size(), 0,
                           reinterpret_cast<sockaddr*>(&target), sizeof(target)) == ssize_t(packet.size());
        ::close(output);
        return ok;
    }
private:
    int fd_ = -1;
    in_addr local_{};
};
