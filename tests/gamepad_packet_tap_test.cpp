// Standalone: no ROS, policy, motor publishers, or real gamepad commands.
#include "gamepad_packet_tap.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <sys/poll.h>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static bool readable(int fd) { pollfd p{fd, POLLIN, 0}; return poll(&p, 1, 200) > 0; }
static int receiver(bool reuse) {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    int yes = 1;
    if (reuse) setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(30000);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    require(bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0, "test port occupied");
    return fd;
}
int main(int argc, char** argv) {
    try {
        // Optional read-only capture on robot interface: no replies/ROS publisher.
        if ((argc == 3 || argc == 4) && std::string(argv[1]) == "--observe") {
            GamepadPacketTap tap(argv[2]);
            int seconds = argc == 4 ? std::stoi(argv[3]) : 8;
            require(seconds > 0 && seconds <= 600, "invalid observation duration");
            auto end = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
            int valid = 0;
            while (std::chrono::steady_clock::now() < end) {
                char buf[65536]; sockaddr_ll address{};
                socklen_t n = sizeof(address);
                auto count = recvfrom(tap.fd(), buf, sizeof(buf), 0, reinterpret_cast<sockaddr*>(&address), &n);
                sockaddr_in peer{}; size_t offset, length;
                if (count > 0 && address.sll_pkttype == PACKET_HOST &&
                    GamepadPacketTap::Decode(buf, count, tap.local(), peer, offset, length)) {
                    ++valid;
                    if (valid <= 3) std::cout << "source=" << inet_ntoa(peer.sin_addr) << ":" << ntohs(peer.sin_port) << " bytes=" << length << std::endl;
                }
            }
            std::cout << "passive_packets=" << valid << " no_joint_commands=true no_replies=true\n";
            return valid > 0 ? 0 : 2;
        }
        char current_ns[128]{},init_ns[128]{};
        auto current_size=readlink("/proc/self/ns/net",current_ns,sizeof(current_ns));
        auto init_size=readlink("/proc/1/ns/net",init_ns,sizeof(init_ns));
        require(current_size>0&&init_size>0&&
                (current_size!=init_size||std::memcmp(current_ns,init_ns,current_size)!=0),
                "offline packet test requires an isolated network namespace; never run it on the robot network");
        int vendor = receiver(true), candidate = receiver(true);
        int sender = socket(AF_INET, SOCK_DGRAM, 0);
        sockaddr_in dst{}; dst.sin_family = AF_INET; dst.sin_port = htons(30000);
        dst.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        int no_checksum = 1; setsockopt(sender, SOL_SOCKET, SO_NO_CHECK, &no_checksum, sizeof(no_checksum));
        auto send = [&] { require(sendto(sender, "tap-test", 8, 0, reinterpret_cast<sockaddr*>(&dst), sizeof(dst)) == 8, "send"); };
        send();
        require(readable(candidate) && !readable(vendor), "old duplicate-bind conflict not reproduced");
        std::cout << "old_duplicate_bind: vendor=0 candidate=1\n";
        close(candidate); close(vendor);
        vendor = receiver(false);
        GamepadPacketTap tap("lo");
        send();
        require(readable(vendor) && readable(tap.fd()), "passive capture stole vendor data");
        char buffer[65536]; sockaddr_ll address{};
        alignas(cmsghdr) char control[CMSG_SPACE(sizeof(timespec))]{};
        iovec io{buffer, sizeof(buffer)}; msghdr msg{};
        msg.msg_name = &address; msg.msg_namelen = sizeof(address);
        msg.msg_iov = &io; msg.msg_iovlen = 1; msg.msg_control = control; msg.msg_controllen = sizeof(control);
        ssize_t count = recvmsg(tap.fd(), &msg, 0);
        bool timestamp = false;
        for (auto* c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c))
            timestamp |= c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_TIMESTAMPNS;
        require(timestamp && count > 0, "missing original kernel timestamp");
        sockaddr_in peer{}; size_t offset, length;
        require(GamepadPacketTap::Decode(buffer, count, tap.local(), peer, offset, length), "decode");
        require(length == 8 && std::string(buffer + offset, length) == "tap-test", "payload changed");
        auto reject = [&](size_t index, unsigned char value) {
            auto original = buffer[index]; buffer[index] = value;
            sockaddr_in unused{}; size_t off, len;
            require(!GamepadPacketTap::Decode(buffer, count, tap.local(), unused, off, len), "malformed packet accepted");
            buffer[index] = original;
        };
        reject(10, static_cast<unsigned char>(buffer[10]) ^ 1); // corrupt IP checksum
        reject(26, 1); // nonzero incorrect UDP checksum
        reject(0, 0x44); // invalid IHL
        reject(6, 0x20); // fragmented IP
        reject(9, 6); // TCP
        reject(16, 1); // wrong destination
        reject(22, 0); // wrong destination UDP port
        reject(25, 7); // impossible UDP length
        const auto* bytes = reinterpret_cast<const unsigned char*>(buffer);
        unsigned checksum = (~GamepadPacketTap::Sum(bytes + 20, 16,
            GamepadPacketTap::Sum(bytes + 12, 8, IPPROTO_UDP + 16))) & 0xffff;
        if (!checksum) checksum = 0xffff;
        buffer[26] = checksum >> 8; buffer[27] = checksum & 255;
        sockaddr_in unused{}; size_t off, len;
        require(GamepadPacketTap::Decode(buffer, count, tap.local(), unused, off, len), "valid UDP checksum rejected");
        reject(28, static_cast<unsigned char>(buffer[28]) ^ 1); // corruption with unchanged checksum
        require(!GamepadPacketTap::Decode(buffer, 27, tap.local(), unused, off, len), "truncated packet accepted");
        require(tap.Reply("ack-test", peer), "raw private reply");
        require(readable(sender), "reply missing");
        sockaddr_in reply{}; socklen_t reply_size = sizeof(reply);
        auto n = recvfrom(sender, buffer, sizeof(buffer), 0, reinterpret_cast<sockaddr*>(&reply), &reply_size);
        require(n == 8 && ntohs(reply.sin_port) == 30000 && reply.sin_addr.s_addr == dst.sin_addr.s_addr, "reply source changed");
        std::cout << "passive: vendor=1 tap=1 kernel_timestamp=1 reply_source=30000\n";
        close(vendor); close(sender);
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
