
#pragma once

#include <Eigen/Dense>
#include "cts_feedback.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <deque>
#include <map>
#include <cmath>
#include <memory>
#include <thread>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <sys/timerfd.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <stdio.h>
#include <csignal>
#include <pthread.h>
#include <mutex>
#include <limits>


namespace types {
using Vec3f = Eigen::Vector3f;
using Vec3d = Eigen::Vector3d;
using Vec4f = Eigen::Vector4f;
using Vec4d = Eigen::Vector4d;
using VecXf = Eigen::VectorXf;
using VecXd = Eigen::VectorXd;

using Mat3f = Eigen::Matrix3f;
using Mat3d = Eigen::Matrix3d;
using MatXf = Eigen::MatrixXf;
using MatXd = Eigen::MatrixXd;

const float gravity = 9.815;

struct RobotBasicState {
    m20_cts::LandingFeedback landing_feedback;
    double body_receive_age_s = std::numeric_limits<double>::infinity();
    double snapshot_steady_s = std::numeric_limits<double>::quiet_NaN();
    Vec3f base_rpy;
    Vec4f base_quat;
    Mat3f base_rot_mat;
    Vec3f base_omega;
    Vec3f base_acc;
    MatXf flt_base_acc_mat;
    VecXf joint_pos;
    VecXf joint_vel;
    VecXf joint_tau;
    VecXf lidar_observation;
    bool lidar_valid;
    float lidar_age_seconds;
    VecXf depth_observation;
    bool depth_valid;
    float depth_age_seconds;
    VecXf v11_observation;
    bool v11_valid;
    float v11_age_seconds;

    RobotBasicState(int dof = 16) {
        base_rpy.setZero();
        base_quat.setZero();
        base_rot_mat.setIdentity();
        base_omega.setZero();
        base_acc.setZero();
        flt_base_acc_mat = Eigen::MatrixXf::Zero(20, 3);
        joint_pos = VecXf::Zero(dof);
        joint_vel = VecXf::Zero(dof);
        joint_tau = VecXf::Zero(dof);
        lidar_observation.resize(0);
        lidar_valid = false;
        lidar_age_seconds = std::numeric_limits<float>::infinity();
        depth_observation.resize(0);
        depth_valid = false;
        depth_age_seconds = std::numeric_limits<float>::infinity();
        v11_observation.resize(0);
        v11_valid = false;
        v11_age_seconds = std::numeric_limits<float>::infinity();
    }
};

struct RobotAction {
    VecXf goal_joint_pos;
    VecXf goal_joint_vel;
    VecXf kp;
    VecXf kd;
    VecXf tau_ff;

    MatXf ConvertToMat() const {
        MatXf res(goal_joint_pos.rows(), 5);
        res.col(0) = kp;
        res.col(1) = goal_joint_pos;
        res.col(2) = kd;
        res.col(3) = goal_joint_vel;
        res.col(4) = tau_ff;
        return res;
    }
};


struct UserCommand {
    std::atomic<double> time_stamp{0.0};
    std::atomic<uint8_t> safe_control_mode{0}; //0 normal, 1 stand, 2 sit, 3 joint damping
    std::atomic<uint8_t> target_mode{0};
    std::atomic<float> forward_vel_scale{0.0f};
    std::atomic<float> side_vel_scale{0.0f};
    std::atomic<float> turnning_vel_scale{0.0f};
    std::atomic<float> reserved_scale{0.0f};

    void Reset() {
        time_stamp.store(0.0);
        safe_control_mode.store(0);
        target_mode.store(0);
        forward_vel_scale.store(0.0f);
        side_vel_scale.store(0.0f);
        turnning_vel_scale.store(0.0f);
        reserved_scale.store(0.0f);
    }
};

enum FromType {
    inside = 0,
    user,
};
struct NetInfo {
    std::string ip;
    int port;
    FromType fromType;
};

struct DataInfo : public NetInfo {
    const char* buffer;
    ssize_t len;
    std::chrono::steady_clock::time_point command_deadline{};
};
struct CmdInfo : public NetInfo {
    int type;
    int command;
    std::chrono::steady_clock::time_point command_deadline{};
};
};  // namespace types
