/**
 * @file dds_hardware_interface.hpp
 * @brief hardware interface for dds
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */

#pragma once


#include "common_types.h"
#include "robot_interface.h"
#include "dds_types.h"
#include "drdds/msg/joints_data.hpp"
#include "drdds/msg/joints_data_cmd.hpp"
#include "drdds/msg/imu_data.hpp"
#include "drdds/msg/battery_data.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>
#include <thread>
#include <unistd.h>
#include "command_ownership.hpp"

using namespace dds;

struct JointConfig {
    float dir;
    float offset;
};


class DdsInterface : public RobotInterface{
protected:
    std::atomic<double> ri_ts_{0.0}, imu_ts_{0.0};
    Vec3f omega_body_, rpy_, acc_;
    VecXf joint_pos_, joint_vel_, joint_tau_;
    VecXf motor_temperture_, driver_temperture_;
    float *pos_offset_;
    float *joint_dir_;
    bool *data_updated_;
    uint16_t *control_word;

    JointConfig *joint_config_;
    std::vector<uint16_t> driver_status_;
    std::vector<uint16_t> joint_data_id_;
    BatteryInfo_dds battery_info_[2];
    std::vector<uint16_t> battery_data_;

    // DDS callbacks run in the ROS executor while the control and safety loops
    // read the same state from their own threads.  Keep those snapshots free
    // from C++ data races; on aarch64 an Eigen copy racing a callback can
    // otherwise produce a torn state and trip the real-robot safety latch.
    mutable std::mutex joint_state_mutex_;
    mutable std::mutex imu_state_mutex_;
    mutable std::mutex health_state_mutex_;

    double time_stamp_;
    float remain_bat;

    rclcpp::Publisher<drdds::msg::JointsDataCmd>::SharedPtr joint_cmd_pub_;
    rclcpp::Subscription<drdds::msg::JointsDataCmd>::SharedPtr command_monitor_;
    rclcpp::TimerBase::SharedPtr command_owner_timer_;
    std::atomic<bool> foreign_command_seen_{false}, command_authority_lost_{false};
    std::atomic<bool> command_output_started_{false};
    std::mutex command_identity_mutex_;
    std::array<uint8_t,RMW_GID_STORAGE_SIZE> startup_data_gid_{};
    std::vector<std::array<uint8_t,RMW_GID_STORAGE_SIZE>> startup_command_gids_;
    void CheckCommandAuthority(bool initializing=false){
        std::lock_guard<std::mutex> lock(command_identity_mutex_);
        if(foreign_command_seen_.load()||command_authority_lost_.load())throw std::runtime_error("external joint command observed; control ownership lost");
        auto data=node_->get_publishers_info_by_topic("/JOINTS_DATA");
        auto commands=node_->get_publishers_info_by_topic("/JOINTS_CMD");
        if(data.size()!=1)throw std::runtime_error("JOINTS_DATA source is ambiguous");
        const char* verified_pid=std::getenv("M20_VERIFIED_VENDOR_PID");
        const int pid=verified_pid?std::atoi(verified_pid):0;
        const auto& prefix=data.front().endpoint_gid();
        if(pid<=0||pid>=65536||data.front().node_name()!="_CREATED_BY_BARE_DDS_APP_"||
           prefix[4]!=(pid&255)||prefix[5]!=((pid>>8)&255))throw std::runtime_error("data endpoint does not match verified vendor worker");
        const auto own=joint_cmd_pub_->get_gid();size_t external=0,self=0;
        for(const auto& p:commands){
            if(std::memcmp(p.endpoint_gid().data(),own.data,RMW_GID_STORAGE_SIZE)==0){++self;continue;}
            ++external;
            if(!official_command_identity(p.node_name(),p.endpoint_gid().data(),data.front().endpoint_gid().data()))throw std::runtime_error("unknown external command endpoint");
        }
        if(self!=1||external>1)throw std::runtime_error("joint command endpoint set is not exclusive");
        std::vector<std::array<uint8_t,RMW_GID_STORAGE_SIZE>> ids;
        for(const auto& p:commands)ids.push_back(p.endpoint_gid());
        if(initializing){startup_data_gid_=prefix;startup_command_gids_=ids;}
        else if(startup_data_gid_!=prefix||!same_endpoint_set(ids,startup_command_gids_))throw std::runtime_error("joint endpoint identities changed after SDK admission");
    }
    rclcpp::Subscription<drdds::msg::JointsData>::SharedPtr joint_data_sub_;
    rclcpp::Subscription<drdds::msg::ImuData>::SharedPtr imu_data_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sensor_imu_data_sub_;
    rclcpp::CallbackGroup::SharedPtr joint_state_callback_group_;
    rclcpp::CallbackGroup::SharedPtr imu_state_callback_group_;
    rclcpp::Subscription<drdds::msg::BatteryData>::SharedPtr health_data_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr lidar_observation_sub_;
    VecXf lidar_observation_;
    bool lidar_observation_valid_ = false;
    std::chrono::steady_clock::time_point lidar_last_update_;
    float lidar_timeout_seconds_ = 0.30f;
    std::mutex lidar_mutex_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr depth_observation_sub_;
    VecXf depth_observation_;
    bool depth_observation_valid_ = false;
    std::chrono::steady_clock::time_point depth_last_update_;
    float depth_timeout_seconds_ = 0.30f;
    std::mutex depth_mutex_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr v11_observation_sub_;
    VecXf v11_observation_;
    bool v11_observation_valid_ = false;
    std::chrono::steady_clock::time_point v11_last_update_;
    float v11_timeout_seconds_ = 0.30f;
    std::mutex v11_mutex_;
    void WaitForStableRobotState() {
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(30);
        auto last_joint_update = std::chrono::steady_clock::time_point{};
        auto last_imu_update = std::chrono::steady_clock::time_point{};
        auto stable_since = std::chrono::steady_clock::time_point{};
        double last_joint_stamp = 0.0;
        double last_imu_stamp = 0.0;
        while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
            rclcpp::spin_some(node_);
            const auto now = std::chrono::steady_clock::now();
            const double joint_stamp = GetInterfaceTimeStamp();
            const double imu_stamp = GetImuTimestamp();
            if (joint_stamp > 0.0 && joint_stamp != last_joint_stamp) {
                last_joint_stamp = joint_stamp;
                last_joint_update = now;
            }
            if (imu_stamp > 0.0 && imu_stamp != last_imu_stamp) {
                last_imu_stamp = imu_stamp;
                last_imu_update = now;
            }
            const bool fresh = last_joint_update !=
                    std::chrono::steady_clock::time_point{} &&
                last_imu_update != std::chrono::steady_clock::time_point{} &&
                now - last_joint_update < std::chrono::milliseconds(50) &&
                now - last_imu_update < std::chrono::milliseconds(50);
            if (fresh) {
                if (stable_since == std::chrono::steady_clock::time_point{}) {
                    stable_since = now;
                } else if (now - stable_since >= std::chrono::seconds(2)) {
                    std::cout << "Robot state stable for 2 seconds; "
                              << "hardware initialization may continue." << std::endl;
                    return;
                }
            } else {
                stable_since = {};
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        throw std::runtime_error(
            "joint/IMU sources did not remain stable for 2 seconds");
    }


    bool IsDataUpdatedFinished() {
        bool res = data_updated_[0];
        for (int i = 0; i < dof_num_; ++i) {
            res = res & data_updated_[i];
        }
        return res;
    }

    void ResetPositionOffset() {
        memset(data_updated_, 0, dof_num_ * sizeof(bool));
        this->SetJointCommand(MatXf::Zero(dof_num_, 5));
        usleep(100 * 1000);
        VecXf last_joint_pos = this->GetJointPosition();
        VecXf current_joint_pos = this->GetJointPosition();
        int cnt = 0;
        while (!IsDataUpdatedFinished()) {
            ++cnt;
            this->RefreshRobotData();
            rclcpp::spin_some(this->get_node());
            current_joint_pos = this->GetJointPosition();

            for (int i = 0; i < dof_num_; ++i) {
                if (!data_updated_[i] && current_joint_pos(i) != last_joint_pos(i)) {
                    data_updated_[i] = true;
                }
            }
            last_joint_pos = current_joint_pos;
            usleep(1000);
            if (cnt == 10000) {
                std::cout << data_updated_ << std::endl;
                std::cout << "joint data update is not finished\n";
            }
        }
        for (int i = 1; i < dof_num_; i += 4) {
            if (current_joint_pos(i) < -210. / 180. * M_PI) {
                pos_offset_[i] = pos_offset_[i] + 360.;
                std::cout << "joint " << i << " offset is changed to " << pos_offset_[i] << "\n";
            } else if (current_joint_pos(i) > 90. / 180. * M_PI) {
                pos_offset_[i] = pos_offset_[i] - 360.;
                std::cout << "joint " << i << " offset is changed to " << pos_offset_[i] << "\n";
            }
        }
        for (int i = 0; i < dof_num_; ++i) {
            joint_config_[i].dir = joint_dir_[i];
            joint_config_[i].offset = Deg2Rad(pos_offset_[i]);
        }
    }

public:
    int run_cnt_ = 0;

    DdsInterface(const std::string &robot_name, const int &dof_num) : RobotInterface(robot_name, dof_num) {
        std::cout << robot_name << " is using Ecan Hardware Interface" << std::endl;
        ri_ts_.store(0.0, std::memory_order_relaxed);
        imu_ts_.store(0.0, std::memory_order_relaxed);
        omega_body_.setZero();
        rpy_.setZero();
        acc_.setZero();
        joint_pos_ = VecXf::Zero(dof_num_);
        joint_vel_ = VecXf::Zero(dof_num_);
        joint_tau_ = VecXf::Zero(dof_num_);
        motor_temperture_ = VecXf::Zero(dof_num_);
        driver_temperture_ = VecXf::Zero(dof_num_);

        pos_offset_ = new float[dof_num_];
        joint_dir_ = new float[dof_num_];
        data_updated_ = new bool[dof_num_];
        joint_config_ = new JointConfig[dof_num_];
        control_word = new uint16_t[4];

        driver_status_.resize(dof_num_);
        joint_data_id_.resize(dof_num_);
        battery_data_.resize(BATTERY_DATA_SIZE);

        joint_cmd_pub_ = node_->create_publisher<drdds::msg::JointsDataCmd>("/JOINTS_CMD", 10);
        command_monitor_=node_->create_subscription<drdds::msg::JointsDataCmd>("/JOINTS_CMD",rclcpp::SensorDataQoS(),
            [this](const drdds::msg::JointsDataCmd::SharedPtr,const rclcpp::MessageInfo& info){
                auto own=joint_cmd_pub_->get_gid();
                if(std::memcmp(info.get_rmw_message_info().publisher_gid.data,own.data,RMW_GID_STORAGE_SIZE)!=0){
                    foreign_command_seen_.store(true);
                    if(!command_authority_lost_.exchange(true))RCLCPP_ERROR(node_->get_logger(),"External JOINTS_CMD received; student output latched off. Operator must restore ownership.");
                }
            });
        command_owner_timer_=node_->create_wall_timer(std::chrono::milliseconds(100),[this](){
            // Discovery is checked before SDK initialization as well. Runtime
            // graph changes latch output off; never stop a vendor service here.
            if(!command_output_started_.load())return;
            try{CheckCommandAuthority();}catch(const std::exception& e){
                if(!command_authority_lost_.exchange(true))RCLCPP_ERROR(node_->get_logger(),"Student output disabled: %s",e.what());
            }
        });
        // Joint state is high-rate, latest-value data.  A reliable depth-10
        // reader can back-pressure the board's bare-DDS writer during a short
        // NX scheduling pause, which then starves both joint and IMU streams.
        // Best-effort keep-last avoids that feedback loop; the safety watchdog
        // still rejects an actual loss of fresh samples.
        auto joint_state_qos = rclcpp::QoS(rclcpp::KeepLast(1))
                                   .best_effort().durability_volatile();
        // A single executor lane can be starved by the 200 Hz joint
        // stream on NX.  Keep joint and IMU callbacks in separate lanes so
        // neither can cross the 200 ms safety deadline while both are active.
        joint_state_callback_group_ = node_->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        imu_state_callback_group_ = node_->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions joint_options;
        joint_options.callback_group = joint_state_callback_group_;
        rclcpp::SubscriptionOptions imu_options;
        imu_options.callback_group = imu_state_callback_group_;
        joint_data_sub_ = node_->create_subscription<drdds::msg::JointsData>("/JOINTS_DATA", joint_state_qos,
                                    std::bind(&DdsInterface::Handler, this,std::placeholders::_1),
                                    joint_options);
        const char* imu_source_env = std::getenv("M20_IMU_SOURCE");
        const std::string imu_source = imu_source_env == nullptr ? "bridge" : imu_source_env;
        if (imu_source == "bridge" || imu_source == "simulation" ||
            imu_source == "native") {
            // NX consumes its release-local /M20_IMU_BRIDGE; AOS and simulation
            // consume /IMU_DATA directly. RPY is degrees on either wire and
            // HandlerIMU converts it exactly once for the actor.
            const std::string imu_topic =
                imu_source == "bridge" ? "/M20_IMU_BRIDGE" : "/IMU_DATA";
            imu_data_sub_ = node_->create_subscription<drdds::msg::ImuData>(
                imu_topic, joint_state_qos,
                std::bind(&DdsInterface::HandlerIMU, this, std::placeholders::_1),
                imu_options);
            std::cout << "IMU source: "
                      << (imu_source == "bridge" ? "NX bridge /M20_IMU_BRIDGE"
                          : (imu_source == "native" ? "AOS native /IMU_DATA"
                                                    : "simulation /IMU_DATA"))
                      << std::endl;
        } else if (imu_source == "direct") {
            sensor_imu_data_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
                "/IMU", joint_state_qos,
                std::bind(&DdsInterface::HandlerSensorIMU, this, std::placeholders::_1),
                imu_options);
            std::cout << "IMU source: direct /IMU (diagnostic fallback)" << std::endl;
        } else {
            throw std::runtime_error(
                "M20_IMU_SOURCE must be bridge, native, simulation, or direct");
        }
        health_data_sub_ = node_->create_subscription<drdds::msg::BatteryData>("/BATTERY_DATA", 10,
                                    std::bind(&DdsInterface::HandlerHealth, this, std::placeholders::_1));
        lidar_observation_sub_ = node_->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/M20_LIDAR_OBS", 10,
            std::bind(&DdsInterface::HandlerLidarObservation, this, std::placeholders::_1));
        const char* depth_source = std::getenv("M20_DEPTH_SOURCE");
        const bool direct_realsense =
            depth_source != nullptr && std::string(depth_source) == "realsense";
        if (!direct_realsense) {
            auto depth_qos = rclcpp::QoS(rclcpp::KeepLast(2)).best_effort().durability_volatile();
            depth_observation_sub_ = node_->create_subscription<std_msgs::msg::Float32MultiArray>(
                "/M20_DEPTH_OBS", depth_qos,
                std::bind(&DdsInterface::HandlerDepthObservation, this, std::placeholders::_1));
        } else {
            std::cout << "Depth ROS subscription disabled: using in-process librealsense2"
                      << std::endl;
        }
        const auto v11_observation_qos = rclcpp::QoS(rclcpp::KeepLast(2))
                                             .best_effort().durability_volatile();
        v11_observation_sub_ = node_->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/M20_V11_OBS", v11_observation_qos,
            std::bind(&DdsInterface::HandlerV11Observation, this, std::placeholders::_1));

        if (const char* timeout = std::getenv("M20_LIDAR_TIMEOUT_S")) {
            try {
                const float configured_timeout = std::stof(timeout);
                if (configured_timeout > 0.0f && configured_timeout <= 0.5f) {
                    lidar_timeout_seconds_ = configured_timeout;
                } else {
                    std::cerr << "Ignoring M20_LIDAR_TIMEOUT_S outside (0, 0.5]: "
                              << timeout << std::endl;
                }
            } catch (const std::exception&) {
                std::cerr << "Ignoring invalid M20_LIDAR_TIMEOUT_S: " << timeout << std::endl;
            }
        }
        if (const char* timeout = std::getenv("M20_DEPTH_TIMEOUT_S")) {
            try {
                const float configured_timeout = std::stof(timeout);
                if (configured_timeout > 0.0f && configured_timeout <= 1.0f) {
                    depth_timeout_seconds_ = configured_timeout;
                } else {
                    std::cerr << "Ignoring M20_DEPTH_TIMEOUT_S outside (0, 1]: "
                              << timeout << std::endl;
                }
            } catch (const std::exception&) {
                std::cerr << "Ignoring invalid M20_DEPTH_TIMEOUT_S: " << timeout << std::endl;
            }
        }
        
        control_word[0] = kIndexDisable;
        control_word[1] = kIndexErrorReset;
        control_word[2] = kIndexEnable;
        control_word[3] = kIndexGetStatusWord;
    }

    virtual ~DdsInterface() {
        delete[] pos_offset_;
        delete[] joint_dir_;
        delete[] data_updated_;
        delete[] joint_config_;
    }

    virtual void Start() {
        // Start() runs after policy/controller construction but before the
        // safety watchdog.  Let this long-lived DDS participant finish
        // discovery and require stable state before any command is published.
        WaitForStableRobotState();
        ResetJointError();
    }

    virtual void Stop() {
    }

    virtual double GetInterfaceTimeStamp() {
        return ri_ts_.load(std::memory_order_acquire);
    }

    virtual VecXf GetJointPosition() {
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        return joint_pos_;
    }

    virtual VecXf GetJointVelocity() {
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        return joint_vel_;
    }

    virtual VecXf GetJointTorque() {
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        return joint_tau_;
    }

    virtual Vec3f GetImuRpy() {
        std::lock_guard<std::mutex> lock(imu_state_mutex_);
        return rpy_;
    }

    virtual Vec3f GetImuAcc() {
        std::lock_guard<std::mutex> lock(imu_state_mutex_);
        return acc_;
    }

    virtual Vec3f GetImuOmega() {
        std::lock_guard<std::mutex> lock(imu_state_mutex_);
        return omega_body_;
    }

    virtual VecXf GetContactForce() {
        return VecXf::Zero(4);
    }

    virtual VecXf GetLidarObservation() {
        std::lock_guard<std::mutex> lock(lidar_mutex_);
        return lidar_observation_;
    }

    virtual bool IsLidarObservationValid() {
        std::lock_guard<std::mutex> lock(lidar_mutex_);
        if (!lidar_observation_valid_) {
            return false;
        }
        const float age = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - lidar_last_update_).count();
        return age <= lidar_timeout_seconds_;
    }

    virtual float GetLidarObservationAgeSeconds() {
        std::lock_guard<std::mutex> lock(lidar_mutex_);
        if (!lidar_observation_valid_) {
            return std::numeric_limits<float>::infinity();
        }
        return std::chrono::duration<float>(
            std::chrono::steady_clock::now() - lidar_last_update_).count();
    }

    virtual VecXf GetDepthObservation() {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        return depth_observation_;
    }

    virtual bool IsDepthObservationValid() {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        if (!depth_observation_valid_) {
            return false;
        }
        const float age = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - depth_last_update_).count();
        return age <= depth_timeout_seconds_;
    }

    virtual float GetDepthObservationAgeSeconds() {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        if (!depth_observation_valid_) {
            return std::numeric_limits<float>::infinity();
        }
        return std::chrono::duration<float>(
            std::chrono::steady_clock::now() - depth_last_update_).count();
    }

    virtual VecXf GetV11Observation() {
        std::lock_guard<std::mutex> lock(v11_mutex_);
        return v11_observation_;
    }

    virtual bool IsV11ObservationValid() {
        std::lock_guard<std::mutex> lock(v11_mutex_);
        if (!v11_observation_valid_) {
            return false;
        }
        return std::chrono::duration<float>(
            std::chrono::steady_clock::now() - v11_last_update_).count() <= v11_timeout_seconds_;
    }

    virtual float GetV11ObservationAgeSeconds() {
        std::lock_guard<std::mutex> lock(v11_mutex_);
        if (!v11_observation_valid_) {
            return std::numeric_limits<float>::infinity();
        }
        return std::chrono::duration<float>(
            std::chrono::steady_clock::now() - v11_last_update_).count();
    }

    virtual void ResetJointError() {
        auto msg = drdds::msg::JointsDataCmd();

        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < dof_num_; ++i) {
                msg.data.joints_data[i].position = 0;
                msg.data.joints_data[i].velocity = 0;
                msg.data.joints_data[i].torque = 0;
                msg.data.joints_data[i].kp = 0;
                msg.data.joints_data[i].kd = 0;
                msg.data.joints_data[i].control_word = control_word[j];
            }
            std::cout << "ResetJointError publish!! \n" << std::endl;
            joint_cmd_pub_->publish(msg);
            usleep(10 * 1000);
        }
    }

    virtual void SetJointCommand(Eigen::Matrix<float, Eigen::Dynamic, 5> input) {
        joint_cmd_ = input;
        auto msg = drdds::msg::JointsDataCmd();

        for (int i = 0; i < dof_num_; ++i) {
            msg.data.joints_data[i].position =
                    (input(i, 1) - joint_config_[i].offset) * joint_config_[i].dir;
            msg.data.joints_data[i].velocity = input(i, 3) * joint_dir_[i];
            msg.data.joints_data[i].torque = input(i, 4) * joint_dir_[i];
            msg.data.joints_data[i].kp = input(i, 0);
            msg.data.joints_data[i].kd = input(i, 2);
            msg.data.joints_data[i].control_word = kIndexMotorControl;
        }
        std::cout << "SetJointCommand publish!! \n" << std::endl;
        joint_cmd_pub_->publish(msg);
    }

    virtual VecXf GetMotorTemperture() {
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        return motor_temperture_;
    }

    virtual VecXf GetDriverTemperture() {
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        return driver_temperture_;
    }

    virtual double GetImuTimestamp() {
        return imu_ts_.load(std::memory_order_acquire);
    }

    virtual std::vector<uint16_t> GetBatteryData() {
        std::lock_guard<std::mutex> lock(health_state_mutex_);
        return battery_data_;
    }

    virtual std::vector<uint16_t> GetDriverStatusWord() {
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        return driver_status_;
    }

    virtual std::vector<uint16_t> GetJointDataID() {
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        return joint_data_id_;
    }

    virtual void RefreshRobotData() {
    }

    std::atomic<double> joint_receive_time_{0.}, imu_receive_time_{0.};
    static double ReceiveSteadyNow(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
    double GetBodyReceiveAgeSeconds() override {
        if(command_authority_lost_.load()||foreign_command_seen_.load())return std::numeric_limits<double>::infinity();
        const double oldest=std::min(joint_receive_time_.load(std::memory_order_acquire),imu_receive_time_.load(std::memory_order_acquire));
        return oldest>0?ReceiveSteadyNow()-oldest:std::numeric_limits<double>::infinity();
    }

    virtual void Handler(const drdds::msg::JointsData::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(joint_state_mutex_);
        const double source_stamp=rclcpp::Time(msg->header.stamp).seconds();
        if(source_stamp<=ri_ts_.load(std::memory_order_acquire))return;
        joint_receive_time_.store(ReceiveSteadyNow(),std::memory_order_release);
        ++run_cnt_;
        for (int i = 0; i < dof_num_; ++i) {
            joint_pos_(i) = msg->data.joints_data[i].position * joint_dir_[i] + pos_offset_[i] / 180. * M_PI;
            joint_vel_(i) = msg->data.joints_data[i].velocity * joint_dir_[i];
            joint_tau_(i) = msg->data.joints_data[i].torque * joint_dir_[i];
            motor_temperture_(i) = float(msg->data.joints_data[i].motion_temp);
            driver_temperture_(i) = float(msg->data.joints_data[i].driver_temp);
            driver_status_[i] = msg->data.joints_data[i].status_word;
            joint_data_id_[i] = uint16_t(run_cnt_);
        }
        ri_ts_.store(rclcpp::Time(msg->header.stamp).seconds(), std::memory_order_release);
    }

    void HandlerIMU(const drdds::msg::ImuData::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(imu_state_mutex_);
        if(rclcpp::Time(msg->header.stamp).seconds()<=imu_ts_.load(std::memory_order_acquire))return;
        imu_receive_time_.store(ReceiveSteadyNow(),std::memory_order_release);
        rpy_ = Vec3f(Deg2Rad(msg->data.roll), Deg2Rad(msg->data.pitch), Deg2Rad(msg->data.yaw));
        acc_ << msg->data.acc_x, msg->data.acc_y, msg->data.acc_z;
        omega_body_ << msg->data.omega_x, msg->data.omega_y, msg->data.omega_z;
        imu_ts_.store(rclcpp::Time(msg->header.stamp).seconds(), std::memory_order_release);
    }

    void HandlerSensorIMU(const sensor_msgs::msg::Imu::SharedPtr msg) {
        double w = msg->orientation.w;
        double x = msg->orientation.x;
        double y = msg->orientation.y;
        double z = msg->orientation.z;
        const double norm = std::sqrt(w * w + x * x + y * y + z * z);
        if (!std::isfinite(norm) || std::abs(norm-1.0)>.02) {
            return;
        }
        std::lock_guard<std::mutex> lock(imu_state_mutex_);
        if(rclcpp::Time(msg->header.stamp).seconds()<=imu_ts_.load(std::memory_order_acquire))return;
        imu_receive_time_.store(ReceiveSteadyNow(),std::memory_order_release);
        w /= norm;
        x /= norm;
        y /= norm;
        z /= norm;

        const double sin_pitch = std::clamp(2.0 * (w * y - z * x), -1.0, 1.0);
        rpy_ << static_cast<float>(std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))),
                static_cast<float>(std::asin(sin_pitch)),
                static_cast<float>(std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)));
        acc_ << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
        omega_body_ << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
        imu_ts_.store(rclcpp::Time(msg->header.stamp).seconds(), std::memory_order_release);
    }

    void HandlerHealth(const drdds::msg::BatteryData::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(health_state_mutex_);
        battery_data_[0] = uint8_t(msg->data[0].battery_level);
        remain_bat = battery_data_[0] / 100;
        battery_info_[0].battery_level = msg->data[0].battery_level;
    }


    void HandlerLidarObservation(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(lidar_mutex_);
        lidar_observation_.resize(msg->data.size());
        bool all_finite = !msg->data.empty();
        for (size_t i = 0; i < msg->data.size(); ++i) {
            lidar_observation_(i) = msg->data[i];
            all_finite = all_finite && std::isfinite(msg->data[i]);
        }
        lidar_observation_valid_ = all_finite;
        lidar_last_update_ = std::chrono::steady_clock::now();
    }

    void HandlerDepthObservation(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        depth_observation_.resize(msg->data.size());
        bool all_finite = !msg->data.empty();
        for (size_t i = 0; i < msg->data.size(); ++i) {
            depth_observation_(i) = msg->data[i];
            all_finite = all_finite && std::isfinite(msg->data[i]);
        }
        depth_observation_valid_ = all_finite;
        depth_last_update_ = std::chrono::steady_clock::now();
    }

    void HandlerV11Observation(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(v11_mutex_);
        v11_observation_.resize(msg->data.size());
        const char* contract_env = std::getenv("M20_V11_FEATURE_CONTRACT");
        const std::string contract = contract_env == nullptr
            ? "v11_positive_risers_v1" : std::string(contract_env);
        const bool signed_edges = contract == "v11_signed_edges_v2" ||
            contract == "v11_signed_edges_rear_axle_v3";
        const std::string expected_label = signed_edges
            ? "B1,signedH1,D1,B2,signedH2,D2" : "B1,H1,D1,B2,H2,D2";
        const bool layout_matches = msg->layout.dim.size() == 1 &&
            msg->layout.dim[0].size == 6 && msg->layout.dim[0].stride == 6 &&
            msg->layout.dim[0].label == expected_label;
        bool all_finite = msg->data.size() == 6 && layout_matches;
        for (size_t i = 0; i < msg->data.size(); ++i) {
            v11_observation_(i) = msg->data[i];
            all_finite = all_finite && std::isfinite(msg->data[i]);
        }
        v11_observation_valid_ = all_finite;
        v11_last_update_ = std::chrono::steady_clock::now();
    }
};
