/**
 * @file safe_controller.hpp
 * @brief safe controller to make robot safe
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */

/**
 * 一、关节端
 *      1.关节数据异常（nan值）
 *      2.关节数据不更新
 *      3.关节温度过温
 *      4.驱动器报错
 * 二、imu
 *      1.imu数据异常
 *      2.imu数据不更新
 * 三、电池
 *      1.电池电量过低
 *      2.电池数据不上报？
 *      3.电池温度？
 * 四、手柄
 *      1.手柄断连
 *      
 */

#pragma once

#include <algorithm>

#include "robot_interface.h"
#include "user_command_interface.h"


union RobotErrorState {
    struct {
        uint32_t imu_num_error: 1;
        uint32_t imu_update_overtime: 1;
        uint32_t command_heartbeat_overtime: 1;
        uint32_t driver_error: 1;
        uint32_t motor_heat_warn: 1;
        uint32_t joint_num_error: 1;
        uint32_t joint_update_error: 1;
        uint32_t battery_low_warn: 1;
        uint32_t battery_heat_warn: 1;
        uint32_t cpu_heat_warn: 1;
        uint32_t cpu_freq_warn: 1;
        uint32_t reserved: 21;
    };
    uint32_t error_code;
};

class SafeController {
private:
    RobotType robot_type_;
    std::shared_ptr<interface::RobotInterface> ri_ptr_;
    std::shared_ptr<interface::UserCommandInterface> uc_ptr_;
    std::thread judge_thread_;
    int run_cnt_ = 0;

    double last_imu_ts_, imu_check_time_;
    bool imu_check_flag_ = true;

    double ri_check_time_, last_ri_ts_;
    VecXf last_joint_pos_, last_joint_vel_, last_joint_tau_;
    std::vector<int> joint_data_same_cnt_;
    bool joint_check_flag_ = true;

    float max_temp_;
    int max_temp_idx_;

    double last_cmd_ts_, cmd_check_time_;
    bool cmd_check_flag_ = true;
    bool cmd_seen_ = false;
    bool safety_armed_ = false;
    int startup_healthy_cycles_ = 0;
    bool temperature_ok_ = true;

    static constexpr double sensor_timeout_seconds_ = 0.10;
    static constexpr double command_timeout_seconds_ = 2.50;
    static constexpr int startup_required_healthy_cycles_ = 500;

    types::UserCommand *usr_cmd_;
    std::atomic<bool> start_thread_flag_{false};

    double driver_error_ts_;
    double joint_data_error_ts_;
    double imu_data_error_ts_;
    double current_time_;

    RobotErrorState robot_error_state_;
    uint32_t last_error_code_;
    std::map<uint8_t, uint16_t> error_driver_map_;
    uint32_t joint_update_state_;


    bool IsImuDataNormal() {
        double current_imu_ts_ = ri_ptr_->GetImuTimestamp();
        Vec3f rpy = ri_ptr_->GetImuRpy();
        Vec3f acc = ri_ptr_->GetImuAcc();
        Vec3f omg = ri_ptr_->GetImuOmega();
        if (imu_check_flag_) {
            last_imu_ts_ = current_imu_ts_;
            imu_check_time_ = current_time_;
            imu_check_flag_ = false;
        }

        bool res = true;
        robot_error_state_.imu_update_overtime = 0;
        robot_error_state_.imu_num_error = 0;
        if (current_imu_ts_ != last_imu_ts_) {
            last_imu_ts_ = current_imu_ts_;
            imu_check_time_ = current_time_;
        }
        if (current_time_ - imu_check_time_ > sensor_timeout_seconds_) {
            robot_error_state_.imu_update_overtime = 1;
            res = false;
        }

        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(rpy(i)) || !std::isfinite(acc(i)) || !std::isfinite(omg(i))) {
                res = false;
                robot_error_state_.imu_num_error = 1;
            }
        }
        return res;
    }

    bool IsJointDataNormal() {
        double ri_ts_ = ri_ptr_->GetInterfaceTimeStamp();
        VecXf joint_pos = ri_ptr_->GetJointPosition();
        VecXf joint_vel = ri_ptr_->GetJointVelocity();
        VecXf joint_tau = ri_ptr_->GetJointTorque();
        std::vector<uint16_t> joint_id = ri_ptr_->GetJointDataID();
        if (joint_check_flag_) {
            joint_check_flag_ = false;
            last_ri_ts_ = ri_ts_;
            last_joint_pos_ = joint_pos;
            last_joint_vel_ = joint_vel;
            last_joint_tau_ = joint_tau;
            ri_check_time_ = current_time_;
        }
        bool res = true;
        robot_error_state_.joint_update_error = 0;
        if (last_ri_ts_ != ri_ts_) {
            last_ri_ts_ = ri_ts_;
            ri_check_time_ = current_time_;
        }
        if (current_time_ - ri_check_time_ > sensor_timeout_seconds_) {
            robot_error_state_.joint_update_error = 1;
            res = false;
        }


        robot_error_state_.joint_num_error = 0;
        joint_update_state_ = 0;
        for (int i = 0; i < joint_pos.size(); ++i) {
            if (!std::isfinite(joint_pos(i)) || !std::isfinite(joint_vel(i)) || !std::isfinite(joint_tau(i))) {
                robot_error_state_.joint_num_error = 1;
                res = false;
            }
            if (joint_pos(i) == last_joint_pos_(i)
                && joint_vel(i) == last_joint_vel_(i)
                && joint_tau(i) == last_joint_tau_(i)) {
                joint_data_same_cnt_[i]++;
            } else {
                joint_data_same_cnt_[i] = 0;
            }
            // if (joint_data_same_cnt_[i] > 200) {
            //     robot_error_state_.joint_num_error = 1;
            //     joint_update_state_ |= (1 << i);
            //     res = false;
            // }
        }
        last_joint_pos_ = joint_pos;
        last_joint_vel_ = joint_vel;
        last_joint_tau_ = joint_tau;
        if (run_cnt_ % 5000 == 0) {
            std::cout << "joint_pos: " << joint_pos.transpose() << std::endl;
            std::cout << "joint_vel: " << joint_vel.transpose() << std::endl;
        }
        return res;
    }

    bool IsDriverStatusNormal() {
        std::vector<uint16_t> status_word = ri_ptr_->GetDriverStatusWord();
        if (ri_ptr_->GetInterfaceTimeStamp() <= 0.0) {
            robot_error_state_.driver_error = 0;
            return true;
        }
        bool res = true;
        robot_error_state_.driver_error = 0;
        for (int i = 0; i < status_word.size(); ++i) {
            if (status_word[i] != 1) {
                error_driver_map_[uint8_t(i)] = status_word[i];
                robot_error_state_.driver_error = 1;
                res = false;
            }
        }
        if (robot_error_state_.driver_error == 0) {
            error_driver_map_.clear();
        }
        return res;
    }

    bool IsMotorTempertureNormal() {
        VecXf m_t = ri_ptr_->GetMotorTemperture();
        VecXf::Index max_idx, min_idx;
        float max_temp = m_t.maxCoeff(&max_idx);
        float min_temp = m_t.minCoeff(&min_idx);
        max_temp_ = max_temp;
        max_temp_idx_ = max_idx;
        if (run_cnt_ % 5000 == 0) {
            std::cout << "Motor Temperture: " << m_t.transpose() << std::endl;
            std::cout << "Max&Min:          " << max_idx << " : " << max_temp << "  |  " << min_idx << " : " << min_temp
                      << std::endl;
        }
        if (max_temp > 100) {
            robot_error_state_.motor_heat_warn = 1;
            return false;
        } else {
            robot_error_state_.motor_heat_warn = 0;
        }

        return true;
    }

    bool IsBatteryNormal() {
        std::vector<uint16_t> battery_data = ri_ptr_->GetBatteryData();
        if (battery_data[0] < 10) {
            robot_error_state_.battery_low_warn = 1;
            return false;
        } else {
            robot_error_state_.battery_low_warn = 0;
        }
        return true;
    }

    bool IsUserCommandNormal() {
        double cmd_ts_ = uc_ptr_->GetUserCommand()->time_stamp;
        robot_error_state_.command_heartbeat_overtime = 0;
        if (!std::isfinite(cmd_ts_)) {
            robot_error_state_.command_heartbeat_overtime = 1;
            return false;
        }
        if (!cmd_seen_ && cmd_ts_ == 0.0) {
            return true;
        }
        if (!cmd_seen_) {
            cmd_seen_ = true;
            cmd_check_flag_ = true;
        }
        if (cmd_check_flag_) {
            cmd_check_flag_ = false;
            last_cmd_ts_ = cmd_ts_;
            cmd_check_time_ = current_time_;
        }
        if (last_cmd_ts_ != cmd_ts_) {
            last_cmd_ts_ = cmd_ts_;
            cmd_check_time_ = current_time_;
        }
        if (current_time_ - cmd_check_time_ > command_timeout_seconds_) {
            robot_error_state_.command_heartbeat_overtime = 1;
            return false;
        }
        return true;
    }


    void PrintRobotErrorState() {
        std::cout << "Error occurred!" << std::endl;

        if (robot_error_state_.imu_num_error) {
            std::cout << "IMU number error: invalid IMU data count | "
                    << ri_ptr_->GetImuRpy().transpose() << " | "
                    << ri_ptr_->GetImuAcc().transpose() << " | "
                    << ri_ptr_->GetImuOmega().transpose() << std::endl;
        }

        if (robot_error_state_.imu_update_overtime) {
            std::cout << "IMU update overtime: no IMU data received since timestamp " << last_imu_ts_ << std::endl;
        }

        if (robot_error_state_.command_heartbeat_overtime) {
            std::cout << "Heartbeat overtime: lost communication with controller (no heartbeat signal received)" << std::endl;
        }

        if (robot_error_state_.driver_error) {
            std::cout << "Driver error: motor driver reported fault(s):" << std::endl;
            for (const auto &pair : error_driver_map_) {
                std::cout << "  Driver " << int(pair.first) << ": error code 0x" << std::hex << std::uppercase << int(pair.second) << std::dec << std::endl;
            }
        }

        if (robot_error_state_.motor_heat_warn) {
            std::cout << "Motor overheat warning: motor " << max_temp_idx_ << " reached " << max_temp_ << "°C" << std::endl;
        }

        if (robot_error_state_.joint_num_error) {
            std::cout << "Joint number error: mismatched joint count | pos: "
                    << ri_ptr_->GetJointPosition().transpose() << " | vel: "
                    << ri_ptr_->GetJointVelocity().transpose() << " | tau: "
                    << ri_ptr_->GetJointTorque().transpose() << std::endl;
        }

        if (robot_error_state_.joint_update_error) {
            std::cout << "Joint update error: no joint data update (current_time 0x" << std::hex << current_time_
                    << " vs last_check 0x" << ri_check_time_ << ")" << std::dec << std::endl;
        }

        if (robot_error_state_.battery_low_warn) {
            std::cout << "Battery low warning: battery voltage/level critically low" << std::endl;
        }

        if (robot_error_state_.battery_heat_warn) {
            std::cout << "Battery overheat warning: battery temperature too high" << std::endl;
        }

        if (robot_error_state_.cpu_heat_warn) {
            std::cout << "CPU overheat warning: CPU temperature exceeds safe limit" << std::endl;
        }

        if (robot_error_state_.cpu_freq_warn) {
            std::cout << "CPU frequency warning: CPU throttling due to thermal or power limits" << std::endl;
        }
    }

public:
    SafeController(RobotType robot_type, const std::string &path) : robot_type_(robot_type) {
        robot_error_state_.error_code = 0;
        last_error_code_ = 0;
        driver_error_ts_ = -std::numeric_limits<double>::infinity();
        joint_data_error_ts_ = -std::numeric_limits<double>::infinity();
        imu_data_error_ts_ = -std::numeric_limits<double>::infinity();
        error_driver_map_.clear();
    }

    ~SafeController() {
    }

    void SetRobotDataSource(std::shared_ptr<interface::RobotInterface> ri) {
        ri_ptr_ = ri;
        joint_data_same_cnt_.resize(ri_ptr_->dof_num_, 0);
    }

    void SetUserCommandDataSource(std::shared_ptr<interface::UserCommandInterface> uc) {
        uc_ptr_ = uc;
        usr_cmd_ = uc_ptr_->GetUserCommand();
    }

    void Start() {
        // Do not accept a stand request while this new process is still
        // discovering its own joint, IMU, and gamepad streams.  Once all
        // inputs have stayed healthy, return to Idle and require a fresh L1.
        usr_cmd_->safe_control_mode = 3;
        usr_cmd_->target_mode = uint8_t(RobotMotionState::JointDamping);
        start_thread_flag_.store(true, std::memory_order_release);
        judge_thread_ = std::thread(std::bind(&SafeController::Run, this));
    }

    void Run() {
        int tfd;    //定时器描述符
        int efd;    //epoll描述符
        int fds;
        uint64_t value;
        struct epoll_event ev, *evptr;
        struct itimerspec time_intv; //用来存储时间

        tfd = timerfd_create(CLOCK_MONOTONIC, 0);   //创建定时器
        if (tfd == -1) return;

        time_intv.it_value.tv_sec = 1.;
        time_intv.it_value.tv_nsec = 1000 * 1000;
        time_intv.it_interval.tv_sec = 0;
        time_intv.it_interval.tv_nsec = time_intv.it_value.tv_nsec;

        timerfd_settime(tfd, 0, &time_intv, NULL);  //启动定时器

        efd = epoll_create1(0); //创建epoll实例
        if (efd == -1) close(tfd);

        evptr = (struct epoll_event *) calloc(1, sizeof(struct epoll_event));
        if (evptr == NULL) {
            close(tfd);
            close(efd);
        }

        ev.data.fd = tfd;
        ev.events = EPOLLIN;    //监听定时器读事件，当定时器超时时，定时器描述符可读。
        epoll_ctl(efd, EPOLL_CTL_ADD, tfd, &ev); //添加到epoll监听队列中
        while (start_thread_flag_.load(std::memory_order_acquire)) {
            fds = epoll_wait(efd, evptr, 1, -1);    //阻塞监听，直到有事件发生
            if (evptr[0].events & EPOLLIN) {
                int ret = read(evptr->data.fd, &value, sizeof(uint64_t));
                if (ret == -1) {
                    continue;
                } else {
                }
            }
            run_cnt_++;
            current_time_ = GetTimestampMs() / 1000;
            uint8_t requested_safe_mode = 0;
            const bool driver_ok = IsDriverStatusNormal();
            const bool joint_ok = IsJointDataNormal();
            const bool imu_ok = IsImuDataNormal();
            const bool command_ok = IsUserCommandNormal();
            if (run_cnt_ % 1000 == 0) {
                temperature_ok_ = IsMotorTempertureNormal();
            }

            if (!driver_ok) {
                requested_safe_mode = std::max<uint8_t>(requested_safe_mode, 2);
                if (current_time_ - driver_error_ts_ >= 1.0) {
                    driver_error_ts_ = current_time_;
                    std::cout << "Driver status error!" << std::endl;
                }
            }
            if (!joint_ok) {
                requested_safe_mode = std::max<uint8_t>(requested_safe_mode, 3);
                if (current_time_ - joint_data_error_ts_ >= 1.0) {
                    joint_data_error_ts_ = current_time_;
                    std::cout << "Joint data error!" << std::endl;
                }
            }
            if (!temperature_ok_) {
                requested_safe_mode = std::max<uint8_t>(requested_safe_mode, 2);
                if (run_cnt_ % 1000 == 0) {
                    std::cout << "Motor temperture error!" << std::endl;
                }
            }
            if (!imu_ok) {
                requested_safe_mode = std::max<uint8_t>(requested_safe_mode, 2);
                if (current_time_ - imu_data_error_ts_ >= 1.0) {
                    imu_data_error_ts_ = current_time_;
                    std::cout << "IMU error!" << std::endl;
                }
            }
            if (!command_ok) {
                requested_safe_mode = std::max<uint8_t>(requested_safe_mode, 3);
            }

            if (!safety_armed_) {
                if (requested_safe_mode == 0 && cmd_seen_) {
                    ++startup_healthy_cycles_;
                } else {
                    startup_healthy_cycles_ = 0;
                }
                usr_cmd_->safe_control_mode = 3;
                usr_cmd_->target_mode = uint8_t(RobotMotionState::JointDamping);
                usr_cmd_->forward_vel_scale = 0.0f;
                usr_cmd_->side_vel_scale = 0.0f;
                usr_cmd_->turnning_vel_scale = 0.0f;
                if (startup_healthy_cycles_ >= startup_required_healthy_cycles_) {
                    safety_armed_ = true;
                    usr_cmd_->safe_control_mode = 0;
                    std::cout << "Safety inputs stable; controls armed. "
                              << "A fresh L1 is required to stand." << std::endl;
                }
            } else if (requested_safe_mode != 0) {
                usr_cmd_->forward_vel_scale = 0.0f;
                usr_cmd_->side_vel_scale = 0.0f;
                usr_cmd_->turnning_vel_scale = 0.0f;
                usr_cmd_->safe_control_mode = requested_safe_mode;
                usr_cmd_->target_mode = uint8_t(RobotMotionState::JointDamping);
            } else {
                // A recovered transient remains in/returns through damping and
                // Idle because target_mode was reset above.  It cannot resume
                // standing or RL without a new physical L1/L2 sequence.
                usr_cmd_->safe_control_mode = 0;
            }
            if (last_error_code_ != robot_error_state_.error_code) {
                if (robot_error_state_.error_code != 0) {
                    std::cout << "========== error_code ========>> " << 
                               std::hex  << robot_error_state_.error_code << std::endl;
                    PrintRobotErrorState();
                }
                last_error_code_ = robot_error_state_.error_code;
            }
        }
    }

    void Stop() {
        start_thread_flag_.store(false, std::memory_order_release);
        if (judge_thread_.joinable()) {
            judge_thread_.join();
        }
    }

};
