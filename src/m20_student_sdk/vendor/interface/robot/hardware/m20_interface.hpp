#pragma once

#include "robot_interface.h"
#include "dds_interface.hpp"
#include "student_trace.hpp"

class M20Interface : public DdsInterface {
protected:
    void ResetPositionOffset() {
        this->SetJointCommand(MatXf::Zero(dof_num_, 5));

        VecXf current_joint_pos = this->GetJointPosition();
        const double initial_joint_stamp = this->GetInterfaceTimeStamp();
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(2);
        bool received_fresh_joint_state = false;
        while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
            rclcpp::spin_some(this->get_node());
            current_joint_pos = this->GetJointPosition();
            const double current_joint_stamp = this->GetInterfaceTimeStamp();
            if (current_joint_stamp > 0.0 &&
                current_joint_stamp != initial_joint_stamp &&
                current_joint_pos.allFinite()) {
                received_fresh_joint_state = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!received_fresh_joint_state) {
            throw std::runtime_error(
                "no fresh finite joint state received after reset command");
        }
        std::cout << "Fresh joint state received after reset command." << std::endl;

        for (int i = 1; i < dof_num_; i += 4) {
            if (current_joint_pos(i) < -140. / 180. * M_PI) {
                pos_offset_[i] = pos_offset_[i] + 360.;
                std::cout << "joint " << i << " offset is changed to " << pos_offset_[i] << "\n";
            } else if (current_joint_pos(i) > 140. / 180. * M_PI) {
                pos_offset_[i] = pos_offset_[i] - 360.;
                std::cout << "joint " << i << " offset is changed to " << pos_offset_[i] << "\n";
            }
        }
        for (int i = 2; i < dof_num_; i += 4) {
            if (current_joint_pos(i) < -164. / 180. * M_PI) {
                pos_offset_[i] = pos_offset_[i] + 360.;
                std::cout << "joint " << i << " offset is changed to " << pos_offset_[i] << "\n";
            } else if (current_joint_pos(i) > 164. / 180. * M_PI) {
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
    M20Interface(const std::string &robot_name) : DdsInterface(robot_name, 16) {
        battery_data_.resize(2 * BATTERY_DATA_SIZE);

        float init_pos_offset[16] = {-25, -131, 160, 0.,
                                     25, -131, 160, 0,
                                     -25, 131, -160, 0,
                                     25, 131, -160, 0};
        float joint_dir[16] = {1, 1, -1, 1,
                               1, -1, 1, -1,
                               -1, 1, -1, 1.,
                               -1, -1, 1, -1};
        for (int i = 0; i < dof_num_; ++i) {
            pos_offset_[i] = init_pos_offset[i];
            joint_dir_[i] = joint_dir[i];
            data_updated_[i] = false;
            joint_config_[i].dir = joint_dir_[i];
            joint_config_[i].offset = Deg2Rad(pos_offset_[i]);
        }
    }

    ~M20Interface() {
    }

    virtual void Start() {
        // Do not publish the offset-initialization command until both state
        // streams have remained fresh.  The external preflight checks the same
        // condition, and this closes the race between preflight and startup.
        WaitForStableRobotState();
        // This node owns one endpoint already. Recheck after discovery settles,
        // before the first initialization command (the external check saw zero).
        CheckCommandAuthority(true);
        command_output_started_.store(true);
        time_stamp_ = GetTimestampMs();
        ResetPositionOffset();
    }

    virtual void Stop() {
        MatXf damping_cmd = MatXf::Zero(dof_num_, 5);
        for (int i = 0; i < dof_num_; ++i) {
            damping_cmd(i, 2) = (i % 4 == 3) ? 1.0f : 2.0f;
        }
        std::cout << "[M20Interface] Stop: publishing joint damping command" << std::endl;
        for (int i = 0; i < 20; ++i) {
            SetJointCommand(damping_cmd);
            usleep(10 * 1000);
        }
    }

    /**
     * Sends joint commands to the robot.
     * Input matrix layout: [dof × 5] → [kp, pos, kd, vel, torque_ff]
     * 
     * ⚠ Torque values (column 4) are protected by low-level hardware torque limits
     *    → commands exceeding limits are saturated at the firmware/hardware level
     * 
     * Joints with index % 4 == 3 have kp forced to 0 (velocity/torque mode)
     */
    virtual void SetJointCommand(Eigen::Matrix<float, Eigen::Dynamic, 5> input) {
        (void)SetJointCommandWithReceipt(input);
        FlushJointCommandTrace();
    }
    virtual interface::JointCommandReceipt SetJointCommandWithReceipt(MatXf input) override {
        // A foreign stream must not race this SDK's damping/position writes.
        // Latch until operator intervention; hardware exit/hold semantics still
        // require commissioning and are never inferred from endpoint silence.
        if(foreign_command_seen_.load()||command_authority_lost_.load()){
            if(!command_authority_lost_.exchange(true))RCLCPP_ERROR(node_->get_logger(),"Control ownership lost; SDK command output disabled");
            TraceCommandEvent(input,nullptr,"blocked",nlohmann::json(),nlohmann::json(),"control ownership lost");
            return {};
        }
        auto msg = drdds::msg::JointsDataCmd();
        for (int i = 0; i < dof_num_; ++i) {
            msg.data.joints_data[i].position =
                    (input(i, 1) - joint_config_[i].offset) * joint_config_[i].dir;
            msg.data.joints_data[i].velocity = input(i, 3) * joint_dir_[i];
            msg.data.joints_data[i].torque = input(i, 4) * joint_dir_[i];
            msg.data.joints_data[i].kp = input(i, 0);
            if (i % 4 == 3) {
                msg.data.joints_data[i].kp = 0;
            }
            msg.data.joints_data[i].kd = input(i, 2);
            msg.data.joints_data[i].control_word = kIndexMotorControl;
        }
        nlohmann::json publish_start,publish_end;
        try{publish_start=StudentTrace::Stamp();}catch(...){}
        try{
            joint_cmd_pub_->publish(msg);
        }catch(...){
            try{publish_end=StudentTrace::Stamp();}catch(...){}
            TraceCommandEvent(input,&msg,"publish_failed",publish_start,publish_end,"DDS publisher threw; delivery is unknown");
            throw;
        }
        const double returned_steady_s=m20_cts::SteadySeconds();
        try{publish_end=StudentTrace::Stamp();}catch(...){}
        MatXf trace_sdk_input;
        try{trace_sdk_input=input;}catch(...){}
        // Retain the model-frame command actually published for FSM handoff.
        interface::JointCommandReceipt receipt;
        {
            std::lock_guard<std::mutex> lock(command_cache_mutex_);
            for(int j=3;j<dof_num_;j+=4)input(j,0)=0;
            joint_cmd_=input;
            receipt={true,++command_sequence_,returned_steady_s,input};
        }
        // This interface has one FSM writer. Defer serialization until that
        // worker has released its publication gate and decided the deadline.
        pending_trace_input_=std::move(trace_sdk_input);pending_trace_wire_=std::move(msg);
        pending_trace_start_=std::move(publish_start);pending_trace_end_=std::move(publish_end);
        pending_trace_=true;
        return receipt;
    }
    void FlushJointCommandTrace() noexcept override {
        if(!pending_trace_)return;
        pending_trace_=false;
        TraceCommandEvent(pending_trace_input_,&pending_trace_wire_,"publish_returned",pending_trace_start_,pending_trace_end_,
                          "DDS publish returned; motor execution is not acknowledged");
    }

    virtual MatXf GetJointCommand() override {
        std::lock_guard<std::mutex> lock(command_cache_mutex_);
        return joint_cmd_;
    }
private:
    std::mutex command_cache_mutex_;
    uint64_t command_sequence_=0;
    bool pending_trace_=false;
    MatXf pending_trace_input_;
    drdds::msg::JointsDataCmd pending_trace_wire_;
    nlohmann::json pending_trace_start_,pending_trace_end_;
    void TraceCommandEvent(const MatXf& input,const drdds::msg::JointsDataCmd* wire,
                           const char* status,const nlohmann::json& publish_start,
                           const nlohmann::json& publish_end,const char* reason) noexcept {
        try{
            nlohmann::json row={{"event","joint_command"},{"status",status},{"reason",reason},
                {"sdk_command_columns",{"kp","position","kd","velocity","torque_ff"}},
                {"sdk_command_snapshot_available",input.rows()==dof_num_&&input.cols()==5},
                {"sdk_command",nlohmann::json::array()},
                {"publish_start_stamp",publish_start},{"publish_end_stamp",publish_end}};
            for(int i=0;i<input.rows();++i){
                nlohmann::json joint=nlohmann::json::array();
                for(int j=0;j<input.cols();++j)joint.push_back(input(i,j));
                row["sdk_command"].push_back(std::move(joint));
            }
            if(wire){
                row["wire_command"]=nlohmann::json::array();
                for(int i=0;i<dof_num_;++i){
                    const auto& joint=wire->data.joints_data[i];
                    row["wire_command"].push_back({{"position",joint.position},{"velocity",joint.velocity},
                        {"torque",joint.torque},{"kp",joint.kp},{"kd",joint.kd},{"control_word",joint.control_word}});
                }
            }
            if(publish_end.is_object())row.update(publish_end);
            if(StudentTrace::Context().empty()){
                row["theoretical_monotonic_ns"]=nullptr;row["theoretical_wall_ns"]=nullptr;
                row["theoretical_time_basis"]="not_applicable_nonpolicy_command";
            }
            StudentTrace::Instance().Push(std::move(row));
        }catch(...){try{StudentTrace::Instance().RecordConstructionFailure();}catch(...){}}
    }

};
