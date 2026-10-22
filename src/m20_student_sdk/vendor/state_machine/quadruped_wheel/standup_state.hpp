/**
 * @file standup_state.hpp
 * @brief from sit state to stand state
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */
#pragma once

#include "state_base.h"
#include "json.hpp"
#include "student_control_core.hpp"
#include <fstream>

namespace qw{
class StandUpState : public StateBase{
private:
    VecXf init_joint_pos_, init_joint_vel_, current_joint_pos_, current_joint_vel_;
    double time_stamp_record_, run_time_;
    VecXf pre_joint_pos_, policy_default_joint_pos_, kp_, kd_;
    MatXf joint_cmd_;
    nlohmann::json robot_;
    VecXf policy_kp_,policy_kd_;
    bool warned_handoff_=false;
    double stand_duration_ = 2.;

    const float init_hipx_pos_ = Deg2Rad(0.);
    const float set_wheel_kd_ = 1.;
    const float standup_kp_scale_ = 0.5;
    const float min_stand_duration_ = 2.5;

    void GetRobotJointValue(){
        current_joint_pos_ = ri_ptr_->GetJointPosition();
        current_joint_vel_ = ri_ptr_->GetJointVelocity();
        run_time_ = ri_ptr_->GetInterfaceTimeStamp();
    }

    void RecordJointData(){
        init_joint_pos_ = current_joint_pos_;
        init_joint_vel_ = current_joint_vel_;
        time_stamp_record_ = run_time_;
    }

    float GetHipYPosByHeight(float h){
        float l1 = cp_ptr_->thigh_len_;
        float l2 = cp_ptr_->shank_len_;
        float default_pos = (cp_ptr_->fl_joint_lower_(1)+cp_ptr_->fl_joint_upper_(1)) / 2.;
        if(fabs(h) >= l1 + l2) {
            std::cerr << "error height input" << std::endl;
            return 0;
        }
        float theta = -acos((l1*l1+h*h-l2*l2)/(2.*h*l1));
        theta = LimitNumber(theta, cp_ptr_->fl_joint_lower_(1), cp_ptr_->fl_joint_upper_(1));
        return theta;
    }

    float GetKneePosByHeight(float h){
        float l1 = cp_ptr_->thigh_len_;
        float l2 = cp_ptr_->shank_len_;
        float default_pos = (cp_ptr_->fl_joint_lower_(2)+cp_ptr_->fl_joint_upper_(2)) / 2.;
        if(fabs(h) >= l1 + l2) {
            std::cerr << "error height input" << std::endl;
            return 0;
        }
        float theta = M_PI-acos((l1*l1+l2*l2-h*h)/(2*l1*l2));
        theta = LimitNumber(theta, cp_ptr_->fl_joint_lower_(2), cp_ptr_->fl_joint_upper_(2));
        return theta;
    }

public:
    StandUpState(const RobotName& robot_name, const std::string& state_name, 
        std::shared_ptr<ControllerData> data_ptr):StateBase(robot_name, state_name, data_ptr){
            pre_joint_pos_ = Vec4f(init_hipx_pos_, GetHipYPosByHeight(cp_ptr_->pre_height_), GetKneePosByHeight(cp_ptr_->pre_height_), 0).replicate(4, 1);
            pre_joint_pos_(4) = -init_hipx_pos_; pre_joint_pos_(12) = -init_hipx_pos_;
            // 后髋膝取反
            pre_joint_pos_(9) = -pre_joint_pos_(9);
            pre_joint_pos_(10) = -pre_joint_pos_(10);
            pre_joint_pos_(13) = -pre_joint_pos_(13);
            pre_joint_pos_(14) = -pre_joint_pos_(14);

            const char* robot_config=std::getenv("M20_ROBOT_CONFIG");
            if(!robot_config)throw std::runtime_error("M20_ROBOT_CONFIG required for training stand pose");
            std::ifstream robot_file(robot_config);nlohmann::json robot;robot_file>>robot;
            robot_=robot;policy_kp_.resize(16);policy_kd_.resize(16);
            for(int j=0;j<16;++j){policy_kp_(j)=robot["joints"]["kp_sdk"][j];policy_kd_(j)=robot["joints"]["kd_sdk"][j];}
            policy_default_joint_pos_.resize(16);
            for(int sdk=0;sdk<16;++sdk){int policy=robot.at("joints").at("sdk_from_policy_indices").at(sdk);policy_default_joint_pos_(sdk)=robot.at("joints").at("default_position_policy_rad").at(policy);}

            Vec4f one_leg_kp, one_leg_kd;
            one_leg_kp << cp_ptr_->swing_leg_kp_ * standup_kp_scale_, 0;
            one_leg_kd << Vec3f(5., 5., 5.), 0;
            kp_ = one_leg_kp.replicate(4, 1);
            kd_ = one_leg_kd.replicate(4, 1);
            joint_cmd_ = MatXf::Zero(16, 5);
            joint_cmd_.col(0) = kp_;
            joint_cmd_.col(2) = kd_;
            stand_duration_ = cp_ptr_->stand_duration_;
            if (stand_duration_ < min_stand_duration_) {
                stand_duration_ = min_stand_duration_;
            }
        }
    ~StandUpState(){}

    virtual void OnEnter() {
        warned_handoff_=false;
        GetRobotJointValue();
        RecordJointData();
        StateBase::msfb_.UpdateCurrentState(RobotMotionState::StandingUp);
    };
    virtual void OnExit() {
    }
    virtual void Run() {
        GetRobotJointValue();
        VecXf planning_joint_pos(current_joint_pos_.rows());
        VecXf planning_joint_vel(current_joint_pos_.rows());
        if(run_time_ - time_stamp_record_ <= stand_duration_){
            for(int i=0;i<current_joint_pos_.rows();++i){
                planning_joint_pos(i) = GetCubicSplinePos(init_joint_pos_(i), init_joint_vel_(i), pre_joint_pos_(i), 0, 
                                                run_time_ - time_stamp_record_, stand_duration_);
                planning_joint_vel(i) = GetCubicSplineVel(init_joint_pos_(i), init_joint_vel_(i), pre_joint_pos_(i), 0, 
                                                run_time_ - time_stamp_record_, stand_duration_);
                if(i%4==3){
                    kd_(i) = 0;
                }
            }
            
        }else{
            float new_time = run_time_ - time_stamp_record_ - stand_duration_;
            for(int i=0;i<current_joint_pos_.rows();++i){
                planning_joint_pos(i) = GetCubicSplinePos(pre_joint_pos_(i), 0, policy_default_joint_pos_(i), 0,
                                                new_time, stand_duration_);
                planning_joint_vel(i) = GetCubicSplineVel(pre_joint_pos_(i), 0, policy_default_joint_pos_(i), 0,
                                                new_time, stand_duration_);
                if(i%4==3){
                    kd_(i) = set_wheel_kd_;
                }
            }
        }

        joint_cmd_.col(1) = planning_joint_pos;
        joint_cmd_.col(3) = planning_joint_vel;
        joint_cmd_.col(2) = kd_;
        // During the final stand spline, converge to the policy's gains too.
        // RL handoff then preserves target AND stiffness instead of dropping it.
        float blend=std::clamp(static_cast<float>((run_time_-time_stamp_record_-stand_duration_)/stand_duration_),0.f,1.f);
        joint_cmd_.col(0)=(1.f-blend)*kp_+blend*policy_kp_;
        joint_cmd_.col(2)=(1.f-blend)*kd_+blend*policy_kd_;
        ri_ptr_->SetJointCommand(joint_cmd_);
    }
    virtual bool LoseControlJudge() {
        if(uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::JointDamping)) return true;
        return false;
    }
    virtual StateName GetNextStateName() {
        if(uc_ptr_->GetUserCommand()->safe_control_mode!=0) return StateName::kJointDamping;
        if(run_time_ - time_stamp_record_ <= 2.*stand_duration_){
            return StateName::kStandUp;
        }else{
            if(uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::RLControlMode)){
                try{StudentControlCore::ValidateHandoff(ri_ptr_->GetJointCommand(),robot_);}
                catch(const std::exception& error){
                    if(!warned_handoff_){std::cerr<<"[STUDENT_ENTRY_REJECTED] "<<error.what()<<std::endl;warned_handoff_=true;}
                    return StateName::kStandUp;
                }
                return StateName::kRLControl;
            }else if(uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::LieDown)){
                return StateName::kLieDown;
            }
        }
        return StateName::kStandUp;
    }
};

};
