/**
 * @file motion_state_feedback.hpp
 * @brief motion state feedback 
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */
#pragma once
#include "common_types.h"

class MotionStateFeedback
{
private:
    std::atomic<uint8_t> current_state_{0},current_gait_{0},last_state_{0},last_gait_{0};

    Vec3f current_vel_;
    Vec3f cmd_vel_;
    Vec3f max_vel_;

    std::atomic<uint32_t> error_code_{0};
public:
    MotionStateFeedback(/* args */){
        Reset();
        std::cout << "Motion State Feedback" << std::endl;
    }
    ~MotionStateFeedback(){

    }
    void Reset(){
        current_state_=0;current_gait_=0;last_state_=0;last_gait_=0;error_code_=0;
        current_vel_.setZero();cmd_vel_.setZero();max_vel_.setZero();
    }

    void UpdateCurrentState(int state){
        if(uint8_t(state) != current_state_){
            last_state_ = current_state_.load();
            current_state_ = uint8_t(state);
        }
    }

    void UpdateCurrentGait(uint8_t gait){
        if(gait != current_gait_){
            last_gait_ = current_gait_.load();
            current_gait_ = gait;
        }
    }

    void ClearMotionError(){error_code_ = 0;}
    void SetJointPosLimitError(){
        error_code_.fetch_or(1);
    }
    void SetPostureLimitError(){
        error_code_.fetch_or(2);
    }

    inline uint8_t GetCurrentState(){return current_state_;}
    inline uint8_t GetCurrentGait(){return current_gait_;}
    inline uint8_t GetLastState(){return last_state_;}
    inline uint8_t GetLastGait(){return last_gait_;}
};
