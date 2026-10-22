#pragma once
#include "common_types.h"
#include "json.hpp"
#include <array>
#include <stdexcept>

// Pure command shaping, shared by the real runner and offline tests. No I/O.
class StudentControlCore {
  nlohmann::json robot_;
  types::RobotAction last_;
  bool seeded_=false,latched_=false;
  bool leg_target_slew_enabled_=true;
  bool joint_target_position_clip_enabled_=true;
  bool wheel_target_velocity_clip_enabled_=true;
  std::string reason_;
  std::array<float,16> previous_{};
public:
  explicit StudentControlCore(const nlohmann::json& robot):robot_(robot),
      leg_target_slew_enabled_(robot.at("policy").value("leg_target_slew_enabled",true)),
      joint_target_position_clip_enabled_(robot.at("policy").value("joint_target_position_clip_enabled",true)),
      wheel_target_velocity_clip_enabled_(robot.at("policy").value("wheel_target_velocity_clip_enabled",true)){}
  static void ValidateHandoff(const types::MatXf& command,const nlohmann::json& robot) {
    if(command.rows()!=16||command.cols()!=5||!command.allFinite())throw std::runtime_error("invalid standing handoff shape/value");
    for(int j=0;j<16;++j){
      if(command(j,0)<0||command(j,2)<0)throw std::runtime_error("negative handoff gains");
      if(j%4!=3){
        const auto& limits=robot.at("hardware").at("software_position_limits_sdk_rad").at(j);
        if(command(j,0)<=0||command(j,1)<limits[0].get<float>()||command(j,1)>limits[1].get<float>())throw std::runtime_error("standing handoff has no support or invalid target");
      }
    }
  }
  void Begin(const types::MatXf& command) {
    ValidateHandoff(command,robot_);
    last_.kp=command.col(0);last_.goal_joint_pos=command.col(1);
    last_.kd=command.col(2);last_.goal_joint_vel=types::VecXf::Zero(16);last_.tau_ff=types::VecXf::Zero(16);
    previous_.fill(0);seeded_=true;latched_=false;reason_.clear();
  }
  void CheckBody(const types::RobotBasicState& ro,double max_age)const {
    if(ro.joint_pos.size()!=16||ro.joint_vel.size()!=16||!ro.joint_pos.allFinite()||!ro.joint_vel.allFinite()||!ro.base_omega.allFinite()||!ro.base_rot_mat.allFinite()||!std::isfinite(ro.body_receive_age_s)||ro.body_receive_age_s>max_age||ro.body_receive_age_s<0)throw std::runtime_error("invalid or stale SDK body state");
    for(int j=0;j<16;++j)if(j%4!=3){
      const auto& hardware=robot_.at("hardware");
      const auto& limits=hardware.contains("software_feedback_position_limits_sdk_rad")
          ?hardware.at("software_feedback_position_limits_sdk_rad").at(j)
          :hardware.at("software_position_limits_sdk_rad").at(j);
      if(ro.joint_pos(j)<limits[0].get<float>()||ro.joint_pos(j)>limits[1].get<float>())
        throw std::runtime_error("measured position outside software feedback limits: joint="+std::to_string(j)+
          " q="+std::to_string(ro.joint_pos(j))+" lower="+std::to_string(limits[0].get<float>())+
          " upper="+std::to_string(limits[1].get<float>()));
    }
  }
  types::RobotAction Hold(const std::string& reason) {
    if(!seeded_)throw std::runtime_error("no verified standing handoff for HOLD");
    if(!latched_){latched_=true;reason_=reason;}
    // Preserve the issued support target and gains, never rebase onto feedback.
    last_.goal_joint_vel.setZero();last_.tau_ff.setZero();
    return last_;
  }
  types::RobotAction Map(const float* action,double elapsed_s) {
    if(!seeded_)throw std::runtime_error("policy has no standing handoff");
    if(latched_)return Hold(reason_);
    if(!std::isfinite(elapsed_s)||elapsed_s<=0)return Hold("invalid policy interval");
    const float dt=std::min(static_cast<float>(elapsed_s),robot_["policy"]["dt_s"].get<float>());
    types::RobotAction target=last_;
    target.goal_joint_vel.setZero();target.tau_ff.setZero();
    std::array<float,16> next_previous{};
    for(int i=0;i<16;++i)if(!std::isfinite(action[i]))return Hold("nonfinite policy output");
    for(int j=0;j<16;++j){
      int i=robot_["joints"]["sdk_from_policy_indices"][j];
      float raw=std::clamp(action[i],robot_["policy"]["normalized_action_clip"][0].get<float>(),robot_["policy"]["normalized_action_clip"][1].get<float>());
      float baseline=robot_["joints"]["default_position_policy_rad"][i],scale=robot_["joints"]["action_scale_policy"][i];
      float value=baseline+raw*scale;
      const bool clip_target=j%4==3?wheel_target_velocity_clip_enabled_:joint_target_position_clip_enabled_;
      if(clip_target)value=std::clamp(value,robot_["policy"]["processed_target_clip_policy"][i][0].get<float>(),robot_["policy"]["processed_target_clip_policy"][i][1].get<float>());
      float speed=robot_["hardware"]["software_velocity_limits_sdk_rad_s"][j];
      if(j%4==3){target.goal_joint_vel(j)=clip_target?std::clamp(value,-speed,speed):value;target.kp(j)=0;}
      else{
        float lower=robot_["hardware"]["software_position_limits_sdk_rad"][j][0],upper=robot_["hardware"]["software_position_limits_sdk_rad"][j][1];
        if(clip_target)value=std::clamp(value,lower,upper);
        if(leg_target_slew_enabled_){
          float delta=speed*dt;
          target.goal_joint_pos(j)=std::clamp(value,last_.goal_joint_pos(j)-delta,last_.goal_joint_pos(j)+delta);
        }else{
          target.goal_joint_pos(j)=value;
        }
        target.kp(j)=robot_["joints"]["kp_sdk"][j];
      }
      target.kd(j)=robot_["joints"]["kd_sdk"][j];
      // Match training last_action: policy request, before physical shaping.
      next_previous[i]=raw;
    }
    last_=target;previous_=next_previous;return target;
  }
  bool Latched()const{return latched_;}
  bool LegTargetSlewEnabled()const{return leg_target_slew_enabled_;}
  bool JointTargetPositionClipEnabled()const{return joint_target_position_clip_enabled_;}
  bool WheelTargetVelocityClipEnabled()const{return wheel_target_velocity_clip_enabled_;}
  const std::string& Reason()const{return reason_;}
  const std::array<float,16>& Previous()const{return previous_;}
};
