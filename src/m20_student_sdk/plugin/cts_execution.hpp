#pragma once
#include "cts_feedback.hpp"
#include "cts_terrain_memory.hpp"
#include "common_types.h"
#include <set>

namespace m20_cts {
constexpr double servo_period_s=.005;
// Consecutive real SDK generations may wake early after one late dispatch.
// Tolerance changes interval admission only, never the 5ms publish deadline.
constexpr double servo_jitter_tolerance_s=.0035;
constexpr double servo_interval_min_s=servo_period_s-servo_jitter_tolerance_s;
constexpr double servo_interval_max_s=servo_period_s+servo_jitter_tolerance_s;
static_assert(servo_interval_min_s>0&&servo_interval_max_s<2*servo_period_s,
              "servo jitter must not admit a full missed period");
constexpr int policy_decimation=4;
class SchedulingFault : public std::runtime_error {
public: using std::runtime_error::runtime_error;
};

// Checked by the real SDK worker, also used by the offline execution probe.
// Missing ticks are faults, never an invitation to simulate catch-up substeps.
class SubstepClock {
  bool seen_=false;int64_t last_tick_=0;double last_time_=0;
public:
  void Reset(){seen_=false;}
  bool Advance(int64_t tick,double now){
    if(!std::isfinite(now)||tick<0)throw std::runtime_error("CTS invalid servo clock");
    if(seen_&&now<=last_time_)throw std::runtime_error("CTS nonmonotonic servo clock");
    if(!seen_&&tick!=0)throw SchedulingFault("CTS first servo tick was missed");
    const double interval=now-last_time_;
    if(seen_&&(tick!=last_tick_+1||interval<servo_interval_min_s||interval>servo_interval_max_s))
      throw SchedulingFault("CTS missed/duplicate servo substep");
    seen_=true;last_tick_=tick;last_time_=now;
    return tick%policy_decimation==0;
  }
};

class Execution {
  LandingSettings settings_;
  LandingController landing_;
  TerrainMemory memory_;
  std::array<int,16> policy_from_sdk_{};
  std::array<std::array<float,2>,16> position_bounds_{};
  types::RobotAction last_issued_;
  bool have_issued_=false;
  double epoch_=0;
  float offset_;
  double maximum_body_age_;
  std::string world_frame_;
  uint64_t feedback_session_=0;
  uint64_t map_session_=0,map_sequence_=0;
  std::set<uint64_t> retired_maps_;
  std::array<float,3> command_{};
  LandingOutput last_output_;
public:
  Execution(const nlohmann::json& contract,const nlohmann::json& robot,
            std::string world_frame,double maximum_body_age):
      settings_(contract.at("quiet_execution_control")),landing_(settings_),memory_(187,settings_),
      offset_(contract.at("height").at("offset_m").get<float>()),
      maximum_body_age_(maximum_body_age),world_frame_(std::move(world_frame)){
    if(contract.at("deployment_family")!="m20_cts_student"||contract.at("policy_dt_s")!=servo_period_s*policy_decimation||
       contract.at("standalone_robot_controller")!=false||settings_.mode=="off"||
       settings_.contact_delay_steps!=0||settings_.contact_dropout_probability!=0)
      throw std::runtime_error("CTS execution settings unsupported");
    const auto mapping=robot.at("joints").at("policy_from_sdk_indices").get<std::vector<int>>();
    if(mapping.size()!=16)throw std::runtime_error("CTS execution joint mapping shape");
    std::array<bool,16> seen{};
    for(int i=0;i<16;++i){int j=mapping[i];
      if(j<0||j>=16||seen[j]||((i<12)==(j%4==3)))throw std::runtime_error("CTS execution joint mapping invalid");
      seen[j]=true;policy_from_sdk_[i]=j;}
    for(int j=0;j<16;++j){position_bounds_[j]=robot.at("hardware").at("software_position_limits_sdk_rad").at(j).get<std::array<float,2>>();
      if(!std::isfinite(position_bounds_[j][0])||!std::isfinite(position_bounds_[j][1])||position_bounds_[j][0]>=position_bounds_[j][1])
        throw std::runtime_error("CTS software position limits invalid");}
    if(!std::isfinite(offset_)||!std::isfinite(maximum_body_age_)||maximum_body_age_<=0||world_frame_.empty())
      throw std::runtime_error("CTS execution geometry/timing invalid");
  }
  void Reset(double now){epoch_=now;feedback_session_=map_session_=map_sequence_=0;retired_maps_.clear();
    landing_.Reset();memory_.Reset();have_issued_=false;command_.fill(0);}
  void SetRawCommand(const std::array<float,3>& command){command_=command;}
  const LandingOutput& LastOutput()const{return last_output_;}
  const LandingController& Controller()const{return landing_;}
  const TerrainMemory& Memory()const{return memory_;}
  void CheckFeedback(const LandingFeedback& f,double now){
    f.RequireFresh(now,maximum_body_age_);
    if(f.world_frame!=world_frame_||!f.session)throw std::runtime_error("CTS feedback world/session mismatch");
    if(feedback_session_&&feedback_session_!=f.session)throw std::runtime_error("CTS feedback source session changed");
    feedback_session_=f.session;
    // Catch current LIO pose jumps even between policy updates or when no
    // matching map/query pose is available. Preserve the post-jump capture cutoff.
    memory_.ObservePose(f.base_position_world,f.base_yaw_world,float(now-epoch_));
  }
  void SelectMapSource(uint64_t session,const LandingFeedback& f,double now){
    CheckFeedback(f,now);
    if(!session||retired_maps_.count(session))throw std::runtime_error("CTS terrain source retired/invalid");
    if(map_session_&&map_session_!=session){
      if(retired_maps_.size()>=128)throw std::runtime_error("CTS terrain source epoch history exhausted");
      retired_maps_.insert(map_session_);
      memory_.ResetAt(f.base_position_world,f.base_yaw_world,float(now-epoch_));map_sequence_=0;
    }
    map_session_=session;
  }
  bool ObserveMap(const std::vector<float>& obs,const std::array<float,187>& age,
                  const LandingFeedback& f,uint64_t map_session,uint64_t map_sequence,
                  double query_stamp,double now){
    CheckFeedback(f,now);
    if(obs.size()!=789)throw std::runtime_error("CTS execution map observation size");
    SelectMapSource(map_session,f,now);
    if(!map_sequence||map_sequence<map_sequence_)throw std::runtime_error("CTS terrain sequence regressed");
    map_sequence_=map_sequence;
    if(!f.query_pose_valid||f.map_session!=map_session||f.map_sequence!=map_sequence||
       std::abs(f.query_stamp_ros_s-query_stamp)>1e-6)return false;
    std::vector<V3> points(187);std::vector<bool> valid(187);std::vector<float> ages(age.begin(),age.end());
    const float co=std::cos(f.query_yaw_world),si=std::sin(f.query_yaw_world);
    for(int i=0;i<187;++i){const float x=(i%17-8)*.1f,y=(i/17-5)*.1f,h=obs[57+i];
      points[i]=V3(f.query_position_world.x()+co*x-si*y,f.query_position_world.y()+si*x+co*y,
                   f.query_position_world.z()-offset_-h);
      valid[i]=obs[244+i]>.5f&&obs[431+i]>0&&std::abs(h)<1&&points[i].allFinite()&&std::isfinite(age[i])&&age[i]>=0;
    }
    memory_.Update(points,valid,ages,f.base_position_world,f.base_yaw_world,float(now-epoch_));return true;
  }
  types::RobotAction Apply(const types::RobotAction& nominal,const LandingFeedback& f,double now){
    CheckFeedback(f,now);
    auto matrix=nominal.ConvertToMat();
    if(matrix.rows()!=16||matrix.cols()!=5||!matrix.allFinite())throw std::runtime_error("CTS corrupt nominal SDK target");
    const auto terrain=memory_.Query(f.support_world,float(now-epoch_));
    LandingInput in;in.command_active=std::max({std::abs(command_[0]),std::abs(command_[1]),std::abs(command_[2])})>settings_.command_active_threshold;
    for(int leg=0;leg<4;++leg){
      for(int k=0;k<3;++k){int j=policy_from_sdk_[3*leg+k];
        in.nominal_leg_target[leg][k]=nominal.goal_joint_pos[j];
        in.joint_position[leg][k]=f.joint_position_sdk[j];in.joint_velocity[leg][k]=f.joint_velocity_sdk[j];}
      in.nominal_wheel_velocity[leg]=nominal.goal_joint_vel[policy_from_sdk_[12+leg]];
      in.point_jacobian[leg]=f.point_jacobian_world[leg];in.surface_normal[leg]=V3(0,0,1);
      in.normal_jacobian[leg]=f.point_jacobian_world[leg].row(2).transpose();
      in.normal_velocity[leg]=f.normal_velocity[leg];in.clearance[leg]=f.support_world[leg].z()-terrain.height[leg];
      in.terrain_valid[leg]=terrain.valid[leg];in.contact_force[leg]=f.contact_force[leg];
      const double age=now-f.contact_capture_s[leg];
      in.contact_valid[leg]=f.contact_valid[leg]&&std::isfinite(age)&&age>=-1e-9&&age<=settings_.contact_max_age_s;
    }
    last_output_=landing_.Step(in,float(servo_period_s));
    types::RobotAction output=nominal;
    for(int leg=0;leg<4;++leg)for(int k=0;k<3;++k){int j=policy_from_sdk_[3*leg+k];
      output.goal_joint_pos[j]=last_output_.leg_target[leg][k];
      output.kp[j]=nominal.kp[j]*last_output_.kp_scale[leg];output.kd[j]=nominal.kd[j]*last_output_.kd_scale[leg];}
    for(int j=0;j<16;++j)if(j%4!=3&&(output.goal_joint_pos[j]<position_bounds_[j][0]||
        output.goal_joint_pos[j]>position_bounds_[j][1]))
      throw std::runtime_error("CTS corrected target outside software position limits");
    // Cache only after the one SDK writer confirms DDS publish returned.
    return output;
  }
  void RecordPublished(const types::RobotAction& issued){last_issued_=issued;have_issued_=true;}
  types::RobotAction Hold(const types::RobotAction& nominal)const {
    auto out=have_issued_?last_issued_:nominal;out.goal_joint_vel.setZero();out.tau_ff.setZero();return out;
  }
};
}
