#pragma once
// Single-environment CPU port of task_support/stair_quiet_v1/landing.py.
// Nominal commands and actor history are never modified by this state machine.
#include "json.hpp"
#include <Eigen/Dense>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace m20_cts {
using V3=Eigen::Vector3f;
using M3=Eigen::Matrix3f;
constexpr const char* landing_source="d64002471291e55b2a3afc64ac11e21322f584b7242ce1df199609e927e64400";
constexpr const char* memory_source="6ba196ea6a726485ad0893f182e719f24d8ebbe935759cd773e1328d53eb0243";

#define M20_LANDING_FLOATS(X) \
 X(landing_speed_m_s) X(available_deceleration_m_s2) X(latency_budget_s) \
 X(clearance_margin_m) X(approach_distance_m) X(max_approach_distance_m) \
 X(reference_response_s) X(max_cartesian_correction_m) X(max_joint_correction_rad) \
 X(correction_rate_rad_s) X(jacobian_regularization_m) X(minimum_jacobian_norm_m) \
 X(closing_threshold_m_s) X(penetration_tolerance_m) X(loading_reference_allowance_m) \
 X(landing_kp_scale) X(landing_kd_scale) X(gain_rate_per_s) X(loading_time_s) \
 X(contact_on_force_N) X(contact_off_force_N) X(contact_confirm_s) \
 X(minimum_air_time_s) X(startup_grace_s) X(terrain_max_distance_m) \
 X(terrain_max_height_spread_m) X(terrain_max_age_s) X(terrain_memory_s) \
 X(terrain_memory_update_s) X(terrain_pose_jump_m) X(terrain_yaw_jump_rad) \
 X(terrain_memory_cell_m) X(contact_dropout_probability) X(contact_max_age_s) X(command_active_threshold)

struct LandingSettings {
#define FIELD(name) float name;
  M20_LANDING_FLOATS(FIELD)
#undef FIELD
  std::string mode;
  int minimum_other_supports,terrain_memory_frames,contact_delay_steps;
  explicit LandingSettings(const nlohmann::json& document) {
    if(document.at("schema")!="stair_quiet_v1.landing_control.v1"||
       document.at("source_sha256")!=landing_source||document.at("memory_source_sha256")!=memory_source||
       document.at("action_semantics")!="nominal_16d_before_execution_control_and_actuator_delay"||
       document.at("contact_requirement")!="causal_support_contact_with_validity; simulator_force_is_oracle"||
       document.at("deployment_status")!="requires_identified_actuator_and_validated_contact_inputs")
      throw std::runtime_error("unsupported CTS landing source/semantics");
    const auto& s=document.at("settings");
    size_t fields=4;
#define LOAD(name) \
    if(!s.at(#name).is_number()||s.at(#name).is_boolean())throw std::runtime_error("invalid landing setting " #name); \
    name=s.at(#name).get<float>(); \
    if(!std::isfinite(name)||name<0)throw std::runtime_error("invalid landing setting " #name); ++fields;
    M20_LANDING_FLOATS(LOAD)
#undef LOAD
    if(s.size()!=fields)throw std::runtime_error("unknown landing settings");
    mode=s.at("mode").get<std::string>();
    if(mode!="off"&&mode!="precontact"&&mode!="phase"&&mode!="combined")throw std::runtime_error("invalid landing mode");
    auto integer=[&](const char* key,int lo,int hi){
      if(!s.at(key).is_number_integer()||s.at(key).is_boolean())throw std::runtime_error("invalid landing integer");
      int v=s.at(key).get<int>();if(v<lo||v>hi)throw std::runtime_error("landing integer out of range");return v;
    };
    minimum_other_supports=integer("minimum_other_supports",1,3);
    terrain_memory_frames=integer("terrain_memory_frames",1,32);
    contact_delay_steps=integer("contact_delay_steps",0,1000000);
    for(float v:{available_deceleration_m_s2,reference_response_s,max_cartesian_correction_m,
        max_joint_correction_rad,correction_rate_rad_s,jacobian_regularization_m,minimum_jacobian_norm_m,
        gain_rate_per_s,loading_time_s,contact_confirm_s,terrain_max_distance_m,terrain_max_age_s,
        contact_max_age_s,terrain_memory_update_s,terrain_pose_jump_m,terrain_memory_cell_m,terrain_yaw_jump_rad})
      if(v<=0)throw std::runtime_error("zero positive landing setting");
    if(landing_kp_scale<=0||landing_kp_scale>1||landing_kd_scale<=0||landing_kd_scale>2||
       contact_off_force_N>=contact_on_force_N||max_approach_distance_m<approach_distance_m||
       contact_dropout_probability>1)throw std::runtime_error("landing settings outside source bounds");
  }
};
#undef M20_LANDING_FLOATS

struct LandingInput {
  std::array<V3,4> nominal_leg_target,joint_position,joint_velocity,normal_jacobian,surface_normal;
  std::array<M3,4> point_jacobian;
  std::array<float,4> nominal_wheel_velocity{},normal_velocity{},clearance{},contact_force{};
  std::array<bool,4> terrain_valid{},contact_valid{};
  bool command_active=false,full_jacobian=true;
};
struct LandingOutput {
  std::array<V3,4> leg_target,correction;
  std::array<float,4> wheel_velocity{},kp_scale{},kd_scale{},safe_speed{};
  std::array<int,4> phase{};
  std::array<bool,4> modified{},invalid{},infeasible{};
};
class LandingController {
  LandingSettings s_;
  std::array<bool,4> contact_{};
  std::array<float,4> pending_{},air_{},age_{},kp_{},kd_{};
  std::array<V3,4> correction_;
  float elapsed_=0;
  std::array<uint64_t,6> counts_{};
public:
  explicit LandingController(const LandingSettings& settings):s_(settings){Reset();}
  const LandingSettings& Settings()const{return s_;}
  const std::array<uint64_t,6>& Counts()const{return counts_;}
  void Reset(){contact_.fill(false);pending_.fill(0);air_.fill(0);age_.fill(1e6f);
    kp_.fill(1);kd_.fill(1);for(auto& c:correction_)c.setZero();elapsed_=0;}
  LandingOutput Step(const LandingInput& in,float dt){
    if(!std::isfinite(dt)||dt<=0||dt>.1f)throw std::runtime_error("invalid landing dt");
    for(int i=0;i<4;++i)if(!in.nominal_leg_target[i].allFinite()||!in.joint_position[i].allFinite()||
        !in.joint_velocity[i].allFinite()||!std::isfinite(in.nominal_wheel_velocity[i]))
      throw std::runtime_error("nonfinite nominal command or joint state");
    const auto& s=s_;elapsed_+=dt;
    std::array<bool,4> contact_ok{},touchdown{};
    for(int i=0;i<4;++i){
      contact_ok[i]=in.contact_valid[i]&&std::isfinite(in.contact_force[i])&&in.contact_force[i]>=0;
      const bool old=contact_[i];
      const bool wants=contact_ok[i]&&in.contact_force[i]>=(old?s.contact_off_force_N:s.contact_on_force_N);
      pending_[i]=wants?pending_[i]+dt:0;
      contact_[i]=wants&&(old||pending_[i]>=s.contact_confirm_s);
      touchdown[i]=contact_[i]&&!old&&air_[i]>=s.minimum_air_time_s;
      age_[i]=touchdown[i]?0:age_[i]+dt;
      air_[i]=contact_ok[i]&&!contact_[i]?air_[i]+dt:0;
    }
    const int supports=std::count(contact_.begin(),contact_.end(),true);
    const bool ready=in.command_active&&elapsed_>=s.startup_grace_s;
    LandingOutput out;
    for(int i=0;i<4;++i){
      const bool geometry=in.terrain_valid[i]&&std::isfinite(in.clearance[i])&&
          std::isfinite(in.normal_velocity[i])&&in.normal_jacobian[i].allFinite()&&
          (!in.full_jacobian||(in.point_jacobian[i].allFinite()&&in.surface_normal[i].allFinite()&&
             std::abs(in.surface_normal[i].squaredNorm()-1)<1e-4f));
      const bool valid=geometry&&contact_ok[i],support=supports-int(contact_[i])>=s.minimum_other_supports;
      V3 j=in.normal_jacobian[i];for(int k=0;k<3;++k)if(!std::isfinite(j[k]))j[k]=0;
      const float vn=std::isfinite(in.normal_velocity[i])?in.normal_velocity[i]:0;
      const float height=std::isfinite(in.clearance[i])?in.clearance[i]:0;
      const float j2=j.squaredNorm();const bool jac_ok=j2>=s.minimum_jacobian_norm_m*s.minimum_jacobian_norm_m;
      const bool enabled=ready&&valid&&support&&jac_ok;
      const float stopping=std::abs(vn)*s.latency_budget_s+vn*vn/(2*s.available_deceleration_m_s2)+s.clearance_margin_m;
      const float trigger=std::clamp(stopping,s.approach_distance_m,s.max_approach_distance_m);
      const bool approach=!contact_[i]&&air_[i]>=s.minimum_air_time_s&&vn<-s.closing_threshold_m_s&&
          height<=trigger&&height>=-s.penetration_tolerance_m;
      const bool loading=contact_[i]&&age_[i]<s.loading_time_s;
      const float remaining=std::max(height-s.clearance_margin_m-std::abs(vn)*s.latency_budget_s,0.f);
      out.safe_speed[i]=std::sqrt(s.landing_speed_m_s*s.landing_speed_m_s+2*s.available_deceleration_m_s2*remaining);
      const float desired=j.dot(in.nominal_leg_target[i]-in.joint_position[i]);
      const float base_velocity=vn-j.dot(in.joint_velocity[i]);
      const float pre=std::max((-out.safe_speed[i]-base_velocity)*s.reference_response_s-desired,0.f);
      const float load=std::max(-s.loading_reference_allowance_m-desired,0.f);
      float requested=0;
      if((s.mode=="precontact"||s.mode=="combined")&&approach&&enabled)requested=pre;
      if((s.mode=="phase"||s.mode=="combined")&&loading&&enabled)requested=load;
      const float magnitude=std::min(requested,s.max_cartesian_correction_m);
      V3 proposed;
      if(!in.full_jacobian)proposed=j*(magnitude/(j2+s.jacobian_regularization_m*s.jacobian_regularization_m));
      else{
        M3 full=in.point_jacobian[i];V3 normal=in.surface_normal[i];
        for(int r=0;r<3;++r){if(!std::isfinite(normal[r]))normal[r]=0;
          for(int c=0;c<3;++c)if(!std::isfinite(full(r,c)))full(r,c)=0;}
        const M3 metric=full*full.transpose()+M3::Identity()*s.jacobian_regularization_m*s.jacobian_regularization_m;
        proposed=full.transpose()*metric.ldlt().solve(normal*magnitude);
      }
      proposed*=std::min(1.f,s.max_joint_correction_rad/std::max(proposed.cwiseAbs().maxCoeff(),1e-12f));
      V3 diff=proposed-correction_[i];
      correction_[i]+=diff*std::min(1.f,s.correction_rate_rad_s*dt/std::max(diff.cwiseAbs().maxCoeff(),1e-12f));
      const bool soften=enabled&&(approach||loading)&&(s.mode=="phase"||s.mode=="combined");
      const float recovery=std::clamp(age_[i]/s.loading_time_s,0.f,1.f);
      const float ramp=loading?s.landing_kp_scale+(1-s.landing_kp_scale)*recovery:s.landing_kp_scale;
      const float gain_step=s.gain_rate_per_s*dt;
      kp_[i]+=std::clamp((soften?ramp:1)-kp_[i],-gain_step,gain_step);
      kd_[i]+=std::clamp((soften?s.landing_kd_scale:1)-kd_[i],-gain_step,gain_step);
      if(s.mode=="off"){correction_[i].setZero();kp_[i]=kd_[i]=1;}
      out.invalid[i]=ready&&!valid;
      const bool limited=requested>s.max_cartesian_correction_m+1e-6f||j.dot(correction_[i])+1e-6f<magnitude;
      out.infeasible[i]=ready&&valid&&(approach||loading)&&(!support||!jac_ok||limited);
      out.phase[i]=contact_[i]?3:0;
      if(approach&&enabled)out.phase[i]=1;if(loading&&enabled)out.phase[i]=2;
      if(out.invalid[i])out.phase[i]=4;if(out.infeasible[i])out.phase[i]=5;
      out.modified[i]=correction_[i].cwiseAbs().maxCoeff()>1e-7f||std::abs(kp_[i]-1)>1e-7f||std::abs(kd_[i]-1)>1e-7f;
      out.leg_target[i]=in.nominal_leg_target[i]+correction_[i];out.correction[i]=correction_[i];
      out.wheel_velocity[i]=in.nominal_wheel_velocity[i];out.kp_scale[i]=kp_[i];out.kd_scale[i]=kd_[i];
      ++counts_[0];counts_[1]+=out.modified[i];counts_[2]+=out.invalid[i];counts_[3]+=out.infeasible[i];
      counts_[4]+=touchdown[i];counts_[5]+=out.phase[i]==1;
    }
    return out;
  }
};
}
