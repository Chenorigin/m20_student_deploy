#pragma once
#include "student_heading_feedback.hpp"
#include <deque>

// Sensor-only CTS packer, reusable in replay and a future validated hardware
// binding. Input proprio uses training scales/order; command is overwritten
// from the independent raw sample before applying sensor heading feedback.
class CTSObservation {
  StudentHeadingFeedback heading_;
  std::array<bool,187> mask_{};
  std::deque<std::array<float,57>> history_;
  uint64_t step_=0;
public:
  explicit CTSObservation(const nlohmann::json& contract):heading_(contract.at("command_feedback")) {
    if(contract.at("deployment_family")!="m20_cts_student"||contract.at("history_steps")!=4||
       contract.at("student_observation_dim")!=789||contract.at("action_dim")!=16)
      throw std::runtime_error("CTS sensor observation ABI mismatch");
    const auto& p=contract.at("perception");
    if(!p.at("front_only").get<bool>())throw std::runtime_error("CTS requires front-only perception");
    const auto sensor=p.at("front_sensor_offset_m").get<std::vector<float>>();
    const float fov=p.at("lidar_horizontal_fov_deg").get<float>();
    const float blind=p.at("front_blind_distance_m").get<float>();
    if(sensor.size()!=3||!std::isfinite(fov)||fov<=0||fov>360||!std::isfinite(blind)||blind<0)
      throw std::runtime_error("invalid CTS front sensor mask");
    for(float v:sensor)if(!std::isfinite(v))throw std::runtime_error("nonfinite CTS sensor transform");
    for(int i=0;i<187;++i){
      const float dx=(i%17-8)*.1f-sensor[0],dy=(i/17-5)*.1f-sensor[1];
      mask_[i]=std::hypot(dx,dy)>blind+1.e-6f&&dx>0.f&&
        std::abs(std::atan2(dy,dx))<=fov*.5f*float(M_PI)/180.f+1.e-7f;
    }
    Reset();
  }
  void Reset(){history_.assign(3,{});heading_.Reset();step_=0;}
  std::vector<float> Pack(std::array<float,57> proprio,const std::array<float,561>& terrain,
                           const std::array<float,3>& raw,float yaw) {
    for(float v:proprio)if(!std::isfinite(v))throw std::runtime_error("nonfinite CTS proprioception");
    for(float v:raw)if(!std::isfinite(v))throw std::runtime_error("nonfinite CTS raw command");
    if(!std::isfinite(yaw))throw std::runtime_error("nonfinite CTS estimated yaw");
    std::vector<float> obs(789,0.f);
    for(int i=0;i<187;++i){
      const float h=terrain[i],m=terrain[187+i],c=terrain[374+i];
      if(!std::isfinite(h)||!std::isfinite(m)||!std::isfinite(c)||std::abs(h)>1||
         (m!=0&&m!=1)||c<0||c>1||(m==0&&(h!=0||c!=0)))
        throw std::runtime_error("invalid CTS terrain channels");
      if(mask_[i]){obs[57+i]=h;obs[244+i]=m;obs[431+i]=c;}
    }
    const auto effective=heading_.Apply(raw,yaw,proprio[2]/.25f,obs,step_);
    for(int i=0;i<3;++i)proprio[6+i]=effective[i];
    std::copy(proprio.begin(),proprio.end(),obs.begin());
    for(int frame=0;frame<3;++frame)
      std::copy(history_[frame].begin(),history_[frame].end(),obs.begin()+618+57*frame);
    history_.pop_front();history_.push_back(proprio);++step_;
    return obs;
  }
};
