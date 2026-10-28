#pragma once
#include "cts_landing.hpp"
#include <chrono>
#include <vector>

namespace m20_cts {
inline double SteadySeconds(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct LandingFeedback {
  bool available=false,query_pose_valid=false;
  std::string fault="CTS measured landing feedback unavailable",world_frame;
  uint64_t session=0,sequence=0,map_session=0,map_sequence=0;
  double capture_s=std::numeric_limits<double>::quiet_NaN(),received_s=0,query_stamp_ros_s=0;
  std::array<double,4> contact_capture_s{};
  std::array<bool,4> contact_valid{};
  std::array<float,4> contact_force{},normal_velocity{};
  std::array<float,16> joint_position_sdk{},joint_velocity_sdk{};
  std::array<V3,4> support_world;
  std::array<M3,4> point_jacobian_world;
  V3 base_position_world=V3::Zero(),query_position_world=V3::Zero();
  float base_yaw_world=0,query_yaw_world=0;
  LandingFeedback(){for(auto& v:support_world)v.setConstant(std::numeric_limits<float>::quiet_NaN());
    for(auto& v:point_jacobian_world)v.setConstant(std::numeric_limits<float>::quiet_NaN());}
  void RequireFresh(double now,double maximum_age)const {
    if(!available||!fault.empty())throw std::runtime_error(fault.empty()?"CTS measured landing feedback unavailable":fault);
    // Integer nanoseconds converted to double seconds can differ by <1 ns.
    if(!std::isfinite(now)||!std::isfinite(capture_s)||now+1e-9<capture_s||now-capture_s>maximum_age||
       now+1e-9<received_s||now-received_s>maximum_age)
      throw std::runtime_error("CTS landing kinematics stale or from future");
  }
};

// A coherent snapshot from a measured external adapter. Parsing this interface
// does NOT approve a hardware binding. The pre-SDK capability guard stays closed
// until contact and actuator identification have independent hardware evidence.
class LandingFeedbackReceiver {
  nlohmann::json expected_;
  LandingFeedback current_;
  bool latched_=false;
  template<size_t N> static std::array<float,N> Floats(const nlohmann::json& j){
    if(!j.is_array()||j.size()!=N)throw std::runtime_error("CTS feedback array shape mismatch");
    std::array<float,N> v{};for(size_t i=0;i<N;++i){
      if(!j[i].is_number()||j[i].is_boolean())throw std::runtime_error("CTS feedback is not numeric");
      v[i]=j[i].get<float>();if(!std::isfinite(v[i]))throw std::runtime_error("CTS feedback contains nonfinite data");}
    return v;
  }
  static double Stamp(const nlohmann::json& j){
    if(!j.is_number_integer()||j.is_boolean()||j.get<int64_t>()<=0)throw std::runtime_error("CTS feedback requires capture monotonic ns");
    return double(j.get<int64_t>())*1e-9;
  }
  static uint64_t Positive(const nlohmann::json& j){
    if(!j.is_number_integer()||j.is_boolean()||
       (j.is_number_unsigned()?j.get<uint64_t>()==0:j.get<int64_t>()<=0))
      throw std::runtime_error("CTS feedback invalid identity counter");
    return j.get<uint64_t>();
  }
public:
  explicit LandingFeedbackReceiver(const nlohmann::json& expected):expected_(expected){
    for(const char* key:{"source_id","hardware_serial","world_frame","binding_sha256","publisher_node"})
      if(!expected.at(key).is_string()||expected.at(key).get<std::string>().empty())throw std::runtime_error("missing CTS measured input identity");
    const auto digest=expected.at("binding_sha256").get<std::string>();
    if(digest.size()!=64||digest.find_first_not_of("0123456789abcdef")!=std::string::npos)throw std::runtime_error("invalid CTS binding digest");
  }
  void Reject(const std::string& reason){latched_=true;current_.available=false;current_.fault=reason;}
  const LandingFeedback& Current()const{return current_;}
  void Consume(const nlohmann::json& j,double received){
    if(latched_)throw std::runtime_error(current_.fault);
    try{
      if(j.at("schema")!="m20_cts_landing_feedback.v1"||j.at("joint_order")!="sdk16"||
         j.at("leg_order")!="FL_FR_HL_HR"||j.at("jacobian_columns")!="policy_leg_joints"||
         j.at("kinematics_valid")!=true||!j.at("kinematics_valid").is_boolean())
        throw std::runtime_error("CTS feedback schema/order/kinematics invalid");
      for(const char* key:{"source_id","hardware_serial","world_frame","binding_sha256"})
        if(j.at(key)!=expected_.at(key))throw std::runtime_error("CTS feedback source/frame/binding changed");
      LandingFeedback next;
      next.world_frame=j.at("world_frame").get<std::string>();
      next.session=Positive(j.at("session"));next.sequence=Positive(j.at("sequence"));
      if(current_.available){
        if(next.session!=current_.session)throw std::runtime_error("CTS feedback session changed; operator restart required");
        if(next.sequence<=current_.sequence)throw std::runtime_error("CTS feedback replay or sequence regression");
      }
      next.capture_s=Stamp(j.at("capture_monotonic_ns"));next.received_s=received;
      if(!std::isfinite(received)||next.capture_s>received+1e-9||
         (current_.available&&next.capture_s<=current_.capture_s))throw std::runtime_error("CTS feedback capture clock regressed/future");
      next.joint_position_sdk=Floats<16>(j.at("joint_position_sdk_rad"));
      next.joint_velocity_sdk=Floats<16>(j.at("joint_velocity_sdk_rad_s"));
      next.contact_force=Floats<4>(j.at("contact_support_force_N"));
      next.normal_velocity=Floats<4>(j.at("support_normal_velocity_world_m_s"));
      const auto support=Floats<12>(j.at("support_position_world_m"));
      const auto jac=Floats<36>(j.at("point_jacobian_world_m_per_rad"));
      for(int i=0;i<4;++i){
        for(int k=0;k<3;++k){next.support_world[i][k]=support[3*i+k];
          for(int c=0;c<3;++c)next.point_jacobian_world[i](k,c)=jac[9*i+3*k+c];}
      }
      const auto& valid=j.at("contact_valid");const auto& stamps=j.at("contact_capture_monotonic_ns");
      if(!valid.is_array()||valid.size()!=4||!stamps.is_array()||stamps.size()!=4)throw std::runtime_error("CTS contact shape mismatch");
      for(int i=0;i<4;++i){
        if(!valid[i].is_boolean())throw std::runtime_error("CTS contact validity requires boolean");
        next.contact_valid[i]=valid[i].get<bool>();
        // Invalid observations may have no capture; they never count as flight.
        next.contact_capture_s[i]=next.contact_valid[i]?Stamp(stamps[i]):0;
        if(next.contact_valid[i]&&(next.contact_capture_s[i]>next.capture_s||next.contact_force[i]<0))
          throw std::runtime_error("CTS contact is noncausal or negative");
      }
      auto pose=Floats<3>(j.at("base_position_world_m"));
      next.base_position_world=V3(pose[0],pose[1],pose[2]);
      next.base_yaw_world=Floats<1>(nlohmann::json::array({j.at("base_yaw_world_rad")}))[0];
      const auto& query=j.at("terrain_query");
      if(!query.at("valid").is_boolean())throw std::runtime_error("CTS query pose validity requires boolean");
      next.query_pose_valid=query.at("valid").get<bool>();
      if(next.query_pose_valid){
        next.map_session=Positive(query.at("session"));next.map_sequence=Positive(query.at("sequence"));
        next.query_stamp_ros_s=query.at("stamp_ros_s").get<double>();
        if(!std::isfinite(next.query_stamp_ros_s))throw std::runtime_error("CTS query timestamp nonfinite");
        pose=Floats<3>(query.at("position_world_m"));next.query_position_world=V3(pose[0],pose[1],pose[2]);
        next.query_yaw_world=Floats<1>(nlohmann::json::array({query.at("yaw_world_rad")}))[0];
      }
      next.available=true;next.fault.clear();current_=next;
    }catch(const std::exception& e){Reject(e.what());throw;}
  }
};
}
