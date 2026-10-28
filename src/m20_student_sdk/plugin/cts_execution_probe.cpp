// Bounded replay of the exact execution core. No ROS, SDK or motor publisher.
#include "cts_execution.hpp"
#include <iostream>
#include <fstream>

using nlohmann::json;
using namespace m20_cts;
static V3 vec(const json& j){if(j.size()!=3)throw std::runtime_error("vector shape");return V3(j[0].get<float>(),j[1].get<float>(),j[2].get<float>());}
static json array(const V3& v){return json::array({v.x(),v.y(),v.z()});}
static json output(const LandingOutput& out){
  json j={{"leg_target",json::array()},{"correction",json::array()},
    {"wheel_velocity",out.wheel_velocity},{"kp_scale",out.kp_scale},{"kd_scale",out.kd_scale},
    {"phase",out.phase},{"modified",out.modified},{"invalid",out.invalid},
    {"infeasible",out.infeasible},{"safe_speed",out.safe_speed}};
  for(int i=0;i<4;++i){j["leg_target"].push_back(array(out.leg_target[i]));j["correction"].push_back(array(out.correction[i]));}
  return j;
}
int main(int argc,char** argv){
 try{
  if(argc!=3)throw std::runtime_error("usage: cts_execution_probe contract.json landing|memory|feedback|clock|execution");
  std::ifstream file(argv[1]);json contract;file>>contract;json request;std::cin>>request;
  LandingSettings settings(contract.at("quiet_execution_control"));
  json rows=json::array();const std::string mode=argv[2];
  if(mode=="landing"){
    LandingController core(settings);
    for(const auto& frame:request){
      if(frame.value("reset",false))core.Reset();
      LandingInput in;
      for(int i=0;i<4;++i){
        in.nominal_leg_target[i]=vec(frame.at("nominal_leg_target")[i]);in.joint_position[i]=vec(frame.at("joint_position")[i]);
        in.joint_velocity[i]=vec(frame.at("joint_velocity")[i]);in.normal_jacobian[i]=vec(frame.at("normal_jacobian")[i]);
        in.nominal_wheel_velocity[i]=frame.at("nominal_wheel_velocity")[i];in.normal_velocity[i]=frame.at("normal_velocity")[i];
        in.clearance[i]=frame.at("clearance")[i];in.contact_force[i]=frame.at("contact_force")[i];
        in.terrain_valid[i]=frame.at("terrain_valid")[i];in.contact_valid[i]=frame.at("contact_valid")[i];
      }
      in.command_active=frame.at("command_active");in.full_jacobian=frame.contains("point_jacobian");
      if(in.full_jacobian)for(int i=0;i<4;++i){in.surface_normal[i]=vec(frame.at("surface_normal")[i]);
        for(int r=0;r<3;++r)for(int c=0;c<3;++c)in.point_jacobian[i](r,c)=frame.at("point_jacobian")[i][r][c];}
      auto row=output(core.Step(in,frame.at("dt")));row["counts"]=core.Counts();rows.push_back(row);
    }
  }else if(mode=="memory"){
    TerrainMemory memory(request.at("points_count").get<size_t>(),settings);
    for(const auto& frame:request.at("frames")){
      if(frame.value("reset",false))memory.Reset();
      if(frame.contains("points_w")){
        std::vector<V3> points;for(const auto& p:frame.at("points_w"))points.push_back(vec(p));
        memory.Update(points,frame.at("valid").get<std::vector<bool>>(),frame.at("age_s").get<std::vector<float>>(),
            vec(frame.at("pose_w")),frame.at("yaw_w"),frame.at("time_s"));
      }
      std::array<V3,4> support;for(int i=0;i<4;++i){support[i]=vec(frame.at("wheel_xy")[i]);}
      auto sample=memory.Query(support,frame.at("time_s"));
      rows.push_back({{"height",sample.height},{"valid",sample.valid},{"age",sample.age},{"counts",memory.Counts()}});
    }
  }else if(mode=="feedback"){
    LandingFeedbackReceiver receiver(request.at("binding"));
    for(const auto& frame:request.at("frames")){
      try{receiver.Consume(frame.at("sample"),frame.at("received_s"));
          receiver.Current().RequireFresh(frame.at("now_s"),request.at("maximum_age_s"));
          rows.push_back({{"passed",true},{"sequence",receiver.Current().sequence}});
      }catch(const std::exception& e){rows.push_back({{"passed",false},{"reason",e.what()}});}
    }
  }else if(mode=="clock"){
    SubstepClock clock;
    for(const auto& frame:request){if(frame.value("reset",false))clock.Reset();
      try{rows.push_back({{"passed",true},{"infer",clock.Advance(frame.at("tick"),frame.at("time_s"))}});}
      catch(const std::exception& e){rows.push_back({{"passed",false},{"reason",e.what()}});}}
  }else if(mode=="execution"){
    const auto& robot=request.at("robot");Execution core(contract,robot,"odom",.04);
    core.Reset(request.at("start_s"));
    LandingFeedbackReceiver receiver(request.at("binding"));
    types::RobotAction nominal;
    auto load=[&](const json& m){nominal.kp.resize(16);nominal.kd.resize(16);nominal.goal_joint_pos.resize(16);
      nominal.goal_joint_vel.resize(16);nominal.tau_ff.resize(16);
      for(int j=0;j<16;++j){nominal.kp[j]=m[j][0];nominal.goal_joint_pos[j]=m[j][1];nominal.kd[j]=m[j][2];
        nominal.goal_joint_vel[j]=m[j][3];nominal.tau_ff[j]=m[j][4];}};
    for(const auto& f:request.at("frames")){
      load(f.at("nominal"));receiver.Consume(f.at("feedback"),f.at("now_s"));
      core.SetRawCommand(f.at("raw_command").get<std::array<float,3>>());
      if(f.contains("terrain_source_session"))core.SelectMapSource(f.at("terrain_source_session"),receiver.Current(),f.at("now_s"));
      if(f.contains("obs"))core.ObserveMap(f.at("obs").get<std::vector<float>>(),f.at("age_s").get<std::array<float,187>>(),
          receiver.Current(),f.at("map_session"),f.at("map_sequence"),f.at("query_stamp_ros_s"),f.at("now_s"));
      auto target=f.value("hold",false)?core.Hold(nominal):core.Apply(nominal,receiver.Current(),f.at("now_s"));
      core.RecordPublished(target);
      json row={{"command",json::array()},{"counts",core.Controller().Counts()}};
      const auto m=target.ConvertToMat();for(int j=0;j<16;++j)row["command"].push_back(json::array({m(j,0),m(j,1),m(j,2),m(j,3),m(j,4)}));
      rows.push_back(row);
    }
  }else throw std::runtime_error("unknown CTS probe mode");
  std::cout<<rows.dump()<<std::endl;return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}
}
