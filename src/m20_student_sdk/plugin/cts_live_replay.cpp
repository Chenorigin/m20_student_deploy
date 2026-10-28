// Exercises the actual M20PolicyRunner with explicit synthetic sensor fixtures.
// No RobotInterface/FSM is constructed; no command publisher exists. Require an
// isolated DDS domain so replay cannot consume the live robot's terrain stream.
#include "student_policy_runner.hpp"
#include <iostream>

static nlohmann::json matrix(const types::RobotAction& target){
  const auto m=target.ConvertToMat();nlohmann::json result=nlohmann::json::array();
  for(int j=0;j<16;++j)result.push_back({m(j,0),m(j,1),m(j,2),m(j,3),m(j,4)});return result;
}
int main(int argc,char** argv){
  bool initialized=false;
  try{
    if(argc!=2)throw std::runtime_error("usage: cts_live_replay selected_policy.onnx < synthetic_fixture.json");
    const char* domain=std::getenv("ROS_DOMAIN_ID"),*loopback=std::getenv("ROS_LOCALHOST_ONLY");
    if(!domain||std::string(domain)!="199"||!loopback||std::string(loopback)!="1")
      throw std::runtime_error("CTS replay requires isolated ROS_DOMAIN_ID=199 and ROS_LOCALHOST_ONLY=1");
    if(std::getenv("FASTRTPS_DEFAULT_PROFILES_FILE")||std::getenv("FASTDDS_DEFAULT_PROFILES_FILE"))
      throw std::runtime_error("CTS replay must not use a robot DDS network profile");
    nlohmann::json request;std::cin>>request;
    if(!request.at("frames").is_array()||request["frames"].empty()||request["frames"].size()>32)
      throw std::runtime_error("CTS replay requires 1..32 bounded synthetic frames");
    std::ifstream robot_file(std::getenv("M20_ROBOT_CONFIG"));nlohmann::json robot;robot_file>>robot;
    std::ifstream cfg_file(std::getenv("M20_HEIGHTMAP_CONFIG"));nlohmann::json cfg;cfg_file>>cfg;
    rclcpp::init(0,nullptr);initialized=true;
    nlohmann::json rows=nlohmann::json::array();
    nlohmann::json checks=nlohmann::json::object();
    {
      M20PolicyRunner runner("cts_readonly_replay",argv[1]);
      if(!runner.IsCTS())throw std::runtime_error("CTS replay refuses a legacy model");
      types::MatXf handoff=types::MatXf::Zero(16,5);
      const auto mapping=robot.at("joints").at("policy_from_sdk_indices").get<std::vector<int>>();
      for(int i=0;i<16;++i){const int j=mapping[i];handoff(j,1)=robot["joints"]["default_position_policy_rad"][i];
        handoff(j,0)=robot["joints"]["kp_sdk"][j];handoff(j,2)=robot["joints"]["kd_sdk"][j];}
      runner.OnEnter();runner.SetStandingHandoff(handoff);
      std::unique_ptr<m20_cts::LandingFeedbackReceiver> receiver;
      if(runner.UsesLandingControl())receiver=std::make_unique<m20_cts::LandingFeedbackReceiver>(request.at("binding"));
      const auto started=std::chrono::steady_clock::now();
      uint64_t capture_sequence=0;
      types::RobotAction last_nominal;types::RobotBasicState last_state;
      for(size_t tick=0;tick<request["frames"].size();++tick){
        std::this_thread::sleep_until(started+std::chrono::microseconds(tick*20000));
        const auto& frame=request["frames"][tick];
        auto terrain=std::make_shared<m20_heightmap_msgs::msg::TerrainObservation>();
        const rclcpp::Time stamp(rclcpp::Clock(RCL_SYSTEM_TIME).now().nanoseconds()-
            int64_t(std::llround(frame.value("query_age_s",0.)*1e9)));terrain->header.stamp=stamp;
        terrain->header.frame_id=cfg["frames"]["output"].get<std::string>();terrain->schema_version=2;
        terrain->profile="v1";terrain->session_id=frame.value("terrain_session",uint64_t(1));terrain->sequence=tick+1;terrain->motion_compensated=true;
        terrain->operator_commissioned=true;terrain->geometry_hash=std::getenv("M20_EXPECTED_GEOMETRY_HASH");
        for(int i=0;i<187;++i){terrain->height[i]=frame.at("height")[i];terrain->validity[i]=1;
          terrain->confidence[i]=1;terrain->variance[i]=0;terrain->age[i]=.01f;}
        runner.ReplayTerrain(terrain);
        runner.ReplayInferenceDelay(frame.value("inference_delay_s",0.f));
        types::RobotBasicState ro;ro.joint_pos=handoff.col(1);ro.joint_vel.setZero();ro.body_receive_age_s=.001;
        const float yaw=frame.at("yaw").get<float>();ro.base_rpy=types::Vec3f(0,0,yaw);
        ro.base_rot_mat=Eigen::AngleAxisf(yaw,types::Vec3f::UnitZ()).toRotationMatrix();ro.base_omega[2]=frame.at("yaw_rate");
        types::UserCommand command;const auto raw=frame.at("raw_command").get<std::array<float,3>>();
        command.forward_vel_scale=raw[0];command.side_vel_scale=raw[1];command.turnning_vel_scale=raw[2];
        auto refresh=[&](){
          ro.snapshot_steady_s=m20_cts::SteadySeconds();ro.body_receive_age_s=.001;
          if(!receiver)return;
          const auto now=std::chrono::steady_clock::now();
          const int64_t ns=std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
          auto sample=frame.at("feedback");sample["sequence"]=++capture_sequence;
          sample["capture_monotonic_ns"]=ns;sample["contact_capture_monotonic_ns"]={ns,ns,ns,ns};
          sample["terrain_query"]["session"]=terrain->session_id;sample["terrain_query"]["sequence"]=tick+1;
          sample["terrain_query"]["stamp_ros_s"]=stamp.seconds();
          sample["joint_position_sdk_rad"]=std::vector<float>(ro.joint_pos.data(),ro.joint_pos.data()+16);
          receiver->Consume(sample,m20_cts::SteadySeconds());ro.landing_feedback=receiver->Current();
          ro.snapshot_steady_s=m20_cts::SteadySeconds();ro.body_receive_age_s=.001;
        };
        refresh();const auto infer_start=std::chrono::steady_clock::now();
        auto nominal=runner.getRobotAction(ro,command);
        const double inference_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-infer_start).count();
        nlohmann::json row={{"obs",runner.ReplayObservation()},{"raw_command",runner.ReplayRawCommand()},
          {"nominal_action",runner.ReplayNominalAction()},{"nominal_command",matrix(nominal)},
          {"inference_and_map_ms",inference_ms},{"substeps",nlohmann::json::array()}};
        for(int sub=0;sub<4;++sub){
          if(sub)std::this_thread::sleep_until(started+std::chrono::microseconds(tick*20000+sub*5000));
          // Use a strictly increasing synthetic capture sequence even though
          // the sensor values are intentionally held fixed for this fixture.
          refresh();
          auto issued=runner.ExecuteCTSSubstep(nominal,ro);
          row["substeps"].push_back(matrix(issued));
          runner.RecordCTSCommandPublished(issued.ConvertToMat());
          runner.FlushCTSTrace();
        }
        row["landing_counts"]=runner.ReplayLandingCounts();
        runner.FlushCTSTrace();
        rows.push_back(row);
        last_nominal=nominal;last_state=ro;
      }
      if(!runner.UsesLandingControl()){
        auto rejects=[](auto operation){try{operation();return false;}catch(const std::exception&){return true;}};
        auto bad=last_nominal;bad.goal_joint_pos[1]=100;
        checks["out_of_bounds_target_rejected"]=rejects([&]{runner.ExecuteCTSSubstep(bad,last_state);});
        bad=last_nominal;bad.goal_joint_vel[3]=NAN;
        checks["nonfinite_target_rejected"]=rejects([&]{runner.ExecuteCTSSubstep(bad,last_state);});
        auto stale=last_state;stale.body_receive_age_s=1;
        checks["stale_body_rejected"]=rejects([&]{runner.ExecuteCTSSubstep(last_nominal,stale);});
        auto invalid=last_state;invalid.joint_pos[0]=NAN;
        checks["nonfinite_body_rejected"]=rejects([&]{runner.ExecuteCTSSubstep(last_nominal,invalid);});
        auto published=last_nominal;published.goal_joint_pos[1]+=.0001f;
        runner.RecordCTSCommandPublished(published.ConvertToMat());
        const auto previous=runner.ReplayNominalAction();
        std::this_thread::sleep_for(std::chrono::milliseconds(35));
        types::UserCommand stopped;
        const auto held=runner.getRobotAction(last_state,stopped);
        published.goal_joint_vel.setZero();published.tau_ff.setZero();
        checks["missed_policy_tick_latches_hold"]=runner.CTSNominalIsHeld();
        checks["hold_preserves_published_target_and_pd"]=held.ConvertToMat().isApprox(published.ConvertToMat());
        checks["hold_stops_wheels_and_feedforward"]=held.goal_joint_vel.isZero()&&held.tau_ff.isZero();
        checks["hold_does_not_advance_previous_action"]=previous==runner.ReplayNominalAction();
        checks["stale_body_rejected_after_hold"]=rejects([&]{runner.ExecuteCTSSubstep(last_nominal,stale);});
        for(const auto& result:checks)if(!result.get<bool>())throw std::runtime_error("nominal_pd replay regression failed: "+checks.dump());
      }
    }
    rclcpp::shutdown();initialized=false;
    std::cout<<nlohmann::json({{"hardware_commands",0},{"command_publishers",0},{"fixture_scope","synthetic isolated live-runner replay"},
        {"execution_mode",m20_cts::ExecutionMode(cfg)},{"checks",checks},
        {"scenario",request.value("scenario",std::string("nominal"))},{"frames",rows}}).dump()<<std::endl;
    return 0;
  }catch(const std::exception& e){if(initialized)rclcpp::shutdown();std::cerr<<e.what()<<std::endl;return 2;}
}
