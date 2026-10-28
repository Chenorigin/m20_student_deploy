#pragma once
#include "policy_runner_base.hpp"
#include "json.hpp"
#include "student_control_core.hpp"
#include "student_trace.hpp"
#include "student_heading_feedback.hpp"
#include "cts_execution.hpp"
#include "cts_execution_mode.hpp"
#include <m20_heightmap_msgs/msg/terrain_observation.hpp>
#include <rclcpp/rclcpp.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <deque>
#include <fstream>
#include <mutex>
#include <set>
#include <thread>
#include <functional>

// Used only inside the existing official SDK FSM after the operator starts it.
// This class does not publish motor commands. The SDK remains the only writer.
class M20PolicyRunner : public PolicyRunnerBase {
  using Terrain=m20_heightmap_msgs::msg::TerrainObservation;
  using Clock=std::chrono::steady_clock;
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING,"m20_student"};
  Ort::Session session_{nullptr};
  Ort::MemoryInfo memory_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault)};
  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Subscription<Terrain>::SharedPtr subscription_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread reader_;
  std::mutex mutex_;
  Terrain::SharedPtr terrain_;
  std::string terrain_fault_;
  Clock::time_point received_,last_tick_,last_target_time_;
  double received_query_age_=0;
  bool have_tick_=false,have_watermark_=false;
  uint64_t source_session_=0,source_sequence_=0;
  uint64_t policy_terrain_session_=0;
  uint64_t source_generation_=0,policy_source_generation_=0;
  bool have_policy_session_=false;
  std::set<uint64_t> retired_;
  std::deque<std::array<float,57>> history_;
  nlohmann::json robot_,cfg_;
  std::string profile_,expected_geometry_;
  std::vector<int> policy_from_sdk_,sdk_from_policy_;
  int dim_=0,history_steps_=0;
  float map_age_=0,align_age_=0,tau_=0,min_fraction_=0,future_=0;
  bool coverage_hold_enabled_=true;
  bool terrain_timing_hold_enabled_=true;
  bool terrain_source_session_hold_enabled_=true;
  bool warned_=false;
  std::unique_ptr<StudentControlCore> control_;
  std::unique_ptr<StudentHeadingFeedback> heading_;
  bool cts_=false;
  bool cts_nominal_pd_=false,have_nominal_issued_=false;
  RobotAction nominal_issued_;
  std::function<void()> pending_cts_trace_;
  bool pending_substep_trace_=false;
#ifdef M20_CTS_OFFLINE_REPLAY
  float replay_after_inference_delay_s_=0;
#endif
  std::unique_ptr<m20_cts::Execution> cts_execution_;
  std::array<bool,187> policy_map_mask_{};
  uint64_t feedback_step_=0;
  double tick_elapsed_s_=0;
  static constexpr double kMinTickIntervalS=.010;
  double max_tick_interval_s_=.025;
  bool tick_interval_ok(double elapsed) const {
    return std::isfinite(elapsed)&&elapsed>=kMinTickIntervalS&&elapsed<=max_tick_interval_s_;
  }
  StudentTrace& trace_=StudentTrace::Instance();
  uint64_t trace_tick_=0;
  uint64_t trace_episode_=0,trace_policy_tick_=0;
  bool trace_have_anchor_=false;
  int64_t trace_anchor_monotonic_ns_=0,trace_anchor_wall_ns_=0;
  nlohmann::json received_stamp_,trace_terrain_received_stamp_;
  Terrain::SharedPtr trace_terrain_frame_;
  double trace_terrain_age_=-1,trace_inference_ms_=0;
  double trace_receive_age_=-1,trace_received_query_age_=-1,trace_query_stamp_=0;
  uint64_t trace_source_session_=0,trace_source_sequence_=0;
  bool trace_have_terrain_=false;
  bool trace_source_session_changed_=false;
  int trace_valid_=-1;
  int trace_front_valid_=-1,trace_rear_valid_=-1;
  std::array<float,16> trace_raw_{};
  std::array<float,3> trace_raw_command_{};
  bool trace_inferred_=false;
  bool trace_full_observation_=true;
  std::vector<float> trace_observation_;
  std::array<float,187> trace_cell_age_{};
  std::string trace_terrain_input_mode_="unknown_no_frame";
  struct InferenceTrace {
    std::vector<float> observation;
    std::array<float,16> raw_action{};
    nlohmann::json start_stamp,end_stamp;
    bool has_output=false;
    std::string disposition="started",reason,terrain_input_mode;
  };
  std::vector<InferenceTrace> trace_attempts_;

  // Instrumentation must never become a new controller fault or safety gate.
  template<class F> static void trace_only(F&& operation) noexcept {
    try{operation();}catch(...){try{StudentTrace::Instance().RecordConstructionFailure();}catch(...){}}
  }
  void trace_mode(const char* mode) noexcept {trace_only([&]{trace_terrain_input_mode_=mode;});}
  static nlohmann::json terrain_payload(const Terrain& t) {
    return {{"schema_version",t.schema_version},{"profile",t.profile},
      {"session",t.session_id},{"sequence",t.sequence},{"frame_id",t.header.frame_id},
      {"source_stamp_ros_ns",int64_t(t.header.stamp.sec)*1000000000LL+t.header.stamp.nanosec},
      {"source_stamp_semantics","query_pose_source_time_not_publication_or_schedule"},
      {"motion_compensated",t.motion_compensated},{"geometry_hash",t.geometry_hash},
      {"operator_commissioned",t.operator_commissioned},
      {"height",t.height},{"validity",t.validity},{"confidence",t.confidence},
      {"variance",t.variance},{"age",t.age}};
  }
  size_t trace_begin_attempt(const std::vector<float>& obs) noexcept {
    size_t index=static_cast<size_t>(-1);
    trace_only([&]{
      InferenceTrace attempt;attempt.observation=obs;attempt.terrain_input_mode=trace_terrain_input_mode_;
      trace_attempts_.push_back(std::move(attempt));index=trace_attempts_.size()-1;
      trace_attempts_[index].start_stamp=StudentTrace::Stamp();
    });
    return index;
  }
  void trace_end_attempt(size_t index,std::vector<Ort::Value>* result,
                         const char* disposition,const char* reason="") noexcept {
    trace_only([&]{
      if(index>=trace_attempts_.size())return;
      auto& attempt=trace_attempts_[index];attempt.end_stamp=StudentTrace::Stamp();
      attempt.disposition=disposition;attempt.reason=reason;
      if(result&&!result->empty()){
        const float* output=(*result)[0].GetTensorMutableData<float>();
        std::copy(output,output+16,attempt.raw_action.begin());attempt.has_output=true;
      }
    });
  }
  void trace_disposition(size_t index,const char* disposition,const char* reason="") noexcept {
    trace_only([&]{if(index<trace_attempts_.size()){
      trace_attempts_[index].disposition=disposition;trace_attempts_[index].reason=reason;
    }});
  }
  void trace_emit_attempts() noexcept {
    trace_only([&]{for(size_t i=0;i<trace_attempts_.size();++i){
      const auto& attempt=trace_attempts_[i];
      nlohmann::json row={{"event","inference_attempt"},{"inference_attempt",i},
        {"observation",attempt.observation},{"input_dim",dim_},
        {"raw_command",trace_raw_command_},{"cts_dual_input",cts_},
        {"inference_start_stamp",attempt.start_stamp},{"inference_end_stamp",attempt.end_stamp},
        {"has_output",attempt.has_output},{"disposition",attempt.disposition},{"reason",attempt.reason},
        {"terrain_input_mode",attempt.terrain_input_mode}};
      if(attempt.has_output)row["raw_action"]=attempt.raw_action;
      if(attempt.end_stamp.is_object())row.update(attempt.end_stamp);
      else if(attempt.start_stamp.is_object())row.update(attempt.start_stamp);
      trace_.Push(std::move(row));
    }});
  }

  static nlohmann::json read(const char* key) {
    const char* value=std::getenv(key);if(!value)throw std::runtime_error(std::string("missing environment ")+key);
    std::ifstream f(value);if(!f)throw std::runtime_error(std::string("cannot read ")+key);
    nlohmann::json j;f>>j;return j;
  }
  void reset() {history_.assign(history_steps_-1,{});have_tick_=false;feedback_step_=0;if(heading_)heading_->Reset();}
  void receive(Terrain::SharedPtr t) {
    nlohmann::json stamp,row;
    trace_only([&]{stamp=StudentTrace::Stamp();row=terrain_payload(*t);
      row["event"]="terrain_received";row["receive_stamp"]=stamp;
      row["theoretical_monotonic_ns"]=nullptr;row["theoretical_wall_ns"]=nullptr;
      row["theoretical_time_basis"]="not_applicable_async_terrain_receive";
      if(stamp.is_object())row.update(stamp);
    });
    auto record=[&](const char* status,const char* reason="") noexcept {
      trace_only([&]{row["acceptance"]=status;row["reason"]=reason;trace_.Push(std::move(row));});
    };
    std::lock_guard<std::mutex> guard(mutex_);
    auto reject=[&](const char* reason){terrain_.reset();if(terrain_fault_.empty())terrain_fault_=reason;record("rejected",reason);};
    const double query=t->header.stamp.sec+t->header.stamp.nanosec*1e-9;
    const double elapsed=node_->now().seconds()-query;
    if(t->schema_version!=2||!t->operator_commissioned||t->geometry_hash!=expected_geometry_||t->profile!=profile_||t->header.frame_id!=cfg_["frames"]["output"].get<std::string>()||!t->motion_compensated){reject("terrain contract or pose source mismatch");return;}
    if(!std::isfinite(elapsed)||elapsed < -future_){reject("terrain clock invalid or from future");return;}
    for(int i=0;i<187;++i){float h=t->height[i],m=t->validity[i],c=t->confidence[i],v=t->variance[i],a=t->age[i];if(!std::isfinite(h)||!std::isfinite(m)||!std::isfinite(c)||!std::isfinite(v)||!std::isfinite(a)||std::abs(h)>1||(m!=0&&m!=1)||c<0||c>1||v<0||(m==0&&(h!=0||c!=0))||(m==1&&a<0)){reject("invalid terrain channel");return;}}
    if(retired_.count(t->session_id)||(have_watermark_&&t->session_id==source_session_&&t->sequence<=source_sequence_)){record("ignored","retired session or nonincreasing sequence");return;}
    if(have_watermark_&&t->session_id!=source_session_) {
      if(retired_.size()>=128){reject("terrain source session history exhausted");return;}
      retired_.insert(source_session_);
      terrain_.reset();
      ++source_generation_;
    }
    source_session_=t->session_id;source_sequence_=t->sequence;have_watermark_=true;
    if(elapsed>align_age_){terrain_.reset();record("rejected","terrain source age exceeds alignment limit");return;}
    terrain_=std::move(t);received_=Clock::now();received_query_age_=elapsed;
    trace_only([&]{received_stamp_=stamp;});
    record("accepted");
  }
  RobotAction hold(const RobotBasicState&,const std::string& reason) {
    auto target=control_->Hold(reason);
    if(cts_execution_)target=cts_execution_->Hold(target);
    if(cts_nominal_pd_&&have_nominal_issued_){
      target=nominal_issued_;target.goal_joint_vel.setZero();target.tau_ff.setZero();
    }
    if(!warned_){std::cerr<<"[STUDENT_HOLD_LATCHED] "<<control_->Reason()<<"; operator re-entry required"<<std::endl;warned_=true;}
    return target;
  }

public:
  M20PolicyRunner(const std::string& name,const std::string& path):PolicyRunnerBase(name) {
    robot_=read("M20_ROBOT_CONFIG");cfg_=read("M20_HEIGHTMAP_CONFIG");
    if(robot_.at("policy").contains("max_tick_interval_s")){
      const auto& value=robot_.at("policy").at("max_tick_interval_s");
      if(!value.is_number())throw std::runtime_error("max_tick_interval_s must be numeric");
      max_tick_interval_s_=value.get<double>();
      if(!std::isfinite(max_tick_interval_s_)||max_tick_interval_s_<.02||max_tick_interval_s_>=.04)
        throw std::runtime_error("max_tick_interval_s must be >=0.02 and <0.04");
    }
    if(!cfg_.at("calibration_confirmed").get<bool>())throw std::runtime_error("physical calibration is not confirmed");
    const char* geometry=std::getenv("M20_EXPECTED_GEOMETRY_HASH");
    if(!geometry||std::string(geometry).size()!=64)throw std::runtime_error("operator geometry identity missing");
    expected_geometry_=geometry;
    profile_=cfg_.at("profile");history_steps_=cfg_.at("history_steps");
    if((profile_=="v0"&&history_steps_!=1)||(profile_=="v1"&&history_steps_!=4)||(profile_!="v0"&&profile_!="v1"))throw std::runtime_error("student profile/history mismatch");
    dim_=profile_=="v0"?618:789;
    if(const char* flag=std::getenv("M20_TRACE_FULL_OBSERVATION")){
      const std::string value(flag);
      if(value!="0"&&value!="1")throw std::runtime_error("M20_TRACE_FULL_OBSERVATION must be 0 or 1");
      trace_full_observation_=value=="1";
    }
    std::filesystem::path contract_path=cfg_["aos"]["model_contract"].get<std::string>();
    if(contract_path.is_relative())contract_path=std::filesystem::path(std::getenv("M20_HEIGHTMAP_CONFIG")).parent_path()/contract_path;
    std::ifstream contract_file(contract_path);nlohmann::json contract_json;contract_file>>contract_json;
    if(contract_json.at("version")!=(profile_=="v0"?"m20_student_v0_618":"m20_student_v1")||contract_json.at("student_observation_dim")!=dim_||contract_json.at("action_dim")!=16||contract_json.at("slices").at("height")!=nlohmann::json({57,244})||contract_json.at("slices").at("validity")!=nlohmann::json({244,431})||contract_json.at("slices").at("confidence")!=nlohmann::json({431,618})||contract_json.at("grid").at("rows_y")!=11||contract_json.at("grid").at("columns_x")!=17)throw std::runtime_error("checkpoint observation contract mismatch");
    if(profile_=="v1"&&(contract_json.at("history_steps")!=history_steps_||contract_json.at("slices").at("past_proprio")!=nlohmann::json({618,789})))throw std::runtime_error("history contract mismatch");
    cts_=contract_json.value("deployment_family",std::string())=="m20_cts_student";
    if(!cts_&&contract_json.at("deployment_family")!="m20_student_v3_5")throw std::runtime_error("requires student-v3.5 or CTS deployment metadata");
    cts_nominal_pd_=cts_&&m20_cts::ExecutionMode(cfg_)=="nominal_pd";
    heading_=std::make_unique<StudentHeadingFeedback>(contract_json.at("command_feedback"));
    const auto& perception=contract_json.at("perception");
    if(!perception.at("front_only").get<bool>())throw std::runtime_error("this deployment requires front-only student sensor contract");
    const auto sensor=perception.at("front_sensor_offset_m").get<std::vector<float>>();
    const float fov=perception.at("lidar_horizontal_fov_deg").get<float>();
    const float blind=perception.at("front_blind_distance_m").get<float>();
    if(sensor.size()!=3||!std::isfinite(fov)||fov<=0||fov>360||!std::isfinite(blind)||blind<0)throw std::runtime_error("invalid sensor output mask");
    for(float value:sensor)if(!std::isfinite(value))throw std::runtime_error("nonfinite front sensor transform");
    for(int i=0;i<187;++i){
      const float x=(i%17-8)*.1f,y=(i/17-5)*.1f;
      const float dx=x-sensor[0],dy=y-sensor[1];
      policy_map_mask_[i]=std::hypot(dx,dy)>blind+1.e-6f&&dx>0.f&&std::abs(std::atan2(dy,dx))<=fov*.5f*float(M_PI)/180.f+1.e-7f;
    }
    policy_from_sdk_=robot_["joints"]["policy_from_sdk_indices"].get<std::vector<int>>();
    sdk_from_policy_=robot_["joints"]["sdk_from_policy_indices"].get<std::vector<int>>();
    if(policy_from_sdk_.size()!=16||sdk_from_policy_.size()!=16)throw std::runtime_error("invalid joint mapping");
    std::set<int> permutation(policy_from_sdk_.begin(),policy_from_sdk_.end());
    if(permutation.size()!=16||*permutation.begin()!=0||*permutation.rbegin()!=15)throw std::runtime_error("joint mapping is not a permutation");
    for(int i=0;i<16;++i)if(sdk_from_policy_.at(policy_from_sdk_[i])!=i)throw std::runtime_error("joint mappings are not inverse");
    for(const char* key:{"default_position_policy_rad","action_scale_policy","kp_sdk","kd_sdk"}){const auto& a=robot_.at("joints").at(key);if(!a.is_array()||a.size()!=16)throw std::runtime_error("joint parameter shape");for(const auto& value:a)if(!value.is_number()||!std::isfinite(value.get<double>()))throw std::runtime_error("nonfinite joint parameter");}
    for(int i=0;i<16;++i)if(robot_["joints"]["action_scale_policy"][i].get<double>()<=0||robot_["joints"]["kp_sdk"][i].get<double>()<0||robot_["joints"]["kd_sdk"][i].get<double>()<0)throw std::runtime_error("invalid scale or PD");
    auto check_bounds=[](const nlohmann::json& b){if(!b.is_array()||b.size()!=2||!b[0].is_number()||!b[1].is_number()||!std::isfinite(b[0].get<double>())||!std::isfinite(b[1].get<double>())||b[0].get<double>()>=b[1].get<double>())throw std::runtime_error("invalid target bounds");};
    check_bounds(robot_["policy"]["normalized_action_clip"]);
    if(robot_["policy"].contains("leg_target_slew_enabled")&&!robot_["policy"]["leg_target_slew_enabled"].is_boolean())throw std::runtime_error("leg_target_slew_enabled must be a boolean");
    for(const char* key:{"joint_target_position_clip_enabled","wheel_target_velocity_clip_enabled"})
      if(robot_["policy"].contains(key)&&!robot_["policy"][key].is_boolean())throw std::runtime_error(std::string(key)+" must be a boolean");
    if(robot_["policy"].contains("terrain_source_session_hold_enabled")&&!robot_["policy"]["terrain_source_session_hold_enabled"].is_boolean())throw std::runtime_error("terrain_source_session_hold_enabled must be a boolean");
    terrain_source_session_hold_enabled_=robot_["policy"].value("terrain_source_session_hold_enabled",true);
    if(robot_["policy"]["processed_target_clip_policy"].size()!=16)throw std::runtime_error("target limit shape");
    for(const auto& b:robot_["policy"]["processed_target_clip_policy"])check_bounds(b);
    if(robot_["hardware"]["software_position_limits_sdk_rad"].size()!=16||robot_["hardware"]["software_velocity_limits_sdk_rad_s"].size()!=16)throw std::runtime_error("software limit shape");
    for(const auto& b:robot_["hardware"]["software_position_limits_sdk_rad"])check_bounds(b);
    if(robot_["hardware"].contains("software_feedback_position_limits_sdk_rad")){
      const auto& feedback=robot_["hardware"]["software_feedback_position_limits_sdk_rad"];
      if(!feedback.is_array()||feedback.size()!=16)throw std::runtime_error("feedback position limit shape");
      for(int j=0;j<16;++j){
        check_bounds(feedback[j]);
        const auto& target=robot_["hardware"]["software_position_limits_sdk_rad"][j];
        if(j%4!=3&&(feedback[j][0].get<double>()>target[0].get<double>()||feedback[j][1].get<double>()<target[1].get<double>()))throw std::runtime_error("feedback limits exclude permitted position targets");
      }
    }
    for(const auto& v:robot_["hardware"]["software_velocity_limits_sdk_rad_s"])if(!v.is_number()||!std::isfinite(v.get<double>())||v.get<double>()<=0)throw std::runtime_error("invalid commissioning speed limit");
    for(const char* key:{"lin_vel_x","lin_vel_y","ang_vel_z"})check_bounds(robot_["command_training_ranges"][key]);
    map_age_=cfg_["estimator"]["max_map_age_s"];align_age_=cfg_["aos"]["max_alignment_age_s"];
    tau_=cfg_["estimator"]["confidence_age_tau_s"];min_fraction_=cfg_["aos"]["min_valid_fraction"];future_=cfg_["runtime"]["future_tolerance_s"];
    coverage_hold_enabled_=cfg_.at("aos").value("coverage_hold_enabled",true);
    terrain_timing_hold_enabled_=cfg_.at("aos").value("terrain_timing_hold_enabled",true);
    double reference=cfg_["estimator"]["confidence_reference_std"],growth=cfg_["estimator"]["variance_growth_per_s"],body_age=cfg_["aos"]["max_proprio_age_s"];
    for(double v:{static_cast<double>(map_age_),static_cast<double>(align_age_),static_cast<double>(tau_),static_cast<double>(min_fraction_),static_cast<double>(future_),reference,growth,body_age})if(!std::isfinite(v))throw std::runtime_error("nonfinite timing/quality limit");
    if(map_age_<=0||align_age_<=0||align_age_>map_age_||tau_<=0||min_fraction_<=0||min_fraction_>1||future_<=0||reference<=0||growth<0||body_age<=0)throw std::runtime_error("invalid deployment timing/quality limits");
    Ort::SessionOptions options;options.SetIntraOpNumThreads(1);options.SetInterOpNumThreads(1);
    options.AddConfigEntry("session.disable_prepacking","1");
    session_=Ort::Session(env_,path.c_str(),options);
    if(session_.GetInputCount()!=(cts_?2u:1u)||session_.GetOutputCount()!=1||session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>({1,dim_})||session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>({1,16}))throw std::runtime_error("ONNX model does not match student profile");
    Ort::AllocatorWithDefaultOptions allocator;
    auto metadata=session_.GetModelMetadata();
    const char* expected_contract=std::getenv("M20_VERIFIED_MODEL_CONTRACT_SHA256");
    auto embedded_contract=metadata.LookupCustomMetadataMapAllocated("contract_sha256",allocator);
    if(!expected_contract||std::string(expected_contract).size()!=64||!embedded_contract||std::string(embedded_contract.get())!=expected_contract)
      throw std::runtime_error("ONNX embedded contract digest differs from verified sidecar");
    auto family=metadata.LookupCustomMetadataMapAllocated("deployment_family",allocator);
    auto feedback=metadata.LookupCustomMetadataMapAllocated("command_feedback",allocator);
    if(!family||std::string(family.get())!=(cts_?"m20_cts_student":"m20_student_v3_5")||!feedback||
       nlohmann::json::parse(feedback.get())!=contract_json.at("command_feedback"))
      throw std::runtime_error("ONNX feedback metadata disagrees with model contract");
    if(cts_){
      const auto raw_type=session_.GetInputTypeInfo(1);const auto raw_info=raw_type.GetTensorTypeAndShapeInfo();
      auto raw_name=session_.GetInputNameAllocated(1,allocator);
      auto embedded_control=metadata.LookupCustomMetadataMapAllocated("quiet_execution_control",allocator);
      auto control_digest=metadata.LookupCustomMetadataMapAllocated("quiet_execution_control_sha256",allocator);
      if(dim_!=789||std::string(raw_name.get())!="raw_command"||raw_info.GetShape()!=std::vector<int64_t>({1,3})||
         raw_info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||!embedded_control||!control_digest||
         nlohmann::json::parse(embedded_control.get())!=contract_json.at("quiet_execution_control")||
         std::string(control_digest.get())!=contract_json.at("quiet_execution_control_sha256").get<std::string>())
        throw std::runtime_error("CTS embedded input/execution metadata mismatch");
      if(!cts_nominal_pd_)
        cts_execution_=std::make_unique<m20_cts::Execution>(contract_json,robot_,cfg_.at("frames").at("world").get<std::string>(),body_age);
      // Initialize this session's lazy inference allocations before the first
      // real servo deadline. Discard outputs; no history, target or PD update.
      std::vector<float> warm_observation(dim_,0.f);
      std::array<float,3> warm_command{};
      const std::array<int64_t,2> warm_shape{1,dim_},warm_command_shape{1,3};
      std::vector<Ort::Value> warm_inputs;
      warm_inputs.push_back(Ort::Value::CreateTensor<float>(memory_,warm_observation.data(),warm_observation.size(),warm_shape.data(),2));
      warm_inputs.push_back(Ort::Value::CreateTensor<float>(memory_,warm_command.data(),warm_command.size(),warm_command_shape.data(),2));
      const char* warm_names[]={"obs","raw_command"};const char* warm_outputs[]={"actions"};
      double first_ms=0,last_ms=0;
      for(int pass=0;pass<8;++pass){
        const auto begin=Clock::now();
        auto result=session_.Run(Ort::RunOptions{nullptr},warm_names,warm_inputs.data(),2,warm_outputs,1);
        last_ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
        if(pass==0)first_ms=last_ms;
        const auto* values=result[0].GetTensorData<float>();
        for(int j=0;j<16;++j)if(!std::isfinite(values[j]))throw std::runtime_error("nonfinite CTS warmup output");
      }
      std::cout<<"[CTS_WARMUP] runs=8 first_ms="<<first_ms<<" last_ms="<<last_ms
               <<" outputs_discarded=true"<<std::endl;
    }
    control_=std::make_unique<StudentControlCore>(robot_);
    if(cts_)control_->RequirePublicationCommit();
    SetDecimation(4);reset();node_=std::make_shared<rclcpp::Node>("m20_student_terrain_reader");
    subscription_=node_->create_subscription<Terrain>(cfg_["topics"]["terrain"].get<std::string>(),rclcpp::QoS(1).best_effort(),[this](Terrain::SharedPtr t){receive(std::move(t));});
    executor_.add_node(node_);reader_=std::thread([this](){executor_.spin();});
  }
  ~M20PolicyRunner(){executor_.cancel();if(reader_.joinable())reader_.join();}
  void DisplayPolicyInfo() override {
    std::cout<<"M20 student-v3.5 sensor heading feedback; ABI "<<profile_<<" "<<dim_<<"D ->16D; 20ms; official SDK FSM"<<std::endl;
    if(cts_)std::cout<<"[CTS_EXECUTION] mode="<<m20_cts::ExecutionMode(cfg_)
                    <<" landing_control="<<(cts_execution_?"enabled":"disabled")
                    <<" policy_period_s="<<m20_cts::servo_period_s*m20_cts::policy_decimation
                    <<" servo_period_s="<<m20_cts::servo_period_s
                    <<" servo_interval_min_s="<<m20_cts::servo_interval_min_s
                    <<" servo_interval_max_s="<<m20_cts::servo_interval_max_s<<std::endl;
    std::cout<<"[STUDENT_CONFIG] coverage_hold_enabled="<<(coverage_hold_enabled_?"true":"false")
             <<" terrain_timing_hold_enabled="<<(terrain_timing_hold_enabled_?"true":"false")
             <<" terrain_source_session_hold_enabled="<<(terrain_source_session_hold_enabled_?"true":"false")
             <<" min_valid_fraction="<<min_fraction_<<" alignment_age_s="<<align_age_
             <<" confidence_step_ambiguity_weight="<<cfg_["estimator"].value("confidence_step_ambiguity_weight",1.)
             <<" leg_target_slew_enabled="<<(control_->LegTargetSlewEnabled()?"true":"false")
             <<" joint_target_position_clip_enabled="<<(control_->JointTargetPositionClipEnabled()?"true":"false")
             <<" wheel_target_velocity_clip_enabled="<<(control_->WheelTargetVelocityClipEnabled()?"true":"false")
             <<" trace_full_observation="<<(trace_full_observation_?"true":"false")
             <<" policy_tick_interval_min_s="<<kMinTickIntervalS<<" policy_tick_interval_max_s="<<max_tick_interval_s_
             <<" configured_speed_limits_sdk_rad_s="<<robot_["hardware"]["software_velocity_limits_sdk_rad_s"]<<std::endl;
  }
  int GetObservationDim()const{return dim_;}
  int GetLidarObservationDim()const{return 0;}
  int GetDepthObservationDim()const{return 0;}
  int GetV11ObservationDim()const{return 0;}
  bool UsesDepthInput()const{return false;}
  void OnEnter() override {
    reset();have_policy_session_=false;warned_=false;run_cnt_=0;
    have_nominal_issued_=false;
    if(cts_execution_)cts_execution_->Reset(m20_cts::SteadySeconds());
    ++trace_episode_;trace_policy_tick_=0;trace_have_anchor_=false;
    std::lock_guard<std::mutex> guard(mutex_);
    if(terrain_){
      const double age=node_->now().seconds()-(terrain_->header.stamp.sec+terrain_->header.stamp.nanosec*1e-9);
      if(std::isfinite(age)&&age>=-future_&&age<=align_age_&&
         std::chrono::duration<double>(Clock::now()-received_).count()<=align_age_)
        terrain_fault_.clear();
    }
  }
  void SetStandingHandoff(const MatXf& command){control_->Begin(command);last_tick_=last_target_time_=Clock::now();
    if(cts_execution_){RobotAction issued;issued.kp=command.col(0);issued.goal_joint_pos=command.col(1);
      issued.kd=command.col(2);issued.goal_joint_vel=command.col(3);issued.tau_ff=command.col(4);
      cts_execution_->RecordPublished(issued);}
    if(cts_nominal_pd_){nominal_issued_.kp=command.col(0);nominal_issued_.goal_joint_pos=command.col(1);
      nominal_issued_.kd=command.col(2);nominal_issued_.goal_joint_vel=command.col(3);
      nominal_issued_.tau_ff=command.col(4);have_nominal_issued_=true;}}
  bool IsCTS()const{return cts_;}
  bool UsesLandingControl()const{return bool(cts_execution_);}
  void CheckCTSBody(const RobotBasicState& state){
    control_->CheckBody(state,cfg_["aos"]["max_proprio_age_s"].get<double>());
    if(cts_execution_)cts_execution_->CheckFeedback(state.landing_feedback,m20_cts::SteadySeconds());
  }
  void LatchCTSTimingHold(const std::string& reason){(void)hold(RobotBasicState{},reason);}
  void FlushCTSTrace() noexcept {
    if(pending_cts_trace_){auto emit=std::move(pending_cts_trace_);pending_cts_trace_=nullptr;trace_only(emit);}
    if(pending_substep_trace_){
      pending_substep_trace_=false;
      trace_only([&]{auto row=StudentTrace::Stamp();row["event"]="cts_execution_substep";
        row["execution_mode"]=m20_cts::ExecutionMode(cfg_);
        if(cts_execution_){row["landing_counts"]=cts_execution_->Controller().Counts();
          row["phase"]=cts_execution_->LastOutput().phase;row["kp_scale"]=cts_execution_->LastOutput().kp_scale;}
        row["publication_scope"]="SDK receipt returned; motor execution is not acknowledged";trace_.Push(std::move(row));});
    }
  }
#ifdef M20_CTS_OFFLINE_REPLAY
  // Compiled only into the read-only replay executable, never student_control.
  void ReplayTerrain(Terrain::SharedPtr value){receive(std::move(value));}
  const std::vector<float>& ReplayObservation()const{return trace_observation_;}
  const std::array<float,3>& ReplayRawCommand()const{return trace_raw_command_;}
  const std::array<float,16>& ReplayNominalAction()const{return control_->CandidatePrevious();}
  const std::array<uint64_t,6>& ReplayLandingCounts()const{static const std::array<uint64_t,6> none{};
    return cts_execution_?cts_execution_->Controller().Counts():none;}
  void ReplayInferenceDelay(float seconds){if(!std::isfinite(seconds)||seconds<0||seconds>.03f)throw std::runtime_error("replay delay out of bounds");replay_after_inference_delay_s_=seconds;}
#endif
  RobotAction ExecuteCTSSubstep(const RobotAction& nominal,const RobotBasicState& state){
    if(!cts_)throw std::runtime_error("CTS substep called for legacy policy");
    control_->CheckBody(state,cfg_["aos"]["max_proprio_age_s"].get<double>());
    if(cts_nominal_pd_){
      const auto target=control_->Latched()?hold(state,control_->Reason()):nominal;
      control_->ValidateTarget(target);return target;
    }
    if(!cts_execution_)throw std::runtime_error("CTS landing execution missing");
    cts_execution_->CheckFeedback(state.landing_feedback,m20_cts::SteadySeconds());
    if(control_->Latched())return cts_execution_->Hold(nominal);
    return cts_execution_->Apply(nominal,state.landing_feedback,m20_cts::SteadySeconds());
  }
  void RecordCTSCommandPublished(const MatXf& command){if(cts_){
    control_->RecordPublished(command);
    RobotAction issued;issued.kp=command.col(0);issued.goal_joint_pos=command.col(1);
    issued.kd=command.col(2);issued.goal_joint_vel=command.col(3);issued.tau_ff=command.col(4);
    if(cts_nominal_pd_){nominal_issued_=issued;have_nominal_issued_=true;}
    if(cts_execution_)cts_execution_->RecordPublished(issued);
    pending_substep_trace_=true;
  }}
  bool CTSNominalIsHeld()const{return control_->Latched();}
  RobotAction getRobotAction(const RobotBasicState& ro,const UserCommand& uc) override {
    FlushCTSTrace();
    // Use the same command sample for the tensor and its audit row.
    const std::array<float,3> command={uc.forward_vel_scale.load(),uc.side_vel_scale.load(),uc.turnning_vel_scale.load()};
    const uint64_t policy_tick=trace_policy_tick_++,legacy_tick=trace_tick_++;
    trace_terrain_age_=-1;trace_valid_=-1;trace_inference_ms_=0;trace_inferred_=false;trace_raw_.fill(0);
    trace_receive_age_=trace_received_query_age_=-1;trace_have_terrain_=false;
    trace_source_session_changed_=false;
    trace_front_valid_=trace_rear_valid_=-1;
    trace_terrain_frame_.reset();trace_observation_.clear();trace_attempts_.clear();trace_cell_age_.fill(-1.f);
    trace_only([&]{trace_terrain_input_mode_="unknown_no_frame";trace_terrain_received_stamp_=nullptr;});
    auto vector=[](const auto& v){return std::vector<float>(v.data(),v.data()+v.size());};
    const double mono=std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
    nlohmann::json row;
    trace_only([&]{
      StudentTrace::Context().clear();
      auto stamp=StudentTrace::Stamp();
      if(!trace_have_anchor_){
        trace_anchor_monotonic_ns_=stamp.at("actual_monotonic_ns").get<int64_t>();
        trace_anchor_wall_ns_=stamp.at("actual_wall_ns").get<int64_t>();trace_have_anchor_=true;
      }
      const int64_t period_ns=std::llround(robot_.at("policy").at("dt_s").get<double>()*1000000000.);
      StudentTrace::SetContext({{"episode",trace_episode_},{"policy_tick",policy_tick},
        {"theoretical_monotonic_ns",trace_anchor_monotonic_ns_+int64_t(policy_tick)*period_ns},
        {"theoretical_wall_ns",trace_anchor_wall_ns_+int64_t(policy_tick)*period_ns},
        {"theoretical_time_basis","nominal_from_first_policy_tick"},{"nominal_policy_period_ns",period_ns}});
      row={{"event","policy_tick"},{"monotonic_s",mono},{"tick",legacy_tick},
        {"q",vector(ro.joint_pos)},{"dq",vector(ro.joint_vel)},{"measured_tau",vector(ro.joint_tau)},
        {"rpy",vector(ro.base_rpy)},{"body_age_s",ro.body_receive_age_s},
        {"body_snapshot_steady_s",ro.snapshot_steady_s},
        {"coverage_hold_enabled",coverage_hold_enabled_},{"terrain_timing_hold_enabled",terrain_timing_hold_enabled_},
        {"previous_action_input",control_->Previous()},{"command",command},
        {"policy_tick_interval_min_s",kMinTickIntervalS},{"policy_tick_interval_max_s",max_tick_interval_s_}};
      row.update(stamp);row["policy_start_stamp"]=stamp;
      row["leg_target_slew_enabled"]=control_->LegTargetSlewEnabled();
      row["joint_target_position_clip_enabled"]=control_->JointTargetPositionClipEnabled();
      row["wheel_target_velocity_clip_enabled"]=control_->WheelTargetVelocityClipEnabled();
      row["terrain_source_session_hold_enabled"]=terrain_source_session_hold_enabled_;
    });
    auto finish_trace=[&]() noexcept {
      trace_only([&]{row["policy_evaluate_end_stamp"]=StudentTrace::Stamp();});
      auto emit=[this,row=std::move(row)]() mutable noexcept {
      trace_emit_attempts();
      trace_only([&]{
        row["terrain_age_s"]=trace_terrain_age_;row["valid_cells"]=trace_valid_;
        row["inference_ms"]=trace_inference_ms_;row["inferred"]=trace_inferred_;
        row["raw_action"]=trace_raw_;row["previous_action"]=control_->Previous();row["tick_elapsed_s"]=tick_elapsed_s_;
        row["terrain_receive_age_s"]=trace_receive_age_;row["terrain_source_age_on_receive_s"]=trace_received_query_age_;
        row["valid_front_cells"]=trace_front_valid_;row["valid_rear_cells"]=trace_rear_valid_;
        row["terrain_input_mode"]=trace_terrain_input_mode_;row["source_session_changed"]=trace_source_session_changed_;
        if(trace_full_observation_&&!trace_observation_.empty()){
          row["observation"]=trace_observation_;row["terrain_cell_age_s"]=trace_cell_age_;
        }
        if(trace_have_terrain_){row["terrain_session"]=trace_source_session_;row["terrain_sequence"]=trace_source_sequence_;row["terrain_query_stamp"]=trace_query_stamp_;}
        if(trace_terrain_frame_){row["terrain_raw"]=terrain_payload(*trace_terrain_frame_);row["terrain_received_stamp"]=trace_terrain_received_stamp_;}
        row["policy_trace_ready_stamp"]=StudentTrace::Stamp();
        trace_.Push(std::move(row));
      });
      };
      // Full observation/terrain JSON belongs after the SDK publication
      // decision. The worker measures the entire getRobotAction call itself.
      if(cts_)trace_only([&]{pending_cts_trace_=std::move(emit);});else emit();
    };
    try{
      auto target=evaluate(ro,command);
      trace_only([&]{
        row["mode"]=control_->Latched()?"hold_latched":"policy";row["reason"]=control_->Reason();
        row["planned_target_q"]=vector(target.goal_joint_pos);row["planned_target_dq"]=vector(target.goal_joint_vel);
        row["kp"]=vector(target.kp);row["kd"]=vector(target.kd);row["planned_tau_ff"]=vector(target.tau_ff);
      });
      finish_trace();return target;
    }catch(const std::exception& error){
      trace_only([&]{row["mode"]="hard_fault";row["reason"]=error.what();});finish_trace();throw;
    }
  }
  RobotAction evaluate(const RobotBasicState& ro,const std::array<float,3>& command) {
    const auto tick=Clock::now();
    // Hard body faults remain visible even while a terrain HOLD is latched.
    control_->CheckBody(ro,cfg_["aos"]["max_proprio_age_s"].get<double>());
    tick_elapsed_s_=std::chrono::duration<double>(tick-last_tick_).count();
    if(control_->Latched())return hold(ro,control_->Reason());
    if(have_tick_&&!tick_interval_ok(tick_elapsed_s_))return hold(ro,"policy tick discontinuity");
    Terrain::SharedPtr terrain;double receive_age;std::string terrain_fault;uint64_t source_generation,source_session;
    {std::lock_guard<std::mutex> guard(mutex_);terrain=terrain_;terrain_fault=terrain_fault_;source_generation=source_generation_;source_session=source_session_;receive_age=terrain?std::chrono::duration<double>(Clock::now()-received_).count():-1;trace_received_query_age_=terrain?received_query_age_:-1;
      trace_only([&]{trace_terrain_received_stamp_=received_stamp_;});}
    trace_receive_age_=receive_age;
    if(terrain){trace_have_terrain_=true;trace_source_session_=terrain->session_id;trace_source_sequence_=terrain->sequence;trace_query_stamp_=terrain->header.stamp.sec+terrain->header.stamp.nanosec*1e-9;trace_terrain_frame_=terrain;}
    if(!terrain_fault.empty())return hold(ro,terrain_fault);
    auto accept_source_switch=[&](){
      if(cts_execution_)cts_execution_->SelectMapSource(source_session,ro.landing_feedback,m20_cts::SteadySeconds());
      const uint64_t previous_session=policy_terrain_session_,previous_generation=policy_source_generation_;
      policy_source_generation_=source_generation;policy_terrain_session_=source_session;
      std::cout<<"[STUDENT_TERRAIN_SESSION_SWITCH] from_session="<<previous_session
               <<" to_session="<<source_session<<" from_generation="<<previous_generation
               <<" to_generation="<<source_generation<<std::endl;
    };
    if(have_policy_session_&&source_generation!=policy_source_generation_){
      trace_source_session_changed_=true;
      if(terrain_source_session_hold_enabled_)return hold(ro,"terrain source session changed");
      accept_source_switch();
    }
    if(!terrain||receive_age>align_age_){
      if(terrain_timing_hold_enabled_)return hold(ro,"no fresh terrain");
      trace_mode(terrain?"unknown_receive_timeout":"unknown_no_frame");
      terrain.reset();
    }
    double age=0.;auto terrain_age_sample=Clock::now();
    if(terrain){
      if(have_policy_session_&&terrain->session_id!=policy_terrain_session_){
        trace_source_session_changed_=true;
        if(terrain_source_session_hold_enabled_)return hold(ro,"terrain source session changed");
        accept_source_switch();
      }
      policy_terrain_session_=terrain->session_id;policy_source_generation_=source_generation;have_policy_session_=true;
      if(terrain->schema_version!=2||!terrain->operator_commissioned||terrain->geometry_hash!=expected_geometry_||terrain->profile!=profile_||terrain->header.frame_id!=cfg_["frames"]["output"].get<std::string>()||!terrain->motion_compensated)return hold(ro,"terrain contract or pose source mismatch");
      const double stamp=terrain->header.stamp.sec+terrain->header.stamp.nanosec*1e-9;
      terrain_age_sample=Clock::now();age=node_->now().seconds()-stamp;trace_terrain_age_=age;
      if(!std::isfinite(age)||age < -future_)return hold(ro,"terrain clock invalid or from future");
      if(age>align_age_){
        if(terrain_timing_hold_enabled_)return hold(ro,"terrain clock alignment");
        trace_mode("unknown_frame_expired");terrain.reset();
      }else trace_mode("fresh");
    }
    std::vector<float> obs(dim_,0);std::array<float,57> proprio{};
    Vec3f gravity=ro.base_rot_mat.transpose()*Vec3f(0,0,-1);
    if(std::abs(gravity.norm()-1)>.02)return hold(ro,"invalid projected gravity");
    for(int i=0;i<3;++i){proprio[i]=std::clamp(ro.base_omega(i),-100.f,100.f)*.25f;proprio[3+i]=gravity(i);}
    const char* cmd_keys[]={"lin_vel_x","lin_vel_y","ang_vel_z"};
    for(int i=0;i<3;++i){if(!std::isfinite(command[i]))return hold(ro,"invalid operator command");proprio[6+i]=std::clamp(command[i],robot_["command_training_ranges"][cmd_keys[i]][0].get<float>(),robot_["command_training_ranges"][cmd_keys[i]][1].get<float>());}
    for(int i=0;i<16;++i){const int j=policy_from_sdk_[i];proprio[9+i]=i<12?std::clamp(ro.joint_pos(j)-robot_["joints"]["default_position_policy_rad"][i].get<float>(),-100.f,100.f):0.f;proprio[25+i]=std::clamp(ro.joint_vel(j),-100.f,100.f)*.05f;proprio[41+i]=control_->Previous()[i];}
    std::copy(proprio.begin(),proprio.end(),obs.begin());int valid=0,front_valid=0,rear_valid=0;
    for(int i=0;terrain&&i<187;++i){float h=terrain->height[i],m=terrain->validity[i],c=terrain->confidence[i],a=terrain->age[i],v=terrain->variance[i];
      if(!std::isfinite(h)||!std::isfinite(m)||!std::isfinite(c)||!std::isfinite(a)||!std::isfinite(v)||(m!=0&&m!=1)||std::abs(h)>1||c<0||c>1||v<0||(m==0&&(h!=0||c!=0)))return hold(ro,"invalid terrain channel");
      if(m==1&&a+std::max(0.,age)>map_age_){h=m=c=0;}
      else if(m==1){
        if(a<0)return hold(ro,"negative valid cell age");
        const double delta=std::max(0.,age),reference=cfg_["estimator"]["confidence_reference_std"].get<double>();
        const double source_decay=std::exp(-static_cast<double>(a)/tau_);
        const double base=std::min(static_cast<double>(c),source_decay);
        const double weight=source_decay>0?base/source_decay:0.;
        const double growth=delta*cfg_["estimator"]["variance_growth_per_s"].get<double>()/(reference*reference);
        c=static_cast<float>(base*std::exp(-delta/tau_)/(1+growth*weight));++valid;
        trace_cell_age_[i]=static_cast<float>(a+delta);
      }
      if(!policy_map_mask_[i]){if(m==1)--valid;h=m=c=0;trace_cell_age_[i]=-1.f;}
      obs[57+i]=h;obs[244+i]=m;obs[431+i]=c;
      if(m==1){if(i%17>8)++front_valid;else if(i%17<8)++rear_valid;}
    }
    trace_valid_=valid;
    trace_front_valid_=front_valid;trace_rear_valid_=rear_valid;
    if(coverage_hold_enabled_&&valid/187.f<min_fraction_)return hold(ro,"terrain coverage below admission threshold");
    std::array<float,3> raw_command={proprio[6],proprio[7],proprio[8]};
    trace_raw_command_=raw_command;
    const float yaw=std::atan2(ro.base_rot_mat(1,0),ro.base_rot_mat(0,0));
    auto apply_feedback=[&](){
      const auto effective=heading_->Apply(raw_command,yaw,proprio[2]/.25f,obs,feedback_step_);
      for(int i=0;i<3;++i){proprio[6+i]=effective[i];obs[6+i]=effective[i];}
    };
    apply_feedback();
    for(int frame=0;frame<history_steps_-1;++frame)std::copy(history_[frame].begin(),history_[frame].end(),obs.begin()+618+57*frame);
    std::array<int64_t,2> shape{1,dim_},raw_shape{1,3};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_,obs.data(),obs.size(),shape.data(),shape.size()));
    if(cts_)inputs.push_back(Ort::Value::CreateTensor<float>(memory_,raw_command.data(),raw_command.size(),raw_shape.data(),raw_shape.size()));
    const char* names[]={"obs","raw_command"};const char* outputs[]={"actions"};
    std::vector<Ort::Value> result;
    size_t trace_attempt=trace_begin_attempt(obs);
    const auto inference_start=Clock::now();
    try{result=session_.Run(Ort::RunOptions{nullptr},names,inputs.data(),inputs.size(),outputs,1);}
    catch(const Ort::Exception& error){trace_end_attempt(trace_attempt,nullptr,"failed",error.what());return hold(ro,"ONNX inference failed");}
    trace_end_attempt(trace_attempt,&result,"candidate");
    trace_inference_ms_=std::chrono::duration<double,std::milli>(Clock::now()-inference_start).count();
#ifdef M20_CTS_OFFLINE_REPLAY
    std::this_thread::sleep_for(std::chrono::duration<float>(replay_after_inference_delay_s_));
#endif
    double processing_age=std::chrono::duration<double>(Clock::now()-tick).count();
    if(ro.body_receive_age_s+processing_age>cfg_["aos"]["max_proprio_age_s"].get<double>()){
      trace_disposition(trace_attempt,"discarded","SDK body state expired during inference");
      throw std::runtime_error("SDK body state expired during inference");
    }
    // Age was sampled after tick; add only time since that sample, otherwise
    // mutex/validation time before it would be counted twice.
    const double terrain_age_after_inference=age+std::chrono::duration<double>(Clock::now()-terrain_age_sample).count();
    if(terrain&&terrain_age_after_inference>align_age_){
      trace_disposition(trace_attempt,"discarded","terrain expired during inference");
      if(terrain_timing_hold_enabled_)return hold(ro,"terrain expired during inference");
      // Discard that result; only the unknown-terrain inference may be issued.
      std::fill(obs.begin()+57,obs.begin()+618,0.f);
      trace_valid_=trace_front_valid_=trace_rear_valid_=0;
      trace_cell_age_.fill(-1.f);
      trace_mode("unknown_expired_during_inference");
      if(coverage_hold_enabled_)return hold(ro,"terrain coverage below admission threshold");
      apply_feedback(); // Same step: restore pre-feedback state before the unknown-map retry.
      trace_attempt=trace_begin_attempt(obs);
      try{result=session_.Run(Ort::RunOptions{nullptr},names,inputs.data(),inputs.size(),outputs,1);}
      catch(const Ort::Exception& error){trace_end_attempt(trace_attempt,nullptr,"failed",error.what());return hold(ro,"ONNX inference failed");}
      trace_end_attempt(trace_attempt,&result,"candidate");
      trace_inference_ms_=std::chrono::duration<double,std::milli>(Clock::now()-inference_start).count();
      processing_age=std::chrono::duration<double>(Clock::now()-tick).count();
      if(ro.body_receive_age_s+processing_age>cfg_["aos"]["max_proprio_age_s"].get<double>()){
        trace_disposition(trace_attempt,"discarded","SDK body state expired during inference");
        throw std::runtime_error("SDK body state expired during inference");
      }
    }
    float* action=result[0].GetTensorMutableData<float>();
    trace_only([&]{if(trace_full_observation_)trace_observation_=obs;});
    std::copy(action,action+16,trace_raw_.begin());trace_inferred_=true;
    const auto target_time=Clock::now();
    const double target_elapsed=std::chrono::duration<double>(target_time-last_target_time_).count();
    RobotAction target=control_->Map(action,target_elapsed);
    if(control_->Latched()){
      trace_disposition(trace_attempt,"discarded",control_->Reason().c_str());
      return hold(ro,control_->Reason());
    }
    trace_disposition(trace_attempt,"mapped_target","publication follows in the official SDK writer");
    last_target_time_=target_time;
    if(cts_execution_){
      cts_execution_->SetRawCommand(raw_command);
      // The hardware binding must supply the original map query pose; no
      // transform is inferred from the robot's current yaw or zero position.
      if(terrain){
        auto ages=trace_cell_age_;
        const float processing=float(std::chrono::duration<double>(Clock::now()-terrain_age_sample).count());
        for(auto& value:ages)if(value>=0)value+=processing;
        cts_execution_->ObserveMap(obs,ages,ro.landing_feedback,terrain->session_id,terrain->sequence,
            terrain->header.stamp.sec+terrain->header.stamp.nanosec*1e-9,m20_cts::SteadySeconds());
      }
    }
    if(history_steps_>1){history_.pop_front();history_.push_back(proprio);}
    ++feedback_step_;last_tick_=tick;have_tick_=true;warned_=false;++run_cnt_;return target;
  }
};
