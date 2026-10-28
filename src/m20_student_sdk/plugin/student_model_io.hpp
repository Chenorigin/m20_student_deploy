#pragma once
// CPU nominal policy inference, shared by offline board probes. No ROS or SDK.
#include <onnxruntime_cxx_api.h>
#include "json.hpp"
#include "cts_execution_mode.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class StudentModelIO {
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING,"student_model_io"};
  Ort::Session session_{nullptr};
  Ort::MemoryInfo memory_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault)};
  bool cts_=false;
  size_t dimension_=0;
public:
  explicit StudentModelIO(const char* path) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);options.SetInterOpNumThreads(1);
    options.AddConfigEntry("session.disable_prepacking","1");
    session_=Ort::Session(env_,path,options);
    if(session_.GetInputCount()<1||session_.GetInputCount()>2||session_.GetOutputCount()!=1)
      throw std::runtime_error("unsupported student input/output count");
    Ort::AllocatorWithDefaultOptions allocator;
    auto family=session_.GetModelMetadata().LookupCustomMetadataMapAllocated("deployment_family",allocator);
    cts_=family&&std::string(family.get())=="m20_cts_student";
    const auto shape=session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if(shape.size()!=2||shape[0]!=1||(shape[1]!=618&&shape[1]!=789))
      throw std::runtime_error("unsupported student observation ABI");
    dimension_=static_cast<size_t>(shape[1]);
    const std::vector<std::string> names=cts_?std::vector<std::string>{"obs","raw_command"}:
                                             std::vector<std::string>{"obs"};
    if(session_.GetInputCount()!=names.size()||(cts_&&dimension_!=789))
      throw std::runtime_error("student family/input ABI mismatch");
    for(size_t i=0;i<names.size();++i){
      auto name=session_.GetInputNameAllocated(i,allocator);
      const auto type=session_.GetInputTypeInfo(i);
      const auto info=type.GetTensorTypeAndShapeInfo();
      if(std::string(name.get())!=names[i]||info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||
         info.GetShape()!=std::vector<int64_t>({1,i==0?static_cast<int64_t>(dimension_):3}))
        throw std::runtime_error("student input name/shape/type mismatch");
    }
    const auto out_type=session_.GetOutputTypeInfo(0);
    const auto out=out_type.GetTensorTypeAndShapeInfo();
    auto out_name=session_.GetOutputNameAllocated(0,allocator);
    if(std::string(out_name.get())!="actions"||out.GetShape()!=std::vector<int64_t>({1,16})||
       out.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      throw std::runtime_error("student action name/shape/type mismatch");
    if(cts_){
      for(const char* key:{"contract_sha256","quiet_execution_control_sha256","command_feedback","quiet_execution_control"}){
        auto value=session_.GetModelMetadata().LookupCustomMetadataMapAllocated(key,allocator);
        if(!value||std::string(value.get()).empty())throw std::runtime_error("CTS embedded contract missing");
      }
    }
  }
  bool IsCTS()const{return cts_;}
  size_t ObservationDim()const{return dimension_;}
  size_t FixtureColumns()const{return dimension_+(cts_?3:0);}
  nlohmann::json Metadata() {
    Ort::AllocatorWithDefaultOptions allocator;nlohmann::json result;
    for(const char* key:{"deployment_family","contract_sha256","command_feedback","quiet_execution_control_sha256","quiet_execution_control"}){
      auto value=session_.GetModelMetadata().LookupCustomMetadataMapAllocated(key,allocator);
      if(value)result[key]=value.get();
    }
    return result;
  }
  std::array<float,16> Infer(float* observation,float* raw_command=nullptr) {
    if(cts_&&!raw_command)throw std::runtime_error("CTS requires independently sampled raw command3");
    for(size_t i=0;i<dimension_;++i)if(!std::isfinite(observation[i]))throw std::runtime_error("nonfinite observation");
    std::array<int64_t,2> shape{1,static_cast<int64_t>(dimension_)},raw_shape{1,3};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_,observation,dimension_,shape.data(),2));
    if(cts_){
      for(int i=0;i<3;++i)if(!std::isfinite(raw_command[i]))throw std::runtime_error("nonfinite raw command");
      inputs.push_back(Ort::Value::CreateTensor<float>(memory_,raw_command,3,raw_shape.data(),2));
    }
    const char* names[]={"obs","raw_command"};const char* outputs[]={"actions"};
    auto values=session_.Run(Ort::RunOptions{nullptr},names,inputs.data(),inputs.size(),outputs,1);
    std::array<float,16> action{};
    const float* data=values[0].GetTensorData<float>();
    for(size_t i=0;i<action.size();++i){
      if(!std::isfinite(data[i]))throw std::runtime_error("nonfinite nominal action");
      action[i]=data[i];
    }
    return action;
  }
};

// Called in main BEFORE ROS, FSM, command publishers or SDK initialization.
// Checks the selected ONNX too, so removing the sidecar marker is insufficient.
inline bool RequiresLandingHardware(const nlohmann::json& document){
  if(!document.is_object())return false;
  if(document.value("deployment_family",std::string())=="m20_cts_student"||
     document.contains("cts_version")||document.contains("quiet_execution_control")||
     document.contains("control_contract"))return true;
  for(const auto& value:document)if(value.is_object()&&RequiresLandingHardware(value))return true;
  return false;
}
inline void RequireStudentHardwareRuntime() {
  const char* config=std::getenv("M20_HEIGHTMAP_CONFIG");
  const char* policy=std::getenv("M20_POLICY_PATH");
  if(!config||!policy)throw std::runtime_error("missing selected model/configuration");
  std::ifstream file(config);nlohmann::json cfg;file>>cfg;
  std::filesystem::path sidecar=cfg.at("aos").at("model_contract").get<std::string>();
  if(sidecar.is_relative())sidecar=std::filesystem::path(config).parent_path()/sidecar;
  std::ifstream contract_file(sidecar);nlohmann::json contract;contract_file>>contract;
  StudentModelIO model(policy);
  const auto embedded=model.Metadata();
  const auto execution_mode=m20_cts::ExecutionMode(cfg);
  if(model.IsCTS()&&execution_mode=="nominal_pd"){
    const char* verified=std::getenv("M20_VERIFIED_MODEL_CONTRACT_SHA256");
    if(contract.value("deployment_family",std::string())!="m20_cts_student"||
       contract.at("model_inputs")!=nlohmann::json({{"obs",{1,789}},{"raw_command",{1,3}}})||
       !verified||embedded.value("contract_sha256",std::string())!=verified||
       embedded.at("quiet_execution_control_sha256")!=contract.at("quiet_execution_control_sha256")||
       nlohmann::json::parse(embedded.at("quiet_execution_control").get<std::string>())!=contract.at("quiet_execution_control")||
       nlohmann::json::parse(embedded.at("command_feedback").get<std::string>())!=contract.at("command_feedback"))
      throw std::runtime_error("CTS nominal_pd selected model/contract identity mismatch");
    std::cout<<"[CTS_EXECUTION] mode=nominal_pd landing_control=disabled original_PD=true"<<std::endl;
    return;
  }
  if(model.IsCTS()||RequiresLandingHardware(contract)||
     embedded.contains("quiet_execution_control"))
    throw std::runtime_error("CTS/Quiet mandatory landing controller has no validated hardware contact/actuator binding; SDK initialization blocked");
}
