// Standalone CPU inference: no ROS, SDK, hardware interface or command publisher.
#include <onnxruntime_cxx_api.h>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include "json.hpp"
int main(int argc,char** argv) {
  try {
    if(argc!=4&&argc!=2) throw std::runtime_error("usage: student_onnx_probe model.onnx input.f32 output.f32");
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"student_offline");
    Ort::SessionOptions options;options.SetIntraOpNumThreads(1);options.SetInterOpNumThreads(1);
    options.AddConfigEntry("session.disable_prepacking","1");
    Ort::Session session(env,argv[1],options);
    auto shape=session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if(session.GetInputCount()!=1||session.GetOutputCount()!=1||shape.size()!=2||shape[0]!=1||(shape[1]!=618&&shape[1]!=789)) throw std::runtime_error("unexpected student ABI");
    if(argc==2){
      Ort::AllocatorWithDefaultOptions allocator;
      auto metadata=session.GetModelMetadata();nlohmann::json record;
      for(const char* key:{"deployment_family","contract_sha256","command_feedback"}){
        auto value=metadata.LookupCustomMetadataMapAllocated(key,allocator);
        if(!value)throw std::runtime_error("missing model metadata");
        record[key]=value.get();
      }
      std::cout<<record.dump()<<std::endl;return 0;
    }
    std::ifstream in(argv[2],std::ios::binary);std::ofstream out(argv[3],std::ios::binary|std::ios::trunc);
    if(!in||!out)throw std::runtime_error("cannot open offline data");
    std::vector<float> obs(shape[1]);size_t count=0;
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    const char* inputs[]={"obs"};const char* outputs[]={"actions"};
    while(in.read(reinterpret_cast<char*>(obs.data()),obs.size()*sizeof(float))) {
      for(float v:obs)if(!std::isfinite(v))throw std::runtime_error("nonfinite observation");
      auto tensor=Ort::Value::CreateTensor<float>(memory,obs.data(),obs.size(),shape.data(),shape.size());
      auto result=session.Run(Ort::RunOptions{nullptr},inputs,&tensor,1,outputs,1);
      auto result_shape=result[0].GetTensorTypeAndShapeInfo().GetShape();
      if(result_shape!=std::vector<int64_t>({1,16}))throw std::runtime_error("unexpected action ABI");
      const float* action=result[0].GetTensorData<float>();
      for(int i=0;i<16;++i)if(!std::isfinite(action[i]))throw std::runtime_error("nonfinite action");
      out.write(reinterpret_cast<const char*>(action),16*sizeof(float));++count;
    }
    if(in.gcount()!=0||!out||count==0)throw std::runtime_error("truncated or empty offline data");
    std::cout<<"{\"offline_frames\":"<<count<<",\"input_dim\":"<<shape[1]<<",\"command_publishers\":0}"<<std::endl;
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
