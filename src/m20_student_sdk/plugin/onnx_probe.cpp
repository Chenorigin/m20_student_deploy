// Standalone CPU inference: no ROS, SDK, hardware interface or command publisher.
#include "student_model_io.hpp"
#include <iostream>
int main(int argc,char** argv) {
  try {
    if(argc==2&&std::string(argv[1])=="--runtime-check"){
      RequireStudentHardwareRuntime();
      std::cout<<"{\"runtime_check_passed\":true,\"hardware_commands\":0,\"command_publishers\":0}"<<std::endl;return 0;
    }
    if(argc!=4&&argc!=2)throw std::runtime_error("usage: student_onnx_probe model.onnx [input.f32 output.f32]");
    StudentModelIO model(argv[1]);
    if(argc==2){std::cout<<model.Metadata().dump()<<std::endl;return 0;}
    std::ifstream in(argv[2],std::ios::binary);std::ofstream out(argv[3],std::ios::binary|std::ios::trunc);
    if(!in||!out)throw std::runtime_error("cannot open offline data");
    std::vector<float> row(model.FixtureColumns());size_t count=0;
    while(in.read(reinterpret_cast<char*>(row.data()),row.size()*sizeof(float))){
      auto action=model.Infer(row.data(),model.IsCTS()?row.data()+model.ObservationDim():nullptr);
      out.write(reinterpret_cast<const char*>(action.data()),action.size()*sizeof(float));++count;
    }
    if(in.gcount()!=0||!out||count==0)throw std::runtime_error("truncated or empty offline data");
    std::cout<<nlohmann::json({{"offline_frames",count},{"input_dim",model.ObservationDim()},
      {"raw_command_dim",model.IsCTS()?3:0},{"command_publishers",0}}).dump()<<std::endl;
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
