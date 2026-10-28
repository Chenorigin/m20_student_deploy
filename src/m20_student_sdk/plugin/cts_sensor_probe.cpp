// Offline sensor observation replay only. No ROS, SDK or motor publishers.
#include "cts_observation.hpp"
#include <fstream>
#include <iostream>
int main(int argc,char** argv){
  try{
    if(argc!=2)throw std::runtime_error("usage: cts_sensor_probe observation_contract.json < frames.json");
    std::ifstream file(argv[1]);nlohmann::json contract;file>>contract;
    CTSObservation packer(contract);nlohmann::json frames;std::cin>>frames;
    nlohmann::json result=nlohmann::json::array();
    for(const auto& frame:frames){
      if(frame.value("reset",false))packer.Reset();
      auto raw=frame.at("raw_command").get<std::array<float,3>>();
      auto obs=packer.Pack(frame.at("proprio").get<std::array<float,57>>(),
        frame.at("terrain").get<std::array<float,561>>(),raw,frame.at("yaw").get<float>());
      result.push_back({{"obs",obs},{"raw_command",raw}});
    }
    std::cout<<result.dump()<<std::endl;return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
