// Production parser only: no Start(), sockets, DDS or robot interface.
#include "custom_types.h"
using namespace types;
#include "user_command_interface.h"
using namespace interface;
#include "gamepad_interface.hpp"
#include <cmath>
#include <stdexcept>
int main(int argc,char** argv){
  try{
    GamepadInterface gamepad(RobotName::M20);MotionStateFeedback feedback;gamepad.SetMotionStateFeedback(&feedback);
    auto parse=[&](const json& items,int type,int command){
      const auto body=json{{"PatrolDevice",{{"Type",type},{"Command",command},{"Items",items}}}}.dump();
      DataInfo in{};in.ip="127.0.0.1";in.port=1234;in.fromType=FromType::user;
      in.buffer=body.data();in.len=body.size();in.command_deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(200);
      std::string reply;if(!gamepad.ParseData(in,reply))throw std::runtime_error("fixture rejected by actual gamepad parser");
    };
    parse(json::object(),100,100);
    int checks=0;const float limits[]={argc==4?std::stof(argv[1]):.7f,argc==4?std::stof(argv[2]):.5f,argc==4?std::stof(argv[3]):.6f};const char* keys[]={"X","Y","Yaw"};
    for(int axis=0;axis<3;++axis)for(float value:{-10.f,-1.f,-.25f,0.f,.25f,1.f,10.f}){
      json items={{"X",0.f},{"Y",0.f},{"Yaw",0.f}};items[keys[axis]]=value;parse(items,2,21);
      const auto* c=gamepad.GetUserCommand();const float actual[]={c->forward_vel_scale,c->side_vel_scale,c->turnning_vel_scale};
      for(int j=0;j<3;++j){const float expected=j==axis?std::clamp(value,-limits[j],limits[j]):0.f;
        if(std::abs(actual[j]-expected)>1.e-6f)throw std::runtime_error("gamepad command saturation differs from MuJoCo");++checks;}
    }
    std::cout<<json{{"passed",true},{"checks",checks},{"motor_commands",0},{"sockets_created",0}}.dump()<<std::endl;
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}
}
