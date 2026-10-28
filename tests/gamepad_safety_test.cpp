#include "custom_types.h"
using namespace types;
#include "user_command_interface.h"
using namespace interface;
#include "gamepad_interface.hpp"
#include <cassert>
int main(){
 const bool forward_only=std::getenv("M20_FORWARD_ONLY")&&std::string(std::getenv("M20_FORWARD_ONLY"))=="1";
 MotionStateFeedback feedback;GamepadInterface gamepad(RobotName::M20);gamepad.SetMotionStateFeedback(&feedback);
 auto parse=[&](const std::string& body,const std::string& ip="10.0.0.1"){
  DataInfo in{};in.ip=ip;in.port=1234;in.fromType=FromType::user;in.buffer=body.data();in.len=body.size();in.command_deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(200);std::string response;return gamepad.ParseData(in,response);
 };
 const std::string hb=R"({"PatrolDevice":{"Type":100,"Command":100,"Items":{}}})";
 const std::string steer=R"({"PatrolDevice":{"Type":2,"Command":21,"Items":{"X":0.8,"Y":0,"Yaw":0.4}}})";
 assert(parse(hb));assert(parse(steer));assert(gamepad.GetUserCommand()->forward_vel_scale==.8f);double stamp=gamepad.GetUserCommand()->time_stamp;
 assert(!parse(R"({"PatrolDevice":{"Type":999,"Command":999,"Items":{}}})"));assert(gamepad.GetUserCommand()->time_stamp==stamp);
 assert(!parse(R"({"PatrolDevice":{"Type":2,"Command":21,"Items":{"X":-0.5,"Y":0}}})","10.0.0.2"));assert(gamepad.GetUserCommand()->forward_vel_scale==.8f);
 assert(parse(R"({"PatrolDevice":{"Type":2,"Command":21,"Items":{"X":-0.5,"Y":0.2,"Yaw":-0.3}}})"));
 assert(gamepad.GetUserCommand()->forward_vel_scale==(forward_only?0.f:-.5f));
 assert(gamepad.GetUserCommand()->side_vel_scale==.2f);assert(gamepad.GetUserCommand()->turnning_vel_scale==-.3f);
 assert(parse(R"({"PatrolDevice":{"Type":1101,"Command":12,"Items":{"KeyList":["G20_KEY_R2","G20_KEY_L1"]}}})"));assert(gamepad.GetUserCommand()->target_mode==uint8_t(RobotMotionState::JointDamping));
 std::this_thread::sleep_for(std::chrono::milliseconds(230));assert(parse(hb));assert(gamepad.GetUserCommand()->forward_vel_scale==0.f);assert(gamepad.GetUserCommand()->turnning_vel_scale==0.f);
 assert(parse(steer));std::this_thread::sleep_for(std::chrono::milliseconds(110));
 parse(R"({"PatrolDevice":{"Type":2,"Command":21,"Items":{"X":"bad","Y":null,"Yaw":{}}}})");
 std::this_thread::sleep_for(std::chrono::milliseconds(110));assert(gamepad.GetUserCommand()->forward_vel_scale==0.f);
 DataInfo short_packet{};short_packet.ip="10.0.0.1";short_packet.port=1234;short_packet.fromType=FromType::user;short_packet.buffer=steer.data();short_packet.len=steer.size();short_packet.command_deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(1);std::string response;
 gamepad.ParseData(short_packet,response);std::this_thread::sleep_for(std::chrono::milliseconds(15));assert(gamepad.GetUserCommand()->forward_vel_scale==0.f);
 std::cout<<"unknown_watchdog_source_priority_and_steer_expiry=PASS no_sockets_no_publishers=true\n";
}
