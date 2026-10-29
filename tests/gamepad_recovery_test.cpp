#include "custom_types.h"
using namespace types;
#include "user_command_interface.h"
using namespace interface;
#include "gamepad_interface.hpp"
#include <cassert>
int main(){
 MotionStateFeedback fb;GamepadInterface g(RobotName::M20);g.SetMotionStateFeedback(&fb);
 auto packet=[&](int type,int command,json items){
  auto body=json({{"PatrolDevice",{{"Type",type},{"Command",command},{"Items",items}}}}).dump();
  DataInfo in{};in.ip="10.0.0.1";in.port=1234;in.fromType=FromType::user;in.buffer=body.data();in.len=body.size();in.command_deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(200);
  std::string response;assert(g.ParseData(in,response));
 };
 auto key=[&](json keys){packet(1101,12,{{"KeyList",keys}});};
 packet(100,100,json::object());g.policy_recovery.Latch();
 fb.UpdateCurrentState(RobotMotionState::WaitingForStand);
 key({"G20_KEY_L1"});key({"G20_KEY_L2"});assert(g.policy_recovery.Active());
 key({"G20_KEY_R1"});key({"G20_KEY_R2"});assert(g.policy_recovery.Active());
 auto begin=std::chrono::steady_clock::now();
 while(std::chrono::steady_clock::now()-begin<std::chrono::milliseconds(2100)){
  packet(100,100,json::object());g.policy_recovery.Health(std::chrono::steady_clock::now(),true);std::this_thread::sleep_for(std::chrono::milliseconds(5));
 }
 key({"G20_KEY_R2"});assert(g.policy_recovery.Active());
 key({"G20_KEY_R1","G20_KEY_R2"});assert(g.policy_recovery.Active());
 key({"G20_KEY_R2"});assert(g.policy_recovery.Active());
 key({"G20_KEY_R1"});assert(g.GetUserCommand()->target_mode==uint8_t(RobotMotionState::JointDamping));
 key({"G20_KEY_R2"});assert(!g.policy_recovery.Active());assert(g.GetUserCommand()->target_mode==uint8_t(RobotMotionState::JointDamping));
 key({"G20_KEY_L2"});assert(g.GetUserCommand()->target_mode!=uint8_t(RobotMotionState::RLControlMode));
 g.policy_recovery.Health(std::chrono::steady_clock::now(),false);
 key({"G20_KEY_L1"});assert(g.GetUserCommand()->target_mode==uint8_t(RobotMotionState::JointDamping));
 begin=std::chrono::steady_clock::now();
 while(std::chrono::steady_clock::now()-begin<std::chrono::milliseconds(2100)){
  packet(100,100,json::object());g.policy_recovery.Health(std::chrono::steady_clock::now(),true);std::this_thread::sleep_for(std::chrono::milliseconds(5));
 }
 assert(g.GetUserCommand()->target_mode==uint8_t(RobotMotionState::JointDamping));
 key({"G20_KEY_R2"});key({"G20_KEY_L1"});assert(g.GetUserCommand()->target_mode==uint8_t(RobotMotionState::StandingUp));
 fb.UpdateCurrentState(RobotMotionState::StandingUp);key({"G20_KEY_L2"});assert(g.GetUserCommand()->target_mode==uint8_t(RobotMotionState::RLControlMode));
 g.policy_recovery.Latch();assert(g.GetUserCommand()->target_mode==uint8_t(RobotMotionState::JointDamping));
 std::cout<<"actual_parser_fault_lock_reset_order=PASS no_motor_commands=true\n";
}
