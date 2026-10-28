#pragma once
// Only the non-I/O base dependencies are faked; RLControlState is production.
#include "common_types.h"
#include "custom_types.h"
#include "robot_interface.h"
#include "user_command_interface.h"
using namespace types;
using namespace interface;
struct ControllerData {
    std::shared_ptr<RobotInterface> ri_ptr;
    std::shared_ptr<UserCommandInterface> uc_ptr;
};
struct OfflineMotionFeedback {void UpdateCurrentState(RobotMotionState) {}};
class StateBase {
public:
    RobotName robot_name_;
    std::string state_name_;
    std::shared_ptr<RobotInterface> ri_ptr_;
    std::shared_ptr<UserCommandInterface> uc_ptr_;
    inline static OfflineMotionFeedback msfb_;
    StateBase(RobotName robot,std::string name,std::shared_ptr<ControllerData> data):
        robot_name_(robot),state_name_(std::move(name)),ri_ptr_(data->ri_ptr),uc_ptr_(data->uc_ptr){}
    virtual ~StateBase()=default;
};
