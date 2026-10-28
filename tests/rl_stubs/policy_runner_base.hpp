#pragma once
#include "common_types.h"
using namespace types;
class PolicyRunnerBase {
public:
    int decimation_=4;
    virtual ~PolicyRunnerBase()=default;
    virtual RobotAction getRobotAction(const RobotBasicState&,const UserCommand&)=0;
    virtual void OnEnter()=0;
    virtual void DisplayPolicyInfo(){}
};
