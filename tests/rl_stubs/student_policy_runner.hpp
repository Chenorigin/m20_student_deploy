#pragma once
// Deterministic policy dependency: real StudentControlCore and CTS clock/types,
// without ONNX, terrain subscriptions, ROS, DDS or a copy of the servo algorithm.
#include "policy_runner_base.hpp"
#include "student_control_core.hpp"
#include "student_trace.hpp"
#include "cts_execution.hpp"
#include <filesystem>
#include <functional>
class M20PolicyRunner : public PolicyRunnerBase {
    static nlohmann::json ReadRobot(){
        std::ifstream input(std::getenv("M20_ROBOT_CONFIG"));nlohmann::json robot;input>>robot;return robot;
    }
public:
    nlohmann::json robot;
    StudentControlCore core;
    int inferences=0,commits=0,flushes=0;
    std::array<float,16> action{};
    std::function<void()> on_infer,on_execute,on_commit;
    M20PolicyRunner(const std::string&,const std::string&):robot(ReadRobot()),core(robot){core.RequirePublicationCommit();}
    void OnEnter()override{}
    void SetStandingHandoff(const types::MatXf& command){core.Begin(command);}
    bool IsCTS()const{return true;}
    bool UsesLandingControl()const{return false;}
    void CheckCTSBody(const types::RobotBasicState& state)const{core.CheckBody(state,.04);}
    bool CTSNominalIsHeld()const{return core.Latched();}
    void LatchCTSTimingHold(const std::string& reason){(void)core.Hold(reason);}
    types::RobotAction getRobotAction(const types::RobotBasicState&,const types::UserCommand&)override{
        ++inferences;auto candidate=core.Map(action.data(),.02);if(on_infer)on_infer();return candidate;
    }
    types::RobotAction ExecuteCTSSubstep(const types::RobotAction& candidate,const types::RobotBasicState& state){
        CheckCTSBody(state);if(on_execute)on_execute();
        auto command=core.Latched()?core.Hold(core.Reason()):candidate;core.ValidateTarget(command);return command;
    }
    void RecordCTSCommandPublished(const types::MatXf& command){++commits;core.RecordPublished(command);if(on_commit)on_commit();}
    void FlushCTSTrace(){++flushes;}
};
