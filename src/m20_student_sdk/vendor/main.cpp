#include "quadruped_wheel/qw_state_machine.hpp"

#include <cstdlib>
#include <string>
#include "student_model_io.hpp"

#ifdef USE_SIMULATION
    #define BACKWARD_HAS_DW 1
    #include "backward.hpp"
    namespace backward{
        backward::SignalHandling sh;
    }
#endif

using namespace types;
MotionStateFeedback StateBase::msfb_ = MotionStateFeedback();

int main(){
    // Building or inspecting the binary never grants control. Only the operator
    // startup wrapper sets this after the live preflight and typed confirmation.
    const char* operator_start=std::getenv("M20_OPERATOR_START");
    if(operator_start==nullptr||std::string(operator_start)!="START_STUDENT"){
        std::cerr<<"Use scripts/operator_start_aos.sh; direct control startup is disabled."<<std::endl;
        return 2;
    }
    const char* transport=std::getenv("M20_GAMEPAD_TRANSPORT");
    const char* sdk_mode=std::getenv("M20_SDK_MODE_CONFIRMED");
    const char* verified_pid=std::getenv("M20_VERIFIED_VENDOR_PID");
    const char* requested_control=std::getenv("M20_REMOTE_CONTROL");
    if(geteuid()!=0||!transport||std::string(transport)!="passive"||
       !sdk_mode||std::string(sdk_mode)!="I_CONFIRMED"||
       !requested_control||std::string(requested_control)!="gamepad"||
       !verified_pid||std::string(verified_pid).empty()||
       std::string(verified_pid).find_first_not_of("0123456789")!=std::string::npos||
       std::strtol(verified_pid,nullptr,10)<=0||std::strtol(verified_pid,nullptr,10)>=65536){
        std::cerr<<"AOS requires root passive input and operator-confirmed SDK mode before ROS/hardware initialization."<<std::endl;
        return 2;
    }
    try { RequireStudentHardwareRuntime(); }
    catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 2;}
    std::cout << "State Machine Start Running" << std::endl;

    // Keep the caller-selected DDS profile.  On the NX real-robot path this is
    // /etc/ros/m20_fastdds.xml, which binds DDS traffic to the dedicated M20
    // Ethernet link.  Simulation selects its own RMW configuration.
    rclcpp::init(0, 0);

    RemoteCommandType remote_command_type = RemoteCommandType::kGamepad;
    if (const char* control = std::getenv("M20_REMOTE_CONTROL")) {
        if (std::string(control) == "keyboard") {
            remote_command_type = RemoteCommandType::kKeyBoard;
        }
    }
    std::cout << "M20 remote control: "
              << (remote_command_type == RemoteCommandType::kGamepad ? "gamepad" : "keyboard")
              << std::endl;
    std::shared_ptr<StateMachineBase> fsm = std::make_shared<qw::QwStateMachine>(
        RobotName::M20, remote_command_type);
    
    fsm->Start();
    fsm->Run();
    fsm->Stop();

    rclcpp::shutdown();
    return 0;
}
