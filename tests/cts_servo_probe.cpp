// Actual production RLControlState::ProcessCTSStep and UpdateRobotObservation.
// Only non-I/O dependencies under rl_stubs/ are replaced; no servo algorithm is
// copied here and no ROS, DDS, RobotInterface hardware or motor is constructed.
#define M20_RL_OFFLINE_TEST 1
#include "quadruped_wheel/rl_control_state.hpp"
#include <condition_variable>
#include <future>
#include <stdexcept>

namespace qw {
struct RLControlStateTest {
    static std::shared_ptr<M20PolicyRunner> Policy(RLControlState& state){return state.m20_policy_;}
    static void Init(RLControlState& state,const MatXf& seed){
        state.start_flag_=true;state.policy_fault_=false;state.last_publication_sequence_=0;
        state.state_run_cnt_=0;state.m20_policy_->SetStandingHandoff(seed);
    }
    static void Generation(RLControlState& state,int generation){state.state_run_cnt_=generation;}
    static void Step(RLControlState& state,const RobotBasicState& observation,int generation,
                     m20_cts::SubstepClock& clock,RobotAction& nominal,double& now){
        state.ProcessCTSStep(observation,generation,clock,nominal,[&]{return now;});
    }
    static nlohmann::json Audit(RLControlState& state){
        return {{"outcome",state.servo_audit_.outcome},{"published",state.servo_audit_.published},
                {"inferred",state.servo_audit_.inferred},{"sequence",state.last_publication_sequence_}};
    }
    static std::pair<int,double> Snapshot(RLControlState& state){
        std::lock_guard<std::mutex> lock(state.rbs_mutex_);
        return {state.state_run_cnt_.load(),state.rbs_[state.getrbsReadIndex()].snapshot_steady_s};
    }
};
}
using Access=qw::RLControlStateTest;
static void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
struct Fixture {
    double now=0;
    std::shared_ptr<interface::RobotInterface> robot=std::make_shared<interface::RobotInterface>();
    std::shared_ptr<interface::UserCommandInterface> command=std::make_shared<interface::UserCommandInterface>();
    std::shared_ptr<ControllerData> data=std::make_shared<ControllerData>();
    std::unique_ptr<qw::RLControlState> state;
    std::shared_ptr<M20PolicyRunner> policy;
    MatXf seed=MatXf::Zero(16,5);
    m20_cts::SubstepClock clock;
    RobotAction nominal;
    explicit Fixture(){
        robot->now=&now;data->ri_ptr=robot;data->uc_ptr=command;
        state=std::make_unique<qw::RLControlState>(RobotName::M20,"offline_real_rl",data);
        policy=Access::Policy(*state);
        for(int j=0;j<16;j++){
            const int i=policy->robot["joints"]["sdk_from_policy_indices"][j];
            seed(j,0)=policy->robot["joints"]["kp_sdk"][j];
            seed(j,1)=policy->robot["joints"]["default_position_policy_rad"][i];
            seed(j,2)=policy->robot["joints"]["kd_sdk"][j];
        }
        robot->cached=seed;Access::Init(*state,seed);
        command->GetUserCommand()->target_mode=uint8_t(RobotMotionState::RLControlMode);
        policy->action[1]=.2f;policy->action[12]=.5f;
        robot->state=Observation();
    }
    RobotBasicState Observation()const{
        RobotBasicState result;result.joint_pos=seed.col(1);result.joint_vel.setZero();
        result.snapshot_steady_s=now;result.body_receive_age_s=.001;return result;
    }
    void Step(int generation){Access::Generation(*state,generation);Access::Step(*state,Observation(),generation,clock,nominal,now);}
    void StepWith(int generation,RobotBasicState observation){Access::Generation(*state,generation);Access::Step(*state,observation,generation,clock,nominal,now);}
    bool Held()const{return policy->CTSNominalIsHeld();}
    bool PreviousZero()const{return std::all_of(policy->core.Previous().begin(),policy->core.Previous().end(),[](float x){return x==0;});}
    void Unpublished()const{
        require(robot->publications.empty(),"expired/refused candidate was published");
        require(PreviousZero(),"unpublished candidate advanced Previous");
    }
    void Hard(std::function<void()> operation){
        bool hard=false;try{operation();}catch(const std::exception&){hard=true;}
        require(hard,"expected a hard fault, got success/timing HOLD");
    }
};
static void run(const std::string& scenario){
    Fixture f;
    if(scenario=="healthy_four_substeps"){
        for(int tick=0;tick<4;++tick){f.now=tick*.005;f.Step(tick);}
        require(f.robot->publications.size()==4&&f.policy->inferences==1&&!f.Held(),"four substep policy decimation failed");
        require(std::abs(f.policy->core.Previous()[1]-.2f)<1e-7,"published action did not commit");
    }else if(scenario=="observed_worker_jitter"){
        // Captured worker and coherent snapshot times, rebased to worker tick 0.
        const double workers[]={0.,.004999812,.009965206,.014899973,.019996334,
            .024868971,.029966791,.037644293,.039910704,.044995106};
        const double snapshots[]={-.000140302,.004861553,.009896368,.014864095,.019858949,
            .024858762,.029893577,.034917017,.039875118,.044873181};
        for(int tick=0;tick<10;++tick){
            f.now=workers[tick];auto observation=f.Observation();
            observation.snapshot_steady_s=snapshots[tick];f.StepWith(tick,observation);
            require(!f.Held(),"observed consecutive jitter incorrectly latched HOLD");
        }
        require(f.robot->publications.size()==10&&f.policy->inferences==3,
                "observed jitter changed publication count or policy decimation");
    }else if(scenario=="interval_above_8_5ms"||scenario=="interval_below_1_5ms"){
        f.Step(0);f.now=scenario=="interval_above_8_5ms"?.008501:.001499;f.Step(1);
        require(f.Held()&&f.robot->publications.size()==1,"out-of-tolerance interval published");
        f.now+=.005;f.Step(2);
        require(f.Held()&&f.robot->publications.size()==2&&f.policy->inferences==1,
                "interval fault did not preserve continuous HOLD without inference");
    }else if(scenario=="duplicate_generation"){
        f.Step(0);f.now=.005;f.Step(0);
        require(f.Held()&&f.robot->publications.size()==1,"duplicate generation published");
    }else if(scenario=="deadline_5ms_boundary"||scenario=="deadline_over_5ms"){
        f.policy->on_infer=[&]{f.now=scenario=="deadline_5ms_boundary"?.005:.005001;};
        f.Step(0);require(f.Held(),"5ms expired candidate did not HOLD");f.Unpublished();
    }else if(scenario=="deadline_under_5ms"){
        f.policy->on_infer=[&]{f.now=.004999;};f.Step(0);
        require(!f.Held()&&f.robot->publications.size()==1,"valid under-5ms candidate rejected");
    }else if(scenario=="snapshot_expired_before_inference"){
        auto observation=f.Observation();observation.snapshot_steady_s=-.005;
        f.StepWith(0,observation);require(f.Held()&&f.policy->inferences==0,"expired snapshot inferred");f.Unpublished();
    }else if(scenario=="snapshot_deadline_after_candidate"){
        auto observation=f.Observation();observation.snapshot_steady_s=-.004;
        f.policy->on_infer=[&]{f.now=.001;};f.StepWith(0,observation);
        require(f.Held(),"short inference with old snapshot failed to HOLD");f.Unpublished();
    }else if(scenario=="generation_advance_during_inference"){
        f.policy->on_infer=[&]{Access::Generation(*f.state,1);};f.Step(0);
        require(f.Held(),"changed generation did not HOLD");f.Unpublished();
    }else if(scenario=="noninference_snapshot_expiry"){
        f.Step(0);f.now=.005;auto observation=f.Observation();observation.snapshot_steady_s=0;
        f.StepWith(1,observation);
        require(f.Held()&&f.robot->publications.size()==1&&f.policy->inferences==1,"non-inference expiry failed");
    }else if(scenario=="missed_first_tick"){
        f.Step(1);require(f.Held()&&f.policy->inferences==0,"missed first tick inferred");f.Unpublished();
    }else if(scenario=="missing_tick_then_continuous_hold"){
        f.Step(0);const auto issued=f.robot->cached;const auto previous=f.policy->core.Previous();
        f.now=.010;f.Step(2);require(f.Held()&&f.robot->publications.size()==1,"missing tick published");
        for(int tick=3;tick<20;++tick){f.now=tick*.005;f.Step(tick);}
        require(f.policy->inferences==1&&f.policy->core.Previous()==previous,"HOLD inferred/changed Previous");
        const auto held=f.robot->cached;
        require(held.col(0).isApprox(issued.col(0))&&held.col(1).isApprox(issued.col(1))&&held.col(2).isApprox(issued.col(2)),"HOLD lost published support/PD");
        require(held.col(3).isZero()&&held.col(4).isZero(),"HOLD retained wheel speed/feedforward");
    }else if(scenario=="discarded_first_candidate_holds_handoff"){
        f.policy->on_infer=[&]{f.now=.005;};f.Step(0);f.Unpublished();
        f.policy->on_infer=nullptr;f.now=.010;f.Step(1);
        require(f.robot->publications.size()==1&&f.robot->cached.col(1).isApprox(f.seed.col(1)),"HOLD used discarded first candidate");
        require(f.robot->cached.col(0).isApprox(f.seed.col(0))&&f.robot->cached.col(2).isApprox(f.seed.col(2)),"handoff PD changed");
        require(f.robot->cached.col(3).isZero()&&f.robot->cached.col(4).isZero()&&f.PreviousZero(),"handoff HOLD advanced action/speed");
    }else if(scenario=="publish_returns_after_deadline"||scenario=="generation_advances_in_publish"){
        if(scenario=="publish_returns_after_deadline")f.robot->publish_duration=.005;
        else f.robot->on_publish=[&]{Access::Generation(*f.state,1);};
        f.Step(0);require(f.Held()&&f.robot->publications.size()==1,"returned publication was discarded");
        const auto issued=f.robot->cached;const auto previous=f.policy->core.Previous();
        require(std::abs(previous[1]-.2f)<1e-7,"late but real publication failed to commit Previous");
        f.robot->publish_duration=0;f.robot->on_publish=nullptr;f.now=.010;f.Step(1);
        require(f.robot->cached.col(1).isApprox(issued.col(1))&&f.policy->core.Previous()==previous,"HOLD reverted a real publication");
        require(f.robot->cached.col(3).isZero()&&f.robot->cached.col(4).isZero()&&f.policy->inferences==1,"late-publication HOLD invalid");
    }else if(scenario=="body_expires_during_publish"){
        f.robot->publish_duration=.041;
        f.Hard([&]{f.Step(0);});
        require(f.robot->publications.size()==1&&f.policy->commits==1,"real publication was not retained before body fault");
        require(std::abs(f.policy->core.Previous()[1]-.2f)<1e-7&&!f.Held(),"expired body became timing HOLD or lost real Previous");
    }else if(scenario=="commit_crosses_deadline"){
        f.policy->on_commit=[&]{f.now+=.005;};f.Step(0);
        require(f.Held()&&f.robot->publications.size()==1,"post-publication commit duration escaped deadline");
        require(std::abs(f.policy->core.Previous()[1]-.2f)<1e-7,"actual publication lost after commit overrun");
    }else if(scenario=="blocked_receipt"||scenario=="publisher_throws"){
        f.robot->blocked=scenario=="blocked_receipt";f.robot->throw_publish=scenario=="publisher_throws";
        f.Hard([&]{f.Step(0);});f.Unpublished();require(!f.Held(),"publish failure became timing HOLD");
    }else if(scenario=="duplicate_receipt"){
        f.Step(0);const auto previous=f.policy->core.Previous();f.now=.005;f.robot->duplicate_receipt=true;
        f.Hard([&]{f.Step(1);});require(f.policy->core.Previous()==previous&&!f.Held(),"invalid receipt committed or became soft");
    }else if(scenario=="receipt_time_in_future"||scenario=="receipt_time_before_publish"||scenario=="receipt_time_nan"){
        f.robot->receipt_time_offset=scenario=="receipt_time_in_future"?.001:scenario=="receipt_time_before_publish"?-.001:NAN;
        f.Hard([&]{f.Step(0);});require(f.PreviousZero()&&!f.Held(),"invalid receipt committed or became soft");
    }else if(scenario=="invalid_body"||scenario=="stale_body"||scenario=="negative_body_age_before_queue"||scenario=="measured_position_out_of_bounds"||scenario=="body_expires_during_inference"||scenario=="body_invalid_even_if_schedule_expired"){
        auto observation=f.Observation();
        if(scenario=="invalid_body"||scenario=="body_invalid_even_if_schedule_expired")observation.joint_pos[0]=NAN;
        if(scenario=="stale_body")observation.body_receive_age_s=.041;
        if(scenario=="negative_body_age_before_queue"){observation.body_receive_age_s=-.001;observation.snapshot_steady_s=-.002;}
        if(scenario=="measured_position_out_of_bounds"){
            const auto& hardware=f.policy->robot.at("hardware");
            const char* key=hardware.contains("software_feedback_position_limits_sdk_rad")?"software_feedback_position_limits_sdk_rad":"software_position_limits_sdk_rad";
            observation.joint_pos[0]=hardware.at(key).at(0).at(1).get<float>()+1.f;
        }
        if(scenario=="body_invalid_even_if_schedule_expired")observation.snapshot_steady_s=-.005;
        if(scenario=="body_expires_during_inference")f.policy->on_infer=[&]{f.now=.041;};
        f.Hard([&]{f.StepWith(0,observation);});f.Unpublished();require(!f.Held(),"body fault became timing HOLD");
    }else if(scenario=="stale_body_after_hold"){
        auto old=f.Observation();old.snapshot_steady_s=-.005;f.StepWith(0,old);require(f.Held(),"setup failed");
        f.now=.010;auto observation=f.Observation();observation.body_receive_age_s=.041;
        f.Hard([&]{f.StepWith(1,observation);});f.Unpublished();
    }else if(scenario=="invalid_snapshot_nan"||scenario=="snapshot_from_future"||scenario=="clock_nan"){
        auto observation=f.Observation();
        if(scenario=="invalid_snapshot_nan")observation.snapshot_steady_s=NAN;
        if(scenario=="snapshot_from_future")observation.snapshot_steady_s=.001;
        if(scenario=="clock_nan")f.now=NAN;
        f.Hard([&]{f.StepWith(0,observation);});f.Unpublished();require(!f.Held(),"invalid clock became timing HOLD");
    }else if(scenario=="clock_regression_between_steps"){
        f.now=.005;f.Step(0);f.now=.004;f.Hard([&]{f.Step(1);});
        require(!f.Held()&&f.robot->publications.size()==1,"nonmonotonic clock became timing HOLD");
    }else if(scenario=="clock_regression_after_hold"){
        f.Step(0);f.now=.010;f.Step(2);require(f.Held(),"setup did not HOLD");
        f.now=.009;f.Hard([&]{f.Step(3);});
        require(f.robot->publications.size()==1,"clock regression in HOLD published");
    }else if(scenario=="r2_after_hold"||scenario=="ownership_lost_after_hold"){
        f.Step(0);const auto previous=f.policy->core.Previous();f.now=.010;f.Step(2);
        require(f.Held(),"setup did not HOLD");f.now=.015;
        if(scenario=="r2_after_hold"){
            f.command->GetUserCommand()->target_mode=uint8_t(RobotMotionState::JointDamping);
            f.Step(3);require(f.state->LoseControlJudge(),"R2 in HOLD did not request exit");
        }else{
            f.robot->blocked=true;f.Hard([&]{f.Step(3);});
        }
        require(f.robot->publications.size()==1&&f.policy->inferences==1&&f.policy->core.Previous()==previous,"HOLD exit/ownership loss published/inferred/changed action");
    }else if(scenario=="r2_during_inference"||scenario=="safe_mode_during_inference"||scenario=="liedown_during_inference"){
        f.policy->on_infer=[&]{
            if(scenario=="safe_mode_during_inference")f.command->GetUserCommand()->safe_control_mode=3;
            else f.command->GetUserCommand()->target_mode=uint8_t(scenario=="r2_during_inference"?RobotMotionState::JointDamping:RobotMotionState::LieDown);
        };
        f.Step(0);f.Unpublished();
        if(scenario=="r2_during_inference")require(f.state->LoseControlJudge(),"R2 did not request exit");
    }else if(scenario=="exit_during_inference"){
        std::promise<void> entered,released;auto ready=entered.get_future();auto release=released.get_future();
        f.policy->on_infer=[&]{entered.set_value();require(release.wait_for(std::chrono::seconds(2))==std::future_status::ready,"exit test release timed out");};
        std::exception_ptr error;
        std::thread worker([&]{try{f.Step(0);}catch(...){error=std::current_exception();}});
        const bool started=ready.wait_for(std::chrono::seconds(2))==std::future_status::ready;
        if(started)f.state->OnExit();released.set_value();worker.join();
        require(started,"fake inference never started");if(error)std::rethrow_exception(error);f.Unpublished();
    }else if(scenario=="generation_snapshot_pair_stress"){
        Access::Generation(*f.state,-1);std::atomic<bool> done{false};std::atomic<int> reads{0},bad{0};int generated=0;
        f.robot->on_snapshot=[&]{f.robot->state.snapshot_steady_s=++generated;};
        std::thread reader([&]{
            do{const auto pair=Access::Snapshot(*f.state);
                if(pair.first>=0){++reads;if(pair.second!=pair.first+1)++bad;}
                std::this_thread::yield();
            }while(!done.load());
        });
        for(int i=0;i<1000;++i){f.state->Run();if(i%8==0)std::this_thread::yield();}
        done=true;reader.join();const auto last=Access::Snapshot(*f.state);
        require(reads>0&&bad==0&&last.first==999&&last.second==1000,"generation/snapshot published incoherently");
        require(f.robot->publications.empty(),"observation update unexpectedly published");
    }else if(scenario=="posture_boundary"){
        f.robot->state.base_rpy[0]=float(31.*M_PI/180.);require(f.state->LoseControlJudge(),"unsafe posture not rejected");
        f.robot->state.base_rpy[0]=NAN;require(f.state->LoseControlJudge(),"nonfinite posture not rejected");f.Unpublished();
    }else throw std::runtime_error("unknown scenario: "+scenario);
}
int main(int argc,char** argv){
    try{
        if(argc!=3)throw std::runtime_error("usage: cts_servo_probe robot.json scenario");
        setenv("M20_ROBOT_CONFIG",argv[1],1);setenv("M20_POLICY_PATH",argv[1],1);unsetenv("M20_CONTROL_TRACE_PATH");
        run(argv[2]);
        std::cout<<nlohmann::json{{"passed",true},{"scenario",argv[2]},
            {"scope","production RLControlState step/update with fake non-I/O dependencies"},
            {"motor_commands",0},{"dds_publishers",0}}.dump()<<std::endl;return 0;
    }catch(const std::exception& error){std::cout<<nlohmann::json{{"passed",false},{"error",error.what()}}.dump()<<std::endl;return 1;}
}
