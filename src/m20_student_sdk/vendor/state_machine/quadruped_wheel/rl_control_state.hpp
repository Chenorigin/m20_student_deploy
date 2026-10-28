/**
 * @file rl_control_state.hpp
 * @brief rl policy runnning state for quadruped-wheel robot
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */
#pragma once
#include "state_base.h"
#include "policy_runner_base.hpp"
#include "student_policy_runner.hpp"
#include "robot_interface.h"
#include "user_command_interface.h"
#include "json.hpp"
#include "basic_function.hpp"
#include <functional>

namespace qw {
    class RLControlState : public StateBase {
    private:
        RobotBasicState rbs_[2];
        std::atomic<int> rbs_write_index_{0};
        std::mutex rbs_mutex_;
        int getrbsReadIndex() const { return 1 - rbs_write_index_.load(std::memory_order_acquire); }

        std::atomic<int> state_run_cnt_{-1};

        std::shared_ptr<PolicyRunnerBase> policy_ptr_;
        std::shared_ptr<M20PolicyRunner> m20_policy_;

        std::thread run_policy_thread_;
        std::atomic<bool> start_flag_{true};
        std::atomic<bool> policy_fault_{false};
        std::mutex publish_gate_;

        float policy_cost_time_ = 1;
        uint64_t last_publication_sequence_=0;
        double last_cts_step_time_=std::numeric_limits<double>::quiet_NaN();
        struct ServoAudit {
            int selected=-1,live=-1;
            double snapshot=0,start=0,evaluate_end=0,publish_start=0,publish_returned=0,end=0;
            bool inferred=false,published=false;
            std::string outcome="pending";
        } servo_audit_;
#ifdef M20_RL_OFFLINE_TEST
        friend struct RLControlStateTest;
#endif

        using Now = std::function<double()>;
        void RequireDeadline(int generation,double start,double snapshot,double now){
            if(!std::isfinite(now)||!std::isfinite(snapshot)||now<snapshot||now<start)
                throw std::runtime_error("CTS invalid snapshot/worker clock");
            servo_audit_.live=state_run_cnt_.load(std::memory_order_acquire);
            if(servo_audit_.live!=generation)throw m20_cts::SchedulingFault("generation advanced before publication");
            if(now-start>=m20_cts::servo_period_s)throw m20_cts::SchedulingFault("worker exceeded 5ms deadline");
            if(now-snapshot>=m20_cts::servo_period_s)throw m20_cts::SchedulingFault("snapshot exceeded 5ms deadline");
        }
        // One real production CTS servo step. The deterministic offline probe
        // supplies time and fake interfaces; runtime always uses SteadySeconds.
        void ProcessCTSStep(RobotBasicState observation,int generation,
                            m20_cts::SubstepClock& clock,RobotAction& nominal,const Now& now){
            const double start=now(),base_age=observation.body_receive_age_s;
            servo_audit_={};servo_audit_.selected=generation;
            servo_audit_.snapshot=observation.snapshot_steady_s;servo_audit_.start=start;
            auto refresh_body=[&](){
                const double t=now();
                if(!std::isfinite(t)||!std::isfinite(observation.snapshot_steady_s)||t<observation.snapshot_steady_s)
                    throw std::runtime_error("CTS invalid snapshot/worker clock");
                observation.body_receive_age_s=base_age+t-observation.snapshot_steady_s;
                m20_policy_->CheckCTSBody(observation);
            };
            try{
                if(!std::isfinite(base_age)||base_age<0)
                    throw std::runtime_error("CTS invalid SDK body receive age");
                if(!std::isfinite(start)||(std::isfinite(last_cts_step_time_)&&start<=last_cts_step_time_))
                    throw std::runtime_error("CTS nonmonotonic servo clock");
                last_cts_step_time_=start;
                // Body safety remains active in HOLD and even on an expired
                // scheduling window. Only scheduling faults preserve support.
                refresh_body();
                RequireDeadline(generation,start,observation.snapshot_steady_s,now());
                if(!m20_policy_->CTSNominalIsHeld()){
                    servo_audit_.inferred=clock.Advance(generation,start);
                    if(servo_audit_.inferred)nominal=policy_ptr_->getRobotAction(observation,*(uc_ptr_->GetUserCommand()));
                }
                servo_audit_.evaluate_end=now();
                refresh_body();
                const auto ra=m20_policy_->ExecuteCTSSubstep(nominal,observation);
                const MatXf res=ra.ConvertToMat();
                if(res.rows()!=16||res.cols()!=5||!res.allFinite())throw std::runtime_error("nonfinite SDK target");
                std::lock_guard<std::mutex> permit(publish_gate_);
                const auto command=uc_ptr_->GetUserCommand();
                if(!start_flag_.load(std::memory_order_acquire)||command->safe_control_mode!=0||
                   command->target_mode==uint8_t(RobotMotionState::JointDamping)||
                   command->target_mode==uint8_t(RobotMotionState::LieDown)){
                    servo_audit_.outcome="exit requested; no publication";servo_audit_.end=now();return;
                }
                refresh_body();
                RequireDeadline(generation,start,observation.snapshot_steady_s,now());
                servo_audit_.publish_start=now();
                const auto receipt=ri_ptr_->SetJointCommandWithReceipt(res);
                const double returned=now();
                if(!receipt.published)throw std::runtime_error("CTS command refused; ownership unavailable");
                if(receipt.sequence<=last_publication_sequence_||!std::isfinite(receipt.returned_steady_s)||
                   receipt.returned_steady_s<servo_audit_.publish_start||receipt.returned_steady_s>returned)
                    throw std::runtime_error("CTS invalid publication receipt; delivery unknown");
                // A successful publish cannot be undone by a later deadline
                // check. Commit the actual issued command/history first.
                m20_policy_->RecordCTSCommandPublished(receipt.command);
                last_publication_sequence_=receipt.sequence;
                servo_audit_.published=true;servo_audit_.publish_returned=receipt.returned_steady_s;
                refresh_body();
                RequireDeadline(generation,start,observation.snapshot_steady_s,now());
                servo_audit_.outcome=m20_policy_->CTSNominalIsHeld()?"hold published":"policy published";
            }catch(const m20_cts::SchedulingFault& error){
                // Discard staged action; keep the last successful publication.
                // Never send a replacement on this expired snapshot. The next
                // fresh generation can publish HOLD without inference/clock.
                const bool already_held=m20_policy_->CTSNominalIsHeld();
                m20_policy_->LatchCTSTimingHold(error.what());
                servo_audit_.outcome=error.what();
                if(!already_held)std::cerr<<"[CTS_TIMING_HOLD] "<<error.what()<<"; support retained; re-entry required"<<std::endl;
            }
            servo_audit_.end=now();
        }
        void TraceServo() noexcept {
            try{
                auto row=StudentTrace::Stamp();row["event"]="cts_servo_decision";
                row["selected_generation"]=servo_audit_.selected;row["live_generation"]=servo_audit_.live;
                row["snapshot_steady_s"]=servo_audit_.snapshot;row["worker_start_s"]=servo_audit_.start;
                row["get_action_returned_s"]=servo_audit_.evaluate_end;row["publish_call_s"]=servo_audit_.publish_start;
                row["publish_returned_s"]=servo_audit_.publish_returned;row["worker_decision_end_s"]=servo_audit_.end;
                row["inferred"]=servo_audit_.inferred;row["published"]=servo_audit_.published;
                row["publication_sequence"]=last_publication_sequence_;row["outcome"]=servo_audit_.outcome;
                StudentTrace::Instance().Push(std::move(row));
            }catch(...){}
        }

        Eigen::MatrixXf acc_rot = Eigen::MatrixXf::Zero(20, 3);
        int acc_rot_count = 0;

        void UpdateRobotObservation() {
            std::lock_guard<std::mutex> lock(rbs_mutex_);
            int write_idx = rbs_write_index_.load(std::memory_order_relaxed);
            RobotBasicState& buffer = rbs_[write_idx];
            if(m20_policy_->IsCTS()){
                auto coherent=ri_ptr_->GetCoherentRobotState();
                buffer.snapshot_steady_s=coherent.snapshot_steady_s;buffer.body_receive_age_s=coherent.body_receive_age_s;
                buffer.base_rpy=coherent.base_rpy;buffer.base_omega=coherent.base_omega;buffer.base_acc=coherent.base_acc;
                buffer.joint_pos=coherent.joint_pos;buffer.joint_vel=coherent.joint_vel;buffer.joint_tau=coherent.joint_tau;
                buffer.landing_feedback=m20_policy_->UsesLandingControl()?ri_ptr_->GetLandingFeedback():m20_cts::LandingFeedback{};
            }else{
                buffer.snapshot_steady_s=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
                buffer.body_receive_age_s=ri_ptr_->GetBodyReceiveAgeSeconds();
                buffer.base_rpy=ri_ptr_->GetImuRpy();buffer.base_omega=ri_ptr_->GetImuOmega();buffer.base_acc=ri_ptr_->GetImuAcc();
                buffer.joint_pos=ri_ptr_->GetJointPosition();buffer.joint_vel=ri_ptr_->GetJointVelocity();buffer.joint_tau=ri_ptr_->GetJointTorque();
            }
            buffer.base_rot_mat = RpyToRm(buffer.base_rpy);
            buffer.lidar_observation = ri_ptr_->GetLidarObservation();
            buffer.lidar_valid = ri_ptr_->IsLidarObservationValid();
            buffer.lidar_age_seconds = ri_ptr_->GetLidarObservationAgeSeconds();
            buffer.depth_observation = ri_ptr_->GetDepthObservation();
            buffer.depth_valid = ri_ptr_->IsDepthObservationValid();
            buffer.depth_age_seconds = ri_ptr_->GetDepthObservationAgeSeconds();
            buffer.v11_observation = ri_ptr_->GetV11Observation();
            buffer.v11_valid = ri_ptr_->IsV11ObservationValid();
            buffer.v11_age_seconds = ri_ptr_->GetV11ObservationAgeSeconds();

            // 储存
            buffer.flt_base_acc_mat.row(acc_rot_count) = buffer.base_acc.transpose();
            acc_rot_count += 1;
            acc_rot_count = acc_rot_count % 20;

            rbs_write_index_.store(1 - write_idx,  std::memory_order_release);
            // Generation and its complete snapshot are published under one
            // mutex. The worker must select both inside this same lock.
            state_run_cnt_.fetch_add(1,std::memory_order_release);
        }

        void PolicyRunner() {
            int run_cnt_record=-1;
            m20_cts::SubstepClock cts_clock;
            RobotAction cts_nominal;
            while(start_flag_.load(std::memory_order_acquire)){
                int generation;
                RobotBasicState observation;
                const bool cts=m20_policy_->IsCTS();
                {
                    std::lock_guard<std::mutex> lock(rbs_mutex_);
                    generation=state_run_cnt_.load(std::memory_order_acquire);
                    if(generation>=0&&generation!=run_cnt_record&&
                       (cts||generation%policy_ptr_->decimation_==0))
                        observation=rbs_[getrbsReadIndex()];
                    else generation=-1;
                }
                if(generation>=0){
                    const double start=m20_cts::SteadySeconds();
                    try{
                        if(cts)ProcessCTSStep(observation,generation,cts_clock,cts_nominal,m20_cts::SteadySeconds);
                        else {
                            observation.body_receive_age_s+=m20_cts::SteadySeconds()-observation.snapshot_steady_s;
                            const auto res=policy_ptr_->getRobotAction(observation,*(uc_ptr_->GetUserCommand())).ConvertToMat();
                            if(res.rows()!=16||res.cols()!=5||!res.allFinite())throw std::runtime_error("nonfinite SDK target");
                            std::lock_guard<std::mutex> permit(publish_gate_);
                            if(!start_flag_.load(std::memory_order_acquire))break;
                            ri_ptr_->SetJointCommand(res);
                        }
                    }catch(const std::exception& error){
                        servo_audit_.outcome=std::string("hard fault: ")+error.what();
                        servo_audit_.end=m20_cts::SteadySeconds();
                        std::cerr<<"[STUDENT_FAULT] "<<error.what()<<"; requesting official FSM joint damping"<<std::endl;
                        policy_fault_.store(true,std::memory_order_release);
                        start_flag_.store(false,std::memory_order_release);
                    }
                    // Expensive JSON is outside the publication decision and
                    // the exit lock. Any resulting missed tick latches HOLD.
                    if(cts){TraceServo();ri_ptr_->FlushJointCommandTrace();m20_policy_->FlushCTSTrace();}
                    run_cnt_record=generation;
                    policy_cost_time_=(m20_cts::SteadySeconds()-start)*1e3;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

    public:
        RLControlState(const RobotName &robot_name, const std::string &state_name,
                       std::shared_ptr<ControllerData> data_ptr) : StateBase(robot_name, state_name, data_ptr) {
            if (robot_name_ == RobotName::M20) {
                namespace fs = std::filesystem;
                fs::path model_path;
                const char* policy_override = std::getenv("M20_POLICY_PATH");
                if (policy_override != nullptr && policy_override[0] != '\0') {
                    model_path = fs::canonical(fs::path(policy_override));
                } else {
                    fs::path base = fs::path(__FILE__).parent_path();
                    model_path = fs::canonical(base / ".." / ".." / "policy" / "policy.onnx");
                }
                std::cout << "Loading M20 policy: " << model_path << std::endl;
                m20_policy_ = std::make_shared<M20PolicyRunner>("m20_policy", model_path.string());
            }

            policy_ptr_ = m20_policy_;
            if (!policy_ptr_) {
                std::cerr << "error policy" << std::endl;
                exit(0);
            }
            policy_ptr_->DisplayPolicyInfo();
        }

        ~RLControlState() {
            start_flag_.store(false, std::memory_order_release);
            if (run_policy_thread_.joinable()) {
                run_policy_thread_.join();
            }
        }

        virtual void OnEnter() {
            state_run_cnt_.store(-1, std::memory_order_release);
            start_flag_.store(true, std::memory_order_release);
            policy_fault_.store(false,std::memory_order_release);
            last_publication_sequence_=0;
            last_cts_step_time_=std::numeric_limits<double>::quiet_NaN();
            policy_ptr_->OnEnter();
            try{m20_policy_->SetStandingHandoff(ri_ptr_->GetJointCommand());}
            catch(const std::exception& error){
                std::cerr<<"[STUDENT_FAULT] invalid standing handoff: "<<error.what()<<std::endl;
                policy_fault_.store(true,std::memory_order_release);
                start_flag_.store(false,std::memory_order_release);return;
            }
            run_policy_thread_ = std::thread(std::bind(&RLControlState::PolicyRunner, this));
            StateBase::msfb_.UpdateCurrentState(RobotMotionState::RLControlMode);
        };

        virtual void OnExit() {
            {std::lock_guard<std::mutex> permit(publish_gate_);
             start_flag_.store(false, std::memory_order_release);}
            if (run_policy_thread_.joinable()) {
                run_policy_thread_.join();
            }
            state_run_cnt_.store(-1, std::memory_order_release);
        }

        virtual void Run() {
            UpdateRobotObservation();
        }

        virtual bool LoseControlJudge() {
            if(policy_fault_.load(std::memory_order_acquire))return true;
            if (uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::JointDamping)) return true;
            return PostureUnsafeCheck();
        }

        bool PostureUnsafeCheck() {
            Vec3f rpy = ri_ptr_->GetImuRpy();
            if (!rpy.allFinite() || std::abs(rpy(0)) > 30. / 180. * M_PI ||
                std::abs(rpy(1)) > 45. / 180. * M_PI) {
                std::cout << "Unsafe RL posture (deg): " << 180. / M_PI * rpy.transpose() << std::endl;
                return true;
            }
            return false;
        }

        virtual StateName GetNextStateName() {
            if (uc_ptr_->GetUserCommand()->safe_control_mode != 0) 
                return StateName::kJointDamping;
            if (uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::LieDown))
                return StateName::kLieDown;
            
            return StateName::kRLControl;
        }
    };
};
