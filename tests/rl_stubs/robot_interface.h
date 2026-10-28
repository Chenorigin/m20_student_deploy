#pragma once
// Offline dependency fake only. No ROS node, DDS endpoint, socket or motor I/O.
#include "common_types.h"
#include <functional>
namespace interface {
struct JointCommandReceipt {
    bool published=false;
    uint64_t sequence=0;
    double returned_steady_s=0;
    types::MatXf command;
};
class RobotInterface {
public:
    virtual ~RobotInterface()=default;
    double* now=nullptr;
    double publish_duration=0;
    uint64_t sequence=0;
    bool blocked=false,throw_publish=false,duplicate_receipt=false;
    double receipt_time_offset=0;
    types::RobotBasicState state;
    types::MatXf cached;
    std::vector<types::MatXf> publications;
    std::function<void()> on_publish,on_snapshot;
    virtual JointCommandReceipt SetJointCommandWithReceipt(types::MatXf command){
        if(blocked)return {};
        if(throw_publish)throw std::runtime_error("fake publisher: delivery unknown");
        if(on_publish)on_publish();
        *now+=publish_duration;
        cached=command;
        for(int j=3;j<16;j+=4)cached(j,0)=0;
        publications.push_back(cached);
        const auto seq=duplicate_receipt?sequence:++sequence;
        return {true,seq,*now+receipt_time_offset,cached};
    }
    virtual void SetJointCommand(types::MatXf command){(void)SetJointCommandWithReceipt(std::move(command));}
    virtual void FlushJointCommandTrace(){}
    virtual types::MatXf GetJointCommand(){return cached;}
    virtual types::RobotBasicState GetCoherentRobotState(){if(on_snapshot)on_snapshot();return state;}
    virtual m20_cts::LandingFeedback GetLandingFeedback(){return {};}
    virtual double GetBodyReceiveAgeSeconds(){return state.body_receive_age_s;}
    virtual types::Vec3f GetImuRpy(){return state.base_rpy;}
    virtual types::Vec3f GetImuOmega(){return state.base_omega;}
    virtual types::Vec3f GetImuAcc(){return state.base_acc;}
    virtual types::VecXf GetJointPosition(){return state.joint_pos;}
    virtual types::VecXf GetJointVelocity(){return state.joint_vel;}
    virtual types::VecXf GetJointTorque(){return state.joint_tau;}
    virtual types::VecXf GetLidarObservation(){return {};}
    virtual bool IsLidarObservationValid(){return false;}
    virtual float GetLidarObservationAgeSeconds(){return INFINITY;}
    virtual types::VecXf GetDepthObservation(){return {};}
    virtual bool IsDepthObservationValid(){return false;}
    virtual float GetDepthObservationAgeSeconds(){return INFINITY;}
    virtual types::VecXf GetV11Observation(){return {};}
    virtual bool IsV11ObservationValid(){return false;}
    virtual float GetV11ObservationAgeSeconds(){return INFINITY;}
};
}
