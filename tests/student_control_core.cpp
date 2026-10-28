#include "student_control_core.hpp"
#include <cassert>
#include <random>
using namespace types;
int main(int argc,char** argv){
 nlohmann::json robot;std::ifstream f(argv[1]);f>>robot;
 MatXf seed=MatXf::Zero(16,5);
 for(int j=0;j<16;j++){int i=robot["joints"]["sdk_from_policy_indices"][j];seed(j,0)=j%4==3?0:100;seed(j,1)=robot["joints"]["default_position_policy_rad"][i];seed(j,2)=j%4==3?1:5;}
 auto rejects=[](auto fun){try{fun();return false;}catch(const std::exception&){return true;}};
 nlohmann::json results;
 StudentControlCore c(robot);float a[16]={};
 results["unseeded_map_rejected"]=rejects([&]{c.Map(a,.02);});
 results["unseeded_hold_rejected"]=rejects([&]{c.Hold("fault");});
 auto bad=seed;bad(1,0)=0;results["zero_support_seed_rejected"]=rejects([&]{c.Begin(bad);});
 bad=seed;bad(1,1)=100;results["out_of_bounds_seed_rejected"]=rejects([&]{c.Begin(bad);});
 bad=seed;bad(1,1)=NAN;results["nonfinite_seed_rejected"]=rejects([&]{c.Begin(bad);});
 c.Begin(seed);a[1]=.1;auto small=c.Map(a,.02);results["small_target_positive_control"]=std::abs(small.goal_joint_pos(1)+.275f)<1e-6;
 c.Begin(seed);a[1]=1;auto current=c.Map(a,.02);for(int n=0;n<20;n++)current=c.Map(a,.02);
 results["fixed_measurement_eventually_reaches_target"]=std::abs(current.goal_joint_pos(1)+.05f)<1e-6;
 results["previous_action_matches_normalized_input"]=std::abs(c.Previous()[1]-1)<1e-6;
 RobotBasicState ro;ro.joint_pos=seed.col(1);ro.body_receive_age_s=.01;c.CheckBody(ro,.08);
 ro.joint_pos(1)=-.1;c.CheckBody(ro,.08);float before=current.goal_joint_pos(1);current=c.Map(a,.02);
 results["target_rate_bound_despite_measurement_jump"]=std::abs(current.goal_joint_pos(1)-before)<=.050001f;
 auto held=c.Hold("first");ro.joint_pos(1)=-.4;c.CheckBody(ro,.08);auto held2=c.Hold("second");
 results["hold_does_not_follow_measurement"]=held.ConvertToMat().isApprox(held2.ConvertToMat());
 results["hold_stops_wheels_positive_control"]=held.goal_joint_vel.isZero();
 a[1]=-10;auto blocked=c.Map(a,.02);results["latch_blocks_future_actions"]=blocked.ConvertToMat().isApprox(held.ConvertToMat())&&c.Reason()=="first";
 ro.body_receive_age_s=.1;results["stale_body_still_rejected_after_latch"]=rejects([&]{c.CheckBody(ro,.08);});
 ro.body_receive_age_s=.01;ro.joint_pos(1)=100;results["invalid_measured_leg_rejected_after_latch"]=rejects([&]{c.CheckBody(ro,.08);});
 c.Begin(seed);results["reentry_clears_latch_and_last_action"]=!c.Latched()&&c.Reason().empty()&&std::all_of(c.Previous().begin(),c.Previous().end(),[](float v){return v==0;});
 auto immediate=c.Hold("before inference");results["initial_fault_preserves_standing_support"]=immediate.goal_joint_pos.isApprox(seed.col(1))&&immediate.kp.isApprox(seed.col(0));
 c.Begin(seed);a[1]=100;auto short_tick=c.Map(a,.015);results["short_tick_honors_wall_time_rate"]=std::abs(short_tick.goal_joint_pos(1)-seed(1,1)-.0375f)<1e-6;
 c.Begin(seed);auto long_tick=c.Map(a,2);results["delayed_tick_does_not_enlarge_step"]=std::abs(long_tick.goal_joint_pos(1)-seed(1,1)-.05f)<1e-6;
 c.Begin(seed);a[12]=20;auto wheel=c.Map(a,.02);results["wheel_clipped_but_previous_raw"]=std::abs(wheel.goal_joint_vel(3)-10)<1e-6&&std::abs(c.Previous()[12]-20)<1e-6&&wheel.kp(3)==0;
 auto before_bad=wheel;a[15]=NAN;auto nonfinite=c.Map(a,.02);results["nonfinite_action_cannot_partially_commit"]=c.Latched()&&nonfinite.goal_joint_pos.isApprox(before_bad.goal_joint_pos)&&nonfinite.goal_joint_vel.isZero()&&c.Previous()[12]==20;
 c.Begin(seed);std::mt19937 rng(143);std::uniform_real_distribution<float> rand(-200,200);auto previous_target=seed.col(1).eval();bool random_ok=true;
 for(int n=0;n<1000;n++){
  for(auto& x:a)x=rand(rng);double dt=n%2?.015:.025;auto t=c.Map(a,dt);random_ok &= t.ConvertToMat().allFinite();
  for(int j=0;j<16;j++){
   int i=robot["joints"]["sdk_from_policy_indices"][j];float speed=robot["hardware"]["software_velocity_limits_sdk_rad_s"][j];
   random_ok &= std::abs(c.Previous()[i]-std::clamp(a[i],-100.f,100.f))<1e-6;
   if(j%4==3)random_ok &= std::abs(t.goal_joint_vel(j))<=speed+1e-5&&t.kp(j)==0;
   else random_ok &= std::abs(t.goal_joint_pos(j)-previous_target(j))<=speed*std::min(dt,.02)+1e-5 && t.goal_joint_pos(j)>=robot["hardware"]["software_position_limits_sdk_rad"][j][0].get<float>()-1e-6 && t.goal_joint_pos(j)<=robot["hardware"]["software_position_limits_sdk_rad"][j][1].get<float>()+1e-6;
  }
  previous_target=t.goal_joint_pos;
 }
 results["random_1000_ticks_obey_all_shaping_invariants"]=random_ok;

 bool all_body_limits=true;
 ro.joint_pos=seed.col(1);ro.body_receive_age_s=.01;
 for(int j=0;j<16;j++)if(j%4!=3){
  for(int side=0;side<2;side++){
   ro.joint_pos=seed.col(1);ro.joint_pos(j)=robot["hardware"]["software_position_limits_sdk_rad"][j][side].get<float>()+(side?1:-1)*.001f;
   all_body_limits &= rejects([&]{c.CheckBody(ro,.08);});
  }
 }
 results["all_12_legs_both_measured_bounds_rejected"]=all_body_limits;
 bool finite_body=true;ro.joint_pos=seed.col(1);ro.body_receive_age_s=.01;
 for(int j=0;j<16;j++){auto q=ro;q.joint_pos(j)=NAN;finite_body &= rejects([&]{c.CheckBody(q,.08);});q=ro;q.joint_vel(j)=INFINITY;finite_body &= rejects([&]{c.CheckBody(q,.08);});}
 for(int j=0;j<3;j++){auto q=ro;q.base_omega(j)=NAN;finite_body &= rejects([&]{c.CheckBody(q,.08);});for(int k=0;k<3;k++){q=ro;q.base_rot_mat(j,k)=NAN;finite_body &= rejects([&]{c.CheckBody(q,.08);});}}
 for(double age:{-1.,INFINITY*1.,NAN*1.}){auto q=ro;q.body_receive_age_s=age;finite_body &= rejects([&]{c.CheckBody(q,.08);});}
 results["all_body_nonfinite_channels_and_invalid_ages_rejected"]=finite_body;
 results["handoff_wrong_shape_rejected"]=rejects([&]{c.Begin(MatXf::Zero(15,5));})&&rejects([&]{c.Begin(MatXf::Zero(16,4));});
 bad=seed;bad(1,2)=-1;results["handoff_negative_kd_rejected"]=rejects([&]{c.Begin(bad);});
 bool dt_ok=true;for(double dt:{0.,-1.,INFINITY*1.,NAN*1.}){c.Begin(seed);auto stopped=c.Map(a,dt);dt_ok &= c.Latched()&&stopped.goal_joint_pos.isApprox(seed.col(1))&&stopped.goal_joint_vel.isZero();}
 results["invalid_intervals_latch_without_target_motion"]=dt_ok;
 c.Begin(seed);auto safe=c.Map(std::array<float,16>{}.data(),.02);c.ValidateTarget(safe);
 auto unsafe=safe;unsafe.goal_joint_pos[1]=100;
 results["nominal_position_bound_rejected_with_clipping_disabled"]=rejects([&]{c.ValidateTarget(unsafe);});
 unsafe=safe;unsafe.goal_joint_vel[3]=NAN;
 results["nominal_nonfinite_command_rejected"]=rejects([&]{c.ValidateTarget(unsafe);});
 results["nominal_target_validation_does_not_change_pd_or_targets"]=safe.ConvertToMat().isApprox(c.Map(std::array<float,16>{}.data(),.02).ConvertToMat());
 int ok=0,badn=0;for(auto& val:results.items())(val.value().get<bool>()?ok:badn)++;
 std::cout<<nlohmann::json{{"ok",ok},{"bad",badn},{"results",results}}.dump(2)<<std::endl;return badn?1:0;
}
