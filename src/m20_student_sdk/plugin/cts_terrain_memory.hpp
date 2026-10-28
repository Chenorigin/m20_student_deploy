#pragma once
#include "cts_landing.hpp"
#include <map>
#include <vector>

namespace m20_cts {
class TerrainMemory {
  static constexpr float inf=std::numeric_limits<float>::infinity();
  struct Cell {int64_t x,y;uint64_t key;size_t slot;};
  struct Slot {V3 point=V3::Zero();uint64_t key=0;float capture=-inf,conflict=-inf;bool occupied=false;};
  struct Live {V3 point=V3::Zero();float capture=-inf;bool valid=false;};
  LandingSettings s_;
  size_t count_,capacity_;
  int radius_;
  std::vector<Slot> slots_;
  std::vector<Live> live_;
  V3 last_pose_=V3::Zero();float last_yaw_=0,last_time_=0,last_admission_=-inf,capture_not_before_=-inf;
  bool seen_=false;
  std::array<uint64_t,6> counts_{};
  Cell Coordinates(int64_t x,int64_t y)const {
    const auto hash=(x*73856093LL)^(y*19349663LL);
    auto mod=hash%int64_t(capacity_);if(mod<0)mod+=capacity_;
    return {x,y,(uint64_t(uint32_t(x))<<32)|uint32_t(y),size_t(mod)};
  }
  Cell Cells(const V3& point)const {
    auto index=[&](float x){if(!std::isfinite(x))return int64_t(0);
      const float quotient=x/s_.terrain_memory_cell_m;
      return int64_t(std::clamp(std::floor(quotient),-float(1LL<<30),float(1LL<<30)));};
    return Coordinates(index(point.x()),index(point.y()));
  }
public:
  struct Sample {std::array<float,4> height,age;std::array<bool,4> valid{};};
  TerrainMemory(size_t points,const LandingSettings& settings):s_(settings),count_(points),
      capacity_(points*settings.terrain_memory_frames),radius_(int(std::ceil(settings.terrain_max_distance_m/settings.terrain_memory_cell_m))),
      slots_(capacity_),live_(points){
    if(points==0||points>187||radius_>16)throw std::runtime_error("unbounded CTS terrain memory");
  }
  void Reset(){for(auto& v:slots_)v=Slot{};for(auto& v:live_)v=Live{};
    seen_=false;last_admission_=capture_not_before_=-inf;}
  void ResetAt(const V3& pose,float yaw,float now){Reset();last_pose_=pose;last_yaw_=yaw;
    last_time_=last_admission_=capture_not_before_=now;seen_=true;}
  const std::array<uint64_t,6>& Counts()const{return counts_;}
  void ObservePose(const V3& pose,float yaw,float now){
    const float delta=std::abs(std::atan2(std::sin(yaw-last_yaw_),std::cos(yaw-last_yaw_)));
    if(!pose.allFinite()||!std::isfinite(yaw)||!std::isfinite(now)||
       (seen_&&((pose-last_pose_).norm()>s_.terrain_pose_jump_m||delta>s_.terrain_yaw_jump_rad||now<last_time_-1e-6f))){
      Update(std::vector<V3>(count_,V3::Zero()),std::vector<bool>(count_,false),std::vector<float>(count_,0),pose,yaw,now);
    }else {last_pose_=pose;last_yaw_=yaw;last_time_=now;seen_=true;}
  }
  void Update(const std::vector<V3>& points,const std::vector<bool>& valid,const std::vector<float>& ages,
              const V3& pose,float yaw,float now){
    if(points.size()!=count_||valid.size()!=count_||ages.size()!=count_)throw std::runtime_error("terrain memory shape mismatch");
    const auto& s=s_;
    const bool pose_ok=pose.allFinite()&&std::isfinite(yaw)&&std::isfinite(now);
    const float delta_yaw=std::abs(std::atan2(std::sin(yaw-last_yaw_),std::cos(yaw-last_yaw_)));
    const bool pose_jump=seen_&&pose_ok&&((pose-last_pose_).norm()>s.terrain_pose_jump_m||delta_yaw>s.terrain_yaw_jump_rad);
    const bool clock_jump=seen_&&now<last_time_-1e-6f,reset=pose_jump||clock_jump||!pose_ok;
    const float safe_now=std::isfinite(now)?now:last_time_;
    if(reset){for(auto& v:slots_){v.occupied=false;v.conflict=-inf;}
      for(auto& v:live_)v.valid=false;capture_not_before_=last_admission_=safe_now;}
    const bool due=safe_now-last_admission_>=s.terrain_memory_update_s-1e-6f;
    if(due&&pose_ok&&!reset)last_admission_=safe_now;
    std::vector<Cell> cells;cells.reserve(count_);
    std::vector<float> capture(count_);
    std::vector<bool> fresh(count_);
    std::vector<float> newest(capacity_,-inf),quarantine(capacity_,-inf);
    std::vector<int> chosen(capacity_,-1);
    std::map<uint64_t,std::vector<size_t>> groups;
    for(size_t i=0;i<count_;++i){
      cells.push_back(Cells(points[i]));capture[i]=safe_now-ages[i];
      fresh[i]=valid[i]&&points[i].allFinite()&&std::isfinite(ages[i])&&ages[i]>=0&&ages[i]<=s.terrain_max_age_s&&
          points[i].head<2>().cwiseAbs().maxCoeff()<s.terrain_memory_cell_m*float(1LL<<30)&&
          capture[i]>=capture_not_before_-1e-6f&&pose_ok&&!reset;
      live_[i].point=points[i];for(int k=0;k<3;++k)if(!std::isfinite(live_[i].point[k]))live_[i].point[k]=0;
      live_[i].capture=fresh[i]?capture[i]:-inf;live_[i].valid=fresh[i];
      const auto& old=slots_[cells[i].slot];
      if(old.occupied&&old.key==cells[i].key&&safe_now-old.capture<=s.terrain_memory_s&&capture[i]<old.capture-1e-6f)
        live_[i].valid=false;
      if(fresh[i]){newest[cells[i].slot]=std::max(newest[cells[i].slot],capture[i]);groups[cells[i].key].push_back(i);}
    }
    for(size_t i=0;i<count_;++i)if(fresh[i]&&capture[i]>=newest[cells[i].slot]-1e-6f)chosen[cells[i].slot]=int(i);
    for(const auto& group:groups){
      float lo=inf,hi=-inf,latest=-inf,contrary=-inf;size_t latest_index=0;
      for(size_t i:group.second){lo=std::min(lo,points[i].z());hi=std::max(hi,points[i].z());latest=std::max(latest,capture[i]);}
      for(size_t i:group.second)if(capture[i]>=latest-1e-6f)latest_index=i;
      for(size_t i:group.second)if(std::abs(points[i].z()-points[latest_index].z())>s.terrain_max_height_spread_m)
        contrary=std::max(contrary,capture[i]);
      if(hi-lo>s.terrain_max_height_spread_m){
        const auto slot=cells[group.second.front()].slot;
        quarantine[slot]=std::max(quarantine[slot],(std::isfinite(contrary)?contrary:latest)+s.terrain_memory_s);
      }
    }
    uint64_t admitted=0,evicted=0,conflicts=0;
    for(size_t b=0;b<capacity_;++b){
      auto& old=slots_[b];const bool alive=old.occupied&&safe_now-old.capture<=s.terrain_memory_s;
      const int chosen_index=chosen[b];const bool exists=chosen_index>=0;
      const size_t i=exists?size_t(chosen_index):0;
      const bool same=alive&&old.key==cells[i].key;
      const bool accept=exists&&(due||same)&&(!alive||newest[b]>old.capture+1e-6f||(same&&newest[b]>=old.capture-1e-6f));
      float until=std::max(old.conflict,quarantine[b]);
      if(accept&&same&&std::abs(points[i].z()-old.point.z())>s.terrain_max_height_spread_m)
        until=std::max(until,old.capture+s.terrain_memory_s);
      conflicts+=until>old.conflict;evicted+=accept&&alive&&!same;
      if(accept){old.point=points[i];old.key=cells[i].key;old.capture=newest[b];old.occupied=true;++admitted;}
      old.conflict=until;
    }
    if(pose_ok){last_pose_=pose;last_yaw_=yaw;}last_time_=safe_now;seen_=pose_ok;
    ++counts_[0];counts_[1]+=admitted;counts_[2]+=evicted;counts_[3]+=conflicts;counts_[4]+=pose_jump;counts_[5]+=clock_jump||!pose_ok;
  }
  Sample Query(const std::array<V3,4>& support,float now)const {
    struct Candidate {V3 point;float age;bool usable;};
    std::vector<Candidate> candidates;candidates.reserve(4*(2*radius_+1)*(2*radius_+1)+count_);
    std::array<bool,4> quarantined{};
    const auto& s=s_;
    for(int wheel=0;wheel<4;++wheel){
      const auto center=Cells(support[wheel]);
      for(int dx=-radius_;dx<=radius_;++dx)for(int dy=-radius_;dy<=radius_;++dy){
        const auto cell=Coordinates(center.x+dx,center.y+dy);const auto& slot=slots_[cell.slot];
        const float age=now-slot.capture;
        const bool known=slot.occupied&&slot.key==cell.key&&age>=-1e-6f&&age<=s.terrain_memory_s;
        candidates.push_back({slot.point,age,known&&slot.conflict<=now});
        const float lx=float(cell.x)*s.terrain_memory_cell_m,ly=float(cell.y)*s.terrain_memory_cell_m;
        const float hx=lx+s.terrain_memory_cell_m,hy=ly+s.terrain_memory_cell_m;
        const float ex=std::max({lx-support[wheel].x(),support[wheel].x()-hx,0.f});
        const float ey=std::max({ly-support[wheel].y(),support[wheel].y()-hy,0.f});
        if(slot.conflict>now&&ex*ex+ey*ey<=s.terrain_max_distance_m*s.terrain_max_distance_m)quarantined[wheel]=true;
      }
    }
    for(const auto& live:live_){const float age=now-live.capture;
      candidates.push_back({live.point,age,live.valid&&age>=-1e-6f&&age<=s.terrain_max_age_s});}
    Sample out;out.height.fill(std::numeric_limits<float>::quiet_NaN());out.age=out.height;
    for(int i=0;i<4;++i){
      float lo=inf,hi=-inf,nearest=inf,height=0,age=0;
      for(const auto& c:candidates){const float dist=(support[i]-c.point).head<2>().squaredNorm();
        if(!c.usable||!c.point.allFinite()||!std::isfinite(dist)||dist>s.terrain_max_distance_m*s.terrain_max_distance_m)continue;
        lo=std::min(lo,c.point.z());hi=std::max(hi,c.point.z());
        if(dist<nearest){nearest=dist;height=c.point.z();age=c.age;}
      }
      out.valid[i]=std::isfinite(nearest)&&support[i].head<2>().allFinite()&&hi-lo<=s.terrain_max_height_spread_m&&
          !quarantined[i]&&std::isfinite(now)&&now>=last_time_-1e-6f;
      if(out.valid[i]){out.height[i]=height;out.age[i]=age;}
    }
    return out;
  }
};
}
