#pragma once
#include <chrono>
#include <mutex>
namespace m20 {
// Only a typed position-limit fault may enter this latch. No motor I/O.
class PolicyFaultRecovery {
public:
 using Clock=std::chrono::steady_clock; using Time=Clock::time_point;
 void Latch(){std::lock_guard<std::mutex> l(mu_);active_=true;ack_=false;healthy_=false;ever_=true;}
 bool Active()const{std::lock_guard<std::mutex> l(mu_);return active_;}
 void Health(Time now,bool safe){std::lock_guard<std::mutex> l(mu_);
   if(!safe||!healthy_||now-last_>std::chrono::milliseconds(40)){since_=now;healthy_=safe;ack_=false;}
   last_=now;
 }
 bool Ready(Time now)const{std::lock_guard<std::mutex> l(mu_);return ReadyLocked(now);}
 bool StandAllowed(Time now)const{std::lock_guard<std::mutex> l(mu_);return !active_&&(!ever_||ReadyLocked(now));}
 bool Acknowledge(Time now){std::lock_guard<std::mutex> l(mu_);ack_=active_&&ReadyLocked(now);ack_time_=now;return ack_;}
 bool Reset(Time now){std::lock_guard<std::mutex> l(mu_);
   const bool accepted=active_&&ack_&&ReadyLocked(now)&&now>=ack_time_&&now-ack_time_<=std::chrono::seconds(5);
   ack_=false; // A denied R2 never turns into a delayed retry.
   if(!accepted)return false;
   active_=false;return true;
 }
private:
 bool ReadyLocked(Time now)const{return healthy_&&now>=last_&&now-last_<=std::chrono::milliseconds(40)&&now-since_>=std::chrono::seconds(2);}
 mutable std::mutex mu_;bool active_=false,ack_=false,healthy_=false,ever_=false;Time since_{},last_{},ack_time_{};
};
} // namespace m20
