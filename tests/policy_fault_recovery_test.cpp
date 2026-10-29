#include "policy_fault_recovery.hpp"
#include <cassert>
int main(){
 m20::PolicyFaultRecovery g;using C=m20::PolicyFaultRecovery::Clock;
 auto t=C::now();assert(g.StandAllowed(t));g.Latch();assert(!g.StandAllowed(t));
 g.Acknowledge(t);assert(!g.Reset(t)); // cannot ack an unhealthy fault
 for(int i=0;i<=410;++i)g.Health(t+std::chrono::milliseconds(5*i),true);
 auto ready=t+std::chrono::milliseconds(2050);
 assert(g.Ready(ready));assert(!g.Reset(ready)); // old ack did not become valid
 g.Acknowledge(ready);g.Health(ready,false);assert(!g.Reset(ready));
 for(int i=1;i<=420;++i)g.Health(ready+std::chrono::milliseconds(5*i),true);
 auto later=ready+std::chrono::milliseconds(2100);
 assert(!g.Reset(later)); // failed R2 consumed ack
 g.Acknowledge(later);g.Health(later+std::chrono::milliseconds(5),false);
 for(int i=2;i<=425;++i)g.Health(later+std::chrono::milliseconds(5*i),true);
 later+=std::chrono::milliseconds(2125);assert(!g.Reset(later));
 g.Acknowledge(later);assert(g.Reset(later));assert(g.StandAllowed(later));
 assert(!g.StandAllowed(later+std::chrono::milliseconds(41)));
 g.Latch();assert(!g.Reset(later));assert(!g.StandAllowed(later));
}
