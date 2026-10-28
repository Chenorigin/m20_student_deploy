#pragma once
#include <pthread.h>
#include <sched.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace m20_cts {
struct WorkerScheduling {int cpu,priority;};
inline int WorkerSetting(const char* value,int lower,int upper){
    const std::string text=value?value:"";
    if(text.empty()||text.find_first_not_of("0123456789")!=std::string::npos)
        throw std::runtime_error("CTS worker scheduling requires unsigned integers");
    size_t used=0;const auto number=std::stoul(text,&used);
    if(used!=text.size()||number<static_cast<unsigned>(lower)||number>static_cast<unsigned>(upper))
        throw std::runtime_error("CTS worker scheduling setting out of range");
    return static_cast<int>(number);
}
inline std::optional<WorkerScheduling> ReadWorkerScheduling(){
    const char* cpu=std::getenv("M20_CTS_WORKER_CPU");
    const char* priority=std::getenv("M20_CTS_WORKER_FIFO_PRIORITY");
    if(!cpu&&!priority)return std::nullopt;
    if(!cpu||!priority)throw std::runtime_error("CTS worker CPU and FIFO priority must be configured together");
    return WorkerScheduling{WorkerSetting(cpu,0,CPU_SETSIZE-1),
        WorkerSetting(priority,sched_get_priority_min(SCHED_FIFO),sched_get_priority_max(SCHED_FIFO))};
}
inline void ConfigureWorkerScheduling(){
    const auto settings=ReadWorkerScheduling();if(!settings)return;
    auto require=[](int code,const char* operation){
        if(code)throw std::runtime_error(std::string("CTS worker ")+operation+": "+std::strerror(code));
    };
    cpu_set_t allowed;CPU_ZERO(&allowed);
    require(pthread_getaffinity_np(pthread_self(),sizeof(allowed),&allowed),"get affinity");
    if(!CPU_ISSET(settings->cpu,&allowed))throw std::runtime_error("CTS worker CPU outside launch affinity");
    cpu_set_t chosen;CPU_ZERO(&chosen);CPU_SET(settings->cpu,&chosen);
    require(pthread_setaffinity_np(pthread_self(),sizeof(chosen),&chosen),"set affinity");
    sched_param requested{};requested.sched_priority=settings->priority;
    require(pthread_setschedparam(pthread_self(),SCHED_FIFO,&requested),"set FIFO scheduling");
    cpu_set_t actual;CPU_ZERO(&actual);int policy=0;sched_param measured{};
    require(pthread_getaffinity_np(pthread_self(),sizeof(actual),&actual),"verify affinity");
    require(pthread_getschedparam(pthread_self(),&policy,&measured),"verify scheduling");
    if(CPU_COUNT(&actual)!=1||!CPU_ISSET(settings->cpu,&actual)||
       policy!=SCHED_FIFO||measured.sched_priority!=settings->priority)
        throw std::runtime_error("CTS worker scheduling readback mismatch");
    require(pthread_setname_np(pthread_self(),"cts_policy"),"name thread");
    std::cout<<"[CTS_WORKER] affinity="<<settings->cpu<<" scheduler=FIFO priority="<<settings->priority
             <<" verified=true; background threads retain launch scheduling"<<std::endl;
}
}
