#pragma once
#include "json.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <string>
#include <stdexcept>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

// One recorder for inputs, inference and the sole SDK command writer.
// Push never waits for the writer or touches the filesystem.
class StudentTrace {
  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<nlohmann::json> queue_;
  std::thread thread_;
  int fd_=-1;
  bool stopping_=false;
  std::atomic<uint64_t> attempted_{0},enqueued_{0},written_{0},dropped_{0},bytes_{0};
  std::atomic<bool> disabled_{false};
  std::atomic<bool> writer_stopped_{false};
  bool partial_tail_=false;
  uint64_t max_bytes_=512ULL*1024*1024,min_free_=1024ULL*1024*1024;
  std::string path_,run_id_,error_;
  nlohmann::json identity_;
  static uint64_t setting(const char* key,uint64_t fallback) {
    const char* s=std::getenv(key);if(!s)return fallback;
    std::string value(s);size_t n=0;auto out=std::stoull(value,&n);
    if(n!=value.size()||!out||value[0]=='-')throw std::runtime_error(std::string("invalid ")+key);
    return out;
  }
  static std::string read_id(const char* path) {
    std::ifstream f(path);std::string value;std::getline(f,value);return value;
  }
  void health(bool final) noexcept {
    try {
      nlohmann::json row={{"schema","m20_trace_health_v2"},{"run_id",run_id_},
        {"attempted",attempted_.load()},{"enqueued",enqueued_.load()},
        {"written",written_.load()},{"dropped",dropped_.load()},
        {"bytes",bytes_.load()},{"max_bytes",max_bytes_},{"disabled",disabled_.load()},
        {"error",error_},{"final",final},{"writer_stopped",writer_stopped_.load()},
        {"partial_tail",partial_tail_},{"identity",identity_},{"at",Stamp()}};
      std::ofstream f(path_+".health.tmp",std::ios::trunc);f<<row.dump()<<'\n';f.close();
      if(f)std::filesystem::rename(path_+".health.tmp",path_+".health.json");
    }catch(...){}
  }
  void run() noexcept {
    bool in_flight=false;
    try {
      auto last_health=std::chrono::steady_clock::now();
      for(;;){
        nlohmann::json row;
        {std::unique_lock<std::mutex> lock(mutex_);
         wake_.wait_for(lock,std::chrono::seconds(1),[this]{return stopping_||!queue_.empty();});
         if(queue_.empty()) {if(stopping_)break;lock.unlock();health(false);continue;}
         row=std::move(queue_.front());queue_.pop_front();in_flight=true;}
        if(disabled_){++dropped_;in_flight=false;continue;}
        auto line=row.dump();line+='\n';
        if(bytes_.load()+line.size()>max_bytes_){error_="file_cap_reached";disabled_=true;++dropped_;in_flight=false;health(false);continue;}
        size_t offset=0;
        while(offset<line.size()){
          const auto n=::write(fd_,line.data()+offset,line.size()-offset);
          if(n<0&&errno==EINTR)continue;
          if(n<=0){partial_tail_=offset!=0;throw std::runtime_error("trace_write_failed");}
          offset+=static_cast<size_t>(n);bytes_+=static_cast<size_t>(n);
        }
        ++written_;in_flight=false;
        const auto now=std::chrono::steady_clock::now();
        if(now-last_health>=std::chrono::seconds(1)){
          if(std::filesystem::space(std::filesystem::path(path_).parent_path()).available<min_free_){error_="disk_reserve_reached";disabled_=true;}
          health(false);last_health=now;
        }
      }
    }catch(const std::exception& e){error_=e.what();disabled_=true;if(in_flight)++dropped_;}
     catch(...){error_="trace_writer_exception";disabled_=true;if(in_flight)++dropped_;}
    {std::lock_guard<std::mutex> lock(mutex_);dropped_+=queue_.size();queue_.clear();}
    if(fd_>=0){if(::fdatasync(fd_)!=0){error_="trace_sync_failed";disabled_=true;}::close(fd_);fd_=-1;}
    writer_stopped_=true;health(false);
  }
public:
  static nlohmann::json Stamp() {
    using namespace std::chrono;
    const auto mono=duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    const auto wall=duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    return {{"actual_monotonic_ns",mono},{"actual_wall_ns",wall}};
  }
  static nlohmann::json& Context(){static thread_local nlohmann::json c=nlohmann::json::object();return c;}
  static void SetContext(nlohmann::json value){Context()=std::move(value);}
  static StudentTrace& Instance(){static StudentTrace value;return value;}
  void RecordConstructionFailure() noexcept {++attempted_;++dropped_;}
  StudentTrace(){
    const char* path=std::getenv("M20_CONTROL_TRACE_PATH");if(!path||!*path)return;
    path_=std::filesystem::absolute(path).string();
    max_bytes_=setting("M20_TRACE_MAX_BYTES",max_bytes_);min_free_=setting("M20_TRACE_MIN_FREE_BYTES",min_free_);
    if(std::filesystem::space(std::filesystem::path(path_).parent_path()).available<min_free_+max_bytes_)
      throw std::runtime_error("insufficient disk budget for control trace");
    fd_=::open(path_.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
    if(fd_<0)throw std::runtime_error("cannot exclusively create control trace");
    auto stamp=Stamp();run_id_=read_id("/proc/sys/kernel/random/boot_id")+":"+std::to_string(stamp["actual_wall_ns"].get<int64_t>());
    char host[256]={};::gethostname(host,sizeof(host)-1);
    identity_={{"hostname",host},{"machine_id",read_id("/etc/machine-id")},
      {"boot_id",read_id("/proc/sys/kernel/random/boot_id")}};
    thread_=std::thread([this]{run();});
    Push({{"event","trace_start"},{"identity",identity_},{"clock_semantics",
      "wall=Unix UTC ns; monotonic is host/boot local; theory is explicitly labelled nominal, never terrain header or motor execution time"}});
  }
  StudentTrace(const StudentTrace&)=delete;
  StudentTrace& operator=(const StudentTrace&)=delete;
  ~StudentTrace(){
    {std::lock_guard<std::mutex> lock(mutex_);stopping_=true;}wake_.notify_one();
    if(thread_.joinable())thread_.join();
    if(!path_.empty())health(true);
  }
  void Push(nlohmann::json row) noexcept {
    const auto seq=attempted_.fetch_add(1);
    try {
      if(!thread_.joinable()||disabled_){++dropped_;return;}
      for(auto it=Context().begin();it!=Context().end();++it)if(!row.contains(it.key()))row[it.key()]=it.value();
      const auto stamp=Stamp();for(auto it=stamp.begin();it!=stamp.end();++it)if(!row.contains(it.key()))row[it.key()]=it.value();
      if(!row.contains("theoretical_monotonic_ns"))row["theoretical_monotonic_ns"]=nullptr;
      if(!row.contains("theoretical_wall_ns"))row["theoretical_wall_ns"]=nullptr;
      row["schema"]="m20_control_trace_v2";row["run_id"]=run_id_;row["event_seq"]=seq;
      row["trace_dropped"]=dropped_.load();
      std::unique_lock<std::mutex> lock(mutex_,std::try_to_lock);
      if(!lock.owns_lock()||queue_.size()>=512||stopping_){++dropped_;return;}
      queue_.push_back(std::move(row));++enqueued_;wake_.notify_one();
    }catch(...){++dropped_;}
  }
};
