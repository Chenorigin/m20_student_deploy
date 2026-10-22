// GOS factory native FastDDS input, loopback UDP only. No DDS writer or command channel.
#include <drdds/core/drdds_core.h>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.h>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include "json.hpp"
#include "native_source_identity.hpp"
#include <cmath>
#include <algorithm>
#include <time.h>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <poll.h>
using namespace eprosima::fastdds::dds;
using nlohmann::json;
static int local_driver_pid(){
  int found=0;
  for(const auto& entry:std::filesystem::directory_iterator("/proc")){
    auto name=entry.path().filename().string();if(name.empty()||name.find_first_not_of("0123456789")!=std::string::npos)continue;
    std::ifstream comm(entry.path()/"comm");std::string value;comm>>value;if(value!="rslidar_network")continue;
    struct stat st{};if(stat(entry.path().c_str(),&st)!=0||st.st_uid!=0)throw std::runtime_error("raw driver must be vendor root process");
    if(std::filesystem::canonical(entry.path()/"exe")!="/opt/robot/share/node_driver/bin/rslidar_network")throw std::runtime_error("unknown raw driver executable");
    if(found)throw std::runtime_error("multiple rslidar_network processes");found=std::stoi(name);
  }
  // This installed native ABI encodes the low 16 PID bits in GUID bytes 4..5.
  // Refuse wraparound instead of guessing identities on a different PID layout.
  if(found<=0||found>=65536)throw std::runtime_error("no uniquely identifiable local rslidar process");return found;
}
static std::string driver_snapshot(int pid){
  const auto root=std::filesystem::path("/proc")/std::to_string(pid);
  struct stat owner{},image{};
  if(stat(root.c_str(),&owner)!=0||owner.st_uid!=0||stat((root/"exe").c_str(),&image)!=0||
      std::filesystem::canonical(root/"exe")!="/opt/robot/share/node_driver/bin/rslidar_network")
    throw std::runtime_error("factory GOS driver identity changed");
  std::ifstream stream(root/"stat");std::string line;std::getline(stream,line);
  auto close=line.rfind(')');if(close==std::string::npos)throw std::runtime_error("invalid driver process stat");
  std::istringstream rest(line.substr(close+2));std::string field,start;
  for(int i=0;i<=19;++i){if(!(rest>>field))throw std::runtime_error("incomplete driver process stat");if(i==19)start=field;}
  return start+":"+std::to_string(image.st_dev)+":"+std::to_string(image.st_ino);
}
class LocalCloud:public DataReaderListener {
  DomainParticipant* participant_=nullptr;Subscriber* subscriber_=nullptr;DataReader* reader_=nullptr;Topic* topic_=nullptr;
  TypeSupport type_{new sensor_msgs::msg::PointCloud2PubSubType()};
  int driver_=0;std::atomic<bool> ready_{false},fault_{false};std::mutex mutex_,packet_mutex_,health_mutex_;std::condition_variable changed_;
  std::string driver_snapshot_,fault_reason_;std::chrono::steady_clock::time_point next_check_{};
  std::vector<uint8_t> latest_;uint64_t sequence_=0,rejected_sources_=0;std::string native_session_,selected_guid_;json samples_=json::array(),observed_sources_=json::object();
  unsigned char host_[4]{};
public:
  LocalCloud(){
    if(geteuid()!=0)throw std::runtime_error("run raw supervisor as root to verify vendor process identity; no SHM permissions are modified");
    driver_=local_driver_pid();driver_snapshot_=driver_snapshot(driver_);DomainParticipantQos qos;qos.name("m20_gos_factory_native_reader");qos.transport().use_builtin_transports=false;
    auto udp=std::make_shared<eprosima::fastdds::rtps::UDPv4TransportDescriptor>();udp->interfaceWhiteList={"127.0.0.1"};udp->sendBufferSize=4*1024*1024;udp->receiveBufferSize=4*1024*1024;qos.transport().user_transports={udp};
    participant_=DomainParticipantFactory::get_instance()->create_participant(0,qos);
    if(!participant_)throw std::runtime_error("cannot create loopback participant");
    const auto& prefix=participant_->guid().guidPrefix.value;std::memcpy(host_,prefix,4);
    std::ostringstream instance;instance<<std::hex<<std::setfill('0');for(int i=0;i<12;++i)instance<<std::setw(2)<<int(prefix[i]);instance<<"000001c1";native_session_=instance.str();
    if(prefix[4]!=(getpid()&255)||prefix[5]!=((getpid()>>8)&255))throw std::runtime_error("native DDS PID identity encoding changed");
    if(type_.register_type(participant_)!=eprosima::fastrtps::types::ReturnCode_t::RETCODE_OK)throw std::runtime_error("cannot register native PointCloud2");
    topic_=participant_->create_topic("rt/LIDAR/POINTS",type_.get_type_name(),TOPIC_QOS_DEFAULT);
    subscriber_=participant_->create_subscriber(SUBSCRIBER_QOS_DEFAULT);
    if(!topic_||!subscriber_)throw std::runtime_error("cannot create native input topic/subscriber");
    DataReaderQos rq;rq.reliability().kind=BEST_EFFORT_RELIABILITY_QOS;rq.history().kind=KEEP_LAST_HISTORY_QOS;rq.history().depth=1;rq.endpoint().history_memory_policy=eprosima::fastrtps::rtps::DYNAMIC_RESERVE_MEMORY_MODE;
    rq.resource_limits().max_samples=2;rq.resource_limits().allocated_samples=1;rq.resource_limits().max_instances=1;rq.resource_limits().max_samples_per_instance=2;
    ready_.store(true);reader_=subscriber_->create_datareader(topic_,rq,this);
    if(!reader_)throw std::runtime_error("cannot create local raw subscriber");
  }
  ~LocalCloud(){ready_.store(false);if(reader_)subscriber_->delete_datareader(reader_);if(subscriber_)participant_->delete_subscriber(subscriber_);if(topic_)participant_->delete_topic(topic_);if(participant_)DomainParticipantFactory::get_instance()->delete_participant(participant_);}
  void fail(const std::string& reason){
    {std::lock_guard<std::mutex> lock(mutex_);if(fault_.load())return;fault_reason_=reason;fault_.store(true);ready_.store(false);latest_.clear();}changed_.notify_all();
  }
  bool health(){
    std::lock_guard<std::mutex> lock(health_mutex_);auto now=std::chrono::steady_clock::now();
    if(now>=next_check_&&!fault_.load()){
      next_check_=now+std::chrono::seconds(1);
      try{if(local_driver_pid()!=driver_||driver_snapshot(driver_)!=driver_snapshot_)fail("factory driver restarted/changed");}
      catch(const std::exception& e){fail(e.what());}
    }
    return !fault_.load();
  }
  void on_data_available(DataReader* reader)override{
    sensor_msgs::msg::PointCloud2 m;SampleInfo info;
    while(reader->take_next_sample(&m,&info)==eprosima::fastrtps::types::ReturnCode_t::RETCODE_OK){
      if(!ready_.load()||!info.valid_data||!health())continue;const auto& id=info.publication_handle.value;
      bool accepted=gos_factory_source_identity(id,host_,driver_);
      std::ostringstream guid;guid<<std::hex<<std::setfill('0');for(int i=0;i<16;++i)guid<<std::setw(2)<<int(id[i]);
#ifdef M20_READONLY_PROBE
      {std::lock_guard<std::mutex> lock(mutex_);observed_sources_[guid.str()]={{"local_host_prefix_matches",std::memcmp(id,host_,4)==0},{"pid",int(id[4])+(int(id[5])<<8)},{"participant_counter",uint32_t(id[8])+(uint32_t(id[9])<<8)+(uint32_t(id[10])<<16)+(uint32_t(id[11])<<24)},{"source_identity_accepted",accepted}};}
#endif
      bool conflict=false;
      {std::lock_guard<std::mutex> lock(mutex_);if(!accepted){++rejected_sources_;continue;}conflict=!selected_guid_.empty()&&selected_guid_!=guid.str();if(selected_guid_.empty())selected_guid_=guid.str();}
      if(conflict){fail("ambiguous factory raw DDS writer; restart after inspection");continue;}
      receive(m,guid.str());
    }
  }
  void receive(const sensor_msgs::msg::PointCloud2& m,const std::string& guid){
    std::lock_guard<std::mutex> packet_lock(packet_mutex_);
    if(fault_.load())return;
    if(m.data().empty()||m.data().size()>16000000||static_cast<uint64_t>(m.width())*m.height()>200000||m.fields().size()>16)return;
#ifdef M20_READONLY_PROBE
    timespec clock{};clock_gettime(CLOCK_REALTIME,&clock);double now=clock.tv_sec+clock.tv_nsec*1e-9;
    double header=m.header().stamp().sec()+m.header().stamp().nanosec()*1e-9,first=INFINITY,last=-INFINITY;
    bool has_time=false;uint32_t offset=0;
    for(const auto& f:m.fields())if(f.name()=="timestamp"&&f.datatype()==8&&f.count()==1&&f.offset()<=m.point_step()&&m.point_step()-f.offset()>=8){has_time=true;offset=f.offset();}
    if(has_time&&!m.is_bigendian()&&m.point_step()>0&&m.row_step()>=uint64_t(m.width())*m.point_step()&&uint64_t(m.row_step())*m.height()==m.data().size()){
      for(uint32_t row=0;row<m.height();++row)for(uint32_t col=0;col<m.width();++col){double t;std::memcpy(&t,m.data().data()+uint64_t(row)*m.row_step()+uint64_t(col)*m.point_step()+offset,8);if(std::isfinite(t)){first=std::min(first,t);last=std::max(last,t);}}
    }
    std::lock_guard<std::mutex> lock(mutex_);++sequence_;
    if(samples_.size()<160)samples_.push_back({{"header",header},{"received_at",now},{"end",std::isfinite(last)?json(last):json(nullptr)},{"begin",std::isfinite(first)?json(first):json(nullptr)},{"header_age_s",now-header},{"newest_point_age_s",std::isfinite(last)?json(now-last):json(nullptr)},{"bytes",m.data().size()},{"points",uint64_t(m.width())*m.height()},{"frame",m.header().frame_id()},{"writer_guid",guid}});
    return;
#endif
    json fields=json::array();for(const auto& f:m.fields())fields.push_back({{"name",f.name()},{"offset",f.offset()},{"datatype",f.datatype()},{"count",f.count()}});
    uint64_t sequence;{std::lock_guard<std::mutex> lock(mutex_);sequence=sequence_+1;}
    json metadata={{"native_session",native_session_},{"native_sequence",sequence},{"sec",m.header().stamp().sec()},{"nanosec",m.header().stamp().nanosec()},{"frame",m.header().frame_id()},{"width",m.width()},{"height",m.height()},{"point_step",m.point_step()},{"row_step",m.row_step()},{"bigendian",m.is_bigendian()},{"dense",m.is_dense()},{"bytes",m.data().size()},{"fields",fields},{"source","GOS_factory_native_UDP"},{"writer_guid",guid},{"driver_pid",driver_}};
    std::string text=metadata.dump();if(text.size()>4096)return;std::vector<uint8_t> packet(4+text.size()+m.data().size());uint32_t length=text.size();
    for(int i=0;i<4;++i)packet[i]=(length>>(8*i))&255;
    std::memcpy(packet.data()+4,text.data(),text.size());std::memcpy(packet.data()+4+text.size(),m.data().data(),m.data().size());
    {std::lock_guard<std::mutex> lock(mutex_);if(fault_.load())return;latest_=std::move(packet);sequence_=sequence;}changed_.notify_one();
  }
  json diagnostics(){std::lock_guard<std::mutex> lock(mutex_);return {{"selected_driver_pid",driver_},{"selected_writer_guid",selected_guid_},{"clouds",sequence_},{"rejected_source_messages",rejected_sources_},{"source_fault_latched",fault_.load()},{"fault_reason",fault_reason_},{"command_writers_created",0},{"transport","native FastDDS UDP loopback, factory GOS rslidar_network"},{"samples",samples_},{"observed_sources",observed_sources_}};}
  void serve(int client){
    timeval timeout{0,200000};setsockopt(client,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));uint64_t last;{std::lock_guard<std::mutex> lock(mutex_);last=sequence_;}
    while(true){
      if(!health())throw std::runtime_error("native source fault: "+fault_reason_);
      pollfd fd{client,POLLIN,0};if(poll(&fd,1,0)!=0)return;
      std::vector<uint8_t> packet;{std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait_for(lock,std::chrono::milliseconds(100),[&](){return fault_.load()||sequence_!=last;});
        if(fault_.load())throw std::runtime_error("native source fault: "+fault_reason_);
        if(sequence_==last)continue;last=sequence_;packet=latest_;}
      size_t sent=0;while(sent<packet.size()){ssize_t n=send(client,packet.data()+sent,packet.size()-sent,MSG_NOSIGNAL);if(n<=0)return;sent+=n;}
    }
  }
};
int main(int argc,char** argv){
  try{
    std::ifstream f("/opt/robot/os_type");std::string board;f>>board;if(board!="CA9B_GOS")throw std::runtime_error("GOS factory native receiver only");
#ifdef M20_READONLY_PROBE
    LocalCloud source;std::this_thread::sleep_for(std::chrono::seconds(10));std::cout<<"M20_PRESTART_RESULT="<<source.diagnostics().dump()<<std::endl;return 0;
#else
    if(argc!=2)throw std::runtime_error("usage: native_cloud_reader release-local-unix-socket");
    sockaddr_un address{};address.sun_family=AF_UNIX;if(std::strlen(argv[1])>=sizeof(address.sun_path))throw std::runtime_error("socket path too long");std::strcpy(address.sun_path,argv[1]);
    struct stat parent{};auto directory=std::filesystem::path(argv[1]).parent_path();
    if(stat(directory.c_str(),&parent)!=0||!S_ISDIR(parent.st_mode)||parent.st_uid!=0||(parent.st_mode&077)!=0)
      throw std::runtime_error("native socket requires a private root-owned directory");
    int server=socket(AF_UNIX,SOCK_STREAM,0);if(server<0||bind(server,reinterpret_cast<sockaddr*>(&address),sizeof(address))<0||chmod(argv[1],0600)<0||listen(server,1)<0)throw std::runtime_error("cannot create exclusive local socket");
    LocalCloud source;std::cout<<"GOS selected factory rslidar_network -> private Unix socket; no command publishers"<<std::endl;
    while(true){int client=accept(server,nullptr,nullptr);if(client<0)break;
      ucred peer{};socklen_t size=sizeof(peer);
      if(getsockopt(client,SOL_SOCKET,SO_PEERCRED,&peer,&size)!=0||size!=sizeof(peer)||peer.uid!=0){close(client);continue;}
      source.serve(client);close(client);}close(server);
#endif
  }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
