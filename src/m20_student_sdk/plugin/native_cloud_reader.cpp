// AOS native FastDDS ABI, loopback UDP only. No command channel or publisher.
#include <drdds/core/drdds_core.h>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.h>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include "json.hpp"
#include "native_source_identity.hpp"
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
using namespace eprosima::fastdds::dds;
using nlohmann::json;
static int local_driver_pid(){
  int found=0;
  for(const auto& entry:std::filesystem::directory_iterator("/proc")){
    auto name=entry.path().filename().string();if(name.empty()||name.find_first_not_of("0123456789")!=std::string::npos)continue;
    std::ifstream comm(entry.path()/"comm");std::string value;comm>>value;if(value!="rslidar")continue;
    struct stat st{};if(stat(entry.path().c_str(),&st)!=0||st.st_uid!=0)throw std::runtime_error("raw driver must be vendor root process");
    if(std::filesystem::canonical(entry.path()/"exe")!="/opt/robot/share/node_driver/bin/rslidar")throw std::runtime_error("unknown raw driver executable");
    if(found)throw std::runtime_error("multiple rslidar processes");found=std::stoi(name);
  }
  // This installed native ABI encodes the low 16 PID bits in GUID bytes 4..5.
  // Refuse wraparound instead of guessing identities on a different PID layout.
  if(found<=0||found>=65536)throw std::runtime_error("no uniquely identifiable local rslidar process");return found;
}
class LocalCloud:public DataReaderListener {
  DomainParticipant* participant_=nullptr;Subscriber* subscriber_=nullptr;DataReader* reader_=nullptr;Topic* topic_=nullptr;
  TypeSupport type_{new sensor_msgs::msg::PointCloud2PubSubType()};
  int driver_=0;std::atomic<bool> ready_{false};std::mutex mutex_;std::condition_variable changed_;
  std::vector<uint8_t> latest_;uint64_t sequence_=0,rejected_sources_=0;std::string selected_guid_;
  unsigned char host_[4]{};
public:
  LocalCloud(){
    if(geteuid()!=0)throw std::runtime_error("run raw supervisor as root to verify vendor process identity; no SHM permissions are modified");
    driver_=local_driver_pid();DomainParticipantQos qos;qos.name("m20_aos_local_raw_reader");qos.transport().use_builtin_transports=false;
    auto udp=std::make_shared<eprosima::fastdds::rtps::UDPv4TransportDescriptor>();udp->interfaceWhiteList={"127.0.0.1"};udp->sendBufferSize=4*1024*1024;udp->receiveBufferSize=4*1024*1024;qos.transport().user_transports={udp};
    participant_=DomainParticipantFactory::get_instance()->create_participant(0,qos);
    if(!participant_)throw std::runtime_error("cannot create loopback participant");
    const auto& prefix=participant_->guid().guidPrefix.value;std::memcpy(host_,prefix,4);
    if(prefix[4]!=(getpid()&255)||prefix[5]!=((getpid()>>8)&255))throw std::runtime_error("native DDS PID identity encoding changed");
    if(type_.register_type(participant_)!=eprosima::fastrtps::types::ReturnCode_t::RETCODE_OK)throw std::runtime_error("cannot register native PointCloud2");
    topic_=participant_->create_topic("rt/LIDAR/POINTS",type_.get_type_name(),TOPIC_QOS_DEFAULT);
    subscriber_=participant_->create_subscriber(SUBSCRIBER_QOS_DEFAULT);
    DataReaderQos rq;rq.reliability().kind=BEST_EFFORT_RELIABILITY_QOS;rq.history().kind=KEEP_LAST_HISTORY_QOS;rq.history().depth=1;rq.endpoint().history_memory_policy=eprosima::fastrtps::rtps::DYNAMIC_RESERVE_MEMORY_MODE;
    ready_.store(true);reader_=subscriber_->create_datareader(topic_,rq,this);
    if(!reader_)throw std::runtime_error("cannot create local raw subscriber");
  }
  ~LocalCloud(){ready_.store(false);if(reader_)subscriber_->delete_datareader(reader_);if(subscriber_)participant_->delete_subscriber(subscriber_);if(topic_)participant_->delete_topic(topic_);if(participant_)DomainParticipantFactory::get_instance()->delete_participant(participant_);}
  void on_data_available(DataReader* reader)override{
    sensor_msgs::msg::PointCloud2 m;SampleInfo info;
    while(reader->take_next_sample(&m,&info)==eprosima::fastrtps::types::ReturnCode_t::RETCODE_OK){
      if(!ready_.load()||!info.valid_data)continue;const auto& id=info.publication_handle.value;
      bool accepted=local_raw_source_identity(id,host_,driver_);
      std::ostringstream guid;guid<<std::hex<<std::setfill('0');for(int i=0;i<16;++i)guid<<std::setw(2)<<int(id[i]);
      {std::lock_guard<std::mutex> lock(mutex_);if(!accepted||(!selected_guid_.empty()&&selected_guid_!=guid.str())){++rejected_sources_;continue;}if(selected_guid_.empty())selected_guid_=guid.str();}
      receive(m,guid.str());
    }
  }
  void receive(const sensor_msgs::msg::PointCloud2& m,const std::string& guid){
    if(m.data().size()>16000000||static_cast<uint64_t>(m.width())*m.height()>200000||m.fields().size()>16)return;
    json fields=json::array();for(const auto& f:m.fields())fields.push_back({{"name",f.name()},{"offset",f.offset()},{"datatype",f.datatype()},{"count",f.count()}});
    json metadata={{"sec",m.header().stamp().sec()},{"nanosec",m.header().stamp().nanosec()},{"frame",m.header().frame_id()},{"width",m.width()},{"height",m.height()},{"point_step",m.point_step()},{"row_step",m.row_step()},{"bigendian",m.is_bigendian()},{"dense",m.is_dense()},{"bytes",m.data().size()},{"fields",fields},{"source","AOS_local_native_UDP"},{"writer_guid",guid},{"driver_pid",driver_}};
    std::string text=metadata.dump();if(text.size()>4096)return;std::vector<uint8_t> packet(4+text.size()+m.data().size());uint32_t length=text.size();
    for(int i=0;i<4;++i)packet[i]=(length>>(8*i))&255;
    std::memcpy(packet.data()+4,text.data(),text.size());std::memcpy(packet.data()+4+text.size(),m.data().data(),m.data().size());
    {std::lock_guard<std::mutex> lock(mutex_);latest_=std::move(packet);++sequence_;}changed_.notify_one();
  }
  json diagnostics(){std::lock_guard<std::mutex> lock(mutex_);return {{"selected_driver_pid",driver_},{"selected_writer_guid",selected_guid_},{"clouds",sequence_},{"rejected_source_messages",rejected_sources_},{"command_writers_created",0},{"transport","UDP loopback only"}};}
  void serve(int client){
    timeval timeout{0,200000};setsockopt(client,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));uint64_t last=0;
    while(true){std::vector<uint8_t> packet;{std::unique_lock<std::mutex> lock(mutex_);changed_.wait(lock,[&](){return sequence_!=last;});last=sequence_;packet=latest_;}
      size_t sent=0;while(sent<packet.size()){ssize_t n=send(client,packet.data()+sent,packet.size()-sent,MSG_NOSIGNAL);if(n<=0)return;sent+=n;}
    }
  }
};
int main(int argc,char** argv){
  try{
    std::ifstream f("/opt/robot/os_type");std::string board;f>>board;if(board!="CA9B_AOS")throw std::runtime_error("AOS native receiver only");
#ifdef M20_READONLY_PROBE
    LocalCloud source;std::this_thread::sleep_for(std::chrono::seconds(6));std::cout<<"M20_PRESTART_RESULT="<<source.diagnostics().dump()<<std::endl;return 0;
#else
    if(argc!=2)throw std::runtime_error("usage: native_cloud_reader release-local-unix-socket");
    sockaddr_un address{};address.sun_family=AF_UNIX;if(std::strlen(argv[1])>=sizeof(address.sun_path))throw std::runtime_error("socket path too long");std::strcpy(address.sun_path,argv[1]);
    int server=socket(AF_UNIX,SOCK_STREAM,0);if(server<0||bind(server,reinterpret_cast<sockaddr*>(&address),sizeof(address))<0||listen(server,1)<0)throw std::runtime_error("cannot create exclusive local socket");
    LocalCloud source;std::cout<<"AOS selected local rslidar -> private Unix socket; no command publishers"<<std::endl;
    while(true){int client=accept(server,nullptr,nullptr);if(client<0)break;source.serve(client);close(client);}close(server);
#endif
  }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
