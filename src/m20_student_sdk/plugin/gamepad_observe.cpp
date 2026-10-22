// Bounded receive-only probe. No Reply, UDP listener, ROS or SDK object.
#include "gamepad_packet_tap.hpp"
#include "gamepad_command_protocol.hpp"
#include "json.hpp"
#include "input_timing.hpp"
#include <chrono>
#include <iostream>
#include <map>
using nlohmann::json;
int main(int argc,char** argv){
 try{
  if(argc!=2)throw std::runtime_error("usage: gamepad_observe interface");
  GamepadPacketTap tap(argv[1]);auto end=std::chrono::steady_clock::now()+std::chrono::seconds(8);
  std::map<std::string,int> heartbeats;std::map<std::string,std::chrono::steady_clock::time_point> last;
  while(std::chrono::steady_clock::now()<end){
   char buffer[32768];sockaddr_ll address{};socklen_t size=sizeof(address);
   alignas(cmsghdr) char control[CMSG_SPACE(sizeof(timespec))]{};iovec io{buffer,sizeof(buffer)};msghdr message{};
   message.msg_name=&address;message.msg_namelen=size;message.msg_iov=&io;message.msg_iovlen=1;message.msg_control=control;message.msg_controllen=sizeof(control);
   auto count=recvmsg(tap.fd(),&message,0);
   if(message.msg_flags&(MSG_TRUNC|MSG_CTRUNC))continue;
   timespec received{},wall{};bool timestamp=false;
   for(auto* c=CMSG_FIRSTHDR(&message);c;c=CMSG_NXTHDR(&message,c))if(c->cmsg_level==SOL_SOCKET&&c->cmsg_type==SCM_TIMESTAMPNS&&c->cmsg_len>=CMSG_LEN(sizeof(received))){std::memcpy(&received,CMSG_DATA(c),sizeof(received));timestamp=true;break;}
   clock_gettime(CLOCK_REALTIME,&wall);double age=double(wall.tv_sec-received.tv_sec)+double(wall.tv_nsec-received.tv_nsec)*1e-9;
   if(!timestamp||age<0||age>std::chrono::duration<double>(m20::kInputPacketLifetime).count())continue;
   sockaddr_in peer{};size_t off=0,len=0;
   if(count<=0||address.sll_pkttype!=PACKET_HOST||!GamepadPacketTap::Decode(buffer,count,tap.local(),peer,off,len))continue;
   m20_gamepad::CommandPacket packet;if(!m20_gamepad::ParseCommandPacket(reinterpret_cast<uint8_t*>(buffer+off),len,packet))continue;
   auto body=json::parse(packet.body,nullptr,false);
   if(body.is_discarded()||!body.contains("PatrolDevice"))continue;const auto& p=body["PatrolDevice"];
   if(!p.is_object()||!p.contains("Items")||!p.contains("Type")||!p.contains("Command")||p["Type"]!=100||p["Command"]!=100)continue;
   auto ip=std::string(inet_ntoa(peer.sin_addr));++heartbeats[ip];last[ip]=std::chrono::steady_clock::now()-std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(age));
  }
  auto now=std::chrono::steady_clock::now();json seen=json::object();bool okay=heartbeats.size()==1;
  for(const auto& item:heartbeats){double age=std::chrono::duration<double>(now-last[item.first]).count();seen[item.first]={{"heartbeats",item.second},{"last_age_s",age}};okay=okay&&item.second>=3&&age<1.;}
  std::cout<<json({{"passed",okay},{"sources",seen},{"commands_sent",0},{"replies_sent",0},{"scope","real passive heartbeat only; not physical button/failsafe validation"}}).dump()<<std::endl;return okay?0:2;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
