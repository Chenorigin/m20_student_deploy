#include "native_source_identity.hpp"
#include "command_ownership.hpp"
#include <cassert>
#include <iostream>
int main(){
 unsigned char host[4]={1,15,26,235};unsigned char id[24]={1,15,26,235,0xbb,0x0d,0x91,0x10,0,0,0,0,0,0,1,3};
 assert(local_raw_source_identity(id,host,3515));assert(!local_raw_source_identity(id,host,3527));assert(!local_raw_source_identity(id,host,0));assert(!local_raw_source_identity(id,host,65536));
 id[8]=1;assert(!local_raw_source_identity(id,host,3515));id[8]=0;id[2]^=1;assert(!local_raw_source_identity(id,host,3515));id[2]^=1;
 auto official=std::string("_CREATED_BY_BARE_DDS_APP_");unsigned char data[24];std::memcpy(data,id,24);
 assert(official_command_identity(official,id,data));assert(!official_command_identity("named_node",id,data));id[0]^=1;assert(!official_command_identity(official,id,data));
 unsigned char zero[24]{};assert(!official_command_identity(official,zero,zero));
 std::array<uint8_t,24> first{},second{};first[0]=1;second[0]=2;
 assert(same_endpoint_set<24>({first,second},{second,first}));assert(!same_endpoint_set<24>({first,second},{first}));second[12]=1;assert(!same_endpoint_set<24>({first,second},{first,first}));
 std::cout<<"native_source_and_official_owner_identity=PASS no_ROS_no_sockets_no_commands=true\n";
}
