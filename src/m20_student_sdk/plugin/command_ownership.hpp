#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <vector>
#include <cstring>
#include <string>
// Pure predicate shared by startup and runtime graph checks.
inline bool official_command_identity(const std::string& name,const uint8_t* command,const uint8_t* data){
  bool nonzero=false;for(int i=0;i<12;++i)nonzero|=data[i]!=0;
  return nonzero&&name=="_CREATED_BY_BARE_DDS_APP_"&&std::memcmp(command,data,12)==0;
}
template<size_t N>
inline bool same_endpoint_set(std::vector<std::array<uint8_t,N>> a,std::vector<std::array<uint8_t,N>> b){
  std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());return a==b;
}
