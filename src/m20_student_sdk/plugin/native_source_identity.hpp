#pragma once
#include <cstring>
inline bool local_raw_source_identity(const unsigned char* id,const unsigned char* host,int pid){
  return pid>0&&pid<65536&&std::memcmp(id,host,4)==0&&id[4]==(pid&255)&&id[5]==((pid>>8)&255)&&id[8]==0&&id[9]==0&&id[10]==0&&id[11]==0;
}
