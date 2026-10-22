#pragma once
#include <cstring>
// This GOS factory UDP participant is counter 1, measured from publication
// handles. It is not the AOS counter-0 contract. A changed ABI fails closed.
inline bool gos_factory_source_identity(const unsigned char* id,const unsigned char* host,int pid){
  return pid>0&&pid<65536&&std::memcmp(id,host,4)==0&&id[4]==(pid&255)&&id[5]==((pid>>8)&255)&&id[8]==1&&id[9]==0&&id[10]==0&&id[11]==0;
}
