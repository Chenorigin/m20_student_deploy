#include "cts_worker_scheduling.hpp"
int main(int argc,char** argv){
    try{
        if(argc!=2)throw std::runtime_error("choose validate or apply");
        const auto settings=m20_cts::ReadWorkerScheduling();
        if(std::string(argv[1])=="apply")m20_cts::ConfigureWorkerScheduling();
        else if(std::string(argv[1])!="validate")throw std::runtime_error("unknown mode");
        std::cout<<(settings?"configured":"unconfigured")<<std::endl;return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}
}
