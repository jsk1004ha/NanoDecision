#include "engine.hpp"
#include <sstream>
#include <iomanip>
// Persistent token-ID protocol: one line = nctx IDs... ncandidates (n IDs...)...
int main(int argc,char** argv){
    try{if(argc<2||argc>3){std::cerr<<"Usage: nd2-native MODEL.nd2 [temperature] < requests.tokens\n";return 2;}
        nd2::Model model(argv[1]);float temperature=argc==3?std::stof(argv[2]):1.0f;
        nd2::require(std::isfinite(temperature)&&temperature>0,"Invalid temperature");
        std::string line;while(std::getline(std::cin,line)){
            try{std::istringstream in(line);int n=0,id=0,c=0;in>>n;
                nd2::require(in&&n>0&&n<=int(model.cfg.max_length),"Bad context length");std::vector<int> state;
                for(int i=0;i<n;++i){in>>id;nd2::require(bool(in),"Missing context token");state.push_back(id);}
                in>>c;nd2::require(in&&c>=2&&c<=256,"Bad candidate count");std::vector<std::vector<int>> opts;
                for(int j=0;j<c;++j){in>>n;nd2::require(in&&n>0&&n<=int(model.cfg.max_length),"Bad candidate length");
                    std::vector<int> row;for(int i=0;i<n;++i){in>>id;nd2::require(bool(in),"Missing candidate token");row.push_back(id);}opts.push_back(row);}
                std::string extra;nd2::require(!(in>>extra),"Trailing input");
                auto ctx=model.prepare(state);auto logits=model.score(ctx,opts);auto scaled=logits;for(float& v:scaled)v/=temperature;
                auto p=nd2::softmax(scaled);std::cout<<std::setprecision(9)<<"{\"logits\":[";
                for(size_t i=0;i<logits.size();++i){if(i)std::cout<<',';std::cout<<logits[i];}std::cout<<"],\"probabilities\":[";
                for(size_t i=0;i<p.size();++i){if(i)std::cout<<',';std::cout<<p[i];}std::cout<<"]}\n"<<std::flush;
            }catch(const std::exception& e){std::cerr<<e.what()<<'\n';std::cout<<"{\"error\":\"invalid request\"}\n"<<std::flush;}
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;
}
