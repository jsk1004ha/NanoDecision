#pragma once
#include "engine.hpp"
namespace nd2 {
// Every block is encoded, with local positions reset. This is not a
// full-attention long-context encoder and does not average block probabilities.
inline Context prepare_blocks(const Model& model,const std::vector<std::vector<int>>& blocks){
    require(!blocks.empty()&&blocks.size()<=256,"Need 1..256 context blocks");
    size_t total=0;
    for(const auto& ids:blocks){model.check_ids(ids);total+=ids.size();require(total<=32768,"Combined context exceeds 32768 tokens");}
    Context combined;combined.length=total;
    combined.keys.reserve(total*model.cfg.h);combined.values.reserve(total*model.cfg.h);
    for(const auto& ids:blocks){
        auto part=model.prepare(ids);
        combined.keys.insert(combined.keys.end(),part.keys.begin(),part.keys.end());
        combined.values.insert(combined.values.end(),part.values.begin(),part.values.end());
    }
    return combined;
}
} // namespace nd2
