#include <histogram_registry_util.hpp>
#include <drawlists.hpp>
#include <histogram_counter_executor.hpp>

namespace util{
namespace histogram_registry{
void check_histogram_update(){
    if(globals::drawlists.changed){
        for(const auto& [dl_id, dl]: globals::drawlists.read()){
            if(!dl.changed)
                continue;
            auto registry_access = dl.read().histogram_registry.const_access();   // automatically locks the registry to avoid multi threading problems
            if(!registry_access->registrators_done && !(dl.read().dataset_read().gpu_stream_infos && !dl.read().dataset_read().gpu_stream_infos->last_block))
                continue;   // changes were not yet applied by the registrators
            if(registry_access->change_request.size()){
                // updating the histograms
                globals::histogram_counter.add_count_task({dl_id, true});
            }
        }
    }
}
}
}