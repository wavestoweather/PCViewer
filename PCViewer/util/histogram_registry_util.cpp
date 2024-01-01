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
            if(dl.read().dataset_registrator && !dl.read().dataset_read().gpu_stream_infos->signal_block_upload_done)
                continue;
            auto registry_access = dl.read().histogram_registry.const_access();   // automatically locks the registry to avoid multi threading problems
            if(registry_access->change_request.size() && dl.read().dataset_registrator && !dl.read().dataset_read().gpu_stream_infos->signal_block_update_request)
                dl.read().dataset_read().gpu_stream_infos->signal_block_update_request = true;
            if(!registry_access->registrators_done && registry_access->dataset_update_done)
                continue;   // changes were not yet applied by the registrators
            if(registry_access->change_request.size()){
                // updating the histograms
                bool clear_counts = true;
                bool last_count_of_dataset = true;
                if(dl.read().dataset_read().gpu_stream_infos){
                    const auto& ds_stream_infos = dl.read().dataset_read().gpu_stream_infos;
                    clear_counts = ds_stream_infos->first_block();
                    last_count_of_dataset = ds_stream_infos->last_block();
                }
                globals::histogram_counter.add_count_task({dl_id, clear_counts, last_count_of_dataset});
            }
        }
    }
}
}
}