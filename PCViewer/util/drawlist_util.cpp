#include <drawlist_util.hpp>
#include <workbench_base.hpp>
#include <priority_sorter.hpp>
#include <stager.hpp>

namespace util{
namespace drawlist{
void download_activation(structures::drawlist& dl){
    auto res = vkDeviceWaitIdle(globals::vk_context.device); util::check_vk_result(res);
    structures::stager::staging_buffer_info staging_info;
    staging_info.data_download = util::memory_view{dl.active_indices_bitset.data(), dl.active_indices_bitset.num_blocks()};
    staging_info.dst_buffer = dl.active_indices_bitset_gpu.buffer;
    globals::stager.add_staging_task(staging_info);
    globals::stager.wait_for_completion();
}

void check_drawlist_deletion(){
    if(globals::drawlists_to_delete.size()){
        // signaling all dependant workbenches
        std::vector<std::string_view> drawlists(globals::drawlists_to_delete.begin(), globals::drawlists_to_delete.end());
        for(auto& workbench: globals::drawlist_dataset_dependencies)
            workbench->remove_drawlists(drawlists);
        
        // deleting drawlists
        bool prev_drawlists_state = globals::drawlists.changed;
        for(auto& dl: globals::drawlists_to_delete){
            globals::drawlists()[dl]().destroy_local_gpu_buffer();
            globals::drawlists().erase(dl);
        }
        globals::drawlists.changed = prev_drawlists_state;

        // removing locally selected drawlist
        if(globals::brush_edit_data.brush_type == structures::brush_edit_data::brush_type::local && util::memory_view(drawlists).contains(globals::brush_edit_data.local_brush_id))
            globals::brush_edit_data.clear();

        // removing the drawlists from the selected drawlists
        globals::selected_drawlists.erase(std::remove_if(globals::selected_drawlists.begin(), globals::selected_drawlists.end(), 
                    [](std::string_view dl){return globals::drawlists_to_delete.count(dl) > 0;}),
                    globals::selected_drawlists.end());
        
        globals::drawlists_to_delete.clear();
    }
}

void check_drawlist_update(){
    if(globals::drawlists.changed){
        std::vector<std::string_view> changed_drawlists;
        for(const auto& [dl_id, dl]: globals::drawlists.read()){
            if(dl.changed)
                changed_drawlists.push_back(dl_id);
        }
        for(auto& workbench: globals::drawlist_dataset_dependencies)
            workbench->signal_drawlist_update(changed_drawlists);
        bool brush_wait{false};
        for(auto id: changed_drawlists){
            if(globals::drawlists.read().at(id).read().local_brushes.changed){
                brush_wait = true;
                continue;
            }
            globals::drawlists.ref_no_track()[id].ref_no_track().clear_change();
            globals::drawlists.ref_no_track()[id].changed = false;
        }
        if(!brush_wait)
            globals::drawlists.changed = false;
    }
}

void check_drawlist_delayed_ops(){
    // checking for priority rendering
    for(const auto& [dl_id, dl]:globals::drawlists.read()){
        if(!dl.changed || !dl.read().delayed_ops.priority_rendering_requested || dl.read().delayed_ops.priority_rendering_sorting_started)
            continue;
        
        if(!dl.read().delayed_ops.priority_rendering_requested || dl.read().delayed_ops.delayed_ops_done)
            continue;
        
        // starting priority render counting, waiting if histogram counting not yet done
        {
            auto access = dl.read().histogram_registry.const_access();
            if(access->registry.size() && !access->dataset_update_done)
                continue;
        }
        auto& drawlist = globals::drawlists.ref_no_track()[dl_id].ref_no_track();
        structures::priority_sorter::sorting_info sort_info{};
        sort_info.dl_id = dl_id;
        sort_info.cpu_signal_flags = {&drawlist.delayed_ops.priority_sorting_done, &drawlist.delayed_ops.delayed_ops_done, &globals::drawlists.ref_no_track()[dl_id].changed, &globals::drawlists.changed};    // order is important, dl update flag has to be last
        sort_info.cpu_unsignal_flags = {&drawlist.delayed_ops.priority_rendering_sorting_started};
        globals::priority_sorter.add_sort_task(sort_info);
        globals::drawlists()[dl_id]().delayed_ops.priority_rendering_sorting_started = true;
    }
}
}
}