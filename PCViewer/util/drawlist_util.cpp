#include <drawlist_util.hpp>
#include <workbench_base.hpp>

namespace util{
namespace drawlist{
void check_drawlist_deletion(){
    if(globals::drawlists_to_delete.size()){
        // signaling all dependant workbenches
        std::vector<std::string_view> drawlists(globals::drawlists_to_delete.begin(), globals::drawlists_to_delete.end());
        for(auto& workbench: globals::drawlist_dataset_dependencies)
            workbench->remove_drawlists(drawlists);
        
        // deleting drawlists
        bool prev_drawlists_state = globals::drawlists.changed;
        for(auto& dl: globals::drawlists_to_delete)
            globals::drawlists().erase(dl);
        globals::drawlists.changed = prev_drawlists_state;

        // removing locally selected drawlist
        if(globals::brush_edit_data.brush_type == structures::brush_edit_data::brush_type::local && util::memory_view(drawlists).contains(globals::brush_edit_data.local_brush_id))
            globals::brush_edit_data.clear();
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
        for(auto id: changed_drawlists){
            globals::drawlists.ref_no_track()[id].ref_no_track().clear_change();
            globals::drawlists.ref_no_track()[id].changed = false;
        }
        globals::drawlists.changed = false;
    }
}
}
}