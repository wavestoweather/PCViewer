#include "violin_attribute_workbench.hpp"
#include <violin_util.hpp>

namespace workbenches{
void violin_attribute_workbench::_update_attribute_histograms(){
    // check for still active histogram update
    for(const auto& [dl_id, dl]: session_state.drawlists){
        if(_registered_histograms.contains(dl.drawlist_id) && _registered_histograms[dl.drawlist_id].size()){
            auto access = dl.drawlist_read().histogram_registry.const_access();
            if(!access->dataset_update_done)    
                return;
        }
    }

    // blending the histogram values according to the settings ---------------------------------------------
    const auto active_drawlist_attributes = get_active_drawlist_attributes();
    std::tie(_global_max, _per_attribute_max, _drawlist_attribute_histograms) = util::violins::update_histograms(active_drawlist_attributes, settings.read().smoothing_std_dev, settings.read().histogram_bin_count, session_state.attributes.size(), settings.read().ignore_zero_bins, session_state.attribute_log);

    settings.changed = false;
    for(auto& [dl, registrators]: _registered_histograms){
        // locking the registry before signaling the reigstrators
        auto registry_lock = globals::drawlists.read().at(dl).read().histogram_registry.const_access();
        for(auto& registrator: registrators)
            registrator.signal_registry_used();
    }
}

void violin_attribute_workbench::_update_registered_histograms(){
    auto active_drawlist_attributes = get_active_drawlist_attributes();
    auto active_attribute_indices = get_active_indices();
    for(const auto& [dl_id, dl]: session_state.drawlists){

        std::vector<bool> registrator_needed(_registered_histograms[dl.drawlist_id].size(), false);
        for(auto a: active_attribute_indices){
            auto registrator_id = util::histogram_registry::get_id_string(a, settings.read().histogram_bin_count, false, false);
            int registrator_index{-1};
            for(int j: util::size_range(_registered_histograms[dl.drawlist_id])){
                if(_registered_histograms[dl.drawlist_id][j].registry_id == registrator_id){
                    registrator_index = j;
                    break;
                }
            }
            if(registrator_index >= 0)
                registrator_needed[registrator_index] = true;
            else{
                // adding the new histogram
                auto& drawlist = dl.drawlist_write();
                _registered_histograms[dl.drawlist_id].emplace_back(*drawlist.histogram_registry.access(), a, settings.read().histogram_bin_count, false, false, true);
                registrator_needed.push_back(true);
            }
        }
        // removing unused registrators
        // locking registry
        auto registry_lock = dl.drawlist_read().histogram_registry.const_access();
        for(int i: util::rev_size_range(_registered_histograms[dl.drawlist_id])){
            if(!registrator_needed[i])
                _registered_histograms[dl.drawlist_id].erase(_registered_histograms[dl.drawlist_id].begin() + i);
        }
        // printing out the registrators
        if(logger.logging_level >= logging::level::l_5){
            logger << logging::info_prefix << " violin_attribute_workbench (" << active_attribute_indices.size() << " attributes, " << registry_lock->registry.size() <<" registrators, " << registry_lock->name_to_registry_key.size() << " name to registry entries), registered histograms: ";
            for(const auto& [key, val]: registry_lock->registry)
                logger << val.hist_id << " ";
            logger << logging::endl;
        }
    }
}

violin_attribute_workbench::violin_attribute_workbench(std::string_view id): workbench(id){
}

void violin_attribute_workbench::show(){
    if(!active)
        return;

    ImGui::Begin(id.data(), &active);

    // violin plots ---------------------------------------------------------------------------------------

    

    // drawlists and settings -----------------------------------------------------------------------------

    ImGui::End();
}

std::vector<violin_attribute_workbench::drawlist_attribute> violin_attribute_workbench::get_active_drawlist_attributes() const{
    std::vector<drawlist_attribute> ret;
    for(const auto& i: session_state.attribute_order_infos){
        if(!i.active)
            continue;
        for(const auto& [dl_id, dl]: session_state.drawlists){
            if(!dl.appearance->read().show)
                continue;
            ret.emplace_back(drawlist_attribute{dl.drawlist_id, i.attribut_index});
        }
    }
    return ret;
}

std::vector<uint32_t> violin_attribute_workbench::get_active_indices() const{
    std::vector<uint32_t> ret;
    for(const auto& i: session_state.attribute_order_infos)
        if(i.active)
            ret.emplace_back(i.attribut_index);
    return ret;
}
}