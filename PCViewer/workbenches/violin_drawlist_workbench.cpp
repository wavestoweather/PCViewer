#include "violin_drawlist_workbench.hpp"
#include <imgui.h>
#include <violin_util.hpp>
#include <histogram_registry_util.hpp>
#include <stager.hpp>

namespace workbenches{
void violin_drawlist_workbench::_update_attribute_histograms(){
    // check for still active histogram update
    for(const auto& dl: session_state.drawlist_infos){
        if(_registered_histograms.contains(dl.drawlist_id) && _registered_histograms[dl.drawlist_id].size()){
            auto access = dl.drawlist_read().histogram_registry.const_access();
            if(!access->dataset_update_done)    
                return;
        }
    }

    // blending the histogram values according to the settings ---------------------------------------------
    const auto active_drawlist_attributes = get_active_drawlist_attributes();

    // downloadig all histograms
    robin_hood::unordered_map<drawlist_attribute, std::vector<float>> original_histograms;
    for(const auto& da: active_drawlist_attributes){
        uint32_t attribute{static_cast<uint32_t>(da.attribute)};
        const auto& registrator = util::memory_view(_registered_histograms[da.dl]).find([&attribute](const auto& h){return util::histogram_registry::id_contains_attributes(h.registry_id, attribute);});
        const auto histogram_access = da.drawlist_read().histogram_registry.const_access();
        original_histograms[da].resize(settings.read().histogram_bin_count);
        structures::stager::staging_buffer_info buffer_info{};
        buffer_info.transfer_dir = structures::stager::transfer_direction::download;
        buffer_info.data_download = util::memory_view(original_histograms[da]);
        buffer_info.dst_buffer = histogram_access->gpu_buffers.at(registrator.registry_id).buffer;
        globals::stager.add_staging_task(buffer_info);
    }
    globals::stager.wait_for_completion();

    float std_dev = settings.read().smoothing_std_dev;
    double std_dev2 = 2 * std_dev * std_dev;
    int k_size = std_dev < 0 ? .2 * settings.read().histogram_bin_count + 1: std::ceil(std_dev * 3);

    // integrating to 3 sigma std dev
    _per_attribute_max = std::vector(session_state.attributes.size(), .0f);
    _global_max = 0;
    for(const auto& da: active_drawlist_attributes){
        const auto& original_histogram = original_histograms[da];
        auto& histogram = _drawlist_attribute_histograms[da];
        histogram.area = histogram.max_val = 0;
        for(int bin: util::size_range(original_histogram)){
            float divisor{}, divider{};

            for(int k: util::i_range(-k_size, k_size + 1)){
                if (bin + k >= ((settings.read().ignore_zero_bins)? 1:0) && bin + k < original_histogram.size()) {
                    float factor = std::exp(-(k * k) / std_dev2);
                    divisor += original_histogram[bin + k] * factor;
                    divider += factor;
                }
            }

            histogram.smoothed_values[bin] = divisor / divider;
            if (session_state.attribute_log[da.attribute]) histogram.smoothed_values[bin] = log(histogram.smoothed_values[bin] + 1);
            histogram.area += histogram.smoothed_values[bin];
            histogram.max_val = std::max(histogram.max_val, histogram.smoothed_values[bin]);
        }
        _per_attribute_max[da.attribute] = std::max(_per_attribute_max[da.attribute], histogram.max_val);
        _global_max = std::max(_global_max, histogram.max_val);
    }


}

void violin_drawlist_workbench::_update_registered_histograms(){

}

violin_drawlist_workbench::violin_drawlist_workbench(std::string_view id): workbench(id){
}

void violin_drawlist_workbench::show(){
    if(!active)
        return;

    ImGui::Begin(id.data(), &active);

    // violin plots ---------------------------------------------------------------------------------------



    // drawlists and settings -----------------------------------------------------------------------------

    ImGui::End();
}

std::vector<violin_drawlist_workbench::drawlist_attribute> violin_drawlist_workbench::get_active_drawlist_attributes(){
    std::vector<violin_drawlist_workbench::drawlist_attribute> ret;
    for(const auto& i: session_state.attribute_order_infos){
        if(!i.active)
            continue;
        for(const auto& dl: session_state.drawlist_infos){
            if(!dl.appearance->read().show)
                continue;
            ret.emplace_back(violin_drawlist_workbench::drawlist_attribute{dl.drawlist_id, static_cast<int>(i.attribut_index)});
        }
    }
    return ret;
}
}