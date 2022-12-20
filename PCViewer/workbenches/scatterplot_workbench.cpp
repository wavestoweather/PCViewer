#include "scatterplot_workbench.hpp"
#include <imgui_util.hpp>
#include <vma_initializers.hpp>

namespace workbenches
{
void scatterplot_workbench::_update_registered_histograms(){
    std::array<int, 2> bucket_sizes{int(settings.read().plot_width), int(settings.read().plot_width)};
    auto active_indices = get_active_ordered_indices();
    for(const auto& dl: drawlist_infos.read()){
        if(dl.templatelist_read().data_size < settings.read().large_vis_threshold){
            _registered_histograms.erase(dl.drawlist_id);
            continue;
        }
        
        // setting up the bin sizes
        std::vector<bool> registrator_needed(_registered_histograms[dl.drawlist_id].size(), false);
        std::array<uint32_t, 2> indices;

        // creating the new registratros and flag the used registrators as true -------------------------
        switch(settings.read().plot_type){
        case plot_type_t::list:

            break;
        case plot_type_t::matrix:
            for(int i: util::i_range(active_indices.size() - 1)){
                for(int j: util::i_range(i + 1, active_indices.size())){
                    indices[0] = active_indices[i];
                    indices[1] = active_indices[j];
                    auto registrator_id = util::histogram_registry::get_id_string(indices, bucket_sizes, false, false);
                    size_t registrator_index = util::memory_view(_registered_histograms[dl.drawlist_id]).index_of([&registrator_id](const registered_histogram& h){return registrator_id == h.registry_id;});
                    if(registrator_index != util::memory_view<>::n_pos)
                        registrator_needed[registrator_index] = true;
                    else{
                        // adding new histogram
                        auto& drawlist = dl.drawlist_write();
                        _registered_histograms[dl.drawlist_id].emplace_back(drawlist.histogram_registry.access()->scoped_registrator(indices, bucket_sizes, false, false, false));
                        registrator_needed.push_back(true);
                    }
                }
            }
            break;
        default:
            throw std::runtime_error{"scatterplot_workbench() Unimplemented plot type"};
        }

        // removing unused registrators -----------------------------------------------------------------
        auto registry_lock = dl.drawlist_read().histogram_registry.const_access();
        for(int i: util::rev_size_range(_registered_histograms[dl.drawlist_id])){
            if(!registrator_needed[i])
                _registered_histograms[dl.drawlist_id].erase(_registered_histograms[dl.drawlist_id].begin() + i);
        }
        // printing out the registrators
        if(logger.logging_level >= logging::level::l_5){
            logger << logging::info_prefix << " scatterplot_workbenche (" << active_indices.size() << " attributes, " << registry_lock->registry.size() << " registrators, " << registry_lock->name_to_registry_key.size() << " name to registry entries), registered histograms: ";
            for(const auto& [key, val]: registry_lock->registry)
                logger << val.hist_id << " ";
            logger << logging::endl;
        }
    }
    // setting update singal flags
    for(const auto& dl: drawlist_infos.read())
        dl.drawlist_write().histogram_registry.access()->request_change_all();
}

void scatterplot_workbench::_update_plot_images(){
    constexpr VkImageUsageFlags image_usage{VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};
    auto active_indices = get_active_ordered_indices();
    // waiting for the device to finish all command buffers using the image before statring to destroy/create plot images
    auto res = vkDeviceWaitIdle(globals::vk_context.device); util::check_vk_result(res);
    
    robin_hood::unordered_set<attribute_pair> used_attribute_pairs;
    std::vector<VkImageMemoryBarrier> image_barriers;
    // creating new pairs / recreating images if size changed
    auto destroy_image = [&](const attribute_pair& p) {
        auto& plot_data = plot_datas[p];
        util::vk::destroy_image(plot_data.image);
        util::vk::destroy_image_view(plot_data.image_view);
        util::imgui::free_image_descriptor_set(plot_data.image_descriptor);
    };
    switch(settings.read().plot_type){
    case plot_type_t::matrix:
        for(uint32_t i: util::i_range(active_indices.size() - 1)){
            for(uint32_t j: util::i_range(i, active_indices.size())){
                // destruction of old image if necessary
                if(plot_datas.contains({i, j}) && (plot_datas[{i, j}].image_width != settings.read().plot_width || plot_datas[{i,j}].image_format != settings.read().plot_format))
                    destroy_image({i, j});
                // creating new image if needed
                if(!plot_datas[{i, j}].image){
                    auto& plot_data = plot_datas[{i, j}];
                    auto image_info = util::vk::initializers::imageCreateInfo(settings.read().plot_format, {settings.read().plot_width, settings.read().plot_width, 1}, image_usage);
                    auto mem_alloc = util::vma::initializers::allocationCreateInfo();
                    std::tie(plot_data.image, plot_data.image_view) = util::vk::create_image_with_view(image_info, mem_alloc);
                    plot_data.image_descriptor = util::imgui::create_image_descriptor_set(plot_data.image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                    plot_data.image_format = settings.read().plot_format;

                    // updating the image layout
                    image_barriers.emplace_back(util::vk::initializers::imageMemoryBarrier(plot_data.image.image, VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, {}, {}, {}, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
                }
                used_attribute_pairs.insert({i, j});
            }
        }
        break;
    case plot_type_t::list:

        break;
    }
    util::vk::convert_image_layouts_execute(image_barriers);

    // removing all unused images
    robin_hood::unordered_set<attribute_pair> unused_attribute_pairs;
    for(const auto& [pair, image_data]: plot_datas) if(!used_attribute_pairs.contains(pair)) unused_attribute_pairs.insert(pair);
    for(const auto& pair: unused_attribute_pairs){
        destroy_image(pair);
        plot_datas.erase(pair);
    }
}

scatterplot_workbench::scatterplot_workbench(std::string_view id):
    workbench(id)
{

}

void scatterplot_workbench::notify_drawlist_dataset_update() 
{
    
}
    
void scatterplot_workbench::show() 
{
    if(!active) 
        return;
    ImGui::Begin(id.data(), &active);

    const auto active_indices = get_active_ordered_indices();
    // plot views ------------------------------------------------------
    switch(settings.read().plot_type){
    case plot_type_t::matrix:
        // matrix should be displayed as a left lower triangular matrix
        for(int i: util::i_range(1, active_indices.size())){
            for(int j: util::i_range(i)){

            }
        }
        break;
    case plot_type_t::list:

        break;
    }

    // settings ---------------------------------------------------------
    ImGui::InputDouble("plot padding", &settings().plot_padding);
    
    ImGui::End();
}

std::vector<uint32_t> scatterplot_workbench::get_active_ordered_indices(){
    std::vector<uint32_t> indices;
    for(const auto& i: attribute_order_infos.read()){
        if(i.active)
            indices.push_back(i.attribut_index);
    }
    return indices;
}
}