#pragma once
#include <workbench_base.hpp>
#include <memory_view.hpp>
#include <imgui.h>
#include <parallel_coordinates_renderer.hpp>

namespace workbenches{

class parallel_coordinates_workbench: public structures::workbench, public structures::drawlist_dependency{
    using appearance_tracker = structures::change_tracker<structures::drawlist::appearance>;
    using drawlist_info = pipelines::parallel_coordinates_renderer::drawlist_info;

    // both are unique_ptrs to avoid issues with the memory_views when data elements are deleted in the vector
    std::vector<std::unique_ptr<appearance_tracker>>         _storage_appearance;
    std::vector<std::unique_ptr<structures::median_type>>    _storage_median_type;
public:
    const std::array<VkFormat, 4>               available_formats{VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT};

    std::vector<drawlist_info>                  drawlist_infos{};     // the order here is the render order of the drawlists
    structures::alpha_mapping_type              alpha_mapping_typ{};
    structures::parallel_coordinates_renderer::render_type render_type{};
    uint32_t                                    plot_width{1024};
    uint32_t                                    plot_height{480};
    structures::image_info                      plot_image{};
    VkImageView                                 plot_image_view{};
    VkSampleCountFlagBits                       plot_image_samples{VK_SAMPLE_COUNT_1_BIT};
    VkFormat                                    plot_image_format{VK_FORMAT_R16G16B16A16_UNORM};
    util::memory_view<structures::attribute>    attributes;

    const std::string                           id;

    parallel_coordinates_workbench(std::string_view id): id(id){}

    void show() override;

    void addDrawlist(std::string_view drawlistId) override{};
    void signalDrawlistUpdate(const std::vector<std::string_view>& drawlistIds) override{};
    void removeDrawlist(std::string_view drawlistId) override{};
};

}