#pragma once
#include <workbench_base.hpp>
#include <memory_view.hpp>
#include <imgui.h>
#include <parallel_coordinates_renderer.hpp>
#include <enum_names.hpp>

namespace workbenches{

class parallel_coordinates_workbench: public structures::workbench, public structures::drawlist_dataset_dependency{
    using appearance_tracker = structures::change_tracker<structures::drawlist::appearance>;
    using drawlist_info = pipelines::parallel_coordinates_renderer::drawlist_info;

    // both are unique_ptrs to avoid issues with the memory_views when data elements are deleted in the vector
    std::vector<std::unique_ptr<appearance_tracker>>         _storage_appearance;
    std::vector<std::unique_ptr<structures::median_type>>    _storage_median_type;
public:
    struct plot_data{
        uint32_t                                    width{1024};
        uint32_t                                    height{480};
        structures::image_info                      image{};
        VkImageView                                 image_view{};
        VkSampleCountFlagBits                       image_samples{VK_SAMPLE_COUNT_1_BIT};
        VkFormat                                    image_format{VK_FORMAT_R16G16B16A16_UNORM};
    };
    enum class render_strategy{
        all,
        batched,
        COUNT
    };
    const structures::enum_names<render_strategy> render_strategy_names{
        "all",
        "batched",
    };

    const std::array<VkFormat, 4>               available_formats{VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT};

    std::vector<drawlist_info>                  drawlist_infos{};     // the order here is the render order of the drawlists
    structures::alpha_mapping_type              alpha_mapping_typ{};
    structures::parallel_coordinates_renderer::render_type render_type{};
    structures::change_tracker<plot_data>       plot_data{};
    util::memory_view<structures::attribute>    attributes{};
    render_strategy                             render_strategy;
    size_t                                      render_batch_size;

    parallel_coordinates_workbench(const std::string_view id);

    void render_plot();
    // overriden methods
    void show() override;

    void addDataset(std::string_view datasetId) override{};
    void signalDatasetUpdate(const util::memory_view<std::string_view>& datasetIds, structures::dataset_dependency::update_flags flags) override{};
    void removeDataset(std::string_view datasetId) override{};
    void addDrawlist(std::string_view drawlistId) override{};
    void signalDrawlistUpdate(const util::memory_view<std::string_view>& drawlistIds) override{};
    void removeDrawlist(std::string_view drawlistId) override{};
};

}