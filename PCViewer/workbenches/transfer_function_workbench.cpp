#include "transfer_function_workbench.hpp"
#include <settings_manager.hpp>
#include <descriptor_set_storage.hpp>
#include <color_brewer_util.hpp>
#include <vma_initializers.hpp>
#include <../vulkan/vk_format_util.hpp>
#include <stager.hpp>
#include <persistent_samplers.hpp>
#include <logger.hpp>
#include <global_descriptor_set_util.hpp>
#include "../storage/colormaps.hpp"

const std::string_view tfs_setting_name{"transfer_functions"};
constexpr VkFormat tf_format{VK_FORMAT_R8G8B8A8_UNORM};
constexpr VkImageAspectFlags tf_aspect{VK_IMAGE_ASPECT_COLOR_BIT};

void workbenches::transfer_function_workbench::_create_tf_function(std::string_view id, util::memory_view<const uint32_t> image_data)
{
    auto image_info = util::vk::initializers::imageCreateInfo(tf_format, {static_cast<uint32_t>(image_data.size()), 1, 1}, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    auto alloc_info = util::vma::initializers::allocationCreateInfo();
    auto [image, view] = util::vk::create_image_with_view(image_info, alloc_info);

    // image upload
    const uint32_t texel_size = static_cast<uint32_t>(FormatSize(image_info.format));
    assert(texel_size == 4);
    structures::stager::staging_image_info staging_info{};
    staging_info.dst_image = image.image;
    staging_info.start_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    staging_info.end_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    staging_info.subresource_layers.aspectMask = tf_aspect;
    staging_info.bytes_per_pixel = texel_size;
    staging_info.image_extent = image_info.extent;
    staging_info.data_upload = util::memory_view(image_data);
    globals::stager.add_staging_task(staging_info);

    // descriptor set creation
    structures::unique_descriptor_info tf_desc{std::make_unique<structures::descriptor_info>()};
    tf_desc->id = id;
    tf_desc->layout = descriptor_set_layout;
    auto descriptor_alloc_info = util::vk::initializers::descriptorSetAllocateInfo(globals::vk_context.general_descriptor_pool, descriptor_set_layout);
    auto res = vkAllocateDescriptorSets(globals::vk_context.device, &descriptor_alloc_info, &tf_desc->descriptor_set); util::check_vk_result(res);
    VkDescriptorImageInfo desc_image_info{globals::persistent_samplers.get(util::vk::initializers::samplerCreateInfo(VK_FILTER_LINEAR)), view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    auto write_desc_set = util::vk::initializers::writeDescriptorSet(tf_desc->descriptor_set, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &desc_image_info);
    vkUpdateDescriptorSets(globals::vk_context.device, 1, &write_desc_set, {}, {});
    globals::descriptor_sets[tf_desc->id] = std::move(tf_desc);
    tf_desc->image_data = structures::descriptor_info::image_data_t{image, view, tf_format, image_info.extent};
    globals::descriptor_sets[tf_desc->id] = std::move(tf_desc);

    structures::unique_tracker<transfer_function_t> transfer_func{};
    transfer_func().name = id;
    transfer_func().editable = false;
    transfer_func().colors = std::vector<col>(image_data.begin(), image_data.end());
    transfer_func.changed = false;
    transfer_functions.emplace_back(std::move(transfer_func));
    transfer_function_index.insert({transfer_functions.back().read().name, transfer_functions.back().ref_no_track()});
}

workbenches::transfer_function_workbench::transfer_function_workbench(std::string_view id, VkDescriptorSetLayout desc_set_layout):
    structures::workbench(id), 
    descriptor_set_layout(desc_set_layout)
{
    // setup default transfer functions(for each color brewer set create tf with alpha = 1)
    structures::unique_tracker<transfer_function_t> transfer_func{};
    transfer_func().name = util::global_descriptors::heatmap_descriptor_id;
    transfer_func().editable = false;
    util::memory_view<const uint32_t> hm = util::memory_view<const uint8_t>(heat_map, sizeof(heat_map));
    transfer_func().colors = std::vector<col>(hm.begin(), hm.end());
    transfer_func.changed = false;
    transfer_functions.emplace_back(std::move(transfer_func));
    transfer_function_index.insert({transfer_functions.back().read().name, transfer_functions.back().ref_no_track()});
    for(const auto& info: brew_palette_infos){
        // always gets the larges palette off a type and creates a transferfunction from it
        auto colors = util::color_brewer::brew_u32(info.name, info.max_colors);
        _create_tf_function(info.name, colors);
    }
    // loading data from the settings manager
    const auto& stored_tfs = globals::settings_manager.get_setting(tfs_setting_name);
    if(!stored_tfs.is_null()){
        if(stored_tfs.is_array()){
            for(size_t i: util::size_range(stored_tfs)){
                auto cur_tf = stored_tfs[i].get<crude_json::object>();
                auto name = cur_tf["name"].get<std::string>();
                auto colors = cur_tf["colors"].get<crude_json::array>();
                std::vector<col> cols(colors.size());
                for(size_t j: util::size_range(colors)){
                    auto col = colors[j].get<crude_json::array>();
                    cols[j].c.r = uint8_t(col[0].get<double>());
                    cols[j].c.g = uint8_t(col[1].get<double>());
                    cols[j].c.b = uint8_t(col[2].get<double>());
                    cols[j].c.a = uint8_t(col[3].get<double>() * 255);
                }
            }
        }
        else if(logger.logging_level >= logging::level::l_2){
            logger << logging::error_prefix << " transfer_function_workbench() Can not load stored transferfunctions, expected array of transfer functions could not be read." << logging::endl;
        }
    }
    globals::stager.wait_for_completion();
}

void workbenches::transfer_function_workbench::show()
{
    if(!active)
        return;

    ImGui::Begin(id.data(), &active);
    // transfer function editing
    ImGui::End();
}
