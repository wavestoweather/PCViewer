#include "global_descriptor_set_util.hpp"
#include "../storage/colormaps.hpp"
#include <vk_context.hpp>
#include <vk_util.hpp>
#include <vk_initializers.hpp>
#include <vma_initializers.hpp>
#include <stager.hpp>
#include "../vulkan/vk_format_util.hpp"
#include <persistent_samplers.hpp>
#include <as_cast.hpp>

namespace util{
namespace global_descriptors{
void setup_default_descriptors(){
    constexpr VkFormat image_format{VK_FORMAT_R8G8B8A8_UNORM};
    assert(FormatSize(image_format) == 4);
    constexpr VkImageAspectFlags image_aspect{VK_IMAGE_ASPECT_COLOR_BIT};
    auto image_info = util::vk::initializers::imageCreateInfo(image_format, {sizeof(heat_map) / sizeof(*heat_map) / 4, 1, 1}, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    auto alloc_info = util::vma::initializers::allocationCreateInfo();
    auto [image, view] = util::vk::create_image_with_view(image_info, alloc_info);

    // uploading image
    const uint32_t texel_size = as<uint32_t>(FormatSize(image_info.format));
    structures::stager::staging_image_info image_staging{};
    image_staging.dst_image = image.image;
    image_staging.start_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_staging.end_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_staging.subresource_layers.aspectMask = image_aspect;
    image_staging.bytes_per_pixel = texel_size;
    image_staging.image_extent.width = sizeof(heat_map) / sizeof(*heat_map) / 4;
    image_staging.data_upload = util::memory_view<const uint8_t>(heat_map, sizeof(heat_map));
    globals::stager.add_staging_task(image_staging);

    // creating descriptor set + layout for heat_map
    structures::unique_descriptor_info heatmap_desc{std::make_unique<structures::descriptor_info>()};
    heatmap_desc->id = std::string(heatmap_descriptor_id);
    auto heat_map_binding = util::vk::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_ALL);
    auto descriptor_info = util::vk::initializers::descriptorSetLayoutCreateInfo(heat_map_binding);
    heatmap_desc->layout = util::vk::create_descriptorset_layout(descriptor_info);
    auto descriptor_alloc_info = util::vk::initializers::descriptorSetAllocateInfo(globals::vk_context.general_descriptor_pool, heatmap_desc->layout);
    auto res = vkAllocateDescriptorSets(globals::vk_context.device, &descriptor_alloc_info, &heatmap_desc->descriptor_set); check_vk_result(res);
    VkDescriptorImageInfo desc_image_info{globals::persistent_samplers.get(util::vk::initializers::samplerCreateInfo(VK_FILTER_LINEAR)), view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    auto write_desc_set = util::vk::initializers::writeDescriptorSet(heatmap_desc->descriptor_set, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &desc_image_info);
    vkUpdateDescriptorSets(globals::vk_context.device, 1, &write_desc_set, {}, {});
    globals::descriptor_sets[heatmap_desc->id] = std::move(heatmap_desc);

    globals::stager.wait_for_completion();  // waiting for all image uploads
}
}
}