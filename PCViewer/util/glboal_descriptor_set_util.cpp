#include "global_descriptor_set_util.hpp"
#include "../storage/colormaps.hpp"
#include <vk_context.hpp>
#include <vk_util.hpp>
#include <vk_initializers.hpp>
#include <vma_initializers.hpp>
#include <stager.hpp>

namespace util{
namespace global_descriptors{
void setup_default_descriptors(){
    auto image_info = util::vk::initializers::imageCreateInfo(VK_FORMAT_R8G8B8A8_UNORM, {sizeof(heat_map) / sizeof(*heat_map) / 4, 1, 1}, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    auto alloc_info = util::vma::initializers::allocationCreateInfo();
    auto [image, view] = util::vk::create_image_with_view(image_info, alloc_info);

}
}
}