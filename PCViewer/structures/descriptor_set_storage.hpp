#pragma once
#include <map>
#include <memory>
#include <string>
#include <vulkan/vulkan.h>
#include <image_info.hpp>
#include <optional>

namespace structures{
// describes a single instance of a descriptor allocated from a descriptor set layout. Serves as a singleton descriptor from a single use descriptor set layout with already updated descriptor set (ready to use)
struct descriptor_info{
    std::string             id;
    VkDescriptorSetLayout   layout;
    VkDescriptorSet         descriptor_set;

    struct flags_t{
        bool drawable_image: 1;
    }                       flags{};

    // optional image infos to be able to destroy the images if needed
    struct image_data_t{
        image_info  image;
        VkImageView image_view;
        VkFormat    image_format;
        VkExtent3D  image_size;
    };
    std::optional<image_data_t> image_data{};

    bool operator==(const descriptor_info & o) const {return layout == o.layout && descriptor_set == o.descriptor_set;}
};
using unique_descriptor_info = std::unique_ptr<descriptor_info>;
}

namespace globals{
extern std::map<std::string_view, structures::unique_descriptor_info> descriptor_sets;
}