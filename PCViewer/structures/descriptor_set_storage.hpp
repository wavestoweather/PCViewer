#pragma once
#include <robin_hood.h>
#include <string>
#include <vulkan/vulkan.h>

namespace structures{
// describes a single instance of a descriptor allocated from a descriptor set layout. Serves as a singleton descriptor from a single use descriptor set layout with already updated descriptor set (ready to use)
struct descriptor_info{
    std::string             id;
    VkDescriptorSetLayout   layout;
    VkDescriptorSet         descriptor_set;

    bool operator==(const descriptor_info & o) const {return layout == o.layout && descriptor_set == o.descriptor_set;}
};
using uniqe_descriptor_info = std::unique_ptr<descriptor_info>;
}

namespace globals{
extern robin_hood::unordered_map<std::string_view, structures::uniqe_descriptor_info> descriptor_sets;
}