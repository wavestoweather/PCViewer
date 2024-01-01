#pragma once
#include <workbench_base.hpp>
#include <vk_util.hpp>
#include <change_tracker.hpp>
#include <robin_hood.h>
#include <descriptor_set_storage.hpp>

namespace workbenches{
class transfer_function_workbench: structures::workbench{
    uint32_t _active_tf{};
    void _create_tf_function(std::string_view id, util::memory_view<const uint32_t> image_data);
public:
    struct u8_col{uint8_t r,g,b,a;};
    union col{uint32_t v; u8_col c; col() = default; col(uint32_t v): v(v){}};
    struct transfer_function_t{
        std::string             name;
        bool                    editable;
        std::vector<col>        colors;
        // the vulkan things are all stored in the global descriptor set storage
    };

    transfer_function_workbench(std::string_view id, VkDescriptorSetLayout desc_set_layout);

    void show() override;

    const VkDescriptorSetLayout                                         descriptor_set_layout;
    std::vector<structures::unique_tracker<transfer_function_t>>        transfer_functions;
    robin_hood::unordered_map<std::string_view, transfer_function_t&>   transfer_function_index;    // references stay valid due to unique_tracker

    VkDescriptorSet get_tf_descriptor(std::string_view id) const {assert(transfer_function_index.contains(id)); return globals::descriptor_sets.at(id)->descriptor_set;};
};
}