#pragma once
#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vk_context.hpp>
#include <vk_initializers.hpp>
#include <stdexcept>
#include <ranges.hpp>

namespace util{
inline void check_vk_result(VkResult err){
    if(err > 0) std::cout << "VkResult " << string_VkResult(err);
    if(err < 0) throw std::runtime_error(std::string("VkResult ") + string_VkResult(err));
}

namespace vk{
template<class T>
struct feature_wrapper{
    T feature;
    std::vector<uint8_t> next_storage;
};
namespace internal{
    template<class T>
    inline void fill_next(void**& curNext, std::vector<uint8_t>*& curStorage){
        void* next = *curNext;
        auto cur = std::vector<uint8_t>(sizeof(feature_wrapper<T>));
        std::memcpy(cur.data(), next, sizeof(T));
        *curNext = cur.data();
        *curStorage = std::move(cur);
        curNext = &(reinterpret_cast<T*>(curStorage->data())->pNext);
        curStorage = &(reinterpret_cast<feature_wrapper<T>*>(curStorage->data())->next_storage);
    }
}

inline feature_wrapper<VkPhysicalDeviceFeatures2> copy_features(const VkPhysicalDeviceFeatures2& in){
    feature_wrapper<VkPhysicalDeviceFeatures2> start{};
    start.feature = in;
    void** curNext = &start.feature.pNext;
    std::vector<uint8_t>* curStorage = &start.next_storage;
    while(*curNext != nullptr){
        void* next = *curNext;
        VkStructureType nextType = *reinterpret_cast<VkStructureType*>(next);
        switch(nextType){
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES:
                internal::fill_next<VkPhysicalDeviceVulkan11Features>(curNext, curStorage);
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
                internal::fill_next<VkPhysicalDeviceVulkan12Features>(curNext, curStorage);
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES:
                internal::fill_next<VkPhysicalDeviceVulkan12Features>(curNext, curStorage);
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT:
                internal::fill_next<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT>(curNext, curStorage);
                break;
            default:
            throw std::runtime_error(std::string("util::copy_features() Unhandled feature type in pNext chain: ") + string_VkStructureType(nextType));
        }
    }
    return std::move(start);
}

template<class T>
inline bool all_features_available(const T& available, const T& required){
    const VkBool32* bool_start_avail = reinterpret_cast<const VkBool32*>(&available.pNext) + sizeof(available.pNext) / sizeof(VkBool32);
    const VkBool32* bool_end_avail = reinterpret_cast<const VkBool32*>(&available) + sizeof(T) / sizeof(VkBool32);
    const VkBool32* bool_start_req = reinterpret_cast<const VkBool32*>(&required.pNext) + sizeof(required.pNext) / sizeof(VkBool32);

    int bool_count = bool_end_avail - bool_start_avail;
    assert(bool_count > 0 && bool_count < 150);
    for(int i: i_range(bool_count)){
        if(bool_start_req[i] && !bool_start_avail[i])
            return false;
    }
    return true;
}

// ----------------------------------------------------------------------------------------------------------------
// Create/destroy helper funcitons which automatically register/unregister the vulkan objects in the context
// ----------------------------------------------------------------------------------------------------------------
inline structures::buffer_info create_buffer(const VkBufferCreateInfo& buffer_info, const VmaAllocationCreateInfo alloc_info,/*out: info for allocated mem*/ VmaAllocationInfo* out_alloc_info = {}){
    structures::buffer_info info{};
    auto res = vmaCreateBuffer(globals::vk_context.allocator, &buffer_info, &alloc_info, &info.buffer, &info.allocation, out_alloc_info);
    check_vk_result(res);
    globals::vk_context.registered_buffers.insert(info);
    return info;
}

inline void destroy_buffer(structures::buffer_info info){
    vmaDestroyBuffer(globals::vk_context.allocator, info.buffer, info.allocation);
    globals::vk_context.registered_buffers.erase(info);
}

inline structures::image_info create_image(const VkImageCreateInfo& image_info, const VmaAllocationCreateInfo& alloc_info, /*out: info for allocated mem*/ VmaAllocationInfo* out_alloc_info = {}){
    structures::image_info info{};
    auto res = vmaCreateImage(globals::vk_context.allocator, &image_info, &alloc_info, &info.image, &info.allocation, out_alloc_info);
    check_vk_result(res);
    globals::vk_context.registered_images.insert(info);
    return info;
}

inline void destroy_image(structures::image_info info){
    vmaDestroyImage(globals::vk_context.allocator, info.image, info.allocation);
    globals::vk_context.registered_images.erase(info);
}

inline VkImageView create_image_view(const VkImageViewCreateInfo& view_info){
    VkImageView view;
    auto res = vkCreateImageView(globals::vk_context.device, &view_info, globals::vk_context.allocation_callbacks, &view);
    check_vk_result(res);
    globals::vk_context.registered_image_views.insert(view);
    return view;
}

inline void destroy_image_view(VkImageView view){
    vkDestroyImageView(globals::vk_context.device, view, globals::vk_context.allocation_callbacks);
    globals::vk_context.registered_image_views.erase(view);
}

inline std::tuple<structures::image_info, VkImageView> create_image_with_view(const VkImageCreateInfo& image_info, const VmaAllocationCreateInfo& alloc_info, /*out: info for allocated mem*/ VmaAllocationInfo* out_alloc_info = {}){
    auto info = create_image(image_info, alloc_info, out_alloc_info);
    auto view_info = initializers::imageViewCreateInfo(info.image, static_cast<VkImageViewType>(image_info.imageType), image_info.format);
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.layerCount = 1;
    view_info.subresourceRange.levelCount = 1;
    return {info, create_image_view(view_info)};
}

inline VkPipelineCache create_pipeline_cache(const VkPipelineCacheCreateInfo& info){
    VkPipelineCache cache;
    auto res = vkCreatePipelineCache(globals::vk_context.device, &info, globals::vk_context.allocation_callbacks, &cache);
    check_vk_result(res);
    globals::vk_context.registered_pipeline_caches.insert(cache);
    return cache;
}

inline VkPipelineLayout create_pipeline_layout(const VkPipelineLayoutCreateInfo& info){
    VkPipelineLayout layout;
    auto res = vkCreatePipelineLayout(globals::vk_context.device, &info, globals::vk_context.allocation_callbacks, &layout);
    check_vk_result(res);
    globals::vk_context.registered_pipeline_layouts.insert(layout);
    return layout;
}

inline void destroy_pipeline_layout(VkPipelineLayout layout){
    vkDestroyPipelineLayout(globals::vk_context.device, layout, globals::vk_context.allocation_callbacks);
    globals::vk_context.registered_pipeline_layouts.erase(layout);
}

inline VkPipeline create_compute_pipline(const VkComputePipelineCreateInfo& info, VkPipelineCache cache = {}){
    VkPipeline pipeline;
    auto res = vkCreateComputePipelines(globals::vk_context.device, cache, 1, &info, globals::vk_context.allocation_callbacks, &pipeline);
    check_vk_result(res);
    globals::vk_context.registered_pipelines.insert(pipeline);
    return pipeline;
}

inline VkPipeline create_graphics_pipline(const VkGraphicsPipelineCreateInfo& info, VkPipelineCache cache = {}){
    VkPipeline pipeline;
    auto res = vkCreateGraphicsPipelines(globals::vk_context.device, cache, 1, &info, globals::vk_context.allocation_callbacks, &pipeline);
    check_vk_result(res);
    globals::vk_context.registered_pipelines.insert(pipeline);
    return pipeline;
}

inline void destroy_pipeline(VkPipeline pipeline){
    vkDestroyPipeline(globals::vk_context.device, pipeline, globals::vk_context.allocation_callbacks);
    globals::vk_context.registered_pipelines.erase(pipeline);
}

inline VkDescriptorPool create_descriptor_pool(const VkDescriptorPoolCreateInfo& info){
    VkDescriptorPool pool;
    auto res = vkCreateDescriptorPool(globals::vk_context.device, &info, globals::vk_context.allocation_callbacks, &pool);
    check_vk_result(res);
    globals::vk_context.registered_descriptor_pools.insert(pool);
    return pool;
}

inline void destroy_descriptor_pool(VkDescriptorPool pool){
    vkDestroyDescriptorPool(globals::vk_context.device, pool, globals::vk_context.allocation_callbacks);
    globals::vk_context.registered_descriptor_pools.erase(pool);
}

inline VkDescriptorSetLayout create_descriptorset_layout(const VkDescriptorSetLayoutCreateInfo& info){
    VkDescriptorSetLayout layout;
    auto res = vkCreateDescriptorSetLayout(globals::vk_context.device, &info, globals::vk_context.allocation_callbacks, &layout);
    check_vk_result(res);
    globals::vk_context.registered_descriptor_set_layouts.insert(layout);
    return layout;
}

inline void destroy_descriptorset_layout(VkDescriptorSetLayout layout){
    vkDestroyDescriptorSetLayout(globals::vk_context.device, layout, globals::vk_context.allocation_callbacks);
    globals::vk_context.registered_descriptor_set_layouts.erase(layout);
}

inline VkCommandPool create_command_pool(const VkCommandPoolCreateInfo& info){
    VkCommandPool pool;
    auto res = vkCreateCommandPool(globals::vk_context.device, &info, globals::vk_context.allocation_callbacks, &pool);
    check_vk_result(res);
    globals::vk_context.registered_command_pools.insert(pool);
    return pool;
}

inline void destroy_command_pool(VkCommandPool pool){
    vkDestroyCommandPool(globals::vk_context.device, pool, globals::vk_context.allocation_callbacks);
    globals::vk_context.registered_command_pools.erase(pool);
}

// ----------------------------------------------------------------------------------------------------------------
// Create helper functions with bundled functionality. No registering in the context going on
// ----------------------------------------------------------------------------------------------------------------

inline VkCommandBuffer create_begin_command_buffer(VkCommandPool pool){
    VkCommandBuffer command;
    VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    info.commandBufferCount = 1;
    info.commandPool = pool;    
    auto res = vkAllocateCommandBuffers(globals::vk_context.device, &info, &command);
    check_vk_result(res);
    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    auto res = vkBeginCommandBuffer(command, &begin_info);
    check_vk_result(res);
    return command;
}

inline void end_commit_command_buffer(VkCommandBuffer commands, VkQueue queue, util::memory_view<VkSemaphore> wait_semaphores = {}, util::memory_view<VkPipelineStageFlags> wait_flags = {}, util::memory_view<VkSemaphore> signal_semaphores = {}, VkFence fence = {}){
    auto res = vkEndCommandBuffer(commands);
    check_vk_result(res);
    VkCommandBufferSubmitInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    info.commandBuffer = commands;
    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.waitSemaphoreCount = wait_semaphores.size();
    submit_info.pWaitSemaphores = wait_semaphores.data();
    submit_info.pWaitDstStageMask = wait_flags.data();
    submit_info.signalSemaphoreCount = signal_semaphores.size();
    submit_info.pSignalSemaphores = signal_semaphores.data();
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &commands;
    vkQueueSubmit(queue, 1, &submit_info, fence);
}

}
}