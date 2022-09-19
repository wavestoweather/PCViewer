#pragma once
#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vk_context.hpp>
#include <vk_initializers.hpp>
#include <file_util.hpp>
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

inline void destroy_pipeline_cache(VkPipelineCache cache){
    vkDestroyPipelineCache(globals::vk_context.device, cache, globals::vk_context.allocation_callbacks);
    globals::vk_context.registered_pipeline_caches.erase(cache);
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
    VkCommandPool pool{};
    auto res = vkCreateCommandPool(globals::vk_context.device, &info, globals::vk_context.allocation_callbacks, &pool);
    check_vk_result(res);
    globals::vk_context.registered_command_pools.insert(pool);
    return pool;
}

inline void destroy_command_pool(VkCommandPool pool){
    vkDestroyCommandPool(globals::vk_context.device, pool, globals::vk_context.allocation_callbacks);
    globals::vk_context.registered_command_pools.erase(pool);
}

inline VkRenderPass create_render_pass(const VkRenderPassCreateInfo& info){
    VkRenderPass render_pass{};
    auto res = vkCreateRenderPass(globals::vk_context.device, &info, globals::vk_context.allocation_callbacks, &render_pass);
    check_vk_result(res);
    globals::vk_context.registered_render_passes.insert(render_pass);
    return render_pass;
}

inline void destroy_render_pass(VkRenderPass renderPass){
    vkDestroyRenderPass(globals::vk_context.device, renderPass, globals::vk_context.allocation_callbacks);
    globals::vk_context.registered_render_passes.erase(renderPass);
}

inline VkFramebuffer create_framebuffer(const VkFramebufferCreateInfo& info){
    VkFramebuffer framebuffer{};
    auto res = vkCreateFramebuffer(globals::vk_context.device, &info, globals::vk_context.allocation_callbacks, &framebuffer);
    check_vk_result(res);
    globals::vk_context.registered_framebuffer.insert(framebuffer);
    return framebuffer;
}

inline void destroy_framebuffer(VkFramebuffer framebuffer){
    vkDestroyFramebuffer(globals::vk_context.device, framebuffer, globals::vk_context.allocation_callbacks);
    globals::vk_context.registered_framebuffer.erase(framebuffer);
}

inline VkSemaphore create_semaphore(const VkSemaphoreCreateInfo& info){
    VkSemaphore semaphore{};
    auto res = vkCreateSemaphore(globals::vk_context.device, &info, globals::vk_context.allocation_callbacks, &semaphore);
    check_vk_result(res);
    globals::vk_context.registered_semaphores.insert(semaphore);
    return semaphore;
}

inline void destroy_semaphore(VkSemaphore semaphore){
    vkDestroySemaphore(globals::vk_context.device, semaphore, globals::vk_context.allocation_callbacks);
    globals::vk_context.registered_semaphores.erase(semaphore);
}

inline VkFence create_fence(const VkFenceCreateInfo& info){
    VkFence fence{};
    auto res = vkCreateFence(globals::vk_context.device, &info, globals::vk_context.allocation_callbacks, &fence);
    check_vk_result(res);
    globals::vk_context.registered_fences.insert(fence);
    return fence;
}

inline void destroy_fence(VkFence fence){
    vkDestroyFence(globals::vk_context.device, fence, globals::vk_context.allocation_callbacks);
    globals::vk_context.registered_fences.erase(fence);
}

inline VkSampler create_sampler(const VkSamplerCreateInfo& info){
    VkSampler sampler{};
    auto res = vkCreateSampler(globals::vk_context.device, &info, globals::vk_context.allocation_callbacks, &sampler);
    check_vk_result(res);
    globals::vk_context.registered_sampler.insert(sampler);
    return sampler;
}

inline void destroy_sampler(VkSampler sampler){
    vkDestroySampler(globals::vk_context.device, sampler, globals::vk_context.allocation_callbacks);
    globals::vk_context.registered_sampler.erase(sampler);
}

inline void setup_debug_report_callback(PFN_vkDebugUtilsMessengerCallbackEXT callback){
	auto vkCreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(globals::vk_context.instance, "vkCreateDebugUtilsMessengerEXT");
	assert(vkCreateDebugUtilsMessengerEXT != NULL);

	VkDebugUtilsMessengerCreateInfoEXT debug_report_ci = {};
	debug_report_ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	debug_report_ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_report_ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug_report_ci.pfnUserCallback = callback;
	debug_report_ci.pUserData = NULL;
	auto res = vkCreateDebugUtilsMessengerEXT(globals::vk_context.instance, &debug_report_ci, globals::vk_context.allocation_callbacks, &globals::vk_context.debug_report_callback);
	check_vk_result(res);
}

// ----------------------------------------------------------------------------------------------------------------
// Create helper functions with bundled functionality. No registering in the context going on
// ----------------------------------------------------------------------------------------------------------------
inline VkShaderModule create_shader_module(std::string_view filename){
    VkShaderModule module;
    auto bytes = util::read_file(filename);
    auto module_info = util::vk::initializers::shaderModuleCreateInfo(bytes);
    auto res = vkCreateShaderModule(globals::vk_context.device, &module_info, globals::vk_context.allocation_callbacks, &module); 
    util::check_vk_result(res);
    return module;
}

inline VkCommandBuffer create_begin_command_buffer(VkCommandPool pool){
    VkCommandBuffer command;
    VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    info.commandBufferCount = 1;
    info.commandPool = pool;    
    auto res = vkAllocateCommandBuffers(globals::vk_context.device, &info, &command);
    check_vk_result(res);
    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    res = vkBeginCommandBuffer(command, &begin_info);
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

inline void wait_semaphore(VkCommandPool commandPool, VkQueue queue, VkSemaphore semaphore){
    auto commands = create_begin_command_buffer(commandPool);
    end_commit_command_buffer(commands, queue, semaphore);
    auto res = vkQueueWaitIdle(queue);
    check_vk_result(res);
}

inline VkDeviceAddress get_buffer_address(const structures::buffer_info& buffer){
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buffer.buffer;
    return vkGetBufferDeviceAddress(globals::vk_context.device, &info);
}
}
}