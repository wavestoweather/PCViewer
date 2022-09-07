#pragma once

#include <vk_mem_alloc.h>
#include <inttypes.h>
#include <mutex>
#include <exception>
#include <robin_hood.h>
#include <memory_view.hpp>

namespace structures{
struct buffer_info{
    VkBuffer        buffer{};
    VmaAllocation   allocation{};
};
struct image_info{
    VkImage         image{};
    VkImageView     image_view{};
    VmaAllocation   allocation{};
};
}

namespace std{
    template <class T>
    inline size_t hash_combine(std::size_t seed, const T& v)
    {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        return seed;
    }

    template<> struct hash<structures::buffer_info>{
        size_t operator()(const structures::buffer_info & x) const
        {
            size_t seed = hash<VkBuffer>{}(x.buffer);
            return hash_combine(seed, x.allocation);
        }
    };

    template<> struct hash<structures::image_info>{
        size_t operator()(const structures::image_info & x) const
        {
            size_t seed = hash<VkImage>{}(x.image);
            seed = hash_combine(seed, x.image_view);
            return hash_combine(seed, x.allocation);
        }
    };
}

namespace structures{
// info struct to insert things such as api version, physical device selection
struct VkContextInitInfo{
    int                                 physical_device_index;  // set to -1 for automatic detection
    uint32_t                            api_version;
    std::string_view                    application_name;
    util::memory_view<const char*>      enabled_instance_layers;         // has to include debug layers. init() does not add this
    util::memory_view<const char*>      enabled_instance_extensions;
    util::memory_view<const char*>      enabled_device_extensions;
    VkPhysicalDeviceFeatures2           device_features;
};
struct VkContextInitReturnInfo{
    int                                 physical_device_index;
    std::vector<std::string>            physical_device_names;
};

struct vk_context{
    VkInstance          instance{};
    VkPhysicalDevice    physical_device{};
    VkDevice            device{};
    VkQueue             graphics_queue{}; // Note: use semaphores for queue sync
    VkQueue             compute_queue{};  // Note: use semaphores for queue sync
    VkQueue             transfer_queue{}; // Note: use semaphores for queue sync
    std::mutex          graphics_mutex{};
    std::mutex          compute_mutex{};
    std::mutex          transfer_mutex{};

    VmaAllocator        allocator{};

    // currently no used
    VkAllocationCallbacks* allocation_callbacks{};

    // section for registrated vulkan resources which have to be destroyed for cleanup
    robin_hood::unordered_set<VkPipeline>       registered_pipelines;
    robin_hood::unordered_set<VkPipelineLayout> registered_pipeline_layouts;
    robin_hood::unordered_set<VkCommandPool>    registered_command_pools;
    robin_hood::unordered_set<VkDescriptorPool> registered_descriptor_pools;
    robin_hood::unordered_set<VkDescriptorSetLayout> registered_descriptor_set_layouts;
    robin_hood::unordered_set<buffer_info>      registered_buffers;
    robin_hood::unordered_set<image_info>       registered_images;

    // initializes this vulkan context. Init function as global object has no well defined lifetime
    VkContextInitReturnInfo init(const VkContextInitInfo& info);

    // cleanup this vulkan context. Cleanup function as global object has no well defined lifetime
    void cleanup();

    // no copy construction
    vk_context(const vk_context&) = delete;
    vk_context& operator=(const vk_context&) = delete;
    // no move construction
    vk_context(vk_context&&) = delete;
    vk_context& operator=(vk_context&&) = delete;
    ~vk_context(){assert(!physical_device && "Missing call to vk_context.cleanup()");}
};
}

namespace globals{
extern structures::vk_context vk_context; 
}