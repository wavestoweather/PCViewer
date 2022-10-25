#pragma once

#include <vk_mem_alloc.h>
#include <inttypes.h>
#include <mutex>
#include <exception>
#include <robin_hood.h>
#include <memory_view.hpp>
#include <std_util.hpp>
#include <buffer_info.hpp>
#include <image_info.hpp>

namespace std{
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
            return hash_combine(seed, x.allocation);
        }
    };
}

namespace structures{
// info struct to insert things such as api version, physical device selection
struct VkContextInitInfo{
    int                                 physical_device_index{-1};  // set to -1 for automatic detection
    uint32_t                            api_version{};
    std::string_view                    application_name{};
    util::memory_view<const char*>      enabled_instance_layers{};         // has to include debug layers. init() does not add this
    util::memory_view<const char*>      enabled_instance_extensions{};
    util::memory_view<const char*>      enabled_device_extensions{};
    VkPhysicalDeviceFeatures2           device_features{};
};
struct VkContextInitReturnInfo{
    int                                 physical_device_index;
    std::vector<std::string>            physical_device_names;
};

struct vk_context{
    struct consume_semaphore{
        VkSemaphore     semaphore{};
        bool            consumed{};
    };

    VkInstance          instance{};
    VkPhysicalDevice    physical_device{};
    VkDevice            device{};
    VkQueue             graphics_queue{}; // Note: use semaphores for queue sync
    VkQueue             compute_queue{};  // Note: use semaphores for queue sync
    VkQueue             transfer_queue{}; // Note: use semaphores for queue sync
    std::mutex*         graphics_mutex{};
    std::mutex*         compute_mutex{};
    std::mutex*         transfer_mutex{};
    std::vector<std::unique_ptr<std::mutex>> mutex_storage{};   // holds the mutexes which can be ealily accessed via transfer/graphics/compute_mutex
    uint32_t            graphics_queue_family_index{};
    uint32_t            compute_queue_family_index{};
    uint32_t            transfer_queue_family_index{};
    //std::vector<consume_semaphore> graphics_semaphores{};  // semaphore queue of the graphics pipelines between a frame (should be cleared after rendering with wait_and_clear_semaphores())
    //std::vector<consume_semaphore> compute_semaphore{};    // semaphore queue of the compute pipelines between a frame (should be cleared after rendering with wait_and_clear_semaphores())
    //std::vector<consume_semaphore> transfer_semaphore{};   // semaphore queue of the transfer pipelines between a frame (should be cleared after rendering with wait_and_clear_semaphores())

    VmaAllocator        allocator{};

    // currently not used
    VkAllocationCallbacks* allocation_callbacks{};

    // section for registrated vulkan resources which have to be destroyed for cleanup
    robin_hood::unordered_set<VkPipeline>       registered_pipelines;
    robin_hood::unordered_set<VkPipelineLayout> registered_pipeline_layouts;
    robin_hood::unordered_set<VkCommandPool>    registered_command_pools;
    robin_hood::unordered_set<VkDescriptorPool> registered_descriptor_pools;
    robin_hood::unordered_set<VkDescriptorSetLayout> registered_descriptor_set_layouts;
    robin_hood::unordered_set<buffer_info>      registered_buffers;
    robin_hood::unordered_set<image_info>       registered_images;
    robin_hood::unordered_set<VkImageView>      registered_image_views;                 // only used when for an image a second image view has to be registered
    robin_hood::unordered_set<VkRenderPass>     registered_render_passes;
    robin_hood::unordered_set<VkFramebuffer>    registered_framebuffer;
    robin_hood::unordered_set<VkSampler>        registered_sampler;
    robin_hood::unordered_set<VkPipelineCache>  registered_pipeline_caches;
    robin_hood::unordered_set<VkSemaphore>      registered_semaphores;
    robin_hood::unordered_set<VkFence>          registered_fences;

    VkCommandPool                               general_graphics_command_pool{};
    VkDescriptorPool                            general_descriptor_pool{};
    buffer_info                                 staging_buffer{};

    VkDebugUtilsMessengerEXT                    debug_report_callback{};

    // initializes this vulkan context. Init function as global object has no well defined lifetime
    VkContextInitReturnInfo init(const VkContextInitInfo& info);

    // cleanup this vulkan context. Cleanup function needed as global object has no well defined lifetime
    void cleanup();

    void wait_and_clear_semaphores();

    // uploads the data into the staging buffer and makes shure the staging buffer can hold the amount of the data
    void upload_to_staging_buffer(const util::memory_view<uint8_t> data);

    vk_context() = default;
    // no copy construction
    vk_context(const vk_context&) = delete;
    vk_context& operator=(const vk_context&) = delete;
    ~vk_context(){assert(!physical_device && "Missing call to vk_context.cleanup()");}
private:
    size_t                                      _staging_buffer_size{};
    void*                                       _staging_buffer_mappped{};
};
}

namespace globals{
extern structures::vk_context vk_context; 
}