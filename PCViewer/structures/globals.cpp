#include "datasets.hpp"
#define VMA_IMPLEMENTATION  // vma allocations cpu
#include "vk_context.hpp"
#include <vk_util.hpp>
#include <ranges.hpp>

namespace structures{
VkContextInitReturnInfo vk_context::init(const VkContextInitInfo& info){
    if(physical_device)
        throw std::runtime_error("vk_context::init() Context was already initailized. Missing call vk_context::cleanup()");
    
    VkResult res;
    VkContextInitReturnInfo ret{};

    // create instance
    VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.apiVersion = info.api_version;
    app_info.pApplicationName = info.application_name.data();

    VkInstanceCreateInfo create_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create_info.enabledExtensionCount = info.enabled_instance_extensions.size();
    create_info.ppEnabledExtensionNames = info.enabled_instance_extensions.data();
    create_info.pApplicationInfo = &app_info;

    res = vkCreateInstance(&create_info, &allocation_callbacks, &instance);
    util::check_vk_result(res);

    // go through all physical devices, check for available physical device features
    int bestFit{-1};
    uint32_t gpu_count;
    res = vkEnumeratePhysicalDevices(instance, &gpu_count, NULL);
    util::check_vk_result(res);

    ret.physical_device_names.resize(gpu_count);
    std::vector<VkPhysicalDevice> physical_devices(gpu_count);
    res = vkEnumeratePhysicalDevices(instance, &gpu_count, physical_devices.data());
    util::check_vk_result(res);

    struct score_index{int score, index;};
    std::vector<score_index> scores(gpu_count);
    for(int i: util::i_range(gpu_count)){
        scores[i].index = i;

        VkPhysicalDeviceFeatures available_features;
        vkGetPhysicalDeviceFeatures(physical_devices[i], &available_features);

        auto feature = util::copy_features(info.device_features);
        vkGetPhysicalDeviceFeatures2(physical_devices[i], &feature.feature);

        void* cur_available = feature.feature.pNext;
        void* cur_required = info.device_features.pNext;
        bool all_features_avail = util::all_features_available<VkPhysicalDeviceFeatures2>(feature.feature, info.device_features);
        while(cur_available){
            switch(*static_cast<VkStructureType*>(cur_available)){
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES:
                all_features_avail &= util::all_features_available<VkPhysicalDeviceVulkan11Features>(*static_cast<VkPhysicalDeviceVulkan11Features*>(cur_available), *static_cast<VkPhysicalDeviceVulkan11Features*>(cur_required));
                cur_available = static_cast<VkPhysicalDeviceVulkan11Features*>(cur_available)->pNext;
                cur_required = static_cast<VkPhysicalDeviceVulkan11Features*>(cur_required)->pNext;
            break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
                all_features_avail &= util::all_features_available<VkPhysicalDeviceVulkan12Features>(*static_cast<VkPhysicalDeviceVulkan12Features*>(cur_available), *static_cast<VkPhysicalDeviceVulkan12Features*>(cur_required));
                cur_available = static_cast<VkPhysicalDeviceVulkan12Features*>(cur_available)->pNext;
                cur_required = static_cast<VkPhysicalDeviceVulkan12Features*>(cur_required)->pNext;
            break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES:
                all_features_avail &= util::all_features_available<VkPhysicalDeviceVulkan13Features>(*static_cast<VkPhysicalDeviceVulkan13Features*>(cur_available), *static_cast<VkPhysicalDeviceVulkan13Features*>(cur_required));
                cur_available = static_cast<VkPhysicalDeviceVulkan13Features*>(cur_available)->pNext;
                cur_required = static_cast<VkPhysicalDeviceVulkan13Features*>(cur_required)->pNext;
            break;
            default:
                throw std::runtime_error(std::string("structures::vk_context::init(...) Unhandled feature type in pNext chain: ") + string_VkStructureType(*static_cast<VkStructureType*>(cur_available)));
            }
        }
        if(!all_features_avail){
            scores[i].score = std::numeric_limits<int>::lowest();
            continue;
        }

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physical_devices[i], &props);
        scores[i].score = props.limits.maxStorageBufferRange;

        ret.physical_device_names[i] = props.deviceName;
    }
    // check selected gpu
    if(info.physical_device_index < scores.size() && scores[info.physical_device_index].score > 0)
        ret.physical_device_index = info.physical_device_index;
    else
        ret.physical_device_index = std::max_element(scores.begin(), scores.end(), [](const score_index& l, const score_index& r){return l.score > r.score;})->index;
    
    if(scores[ret.physical_device_index].score < 0)
        throw std::runtime_error("vk_context::init() No physical device supports all needed features");

    physical_device = physical_devices[ret.physical_device_index];

    // getting the queues
    uint32_t queue_families_count;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_families_count, NULL);
    std::vector<VkQueueFamilyProperties> queue_families(queue_families_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_families_count, queue_families.data());
    std::vector<uint32_t> graphics_queues, compute_queues, transfer_queues;
    for(int i: util::i_range(queue_families_count)){
        if(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            graphics_queues.push_back(i);
        if(queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            compute_queues.push_back(i);
        if(queue_families[i].queueFlags & VK_QUEUE_TRANSFER_BIT)
            transfer_queues.push_back(i);
    }
    std::vector<uint32_t> t_1, t_2;
    std::set_difference(transfer_queues.begin(), transfer_queues.end(), compute_queues.begin(), compute_queues.end(), std::inserter(t_1, t_1.end()));
    std::set_difference(t_1.begin(), t_1.end(), graphics_queues.begin(), graphics_queues.end(), std::inserter(t_2, t_2.end()));
    std::vector<uint32_t> c_1, c_2;
    std::set_difference(compute_queues.begin(), compute_queues.end(), transfer_queues.begin(), transfer_queues.end(), std::inserter(c_1, c_1.end()));
    std::set_difference(c_1.begin(), c_1.end(), graphics_queues.begin(), graphics_queues.end(), std::inserter(c_2, c_2.end()));
    std::vector<uint32_t> g_1, g_2;
    std::set_difference(graphics_queues.begin(), graphics_queues.end(), transfer_queues.begin(), transfer_queues.end(), std::inserter(g_1, g_1.end()));
    std::set_difference(g_1.begin(), g_1.end(), compute_queues.begin(), compute_queues.end(), std::inserter(g_2, g_2.end()));

    uint32_t g_queue_family, c_queue_family, t_queue_family;
    if(t_2.size())
        t_queue_family = t_2[0];
    else
        t_queue_family = transfer_queues[0];
    if(c_2.size())
        c_queue_family = c_2[0];
    else
        c_queue_family = compute_queues[0];
    if(g_2.size())
        g_queue_family = g_2[0];
    else
        g_queue_family = graphics_queues[0];

    // creating the logical device
    
    // getting all available extensions from the selected physical device to enable them
    auto available_device_features = util::copy_features(info.device_features);
    vkGetPhysicalDeviceFeatures2(physical_device, &available_device_features.feature);

    std::set<uint32_t> distinct_queue_families{g_queue_family, c_queue_family, t_queue_family};
    std::vector<uint32_t> distinct_families_v(distinct_queue_families.begin(), distinct_queue_families.end());
    std::vector<VkDeviceQueueCreateInfo> queue_info(distinct_queue_families.size());
    const float queue_priority[] = { 1.0f };
    for(int i: util::size_range(distinct_families_v)){
        queue_info[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info[i].queueFamilyIndex = distinct_families_v[i];
        queue_info[i].queueCount = 1;
        queue_info[i].pQueuePriorities = queue_priority;
    }
    
    VkDeviceCreateInfo device_create_info{};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pNext = &available_device_features.feature;
    device_create_info.queueCreateInfoCount = queue_info.size();
    device_create_info.pQueueCreateInfos = queue_info.data();
    device_create_info.enabledExtensionCount = info.enabled_device_extensions.size();
    device_create_info.ppEnabledExtensionNames = info.enabled_device_extensions.data();
    res = vkCreateDevice(physical_device, &device_create_info, &allocation_callbacks, &device);
    util::check_vk_result(res);
    vkGetDeviceQueue(device, g_queue_family, 0, &graphics_queue);
    vkGetDeviceQueue(device, c_queue_family, 0, &compute_queue);
    vkGetDeviceQueue(device, t_queue_family, 0, &transfer_queue);
}

void vk_context::cleanup(){
    if(!physical_device)
        throw std::runtime_error("vk_context::cleanup() Context was already deleted. Missing call vk_context::init()");

    for(auto& pipeline: registered_pipelines)
        vkDestroyPipeline(device, pipeline, &allocation_callbacks);
    registered_pipelines.clear();
    for(auto& pipeline_layout: registered_pipeline_layouts)
        vkDestroyPipelineLayout(device, pipeline_layout, &allocation_callbacks);
    registered_pipeline_layouts.clear();
    for(auto& command_pool: registered_command_pools)
        vkDestroyCommandPool(device, command_pool, &allocation_callbacks);
    registered_command_pools.clear();
    for(auto& descriptor_pool: registered_descriptor_pools)
        vkDestroyDescriptorPool(device, descriptor_pool, &allocation_callbacks);
    registered_descriptor_pools.clear();
    for(auto& descriptor_set_layout: registered_descriptor_set_layouts)
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, &allocation_callbacks);
    registered_descriptor_set_layouts.clear();
    for(auto& buffer_info: registered_buffers)
        vmaDestroyBuffer(allocator, buffer_info.buffer, buffer_info.allocation);
    registered_buffers.clear();
    for(auto& image_info: registered_images){
        vmaDestroyImage(allocator, image_info.image, image_info.allocation);
        if(image_info.image_view)
            vkDestroyImageView(device, image_info.image_view, &allocation_callbacks);
    }
    registered_images.clear();

    vmaDestroyAllocator(allocator);
    allocator = {};
    vkDestroyDevice(device, &allocation_callbacks);
    device = {};
    transfer_queue = {};
    compute_queue = {};
    graphics_queue = {};
    physical_device = {};
    instance = {};
}
}

namespace globals{
datasets_t datasets{};

structures::vk_context vk_context{};
}
