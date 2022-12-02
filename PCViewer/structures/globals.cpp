#define VMA_IMPLEMENTATION  // vma allocations cpu
#include <vk_mem_alloc.h>
#include <datasets.hpp>
#include "vk_context.hpp"
#include <vk_util.hpp>
#include <vma_initializers.hpp>
#include <vk_initializers.hpp>
#include <ranges.hpp>
#include <brushes.hpp>
#include <drawlists.hpp>
#include <settings_manager.hpp>
#include <fstream>
#include <commandline_parser.hpp>
#include <texture_storage.hpp>
#include <persistent_samplers.hpp>
#include <descriptor_set_storage.hpp>
#include <load_behaviour.hpp>
#include <drawlist_creation_behaviour.hpp>
#include <open_filepaths.hpp>
#include <stager.hpp>
#include <util.hpp>
#include <workbench_base.hpp>
#include <imgui_globals.hpp>
#include <logger.hpp>
#include <sys_info.hpp>
#include <file_loader.hpp>
#include <histogram_counter.hpp>
#include <histogram_counter_executor.hpp>
#include <stopwatch.hpp>
#include <priority_sorter.hpp>
#include <priority_globals.hpp>
#include <distance_calculator.hpp>
#include <radix_sort.hpp>

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

// globals definition
structures::logger<20> logger{};

namespace globals{
structures::vk_context vk_context{};

datasets_t datasets{};
std::set<std::string_view> datasets_to_delete{};

structures::drawlists_t drawlists{};
std::vector<std::string_view> selected_drawlists{};
std::set<std::string_view> drawlists_to_delete{};

structures::global_brushes global_brushes{};
structures::brush_edit_data brush_edit_data{};
std::atomic<structures::brush_id> cur_global_brush_id{1};
std::atomic<structures::range_id> cur_brush_range_id{1};

structures::settings_manager settings_manager{};

structures::commandline_parser commandline_parser{};

robin_hood::unordered_map<std::string, structures::texture> textures{};

structures::persistent_samplers persistent_samplers{};

robin_hood::unordered_map<std::string_view, structures::uniqe_descriptor_info> descriptor_sets{};

structures::load_behaviour load_behaviour{};
structures::drawlist_creation_behaviour drawlist_creation_behaviour{};

std::vector<std::string> paths_to_open{};
std::vector<structures::query_attribute> attribute_query{};

structures::stager stager{};

workbenches_t workbenches{};
structures::workbench* primary_workbench{};
structures::workbench* secondary_workbench{};
dataset_dependencies_t dataset_dependencies{};
drawlist_dataset_dependencies_t drawlist_dataset_dependencies{}; 

structures::imgui_globals imgui{};

structures::sys_info sys_info{};

structures::file_loader file_loader{};

structures::histogram_counter histogram_counter{};

structures::priority_sorter priority_sorter{};

std::atomic<float> priority_center_vealue{};
std::atomic<float> priority_center_distance{};
std::atomic<uint32_t> priority_center_attribute_id{};
const std::string_view priority_drawlist_standard_order{"standard"};
}

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
    create_info.enabledLayerCount = info.enabled_instance_layers.size();
    create_info.ppEnabledLayerNames = info.enabled_instance_layers.data();
    create_info.pApplicationInfo = &app_info;

    res = vkCreateInstance(&create_info, allocation_callbacks, &instance);
    util::check_vk_result(res);

    // go through all physical devices, check for available physical device features
    uint32_t gpu_count;
    res = vkEnumeratePhysicalDevices(instance, &gpu_count, NULL);
    util::check_vk_result(res);

    ret.physical_device_names.resize(gpu_count);
    std::vector<VkPhysicalDevice> physical_devices(gpu_count);
    res = vkEnumeratePhysicalDevices(instance, &gpu_count, physical_devices.data());
    util::check_vk_result(res);

    struct score_index{int64_t score; int index;};
    std::vector<score_index> scores(gpu_count);
    for(int i: util::i_range(gpu_count)){
        scores[i].index = i;

        // checking physical device features
        VkPhysicalDeviceFeatures available_features;
        vkGetPhysicalDeviceFeatures(physical_devices[i], &available_features);

        auto feature = util::vk::copy_features(info.device_features);
        vkGetPhysicalDeviceFeatures2(physical_devices[i], &feature.feature);
        feature.feature.features = available_features;                          // has to be done as vkGetPhysicalDeviceFeatures2 does not fill the base feature

        void* cur_available = feature.feature.pNext;
        void* cur_required = info.device_features.pNext;
        bool all_features_avail = util::vk::all_features_available<VkPhysicalDeviceFeatures2>(feature.feature, info.device_features);
        while(cur_available){
            switch(*static_cast<VkStructureType*>(cur_available)){
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES:
                all_features_avail &= util::vk::all_features_available<VkPhysicalDeviceVulkan11Features>(*static_cast<VkPhysicalDeviceVulkan11Features*>(cur_available), *static_cast<VkPhysicalDeviceVulkan11Features*>(cur_required));
                cur_available = static_cast<VkPhysicalDeviceVulkan11Features*>(cur_available)->pNext;
                cur_required = static_cast<VkPhysicalDeviceVulkan11Features*>(cur_required)->pNext;
            break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
                all_features_avail &= util::vk::all_features_available<VkPhysicalDeviceVulkan12Features>(*static_cast<VkPhysicalDeviceVulkan12Features*>(cur_available), *static_cast<VkPhysicalDeviceVulkan12Features*>(cur_required));
                cur_available = static_cast<VkPhysicalDeviceVulkan12Features*>(cur_available)->pNext;
                cur_required = static_cast<VkPhysicalDeviceVulkan12Features*>(cur_required)->pNext;
            break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES:
                all_features_avail &= util::vk::all_features_available<VkPhysicalDeviceVulkan13Features>(*static_cast<VkPhysicalDeviceVulkan13Features*>(cur_available), *static_cast<VkPhysicalDeviceVulkan13Features*>(cur_required));
                cur_available = static_cast<VkPhysicalDeviceVulkan13Features*>(cur_available)->pNext;
                cur_required = static_cast<VkPhysicalDeviceVulkan13Features*>(cur_required)->pNext;
            break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT:
                all_features_avail &= util::vk::all_features_available<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT>(*static_cast<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT*>(cur_available), *static_cast<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT*>(cur_required));
                cur_available = static_cast<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT*>(cur_available)->pNext;
                cur_required = static_cast<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT*>(cur_required)->pNext;
            break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES:
                all_features_avail &= util::vk::all_features_available<VkPhysicalDevice16BitStorageFeatures>(*static_cast<VkPhysicalDevice16BitStorageFeatures*>(cur_available), *static_cast<VkPhysicalDevice16BitStorageFeatures*>(cur_required));
                cur_available = static_cast<VkPhysicalDevice16BitStorageFeatures*>(cur_available)->pNext;
                cur_required = static_cast<VkPhysicalDevice16BitStorageFeatures*>(cur_required)->pNext;
            break;
            default:
                throw std::runtime_error(std::string("structures::vk_context::init(...) Unhandled feature type in pNext chain: ") + string_VkStructureType(*static_cast<VkStructureType*>(cur_available)));
            }
        }
        if(!all_features_avail){
            scores[i].score = std::numeric_limits<int>::lowest();
            continue;
        }

        // checking device extensions
        for(const auto extension: info.enabled_device_extensions){
            uint32_t _;
            if(vkEnumerateDeviceExtensionProperties(physical_devices[i], extension, &_, {})== VK_ERROR_LAYER_NOT_PRESENT){
                scores[i].score = std::numeric_limits<int>::lowest();
                break;
            }
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
        ret.physical_device_index = std::max_element(scores.begin(), scores.end(), [](const score_index& l, const score_index& r){return l.score < r.score;})->index;
    
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
    auto available_device_features = util::vk::copy_features(info.device_features);
    vkGetPhysicalDeviceFeatures2(physical_device, &available_device_features.feature);

    std::set<uint32_t> distinct_queue_families{g_queue_family, c_queue_family, t_queue_family};
    std::map<uint32_t, std::unique_ptr<std::mutex>> distinct_mutexes;
    for(uint32_t family: distinct_queue_families)
        distinct_mutexes[family] = std::make_unique<std::mutex>();
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
    res = vkCreateDevice(physical_device, &device_create_info, allocation_callbacks, &device);
    util::check_vk_result(res);
    graphics_queue_family_index = g_queue_family;
    compute_queue_family_index = c_queue_family;
    transfer_queue_family_index = t_queue_family;
    vkGetDeviceQueue(device, g_queue_family, 0, &graphics_queue);
    vkGetDeviceQueue(device, c_queue_family, 0, &compute_queue);
    vkGetDeviceQueue(device, t_queue_family, 0, &transfer_queue);
    graphics_mutex = distinct_mutexes[g_queue_family].get();
    compute_mutex = distinct_mutexes[c_queue_family].get();
    transfer_mutex = distinct_mutexes[t_queue_family].get();
    for(auto& [queue, mutex]: distinct_mutexes)
        mutex_storage.push_back(std::move(mutex));

    VmaVulkanFunctions vulkan_functions = {};
    vulkan_functions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
    vulkan_functions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;
    
    VmaAllocatorCreateInfo allocator_create_info = {};
    allocator_create_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    allocator_create_info.vulkanApiVersion = VK_API_VERSION_1_2;
    allocator_create_info.physicalDevice = physical_device;
    allocator_create_info.device = device;
    allocator_create_info.instance = instance;
    allocator_create_info.pVulkanFunctions = &vulkan_functions;

    res = vmaCreateAllocator(&allocator_create_info, &allocator);
    util::check_vk_result(res);

    auto command_pool_info = util::vk::initializers::commandPoolCreateInfo(graphics_queue_family_index);
    general_graphics_command_pool = util::vk::create_command_pool(command_pool_info);
    std::array<VkDescriptorPoolSize, 1> pool_sizes{util::vk::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000)};
    auto descriptor_pool_info = util::vk::initializers::descriptorPoolCreateInfo(pool_sizes, 1000);
    general_descriptor_pool = util::vk::create_descriptor_pool(descriptor_pool_info);

    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    for(int i: util::i_range(memory_properties.memoryHeapCount)){
        if(memory_properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT){
            gpu_mem_size = memory_properties.memoryHeaps[i].size;
            break;
        }
    }

    return ret;
}

void vk_context::cleanup(){
    if(!physical_device)
        throw std::runtime_error("vk_context::cleanup() Context was already deleted. Missing call vk_context::init()");

    for(auto& pipeline: registered_pipelines)
        vkDestroyPipeline(device, pipeline, allocation_callbacks);
    registered_pipelines.clear();
    for(auto& pipeline_layout: registered_pipeline_layouts)
        vkDestroyPipelineLayout(device, pipeline_layout, allocation_callbacks);
    registered_pipeline_layouts.clear();
    for(auto& command_pool: registered_command_pools)
        vkDestroyCommandPool(device, command_pool, allocation_callbacks);
    registered_command_pools.clear();
    for(auto& descriptor_pool: registered_descriptor_pools)
        vkDestroyDescriptorPool(device, descriptor_pool, allocation_callbacks);
    registered_descriptor_pools.clear();
    for(auto& descriptor_set_layout: registered_descriptor_set_layouts)
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, allocation_callbacks);
    registered_descriptor_set_layouts.clear();
    for(auto& buffer_info: registered_buffers)
        vmaDestroyBuffer(allocator, buffer_info.buffer, buffer_info.allocation);
    registered_buffers.clear();
    for(auto& image_info: registered_images)
        vmaDestroyImage(allocator, image_info.image, image_info.allocation);
    registered_images.clear();
    for(auto& image_view: registered_image_views)
        vkDestroyImageView(device, image_view, allocation_callbacks);
    registered_image_views.clear();
    for(auto& render_pass: registered_render_passes)
        vkDestroyRenderPass(device, render_pass, allocation_callbacks);
    registered_render_passes.clear();
    for(auto& framebuffer: registered_framebuffer)
        vkDestroyFramebuffer(device, framebuffer, allocation_callbacks);
    registered_framebuffer.clear();
    for(auto& sampler: registered_sampler)
        vkDestroySampler(device, sampler, allocation_callbacks);
    registered_sampler.clear();
    for(auto& pipeline_cache: registered_pipeline_caches)
        vkDestroyPipelineCache(device, pipeline_cache, allocation_callbacks);
    registered_pipeline_caches.clear();
    for(auto& semaphore: registered_semaphores)
        vkDestroySemaphore(device, semaphore, allocation_callbacks);
    registered_semaphores.clear();
    for(auto& fence: registered_fences)
        vkDestroyFence(device, fence, allocation_callbacks);
    registered_fences.clear();


    vmaDestroyAllocator(allocator);
    allocator = {};
    vkDestroyDevice(device, allocation_callbacks);
    device = {};

    if(debug_report_callback){
        auto vkDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        assert(vkDestroyDebugUtilsMessengerEXT != NULL);
        vkDestroyDebugUtilsMessengerEXT(instance, debug_report_callback, allocation_callbacks);
        debug_report_callback = {};
    }

    transfer_queue = {};
    compute_queue = {};
    graphics_queue = {};
    physical_device = {};
    vkDestroyInstance(instance, allocation_callbacks);
    instance = {};
    
}

void vk_context::upload_to_staging_buffer(const util::memory_view<uint8_t> data){
    if(data.byte_size() > _staging_buffer_size){
        if(staging_buffer)
            util::vk::destroy_buffer(staging_buffer);
        
        auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, data.byte_size());
        auto alloc_create_info = util::vma::initializers::allocationCreateInfo(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);
        VmaAllocationInfo alloc_info{};
        staging_buffer = util::vk::create_buffer(buffer_info, alloc_create_info), &alloc_info;
        _staging_buffer_mappped = alloc_info.pMappedData;
    }

    memcpy(_staging_buffer_mappped, data.data(), data.byte_size());
}

settings_manager::settings_manager()
{
    load_settings(settings_file);
}

settings_manager::~settings_manager()
{
    store_settings(settings_file);
}

bool settings_manager::add_setting(const setting& s, bool autostore)
{
    if (s.id.empty() || s.type.empty()) return false;

    settings[s.id] = s;

    
    if(settings_type.count(s.id) > 0)
        settings_type[s.type].push_back(&settings[s.id]);

    if(autostore)
        store_settings(settings_file);

    return true;
}

bool settings_manager::delete_setting(std::string_view id)
{
    std::string sid(id);
    if (settings.find(sid) == settings.end()) return false;

    auto& s = settings[sid];
    int i = 0;
    for (; i < settings_type[s.type].size(); i++) {
        if (settings_type[s.type][i]->id == id)
            break;
    }

    settings_type[s.type][i] = settings_type[s.type][settings_type[s.type].size()-1];
    settings_type[s.type].pop_back();

    settings.erase(sid);
    return true;
}

settings_manager::setting& settings_manager::get_setting(std::string_view id)
{
    std::string sid;
    if (settings.find(sid) == settings.end()) return notFound;
    return settings[sid];
}

std::vector<settings_manager::setting*>* settings_manager::get_settings_type(std::string type)
{
    return &settings_type[type];
}

void settings_manager::store_settings(std::string_view filename)
{
    std::ofstream file(std::string(filename), std::ifstream::binary);
    for (auto& s : settings) {
        file << "\"" << s.second.id << "\"" << ' ' << "\"" << s.second.type << "\"" << ' ' << s.second.storage.size() << ' ';
        file.write(reinterpret_cast<char*>(s.second.storage.data()), s.second.storage.size());
        file << "\n";
    }
    file.close();
}

void settings_manager::load_settings(std::string_view filename)
{
    std::ifstream file(std::string(filename), std::ifstream::binary);

    if (!file.is_open()) {
        ::logger << "[warning] Settingsfile was not found or no settings exist. Creating empty settings" << logging::endl;
        return;
    }

    setting s = {};
    while (file >> s.id) {
        if (s.id[0] == '\"') {
            if (!(s.id[s.id.size() - 1] == '\"')) {
                s.id = s.id.substr(1);
                std::string nextWord;
                file >> nextWord;
                while (nextWord[nextWord.size() - 1] != '\"') {
                    s.id += " " + nextWord;
                    file >> nextWord;
                }
                s.id += " " + nextWord.substr(0, nextWord.size() - 1);
            }
            else {
                s.id = s.id.substr(1, s.id.size() - 2);
            }
        }
        
        file >> s.type;
        if (s.type[0] == '\"') {
            if (!(s.type[s.type.length() - 1] == '\"')) {
                s.type = s.type.substr(1);
                std::string nextWord;
                file >> nextWord;
                while (nextWord[nextWord.size() - 1] != '\"') {
                    s.type += " " + nextWord;
                    file >> nextWord;
                }
                s.type += " " + nextWord.substr(0, nextWord.size() - 1);
            }
            else {
                s.type = s.type.substr(1, s.type.size() - 2);
            }
        }
        uint32_t byte_length;
        file >> byte_length;
        s.storage = std::vector<uint8_t>(byte_length);
        file.get();
        file.read((char*)s.storage.data(), byte_length);
        file.get();
        if (s.id.size() != 0)
            add_setting(s, false);
    }

    file.close();
}

VkSampler persistent_samplers::get(const VkSamplerCreateInfo& sampler_info){
    if(globals::persistent_samplers._samplers.count(sampler_info) == 0)
        globals::persistent_samplers._samplers[sampler_info] = util::vk::create_sampler(sampler_info);
    return globals::persistent_samplers._samplers[sampler_info];
}

void stager::init(){
    auto fence_info = util::vk::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    _task_fences = {util::vk::create_fence(fence_info), util::vk::create_fence(fence_info)};
    auto command_pool_info = util::vk::initializers::commandPoolCreateInfo(globals::vk_context.transfer_queue_family_index);
    _command_pool = util::vk::create_command_pool(command_pool_info);
    _task_thread = std::thread(&stager::_task_thread_function, this);
}
void stager::cleanup(){
    _thread_finish = true;
    _task_semaphore.release();     // waking up the worker thread
    _task_thread.join();
    _task_thread = {};
}
void stager::add_staging_task(const staging_buffer_info& stage_info){
    std::scoped_lock lock(_task_add_mutex);
    _staging_tasks.push_back(std::make_unique<staging_buffer_info>(stage_info));
    _task_semaphore.release();
}
void stager::add_staging_task(const staging_image_info& stage_info){
    std::scoped_lock lock(_task_add_mutex);
    _staging_tasks.push_back(std::make_unique<staging_image_info>(stage_info));
    _task_semaphore.release();
}
void stager::set_staging_buffer_size(size_t size){
    _staging_buffer_size = util::align<size_t>(size, 512ul);
}
void stager::wait_for_completion(){
    ++_wait_completion;
    _task_semaphore.release();
    _completion_sempahore.acquire();    // waiting for execution thread to finish waiting
}
void stager::_task_thread_function(){
    auto transfer_buffer = [&](staging_buffer_info& cur){
        size_t buffer_size = _staging_buffer_size;
        size_t data_size = cur.transfer_dir == transfer_direction::upload ? cur.data_upload.size() : cur.data_download.size();
        for(structures::min_max<size_t> cur_span{0, buffer_size / 2}; cur_span.min < data_size && !_thread_finish; cur_span.min = cur_span.max, cur_span.max += buffer_size / 2, _fence_index = (++_fence_index) % _task_fences.size()){
            // wait for completion of previous transfer operation (otherwise we write to not yet copied data in the staging bufffer)
            auto res = vkWaitForFences(globals::vk_context.device, 1, &_task_fences[_fence_index], VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
            vkResetFences(globals::vk_context.device, 1, &_task_fences[_fence_index]);

            size_t copy_size = std::min<size_t>(cur_span.max - cur_span.min, data_size - cur_span.min);
            if(cur.transfer_dir == transfer_direction::upload)
                std::memcpy(_staging_buffer_mapped + _fence_index * buffer_size / 2, cur.data_upload.data() + cur_span.min, copy_size);


            if(_command_buffers[_fence_index])
                vkFreeCommandBuffers(globals::vk_context.device, _command_pool, 1, &_command_buffers[_fence_index]);
            _command_buffers[_fence_index] = util::vk::create_begin_command_buffer(_command_pool);

            VkBufferCopy buffer_copy{};
            buffer_copy.size = copy_size;
            if(cur.transfer_dir == transfer_direction::upload){
                buffer_copy.srcOffset = _fence_index * (buffer_size / 2);
                buffer_copy.dstOffset = cur.dst_buffer_offset + cur_span.min;
                vkCmdCopyBuffer(_command_buffers[_fence_index], _staging_buffer.buffer, cur.dst_buffer, 1, &buffer_copy);
            }
            else if(cur.transfer_dir == transfer_direction::download){
                buffer_copy.srcOffset = cur.dst_buffer_offset + cur_span.min;
                buffer_copy.dstOffset = _fence_index * (buffer_size / 2);
                vkCmdCopyBuffer(_command_buffers[_fence_index], cur.dst_buffer, _staging_buffer.buffer, 1, &buffer_copy);
            }
            auto wait_semaphores = cur_span.min < buffer_size ? cur.wait_semaphores: util::memory_view<VkSemaphore>{};
            auto wait_flags = cur_span.min < buffer_size ? cur.wait_flags : util::memory_view<uint32_t>{};
            auto signal_semaphores = cur_span.min + buffer_size >= data_size ? cur.signal_semaphores : util::memory_view<VkSemaphore>{};
            std::scoped_lock lock(*globals::vk_context.transfer_mutex);
            util::vk::end_commit_command_buffer(_command_buffers[_fence_index], globals::vk_context.transfer_queue, wait_semaphores, wait_flags, signal_semaphores, _task_fences[_fence_index]);

            if(cur.transfer_dir == transfer_direction::download){
                res = vkWaitForFences(globals::vk_context.device, 1, &_task_fences[_fence_index], VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
                std::memcpy(cur.data_download.data() + cur_span.min, _staging_buffer_mapped + _fence_index * buffer_size / 2, copy_size);
            }
        }
    };
    auto transfer_image = [&](staging_image_info& cur){
        size_t buffer_size = _staging_buffer_size;
        size_t data_size = cur.transfer_dir == transfer_direction::upload ? cur.data_upload.size() : cur.data_download.size();
        if(cur.image_extent.width * cur.bytes_per_pixel > buffer_size / 2){
            ::logger << "[error] stager::_task_thread_function()::transfer_image() image pixel row is larger than the staging buffer. Aborting upload" << logging::endl;
        }
        size_t upload_size = data_size;
        if(data_size > buffer_size / 2){
            // get max number of rows
            uint32_t row_size = cur.image_extent.width * cur.bytes_per_pixel;
            uint32_t max_rows = buffer_size / 2 / row_size;
            upload_size = max_rows * row_size;
        }
        for(structures::min_max<size_t> cur_span{0, upload_size}; cur_span.min < data_size && !_thread_finish; cur_span.min = cur_span.max, cur_span.max += upload_size, _fence_index = (++_fence_index) % _task_fences.size()){
            // wait for completion of previous transfer operation (otherwise we write to not yet copied data in the staging bufffer)
            auto res = vkWaitForFences(globals::vk_context.device, 1, &_task_fences[_fence_index], VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
            vkResetFences(globals::vk_context.device, 1, &_task_fences[_fence_index]);

            size_t copy_size = std::min(cur_span.max - cur_span.min, data_size - cur_span.min);
            if(cur.transfer_dir == transfer_direction::upload)
                std::memcpy(_staging_buffer_mapped + _fence_index * buffer_size / 2, cur.data_upload.data() + cur_span.min, copy_size);

            if(_command_buffers[_fence_index])
                vkFreeCommandBuffers(globals::vk_context.device, _command_pool, 1, &_command_buffers[_fence_index]);
            _command_buffers[_fence_index] = util::vk::create_begin_command_buffer(_command_pool);

            VkImageMemoryBarrier memory_barrier{};
            if(cur.transfer_dir == transfer_direction::upload)
                memory_barrier = util::vk::initializers::imageMemoryBarrier(cur.dst_image, VkImageSubresourceRange{cur.subresource_layers.aspectMask, cur.subresource_layers.mipLevel, 1, cur.subresource_layers.baseArrayLayer, cur.subresource_layers.layerCount}, VK_ACCESS_NONE, VK_ACCESS_NONE, cur.start_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            else if(cur.transfer_dir == transfer_direction::download)
                memory_barrier = util::vk::initializers::imageMemoryBarrier(cur.dst_image, VkImageSubresourceRange{cur.subresource_layers.aspectMask, cur.subresource_layers.mipLevel, 1, cur.subresource_layers.baseArrayLayer, cur.subresource_layers.layerCount}, VK_ACCESS_NONE, VK_ACCESS_NONE, cur.start_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            vkCmdPipelineBarrier(_command_buffers[_fence_index], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, {}, 0, {}, 0, {}, 1, &memory_barrier);

            VkBufferImageCopy image_copy{};
            image_copy.bufferOffset = _fence_index * (buffer_size / 2);
            image_copy.imageSubresource = cur.subresource_layers;
            image_copy.imageOffset = cur.image_offset;
            image_copy.imageOffset.z += cur_span.min / cur.image_extent.width / cur.image_extent.height / cur.bytes_per_pixel;
            image_copy.imageOffset.y += cur_span.min * cur.bytes_per_pixel % (cur.image_extent.width * cur.image_extent.height) / cur.image_extent.width;
            image_copy.imageExtent = cur.image_extent;
            image_copy.imageExtent.height = copy_size / cur.image_extent.width / cur.bytes_per_pixel;
            image_copy.imageExtent.depth = 1;

            if(cur.transfer_dir == transfer_direction::upload){
                vkCmdCopyBufferToImage(_command_buffers[_fence_index], _staging_buffer.buffer, cur.dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &image_copy);
                memory_barrier = util::vk::initializers::imageMemoryBarrier(cur.dst_image, VkImageSubresourceRange{cur.subresource_layers.aspectMask, cur.subresource_layers.mipLevel, 1, cur.subresource_layers.baseArrayLayer, cur.subresource_layers.layerCount}, VK_ACCESS_NONE, VK_ACCESS_NONE, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, cur.end_layout);
            }
            else if(cur.transfer_dir == transfer_direction::download){
                vkCmdCopyImageToBuffer(_command_buffers[_fence_index], cur.dst_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, _staging_buffer.buffer, 1, &image_copy);
                memory_barrier = util::vk::initializers::imageMemoryBarrier(cur.dst_image, VkImageSubresourceRange{cur.subresource_layers.aspectMask, cur.subresource_layers.mipLevel, 1, cur.subresource_layers.baseArrayLayer, cur.subresource_layers.layerCount}, VK_ACCESS_NONE, VK_ACCESS_NONE, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, cur.end_layout);
            }
            vkCmdPipelineBarrier(_command_buffers[_fence_index], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, {}, 0, {}, 0, {}, 1, &memory_barrier);

            auto wait_semaphores = cur_span.min < buffer_size ? cur.wait_semaphores: util::memory_view<VkSemaphore>{};
            auto wait_flags = cur_span.min < buffer_size ? cur.wait_flags : util::memory_view<uint32_t>{};
            auto signal_semaphores = cur_span.min + buffer_size >= data_size ? cur.signal_semaphores : util::memory_view<VkSemaphore>{};
            std::scoped_lock lock(*globals::vk_context.transfer_mutex);
            util::vk::end_commit_command_buffer(_command_buffers[_fence_index], globals::vk_context.transfer_queue, wait_semaphores, wait_flags, signal_semaphores, _task_fences[_fence_index]);

            if(cur.transfer_dir == transfer_direction::download){
                res = vkWaitForFences(globals::vk_context.device, 1, &_task_fences[_fence_index], VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
                std::memcpy(cur.data_download.data() + cur_span.min, _staging_buffer_mapped + _fence_index * buffer_size / 2, copy_size);
            }
        }
    };

    while(!_thread_finish){
        _task_semaphore.acquire();
        if(_thread_finish)
            return;

        // check staging buffer size change
        size_t buffer_size = _staging_buffer_size;
        if(!_staging_buffer || _staging_buffer.allocation->GetSize() != buffer_size){
            if(_staging_buffer)
                util::vk::destroy_buffer(_staging_buffer);
            auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, buffer_size);
            auto alloc_create_info = util::vma::initializers::allocationCreateInfo(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);
            VmaAllocationInfo alloc_info{};
            _staging_buffer = util::vk::create_buffer(buffer_info, alloc_create_info, &alloc_info);
            _staging_buffer_mapped = reinterpret_cast<uint8_t*>(alloc_info.pMappedData);
        }

        if(_staging_tasks.empty() && _wait_completion > 0){
            auto res = vkWaitForFences(globals::vk_context.device, 2, _task_fences.data(), VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
            auto release_count = _wait_completion.exchange(0);
            // acquiring other released _task_semaphores
            for(int i: util::i_range(release_count - 1))
                _task_semaphore.acquire();
            _completion_sempahore.release(release_count);
            continue;
        }

        // getting the next stage_info
        unique_task cur;
        {
            std::scoped_lock lock(_task_add_mutex);
            cur = std::move(_staging_tasks.front());
            _staging_tasks.erase(_staging_tasks.begin());
        }
        if(_thread_finish)
            return;

        if(staging_buffer_info* buffer = dynamic_cast<staging_buffer_info*>(cur.get()))
            transfer_buffer(*buffer);
        else
            transfer_image(*dynamic_cast<staging_image_info*>(cur.get()));

        if(cur->cpu_semaphore)
            cur->cpu_semaphore->release();

        if(cur->cpu_signal_flags.size()){
            auto res = vkWaitForFences(globals::vk_context.device, 1, &_task_fences[_fence_index], VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
            for(auto& b: cur->cpu_signal_flags)
                *b = true;
            ::logger << logging::info_prefix << " stager::_task_thrad_function() signaled " << cur->cpu_signal_flags.size() << " flags." << logging::endl;
        }
        
        if(cur->cpu_unsignal_flags.size()){
            auto res = vkWaitForFences(globals::vk_context.device, 1, &_task_fences[_fence_index], VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
            for(auto& b: cur->cpu_unsignal_flags)
                *b = false;
            ::logger << logging::info_prefix << " stager::_task_thrad_function() unsignaled " << cur->cpu_unsignal_flags.size() << " flags." << logging::endl;
        }
    }
}

file_loader::file_loader(){
    _task_thread = std::thread(&file_loader::_task_thread_function, this);
}
file_loader::~file_loader(){
    _thread_finish = true;
    _task_semaphore.release();
    _task_thread.join();
    _task_thread = {};
}
void file_loader::add_load_task(const load_info& info){
    std::scoped_lock lock(_task_add_mutex);
    _loading_tasks.push_back(std::make_unique<load_info>(info));
    _task_semaphore.release();
}
void file_loader::wait_for_completion(){
    ++_wait_completion;
    _task_semaphore.release();
    _completion_semaphore.acquire();        // waiting for task thread finish
}
void file_loader::_task_thread_function(){
    while(!_thread_finish){
        _task_semaphore.acquire();          // waiting for work/shutdown/completion_check
        if(_thread_finish)
            return;

        // notify wait_completion threads
        if(_loading_tasks.empty() && _wait_completion > 0){
            auto release_count = _wait_completion.exchange(0);
            for(int i: util::i_range(release_count - 1))
                _task_semaphore.acquire();
            _completion_semaphore.release(release_count);
            continue;
        }

        unique_task cur;
        {
            std::scoped_lock lock(_task_add_mutex);
            cur = std::move(_loading_tasks.front());
            _loading_tasks.erase(_loading_tasks.begin());
        }
        if(_thread_finish)
            return;
        
        // loading the data
        c_file file(cur->src, "rb");
        if(cur->src_offset)
            file.seek(cur->src_offset);
        auto read_bytes = file.read(cur->dst);
        if(read_bytes != cur->dst.byte_size())
            ::logger << logging::warning_prefix << " file_loader::_task_thread_function() Not all bytes for loading task were loaded. Loaded " << float(read_bytes) / (1<<20) << "MBytes from " << float(cur->dst.byte_size()) / (1<<20) << "MBytes requested." << logging::endl;
    }
}

void histogram_counter::init(){
    auto semaphore_info = util::vk::initializers::semaphoreCreateInfo();
    _last_count_semaphore = util::vk::create_semaphore(semaphore_info);
    auto pool_info = util::vk::initializers::commandPoolCreateInfo(globals::vk_context.compute_queue_family_index);
    _wait_semaphore_pool = util::vk::create_command_pool(pool_info);
    _wait_semaphore_command = util::vk::create_begin_command_buffer(_wait_semaphore_pool);
    auto res = vkEndCommandBuffer(_wait_semaphore_command); util::check_vk_result(res);
    _wait_semaphore_fence = util::vk::create_fence(util::vk::initializers::fenceCreateInfo());
    _task_thread = std::thread(&histogram_counter::_task_thread_function, this);
}
void histogram_counter::cleanup(){
    _thread_finish = true;
    _task_semaphore.release();
    _task_thread.join();
    _task_thread = {};
}
void histogram_counter::add_count_task(const histogram_count_info& count_info){
    std::scoped_lock lock(_task_add_mutex);
    _count_tasks.push_back(std::make_unique<histogram_count_info>(count_info));
    _task_semaphore.release();
}
void histogram_counter::wait_for_completion(){
    ++_wait_completion;
    _task_semaphore.release();
    _completion_semaphore.acquire();
}
void histogram_counter::_task_thread_function(){
    while(!_thread_finish){
        _task_semaphore.acquire();          // waiting for work/shutdown/completion_check
        if(_thread_finish)
            return;

        // notify wait_completion threads
        if(_count_tasks.empty() && _wait_completion > 0){
            auto release_count = _wait_completion.exchange(0);
            for(int i: util::i_range(release_count - 1))
                _task_semaphore.acquire();
            _completion_semaphore.release(release_count);
            continue;
        }

        unique_task cur;
        if(::logger.logging_level >= logging::level::l_5)
            ::logger << logging::info_prefix << " histogram_counter::_task_thread_function() starting histogram count" << logging::endl;
        {
            std::scoped_lock lock(_task_add_mutex);
            cur = std::move(_count_tasks.front());
            _count_tasks.erase(_count_tasks.begin());
        }
        if(_thread_finish)
            return;
        
        // counting the histogram
        // avoid rendering of histogram buffers and retrieving the list of histograms
        stopwatch stop_watch;
        std::set<std::string_view> histograms;
        {
            auto histogram_access = globals::drawlists()[cur->dl_id]().histogram_registry.access();
            histograms = std::move(histogram_access->change_request);
            histogram_access->change_request.clear();
            histogram_access->registrators_done = false;    // signaling that no registrator has done its changes
            for(auto hist: histograms){
                if(!histogram_access->name_to_registry_key.contains(hist))
                    continue;
                const auto& key = histogram_access->name_to_registry_key[hist];
                histogram_access->registry[key].registrator_signals.clear();
            }
            stop_watch.start();
        }
        auto& dl = globals::drawlists.ref_no_track()[cur->dl_id].ref_no_track();
        size_t dataset_size = dl.dataset_read().gpu_stream_infos ? dl.dataset_read().gpu_stream_infos->cur_block_size: dl.const_templatelist().data_size;
        
        // calulate priority distances if requested
        if(dl.delayed_ops.priority_rendering_requested){
            if(!dl.priority_colors_gpu || dl.priority_colors_gpu.size < dataset_size){
                if(dl.priority_colors_gpu)
                    util::vk::destroy_buffer(dl.priority_colors_gpu);
                auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, dataset_size);
                auto alloc_info = util::vma::initializers::allocationCreateInfo();
                dl.priority_colors_gpu = util::vk::create_buffer(buffer_info, alloc_info);
            }

            pipelines::distance_calculator::distance_info distance_info{};
            distance_info.data_type = dl.dataset_read().data_flags.data_typ;
            distance_info.data_size = dataset_size;
            distance_info.data_header_address = util::vk::get_buffer_address(dl.dataset_read().gpu_data.header);
            distance_info.index_buffer_address = util::vk::get_buffer_address(dl.const_templatelist().gpu_indices);
            distance_info.distances_address = util::vk::get_buffer_address(dl.priority_colors_gpu);
            distance_info.priority_attribute = globals::priority_center_attribute_id;
            distance_info.priority_center = globals::priority_center_vealue;
            distance_info.priority_distance = globals::priority_center_distance;
            pipelines::distance_calculator::instance().calculate(distance_info);
            pipelines::distance_calculator::instance().wait_for_fence();
        }

        // counting
        int count{};
        for(auto hist: histograms){
            auto key = dl.histogram_registry.const_access()->name_to_registry_key.at(hist);

            if(!dl.histogram_registry.const_access()->gpu_buffers.contains(hist)){
                size_t hist_count{1};
                for(int i: key.bin_sizes) hist_count *= std::abs(i);
                auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, hist_count * sizeof(uint32_t));
                auto alloc_info = util::vma::initializers::allocationCreateInfo();
                dl.histogram_registry.access()->gpu_buffers[hist] = util::vk::create_buffer(buffer_info, alloc_info);
            }

            std::vector<structures::min_max<float>> column_min_max(key.attribute_indices.size());
            for(int i: util::size_range(column_min_max))
                column_min_max[i] = dl.dataset_read().attributes[key.attribute_indices[i]].bounds.read();

            pipelines::histogram_counter::count_info count_info{};
            count_info.data_type = dl.dataset_read().data_flags.data_typ;
            count_info.data_size = dataset_size;
            count_info.data_header_address = util::vk::get_buffer_address(dl.dataset_read().gpu_data.header);
            count_info.index_buffer_address = util::vk::get_buffer_address(dl.const_templatelist().gpu_indices);
            count_info.gpu_data_activations = util::vk::get_buffer_address(dl.active_indices_bitset_gpu);
            count_info.histogram_buffer = dl.histogram_registry.const_access()->gpu_buffers.at(hist);
            count_info.priority_values_address = util::vk::get_buffer_address(dl.priority_colors_gpu);
            count_info.clear_counts = cur->clear_counts;
            count_info.column_indices = key.attribute_indices;
            count_info.bin_sizes = key.bin_sizes;
            count_info.column_min_max = column_min_max;
            if(key.is_max_histogram)
                count_info.reduction_type = structures::histogram_counter_structs::reduction_type_t::max;
            if(key.is_min_histogram)
                count_info.reduction_type = structures::histogram_counter_structs::reduction_type_t::min;
            
            if(cur->histograms_timing_info)
                count_info.gpu_timing_info = cur->histograms_timing_info;
            if(cur->count_timing_info && count == 0)
                count_info.gpu_timing_info = cur->count_timing_info;
            if(++count == histograms.size() - 1)
                count_info.gpu_sync_info.signal_semaphores = cur->signal_semaphores;
            pipelines::histogram_counter::instance().count(count_info);
        }
        // signaling that editing is done (at least all commands are being executed)
        if(cur->cpu_semaphore)
            cur->cpu_semaphore->release();

        // waiting for counting completion before signaling render
        pipelines::histogram_counter::instance().wait_for_fence();
            
        {
            auto histogram_access = globals::drawlists()[cur->dl_id]().histogram_registry.access();
            histogram_access->block_update_done = true;
            if(cur->last_count_of_dataset)
                histogram_access->dataset_update_done = true;
            auto& dl = globals::drawlists()[cur->dl_id]();
            if(dl.dataset_read().registry){
                dl.dataset_read().registry->access()->signal_registrator_used(*dl.dataset_registrator); 
            }
        }

        if(::logger.logging_level >= logging::level::l_5)
            ::logger << logging::info_prefix << " histogram_counter::_task_thread_function() hsitogram counts done, needed " << stop_watch.lap() << " ms" << logging::endl;
    }
}

priority_sorter::priority_sorter(){
    _task_thread = std::thread(&priority_sorter::_task_thread_function, this);
}
priority_sorter::~priority_sorter(){
    _thread_finish = true;
    _task_semaphore.release();
    _task_thread.join();
    _task_thread = {};
}
void priority_sorter::add_sort_task(const sorting_info& sorting_inf){
    std::scoped_lock lock(_task_add_mutex);
    _sort_tasks.push_back(std::make_unique<sorting_info>(sorting_inf));
    _task_semaphore.release();
}
void priority_sorter::wait_for_completion(){
    ++_wait_completion;
    _task_semaphore.release();
    _completion_semaphore.acquire();
}
void priority_sorter::_task_thread_function(){
    while(!_thread_finish){
        _task_semaphore.acquire();      // waiting for a task
        if(_thread_finish)
            return;

        // notify wait_completion thread
        if(_sort_tasks.empty() && _wait_completion > 0){
            auto release_count = _wait_completion.exchange(0);
            for(int i: util::i_range(release_count - 1))
                _task_semaphore.acquire();
            _completion_semaphore.release(release_count);
            continue;
        }

        unique_task cur;
        if(::logger.logging_level >= logging::level::l_4)
            ::logger << logging::info_prefix << " priority_sorter::_task_thread_function() starting priority sorting (including color calc)" << logging::endl;
        {
            std::scoped_lock lock(_task_add_mutex);
            cur = std::move(_sort_tasks.front());
            _sort_tasks.erase(_sort_tasks.begin());
        }
        if(_thread_finish)
            return;
        
        // starting to sort/count according to priority rendering
        const auto& dl = globals::drawlists.read().at(cur->dl_id).read();
        assert(dl.delayed_ops.priority_rendering_requested == true);
        if(dl.histogram_registry.const_access()->is_used()){
            assert(dl.histogram_registry.const_access()->dataset_update_done);
            const structures::histogram_registry_key* last_key{};
            for(const auto& [key, entry]: dl.histogram_registry.const_access()->registry) 
                if((key.is_max_histogram || key.is_min_histogram) && (key.attribute_indices.size() == 2 || key.attribute_indices.size() == 4)) last_key = &key;
            assert(last_key);
            std::vector<std::vector<uint32_t>> order_storage;
            const auto& registry = dl.histogram_registry.const_access()->registry;
            for(const auto& [key, entry]: registry){
                // downloading histograms (2d or 4d) and ordering each of them
                if(!key.is_max_histogram && !key.is_min_histogram && key.attribute_indices.size() != 2 && key.attribute_indices.size() != 4)
                    continue;
                size_t bins_amt{1};
                for(int s: key.bin_sizes) bins_amt *= s;
                std::vector<uint32_t> data(bins_amt);
                stager::staging_buffer_info buffer_info{};
                buffer_info.transfer_dir = stager::transfer_direction::download;
                buffer_info.dst_buffer = dl.histogram_registry.const_access()->gpu_buffers.at(std::string_view(entry.hist_id)).buffer;
                buffer_info.data_download = util::memory_view(data);
                globals::stager.add_staging_task(buffer_info);
                globals::stager.wait_for_completion();
                // sorting and uploading
                order_storage.emplace_back(data.size());
                std::vector<uint32_t>& ordered = order_storage.back();
                std::iota(ordered.begin(), ordered.end(), 0);
                //std::sort(ordered.begin(), ordered.end(), [&](uint32_t a, uint32_t b) {return data[a] < data[b];});
                radix::RadixSortMSDTransform(ordered.data(), ordered.size(), [&](uint32_t i){return data[i];}, 7);
                if(!dl.priority_indices.contains(entry.hist_id)){
                    auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, ordered.size() * sizeof(ordered[0]));
                    auto alloc_info = util::vma::initializers::allocationCreateInfo();
                    globals::drawlists.ref_no_track()[cur->dl_id].ref_no_track().priority_indices[entry.hist_id] = util::vk::create_buffer(buffer_info, alloc_info);
                }
                buffer_info.transfer_dir = stager::transfer_direction::upload;
                buffer_info.dst_buffer = dl.priority_indices.at(entry.hist_id).buffer;
                buffer_info.data_upload = util::memory_view(ordered);
                if(key == *last_key){
                    buffer_info.cpu_signal_flags = std::move(cur->cpu_signal_flags);
                    buffer_info.cpu_unsignal_flags = std::move(cur->cpu_unsignal_flags);
                }
                globals::stager.add_staging_task(buffer_info);
            }
            // waiting as order_storage has to be kept alive until everything was copied
            globals::stager.wait_for_completion();
        }
        else{
            std::string standard_string(globals::priority_drawlist_standard_order);
            // calculating priority colors and ordering the indices
            const auto& tl = dl.const_templatelist();
            const auto& data = std::get<structures::data<float>>(dl.dataset_read().cpu_data.read());
            const auto& dl_read = globals::drawlists.read().at(cur->dl_id).read();
            std::vector<uint8_t> color_index(tl.data_size);
            if(tl.flags.identity_indices){
                for(size_t i: util::size_range(color_index))
                    color_index[i] = std::abs(data(i, globals::priority_center_attribute_id) - globals::priority_center_vealue) / globals::priority_center_distance * 255 + .5;
            }
            else{
                for(size_t i: util::size_range(color_index))
                    color_index[i] = std::abs(data(tl.indices[i], globals::priority_center_attribute_id) - globals::priority_center_vealue) / globals::priority_center_distance * 255 + .5;
            }
            // uploading the colors
            if(!dl_read.priority_colors_gpu){
                auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, color_index.size() * sizeof(color_index[0]));
                auto alloc_info = util::vma::initializers::allocationCreateInfo();
                globals::drawlists.ref_no_track()[cur->dl_id].ref_no_track().priority_colors_gpu = util::vk::create_buffer(buffer_info, alloc_info);
            }
            structures::stager::staging_buffer_info staging_info{};
            staging_info.dst_buffer = dl_read.priority_colors_gpu.buffer;
            staging_info.data_upload = util::memory_view(color_index);
            globals::stager.add_staging_task(staging_info);

            std::vector<uint32_t> order(tl.data_size);
            std::iota(order.begin(), order.end(), 0);
            //std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b){return color_index[a] > color_index[b];});
            radix::RadixSortMSDTransform(order.data(), order.size(), [&](uint32_t i){return uint32_t(255 - color_index[i]);}, 7);

            if(!dl_read.priority_indices.contains(standard_string)){
                auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, order.size() * sizeof(order[0]));
                auto alloc_info = util::vma::initializers::allocationCreateInfo();
                globals::drawlists.ref_no_track()[cur->dl_id].ref_no_track().priority_indices[standard_string] = util::vk::create_buffer(buffer_info, alloc_info);
            }
            staging_info.dst_buffer = dl_read.priority_indices.at(standard_string).buffer;
            staging_info.data_upload = util::memory_view(order);
            staging_info.cpu_signal_flags = std::move(cur->cpu_signal_flags);
            staging_info.cpu_unsignal_flags = std::move(cur->cpu_unsignal_flags);
            globals::stager.add_staging_task(staging_info);
        }
        // the signal flags are set by the last uploadstaging task
        if(::logger.logging_level >= logging::level::l_4)
            ::logger << logging::info_prefix << " priority_sorter::_task_thread_function() priority ordering done" << logging::endl;
    }
}

std::atomic<registrator_id_t> histogram_registry::scoped_registrator_t::registrator_id_counter{};
std::atomic<registrator_id_t> dataset_registry::scoped_registrator_t::registrator_id_counter{};
}
