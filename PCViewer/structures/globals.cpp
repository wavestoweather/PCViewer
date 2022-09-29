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
#include <laod_behaviour.hpp>
#include <open_filepaths.hpp>
#include <stager.hpp>
#include <util.hpp>
#include <workbench_base.hpp>
#include <imgui_globals.hpp>
#include <logger.hpp>

// globals definition
structures::logger<20> logger{};

namespace globals{
structures::vk_context vk_context{};

datasets_t datasets{};

structures::drawlists_t drawlists{};
std::vector<std::string_view> selected_drawlists{};

structures::tracked_brushes global_brushes{};

structures::settings_manager settings_manager{};

structures::commandline_parser commandline_parser{};

robin_hood::unordered_map<std::string, structures::texture> textures{};

structures::persistent_samplers persistent_samplers{};

robin_hood::unordered_map<std::string_view, uniqe_descriptor_info> descriptor_sets{};

structures::load_behaviour load_behaviour{};

std::vector<std::string> paths_to_open{};
std::vector<structures::query_attribute> attribute_query{};

structures::stager stager{};

workbenches_t workbenches{};
structures::workbench* primary_workbench{};
structures::workbench* secondary_workbench{};
dataset_dependencies_t dataset_dependencies{};
drawlist_dataset_dependencies_t drawlist_dataset_dependencies{}; 

structures::imgui_globals imgui{};
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
    int bestFit{-1};
    uint32_t gpu_count;
    res = vkEnumeratePhysicalDevices(instance, &gpu_count, NULL);
    util::check_vk_result(res);

    ret.physical_device_names.resize(gpu_count);
    std::vector<VkPhysicalDevice> physical_devices(gpu_count);
    res = vkEnumeratePhysicalDevices(instance, &gpu_count, physical_devices.data());
    util::check_vk_result(res);

    struct score_index{long score; int index;};
    std::vector<score_index> scores(gpu_count);
    for(int i: util::i_range(gpu_count)){
        scores[i].index = i;

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
    auto available_device_features = util::vk::copy_features(info.device_features);
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
    res = vkCreateDevice(physical_device, &device_create_info, allocation_callbacks, &device);
    util::check_vk_result(res);
    graphics_queue_family_index = g_queue_family;
    compute_queue_family_index = c_queue_family;
    transfer_queue_family_index = t_queue_family;
    vkGetDeviceQueue(device, g_queue_family, 0, &graphics_queue);
    vkGetDeviceQueue(device, c_queue_family, 0, &compute_queue);
    vkGetDeviceQueue(device, t_queue_family, 0, &transfer_queue);

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
    if(data.byteSize() > _staging_buffer_size){
        if(staging_buffer)
            util::vk::destroy_buffer(staging_buffer);
        
        auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, data.byteSize());
        auto alloc_create_info = util::vma::initializers::allocationCreateInfo(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);
        VmaAllocationInfo alloc_info{};
        staging_buffer = util::vk::create_buffer(buffer_info, alloc_create_info), &alloc_info;
        _staging_buffer_mappped = alloc_info.pMappedData;
    }

    memcpy(_staging_buffer_mappped, data.data(), data.byteSize());
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
void stager::add_staging_task(const staging_info& stage_info){
    std::scoped_lock lock(_task_add_mutex);
    _staging_tasks.push_back(stage_info);
    _task_semaphore.release();
}
void stager::set_staging_buffer_size(size_t size){
    _staging_buffer_size = util::align(size, 512ul);
}
void stager::wait_for_completion(){
    _wait_completion = true;
    _task_semaphore.release();
    _completion_sempahore.acquire();    // waiting for execution thread to finish waiting
}
void stager::_task_thread_function(){
    while(!_thread_finish){
        _task_semaphore.acquire();
        if(_thread_finish)
            return;

        // check staging buffer size change
        size_t buffer_size = _staging_buffer_size;
        if(!_staging_buffer || _staging_buffer.allocation->GetSize() != buffer_size){
            if(_staging_buffer)
                util::vk::destroy_buffer(_staging_buffer);
            auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, buffer_size);
            auto alloc_create_info = util::vma::initializers::allocationCreateInfo(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);
            VmaAllocationInfo alloc_info{};
            _staging_buffer = util::vk::create_buffer(buffer_info, alloc_create_info, &alloc_info);
            _staging_buffer_mapped = alloc_info.pMappedData;
        }

        if(_staging_tasks.empty() && _wait_completion){
            auto res = vkWaitForFences(globals::vk_context.device, 2, _task_fences.data(), VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
            _wait_completion = false;
            _completion_sempahore.release(_completion_sempahore.peekCount());
            continue;
        }

        // getting the next stage_info
        staging_info cur;
        {
            std::scoped_lock lock(_task_add_mutex);
            cur = _staging_tasks.front();
            _staging_tasks.erase(_staging_tasks.begin());
        }
        if(_thread_finish)
            return;

        int fence_index = 0;    // also indicates which part of the staging buffer should be used (0 front, 1 back)
        size_t data_size = cur.transfer_dir == transfer_direction::upload ? cur.data_upload.size() : cur.data_download.size();
        for(structures::min_max<size_t> cur_span{0, buffer_size / 2}; cur_span.min < data_size && !_thread_finish; cur_span.min = cur_span.max, cur_span.max += buffer_size / 2, fence_index = (++fence_index) % _task_fences.size()){
            // wait for completion of previous transfer operation (otherwise we write to not yet copied data in the staging bufffer)
            auto res = vkWaitForFences(globals::vk_context.device, 1, &_task_fences[fence_index], VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
            vkResetFences(globals::vk_context.device, 1, &_task_fences[fence_index]);

            size_t copy_size = std::min(cur_span.max - cur_span.min, data_size - cur_span.min);
            if(cur.transfer_dir == transfer_direction::upload)
                std::memcpy(_staging_buffer_mapped, cur.data_upload.data() + cur_span.min, copy_size);


            if(_command_buffers[fence_index])
                vkFreeCommandBuffers(globals::vk_context.device, _command_pool, 1, &_command_buffers[fence_index]);
            _command_buffers[fence_index] = util::vk::create_begin_command_buffer(_command_pool);

            VkBufferCopy buffer_copy{};
            buffer_copy.size = copy_size;
            if(cur.transfer_dir == transfer_direction::upload){
                buffer_copy.srcOffset = fence_index * (buffer_size / 2);
                buffer_copy.dstOffset = cur.dst_buffer_offset + cur_span.min;
                vkCmdCopyBuffer(_command_buffers[fence_index], _staging_buffer.buffer, cur.dst_buffer, 1, &buffer_copy);
            }
            else if(cur.transfer_dir == transfer_direction::download){
                buffer_copy.srcOffset = cur.dst_buffer_offset + cur_span.min;
                buffer_copy.dstOffset = fence_index * (buffer_size / 2);
                vkCmdCopyBuffer(_command_buffers[fence_index], cur.dst_buffer, _staging_buffer.buffer, 1, &buffer_copy);
            }
            auto wait_semaphores = cur_span.min < buffer_size ? cur.wait_semaphores: util::memory_view<VkSemaphore>{};
            auto wait_flags = cur_span.min < buffer_size ? cur.wait_flags : util::memory_view<uint32_t>{};
            auto signal_semaphores = cur_span.min + buffer_size >= data_size ? cur.signal_semaphores : util::memory_view<VkSemaphore>{};
            std::scoped_lock lock(globals::vk_context.transfer_mutex);
            util::vk::end_commit_command_buffer(_command_buffers[fence_index], globals::vk_context.transfer_queue, wait_semaphores, wait_flags, signal_semaphores, _task_fences[fence_index]);

            if(cur.transfer_dir == transfer_direction::download){
                res = vkWaitForFences(globals::vk_context.device, 1, &_task_fences[fence_index], VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
                std::memcpy(cur.data_download.data() + cur_span.min, _staging_buffer_mapped, copy_size);
            }
        }
        if(cur.cpu_semaphore)
            cur.cpu_semaphore->release();
    }
}
}
