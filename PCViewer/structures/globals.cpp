#define VMA_IMPLEMENTATION  // vma allocations cpu
#include <vk_mem_alloc.h>
#include <datasets.hpp>
#include "vk_context.hpp"
#include <vk_util.hpp>
#include <ranges.hpp>
#include <brushes.hpp>
#include <drawlists.hpp>
#include <settings_manager.hpp>
#include <fstream>

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
    vkGetDeviceQueue(device, g_queue_family, 0, &graphics_queue);
    vkGetDeviceQueue(device, c_queue_family, 0, &compute_queue);
    vkGetDeviceQueue(device, t_queue_family, 0, &transfer_queue);

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
    for(auto& image_info: registered_images){
        vmaDestroyImage(allocator, image_info.image, image_info.allocation);
        if(image_info.image_view)
            vkDestroyImageView(device, image_info.image_view, allocation_callbacks);
    }
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

    vmaDestroyAllocator(allocator);
    allocator = {};
    vkDestroyDevice(device, allocation_callbacks);
    device = {};
    transfer_queue = {};
    compute_queue = {};
    graphics_queue = {};
    physical_device = {};
    instance = {};
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
		std::cout << "Settingsfile was not found or no settings exist." << std::endl;
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
}

namespace globals{
structures::vk_context vk_context{};

datasets_t datasets{};

structures::drawlists_t drawlists{};

structures::tracked_brushes global_brushes{};

structures::settings_manager settings_manager{};
}
