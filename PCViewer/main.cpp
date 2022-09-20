#include <vk_context.hpp>
#include <memory_view.hpp>
#include <commandline_parser.hpp>
#include <imgui_impl_sdl.h>
#include <imgui_impl_vulkan.h>
#include <SDL.h>
#include <SDL_vulkan.h>
#include <vk_initializers.hpp>
#include <vk_util.hpp>
#include <data_workbench.hpp>
#include <laod_behaviour.hpp>
#include <imgui_util.hpp>
#include <data_workbench.hpp>
#include <parallel_coordinates_workbench.hpp>

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_report(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,VkDebugUtilsMessageTypeFlagsEXT messageType,const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,void* pUserData)
{
    std::cout << "[Vk validation] " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

std::tuple<std::vector<std::unique_ptr<structures::workbench>>, std::vector<structures::drawlist_dataset_dependency*>, std::vector<structures::drawlist_dataset_dependency*>, structures::workbench*>
setup_worbenches_datasetdeps_drawlist_deps(){
    std::vector<std::unique_ptr<structures::workbench>> workbenches{};
    std::vector<structures::drawlist_dataset_dependency*> dataset_dependecies{};
    std::vector<structures::drawlist_dataset_dependency*> drawlist_dependencies{};
    structures::workbench*                              main_workbench{};
    // register all available workbenches
    auto data_wb = std::make_unique<workbenches::data_workbench>("Data workbench");
    dataset_dependecies.push_back(data_wb.get());
    main_workbench = data_wb.get();
    workbenches.emplace_back(std::move(data_wb));

    auto parallel_coordinates_wb = std::make_unique<workbenches::parallel_coordinates_workbench>("Parallel coordinates workbench");
    dataset_dependecies.push_back(parallel_coordinates_wb.get());
    workbenches.emplace_back(std::move(parallel_coordinates_wb));

    return {std::move(workbenches), std::move(dataset_dependecies), std::move(drawlist_dependencies), main_workbench};
}

int main(int argc,const char* argv[]){
    // variables for all of the execution
    SDL_Window*                                         window ;
    ImGui_ImplVulkanH_Window                            imgui_window_data;
    std::vector<std::unique_ptr<structures::workbench>> workbenches{};
    std::vector<structures::drawlist_dataset_dependency*> dataset_dependecies{};
    std::vector<structures::drawlist_dataset_dependency*> drawlist_dependencies{};
    structures::workbench*                              main_workbench;

    // init global states (including imgui) ---------------------------------------------------------------------

    // command line parsing
    globals::commandline_parser.parse(util::memory_view(argv, static_cast<size_t>(argc)));
    if(globals::commandline_parser.isSet("help")){
        globals::commandline_parser.printHelp();
        return 0;
    }

    // sdl init
    {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
    {
        std::cout << "[error] " << SDL_GetError() << std::endl;
        return -1;
    }
    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_MAXIMIZED);
    window = SDL_CreateWindow("PCViewer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
    // Setup Drag and drop callback
	SDL_EventState(SDL_DROPFILE, SDL_ENABLE);
    uint32_t instance_extension_count = 0;
    SDL_Vulkan_GetInstanceExtensions(window, &instance_extension_count, NULL);
    std::vector<const char*> instance_extensions(instance_extension_count);
    SDL_Vulkan_GetInstanceExtensions(window, &instance_extension_count, instance_extensions.data());
    instance_extensions.push_back("VK_KHR_get_physical_device_properties2");
    std::vector<const char*> instance_layers;
#ifdef USEVKVALIDATIONLAYER
    std::cout << "[info] Using vulkan validation layers" << std::endl;
    instance_extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    instance_layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

    // vulkan init
    std::vector<const char*> device_extensions{ VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_MAINTENANCE3_EXTENSION_NAME, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME};
    VkPhysicalDeviceFeatures2 device_features = util::vk::initializers::physicalDeviceFeatures2();
    int physical_device_index = globals::commandline_parser.getValueAsInt("gpuselection", -1);
    structures::VkContextInitInfo vk_init{
        physical_device_index,
        VK_API_VERSION_1_2,
        "PCViewer",
        instance_layers,
        instance_extensions,
        device_extensions,
        device_features
    };
    auto chosen_gpu = globals::vk_context.init(vk_init);
    if(globals::commandline_parser.isSet("gpulist"))
        std::cout << "[info] Available GPUs: " << util::memory_view(chosen_gpu.physical_device_names) << std::endl;
    std::cout << "[info] Chosen gpu: " << chosen_gpu.physical_device_names[chosen_gpu.physical_device_index] << std::endl;
    std::cout << "[info] If different GPU should be chosen specify --gpu parameter in the command line arguments" << std::endl;
#ifdef USEVKVALIDATIONLAYER
    util::vk::setup_debug_report_callback(debug_report);
#endif

    // imgui init
    if(SDL_Vulkan_CreateSurface(window, globals::vk_context.instance, &imgui_window_data.Surface) == SDL_FALSE){
        std::cout << "[error] SDL_Vulkan_CreateSurface() Failed to create Vulkan surface." << std::endl;
        return -1;
    }
    int w, h; 
    SDL_GetWindowSize(window, &w, &h);
    VkBool32 supported; vkGetPhysicalDeviceSurfaceSupportKHR(globals::vk_context.physical_device, globals::vk_context.graphics_queue_family_index, imgui_window_data.Surface, &supported);
    if(!supported){
        std::cout << "[error] vkGetPhysicalDeviceSurfaceSupportKHR WSI not suported on this physical device." << std::endl;
        return -1;
    }
    const VkFormat requestSurfaceImageFormat[] = { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM };
	const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
	imgui_window_data.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(globals::vk_context.physical_device, imgui_window_data.Surface, requestSurfaceImageFormat, (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat), requestSurfaceColorSpace);

    VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_IMMEDIATE_KHR };   // current workaround, otherwise on linux lagging
    imgui_window_data.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(globals::vk_context.physical_device, imgui_window_data.Surface, &present_modes[0], IM_ARRAYSIZE(present_modes));
    constexpr int min_image_count = 2;
	ImGui_ImplVulkanH_CreateOrResizeWindow(globals::vk_context.instance, globals::vk_context.physical_device, globals::vk_context.device, &imgui_window_data, globals::vk_context.graphics_queue_family_index, globals::vk_context.allocation_callbacks, w, h, min_image_count);
    
    //TODO: recreate export window (not yet setup for export)

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    std::vector<float> font_sizes{5.f, 10.f, 15.f};
    util::imgui::load_fonts("fonts/", font_sizes);

    ImGui_ImplSDL2_InitForVulkan(window);

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = globals::vk_context.instance;
	init_info.PhysicalDevice = globals::vk_context.physical_device;
	init_info.Device = globals::vk_context.device;
	init_info.QueueFamily = globals::vk_context.graphics_queue_family_index;
	init_info.Queue = globals::vk_context.graphics_queue;
	init_info.PipelineCache = {};
	init_info.DescriptorPool = util::imgui::create_desriptor_pool();;
	init_info.Allocator = globals::vk_context.allocation_callbacks;
	init_info.MinImageCount = min_image_count;
	init_info.ImageCount = imgui_window_data.ImageCount;
	init_info.CheckVkResultFn = util::check_vk_result;
	ImGui_ImplVulkan_Init(&init_info, imgui_window_data.RenderPass);

    auto setup_command_pool = util::vk::create_command_pool(util::vk::initializers::commandPoolCreateInfo(globals::vk_context.graphics_queue_family_index));
    auto setup_commands = util::vk::create_begin_command_buffer(setup_command_pool);
    auto setup_fence = util::vk::create_fence(util::vk::initializers::fenceCreateInfo());
    ImGui_ImplVulkan_CreateFontsTexture(setup_commands);
    util::vk::end_commit_command_buffer(setup_commands, globals::vk_context.graphics_queue, {}, {}, {}, setup_fence);
    auto res = vkWaitForFences(globals::vk_context.device, 1, &setup_fence, VK_TRUE, 20e9); util::check_vk_result(res);
    util::vk::destroy_fence(setup_fence);
    util::vk::destroy_command_pool(setup_command_pool);

    // workbench setup
    std::tie(workbenches, dataset_dependecies, drawlist_dependencies, main_workbench) = setup_worbenches_datasetdeps_drawlist_deps();
    }

    // main loop ---------------------------------------------------------------------
    bool done = false;
    while(!done){
        // Poll and handle events (inputs, window resize, etc.)
		// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
		// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
		// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
		// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
		SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT)
                done = true;
            else if(event.type == SDL_DROPFILE) {       // In case if dropped file
                //droppedPaths.push_back(std::string(event.drop.file));
				//droppedPathActive.push_back(1);
                //pathDropped = true;
				//std::string file(event.drop.file);
				//if (droppedPaths.size() == 1) {
				//	queryAttributes = queryFileAttributes(event.drop.file);
				//}
                //SDL_free(event.drop.file);              // Free dropped_filedir memory;
            }
        }
    }

   
    ImGui_ImplVulkan_Shutdown();
	ImGui_ImplSDL2_Shutdown();
	ImGui::DestroyContext();

	ImGui_ImplVulkanH_DestroyWindow(globals::vk_context.instance, globals::vk_context.device, &imgui_window_data, globals::vk_context.allocation_callbacks);
    globals::vk_context.cleanup();

	SDL_DestroyWindow(window);
    SDL_Quit();

	return 0;
}