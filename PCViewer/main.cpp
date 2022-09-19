#include <vk_context.hpp>
#include <memory_view.hpp>
#include <commandline_parser.hpp>
#include <imgui_impl_sdl.h>
#include <imgui_impl_vulkan.h>
#include <SDL.h>
#include <SDL_vulkan.h>
#include <vk_initializers.hpp>
#include <vk_util.hpp>

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_report(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,VkDebugUtilsMessageTypeFlagsEXT messageType,const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,void* pUserData)
{
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

int main(int argc,const char* argv[]){
    // init global states (including imgui) ---------------------------------------------------------------------
    // command line parsing
    globals::commandline_parser.parse(util::memory_view(argv, static_cast<size_t>(argc)));
    if(globals::commandline_parser.isSet("help"))
        globals::commandline_parser.printHelp();

    // sdl init
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
    {
        std::cout << "[error] " << SDL_GetError() << std::endl;
        return -1;
    }
    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_MAXIMIZED);
    SDL_Window* window = SDL_CreateWindow("PCViewer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
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
    //instance_extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    instance_layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

    // vulkan init
    std::vector<const char*> device_extensions;
    VkPhysicalDeviceFeatures2 device_features{};
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

    auto semaphore_info = util::vk::initializers::semaphoreCreateInfo();
    VkSemaphore s;
    vkCreateSemaphore(globals::vk_context.device, &semaphore_info, nullptr, &s);

    // main loop ---------------------------------------------------------------------
    while(false){

    }

   
    //ImGui_ImplVulkan_Shutdown();
	//ImGui_ImplSDL2_Shutdown();
	//ImGui::DestroyContext();

	//CleanupVulkanWindow();
    globals::vk_context.cleanup();

	SDL_DestroyWindow(window);
    SDL_Quit();

	return 0;
}