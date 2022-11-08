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
#include <imgui_internal.h>
#include <data_workbench.hpp>
#include <parallel_coordinates_workbench.hpp>
#include <frame_limiter.hpp>
#include "imgui_file_dialog/ImGuiFileDialog.h"
#include <dataset_util.hpp>
#include <stager.hpp>
#include <workbenches_util.hpp>
#include <imgui_globals.hpp>
#include <logger.hpp>
#include <global_descriptor_set_util.hpp>
#include <brusher.hpp>
#include <brush_util.hpp>
#include <sys_info.hpp>
#include <file_loader.hpp>
#include <histogram_counter_executor.hpp>

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_report(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,VkDebugUtilsMessageTypeFlagsEXT messageType,const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,void* pUserData)
{
    logger << logging::vulkan_validation_prefix << " " << pCallbackData->pMessage << logging::endl;
    return VK_FALSE;
}

int main(int argc, char* argv[]){
    // variables for all of the execution
    SDL_Window*                 window{};
    ImGui_ImplVulkanH_Window    imgui_window_data;
    constexpr int               min_image_count{2};
    const std::string_view      log_window_name{"log window"};

    // init global states (including imgui) ---------------------------------------------------------------------

    // command line parsing
    globals::commandline_parser.parse(util::memory_view(const_cast<const char**>(argv), static_cast<size_t>(argc)));
    if(globals::commandline_parser.isSet("help")){
        globals::commandline_parser.printHelp();
        return 0;
    }
#ifdef USEVKVALIDATIONLAYER
    globals::commandline_parser.set("vulkanvalidation");
#endif

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
    if(globals::commandline_parser.isSet("vulkanvalidation")){
        logger << "[info] Using vulkan validation layers" << logging::endl;
        instance_extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        instance_layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    // vulkan init
    std::vector<const char*> device_extensions{ VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_MAINTENANCE3_EXTENSION_NAME, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME};
    VkPhysicalDeviceVulkan12Features vk_12_features = util::vk::initializers::physicalDeviceVulkan12Features();
    vk_12_features.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceFeatures2 device_features = util::vk::initializers::physicalDeviceFeatures2();
    device_features.pNext = &vk_12_features;
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
        logger << "[info] Available GPUs: " << util::memory_view(chosen_gpu.physical_device_names) << logging::endl;
    logger << "[info] Chosen gpu: " << chosen_gpu.physical_device_names[chosen_gpu.physical_device_index] << logging::endl;
    logger << "[info] If different GPU should be chosen specify --gpu parameter in the command line arguments" << logging::endl;
    if(globals::commandline_parser.isSet("vulkanvalidation"))
        util::vk::setup_debug_report_callback(debug_report);

    // other globals init
    globals::sys_info.init();
    globals::stager.init();
    globals::histogram_counter.init();

    // imgui init
    if(SDL_Vulkan_CreateSurface(window, globals::vk_context.instance, &imgui_window_data.Surface) == SDL_FALSE){
        logger << "[error] SDL_Vulkan_CreateSurface() Failed to create Vulkan surface." << logging::endl;
        return -1;
    }
    int w, h; 
    SDL_GetWindowSize(window, &w, &h);
    VkBool32 supported; vkGetPhysicalDeviceSurfaceSupportKHR(globals::vk_context.physical_device, globals::vk_context.graphics_queue_family_index, imgui_window_data.Surface, &supported);
    if(!supported){
        logger<< "[error] vkGetPhysicalDeviceSurfaceSupportKHR WSI not suported on this physical device." << logging::endl;
        return -1;
    }
    const VkFormat requestSurfaceImageFormat[] = { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM };
    const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    imgui_window_data.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(globals::vk_context.physical_device, imgui_window_data.Surface, requestSurfaceImageFormat, (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat), requestSurfaceColorSpace);

    VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_IMMEDIATE_KHR };   // current workaround, otherwise on linux lagging
    imgui_window_data.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(globals::vk_context.physical_device, imgui_window_data.Surface, &present_modes[0], IM_ARRAYSIZE(present_modes));
    ImGui_ImplVulkanH_CreateOrResizeWindow(globals::vk_context.instance, globals::vk_context.physical_device, globals::vk_context.device, &imgui_window_data, globals::vk_context.graphics_queue_family_index, globals::vk_context.allocation_callbacks, w, h, min_image_count);
    
    //TODO: recreate export window (not yet setup for export)

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_ViewportsEnable;
    ImGui::GetIO().ConfigViewportsNoDecoration = false;
    std::vector<float> font_sizes{10.f, 15.f, 25.f};
    util::imgui::load_fonts("fonts/", font_sizes);
    ImGui::GetIO().FontDefault = ImGui::GetIO().Fonts->Fonts[1];
    if(globals::commandline_parser.isSet("printfontinfo"))
        logger << "[info] Amount of fonts available: " << ImGui::GetIO().Fonts->Fonts.size() / font_sizes.size() << logging::endl;

    ImGui_ImplSDL2_InitForVulkan(window);

    ImGui_ImplVulkan_InitInfo& init_info = globals::imgui.init_info;
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

    // uploading fonts
    auto setup_command_pool = util::vk::create_command_pool(util::vk::initializers::commandPoolCreateInfo(globals::vk_context.graphics_queue_family_index));
    auto setup_commands = util::vk::create_begin_command_buffer(setup_command_pool);
    auto setup_fence = util::vk::create_fence(util::vk::initializers::fenceCreateInfo());
    ImGui_ImplVulkan_CreateFontsTexture(setup_commands);
    util::vk::end_commit_command_buffer(setup_commands, globals::vk_context.graphics_queue, {}, {}, {}, setup_fence);
    auto res = vkWaitForFences(globals::vk_context.device, 1, &setup_fence, VK_TRUE, 20e9); util::check_vk_result(res);
    util::vk::destroy_fence(setup_fence);
    util::vk::destroy_command_pool(setup_command_pool);
    ImGui_ImplVulkan_DestroyFontUploadObjects();

    // workbenches setup
    util::workbench::setup_default_workbenches();
    util::global_descriptors::setup_default_descriptors();
    }

    // main loop ---------------------------------------------------------------------
    structures::frame_limiter   frame_limiter;
    ImGuiIO&                    io = ImGui::GetIO();
    bool                        done = false;
    bool                        rebuild_swapchain = false;
    bool                        first_frame = true;
    int                         swapchain_width = 0, swapchain_height = 0;
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
                globals::paths_to_open.push_back(std::string(event.drop.file));
                SDL_free(event.drop.file);              // Free dropped_filedir memory;
            }
        }

        // disable keyboard navigation if brushes are active
        if(globals::brush_edit_data.selected_ranges.size() && (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_NavEnableKeyboard))
            ImGui::GetIO().ConfigFlags ^= ImGuiConfigFlags_NavEnableKeyboard; //deactivate keyboard navigation
        if (globals::brush_edit_data.selected_ranges.empty() && !(ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_NavEnableKeyboard)) 
            ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; //enable keyboard navigation

        if(rebuild_swapchain && swapchain_width > 0 && swapchain_height > 0){
            ImGui_ImplVulkan_SetMinImageCount(min_image_count);
            ImGui_ImplVulkanH_CreateOrResizeWindow(globals::vk_context.instance, globals::vk_context.physical_device, globals::vk_context.device, &imgui_window_data, globals::vk_context.graphics_queue_family_index, globals::vk_context.allocation_callbacks, swapchain_width, swapchain_height, min_image_count);
            imgui_window_data.FrameIndex = 0;
        }   
            
        // start imgui frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame(window);
        ImGui::NewFrame();  

        // main docking window with menu bar
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGuiWindowFlags dockingWindow_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoSavedSettings;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::Begin("MainDockWindow", NULL, dockingWindow_flags);
        ImGui::PopStyleVar(3);
        ImGuiID main_dock_id = ImGui::GetID("MainDock");
        if (first_frame) {
            ImGui::DockBuilderRemoveNode(main_dock_id);
            ImGuiDockNodeFlags dockSpaceFlags = 0;
            dockSpaceFlags |= ImGuiDockNodeFlags_DockSpace;
            ImGui::DockBuilderAddNode(main_dock_id, dockSpaceFlags);
            ImGui::DockBuilderSetNodeSize(main_dock_id, {viewport->WorkSize.x, viewport->WorkSize.y});
            ImGuiID main_dock_bottom, main_dock_top, main_dock_lowest;
            ImGui::DockBuilderSplitNode(main_dock_id, ImGuiDir_Down, .3f, &main_dock_bottom, &main_dock_top);
            ImGui::DockBuilderSplitNode(main_dock_bottom, ImGuiDir_Down, .1f, &main_dock_lowest, &main_dock_bottom);
            ImGui::DockBuilderDockWindow(globals::primary_workbench->id.data(), main_dock_bottom);
            ImGui::DockBuilderDockWindow(globals::secondary_workbench->id.data(), main_dock_top);
            ImGui::DockBuilderDockWindow(log_window_name.data(), main_dock_lowest);
            ImGuiDockNode* node = ImGui::DockBuilderGetNode(main_dock_bottom);
            node->LocalFlags |= ImGuiDockNodeFlags_NoTabBar;
            node = ImGui::DockBuilderGetNode(main_dock_lowest);
            node->LocalFlags |= ImGuiDockNodeFlags_NoTabBar | ImGuiDockNodeFlags_NoDocking | ImGuiDockNodeFlags_NoResize;
            ImGui::DockBuilderFinish(main_dock_id);
        }
        auto id = ImGui::DockBuilderGetNode(main_dock_id)->SelectedTabId;
        ImGui::DockSpace(main_dock_id, {}, ImGuiDockNodeFlags_None);

        for(const auto& wb: globals::workbenches)
            wb->show();

        ImGui::Begin(log_window_name.data());
        ImGui::SetWindowFontScale(.8f);
        for(int i: util::i_range(logger.buffer_size)){
            auto last_line = logger.get_last_line(logger.buffer_size - 1 - i);
            if(last_line.empty())
                continue;

            if(std::string_view(last_line).substr(0, logging::warning_prefix.size()) == logging::warning_prefix)
                ImGui::TextColored({.8f, .8f, 0, 1}, "%s", last_line.c_str());
            else if(std::string_view(last_line).substr(0, logging::error_prefix.size()) == logging::error_prefix)
                ImGui::TextColored({.8f, 0, .2f, 1}, "%s", last_line.c_str());
            else if(std::string_view(last_line).substr(0, logging::vulkan_validation_prefix.size()) == logging::vulkan_validation_prefix)
                ImGui::TextColored({.8f, .8f, .8f, 1}, "%s", last_line.c_str());
            else
                ImGui::Text("%s", last_line.c_str());
        }
        ImGui::SetScrollHereY(1);
        ImGui::End();   // log window

        if(ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")){
            if(ImGuiFileDialog::Instance()->IsOk()){
                auto selection = ImGuiFileDialog::Instance()->GetSelection();                
                for(auto& [id, path]: selection)
                    globals::paths_to_open.push_back(path);
            }
            ImGuiFileDialog::Instance()->Close();
        }

        ImGui::End();   // main dock

        // check for dataset/drawlist/brush updates --------------------------------------------------------------------------------

        // updating the query attributes if they are not updated to files which should be opened, showing the open dialogue and handling loading
        util::dataset::check_datasets_to_open();

        // check for dataset deletions
        if(globals::datasets_to_delete.size()){
            // signaling all dependant workbenches
            std::vector<std::string_view> datasets(globals::datasets_to_delete.begin(), globals::datasets_to_delete.end());
            for(auto& workbench: globals::dataset_dependencies)
                workbench->remove_datasets(datasets);
            
            // adding all drawlists created from the datasets to the drawlist deletion list
            for(const auto& [dl_id, dl]: globals::drawlists.read()){
                if(globals::datasets_to_delete.count(dl.read().parent_dataset))
                    globals::drawlists_to_delete.insert(dl_id);
            }

            // deleting the datasets
            bool prev_dataset_state = globals::datasets.changed;
            for(auto& ds: globals::datasets_to_delete)
                globals::datasets().erase(ds);
            globals::datasets.changed = prev_dataset_state;
            globals::datasets_to_delete.clear();
        }

        // check for drawlist deletions
        if(globals::drawlists_to_delete.size()){
            // signaling all dependant workbenches
            std::vector<std::string_view> drawlists(globals::drawlists_to_delete.begin(), globals::drawlists_to_delete.end());
            for(auto& workbench: globals::drawlist_dataset_dependencies)
                workbench->remove_drawlists(drawlists);
            
            // deleting drawlists
            bool prev_drawlists_state = globals::drawlists.changed;
            for(auto& dl: globals::drawlists_to_delete)
                globals::drawlists().erase(dl);
            globals::drawlists.changed = prev_drawlists_state;
            globals::drawlists_to_delete.clear();
        }

        // check for dataset updates
        if(globals::datasets.changed){
            std::vector<std::string_view> changed_datasets;
            for(const auto& [ds_id, ds]: globals::datasets.read()){
                if(ds.changed)
                    changed_datasets.push_back(ds_id);
            }
            for(auto& workbench: globals::drawlist_dataset_dependencies)
                workbench->signal_dataset_update(changed_datasets, {});
            for(auto id: changed_datasets){
                globals::datasets.ref_no_track()[id].ref_no_track().clear_change();
                globals::datasets.ref_no_track()[id].changed = false;
            }
            globals::datasets.changed = false;
            // setting the changed flags on drawlists created from this dataset
            for(auto dl: changed_datasets){
                if(globals::drawlists.read().at(dl).read().histogram_registry.const_access()->name_to_registry_key.empty())
                    continue;
                globals::drawlists()[dl]().histogram_registry.access()->request_change_all();
            }
        }

        // updating activations for brushes
        util::brushes::upload_changed_brushes();
        util::brushes::update_drawlist_active_indices();
        // updating histograms for 
        if(globals::drawlists.changed){
            for(const auto& [dl_id, dl]: globals::drawlists.read()){
                if(!dl.changed)
                    continue;
                auto registry_access = globals::drawlists()[dl_id]().histogram_registry.access();   // automatically locks the registry to avoid multi threading problems
                if(registry_access->change_request.size()){
                    // updating the histograms
                    for(auto hist: registry_access->change_request)
                        globals::histogram_counter.add_count_task({dl_id});
                }
            }
        }
        // check for drwawlist updates
        if(globals::drawlists.changed){
            std::vector<std::string_view> changed_drawlists;
            for(const auto& [dl_id, dl]: globals::drawlists.read()){
                if(dl.changed)
                    changed_drawlists.push_back(dl_id);
            }
            for(auto& workbench: globals::drawlist_dataset_dependencies)
                workbench->signal_drawlist_update(changed_drawlists);
            for(auto id: changed_drawlists){
                globals::drawlists.ref_no_track()[id].ref_no_track().clear_change();
                globals::drawlists.ref_no_track()[id].changed = false;
            }
            globals::drawlists.changed = false;
        }

        // final app rendering ---------------------------------------------------------------------------
        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();
        const bool minimized = draw_data->DisplaySize.x <= 0 || draw_data->DisplaySize.y <= 0;
        {   // rendering scope
            std::scoped_lock lock(*globals::vk_context.graphics_mutex);
            if(!minimized)
                util::imgui::frame_render(&imgui_window_data, draw_data);

            if(io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable){
                ImGui::UpdatePlatformWindows();
                ImGui::RenderPlatformWindowsDefault();
            }

            if(!minimized)
                std::tie(rebuild_swapchain, swapchain_width, swapchain_height) = util::imgui::frame_present(&imgui_window_data, window);
        }

        frame_limiter.end_frame();
        first_frame = false;
    }
    auto res = vkDeviceWaitIdle(globals::vk_context.device); util::check_vk_result(res);
   
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    ImGui_ImplVulkanH_DestroyWindow(globals::vk_context.instance, globals::vk_context.device, &imgui_window_data, globals::vk_context.allocation_callbacks);
    globals::histogram_counter.cleanup();
    globals::stager.cleanup();
    globals::vk_context.cleanup();

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}