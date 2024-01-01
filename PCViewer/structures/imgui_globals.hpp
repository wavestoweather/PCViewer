#pragma once
#include <imgui_impl_vulkan.h>

namespace structures{
struct imgui_globals{
    ImGui_ImplVulkan_InitInfo init_info;    // contains all vulkan base resources used for imgui
};
}

namespace globals{
extern structures::imgui_globals imgui;
}