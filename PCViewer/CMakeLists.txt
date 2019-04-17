# CMakeList.txt: CMake-Projekt für "PCViewer". Schließen Sie die Quelle ein, und definieren Sie
# projektspezifische Logik hier.
#
cmake_minimum_required (VERSION 3.8)

# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
add_executable (PCViewer "PCViewer.cpp" "PCViewer.h" "imgui/imconfig.h" "imgui/imgui.cpp" "imgui/imgui.h" "imgui/imgui_draw.cpp" "imgui/imgui_impl_glfw.cpp" "imgui/imgui_impl_glfw.h" "imgui/imgui_impl_vulkan.h" "imgui/imgui_impl_vulkan.cpp" "imgui/imgui_internal.h" "imgui/imgui_widgets.cpp" "imgui/imstb_rectpack.h" "imgui/imstb_textedit.h" "imgui/imstb_truetype.h" "imgui/imgui_demo.cpp")

find_package(Vulkan REQUIRED)

add_subdirectory("imgui/glfw-3.3")

include_directories("imgui/glfw-3.3/include" ; ${Vulkan_INCLUDE_DIR})

target_link_libraries(PCViewer Vulkan::Vulkan glfw)

# TODO: Fügen Sie bei Bedarf Tests hinzu, und installieren Sie Ziele.
