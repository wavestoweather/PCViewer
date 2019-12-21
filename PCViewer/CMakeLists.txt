# CMakeList.txt: CMake-Projekt für "PCViewer". Schließen Sie die Quelle ein, und definieren Sie
# projektspezifische Logik hier.
#
cmake_minimum_required (VERSION 3.7)

# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.

find_path( THRUST_INCLUDE_DIR
  HINTS
    /usr/include/cuda
    /usr/local/include
    /usr/local/cuda/include
    ${CUDA_INCLUDE_DIRS}
  NAMES thrust/version.h
  DOC "Thrust headers"
  )
if( THRUST_INCLUDE_DIR )
  list( REMOVE_DUPLICATES THRUST_INCLUDE_DIR )
endif( THRUST_INCLUDE_DIR )

# Find thrust version
file( STRINGS ${THRUST_INCLUDE_DIR}/thrust/version.h
  version
  REGEX "#define THRUST_VERSION[ \t]+([0-9x]+)"
  )
string( REGEX REPLACE
  "#define THRUST_VERSION[ \t]+"
  ""
  version
  "${version}"
  )

string( REGEX MATCH "^[0-9]" major ${version} )
string( REGEX REPLACE "^${major}00" "" version "${version}" )
string( REGEX MATCH "^[0-9]" minor ${version} )
string( REGEX REPLACE "^${minor}0" "" version "${version}" )
set( THRUST_VERSION "${major}.${minor}.${version}")
set( THRUST_MAJOR_VERSION "${major}")
set( THRUST_MINOR_VERSION "${minor}")

# Check for required components
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( Thrust
  REQUIRED_VARS THRUST_INCLUDE_DIR
  VERSION_VAR THRUST_VERSION
  )

set(THRUST_INCLUDE_DIRS ${THRUST_INCLUDE_DIR})
message(${THRUST_INCLUDE_DIR})

find_package(Vulkan REQUIRED)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory("imgui/glfw-3.3")
add_subdirectory("glm")

include_directories("imgui/glfw-3.3/include" ; ${Vulkan_INCLUDE_DIR} ; ${THRUST_INCLUDE_DIR})

add_executable (PCViewer "PCViewer.cpp" "CameraNav.hpp" "GpuBrusher.h" "PriorityColorUpdater.h" "kd_tree.h" "SettingsManager.cpp" "SettingsManager.h" "SpacialData.h" "BubblePlotter.h" "BubblePlotter.cpp" "Shape.h" "Cylinder.h" "Sphere.h" "PCViewer.h" "PCUtil.h" "PCUtil.cpp" "VkUtil.h" "VkUtil.cpp" "View3d.h" "View3d.cpp" "Color.h" "imgui/imconfig.h" "imgui/imgui.cpp" "imgui/imgui.h" "imgui/imgui_draw.cpp" "imgui/imgui_impl_glfw.cpp" "imgui/imgui_impl_glfw.h" "imgui/imgui_impl_vulkan.h" "imgui/imgui_impl_vulkan.cpp" "imgui/imgui_internal.h" "imgui/imgui_widgets.cpp" "imgui/imstb_rectpack.h" "imgui/imstb_textedit.h" "imgui/imstb_truetype.h" "imgui/imgui_demo.cpp")

target_link_libraries(PCViewer Vulkan::Vulkan glfw glm)

message (" Copying shaders")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/vert.spv
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/frag.spv
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/densVert.spv
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/densFrag.spv
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/histVert.spv
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/histGeo.spv
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/histFrag.spv
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/rectVert.spv
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/rectFrag.spv
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)