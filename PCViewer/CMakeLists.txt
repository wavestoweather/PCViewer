# CMakeList.txt: CMake-Projekt für "PCViewer". Schließen Sie die Quelle ein, und definieren Sie
# projektspezifische Logik hier.
#
cmake_minimum_required (VERSION 3.7)

# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
find_package(Vulkan REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory("imgui/glfw-3.3")
add_subdirectory("glm")
add_subdirectory("eigen-3.3.7")

include_directories("imgui/glfw-3.3/include" ; ${Vulkan_INCLUDE_DIR} ; ${THRUST_INCLUDE_DIR}; ${EIGEN_INCLUDE_DIR})

file(GLOB pcviewer_SRC
    "*.h"
    "*.cpp"
	"*.hpp"
)

file(GLOB imgui_SRC
    "imgui/*.h"
    "imgui/*.cpp"
	"imgui/*.hpp"
)

add_executable (PCViewer ${pcviewer_SRC} ${imgui_SRC})

target_link_libraries(PCViewer Vulkan::Vulkan glfw glm Eigen3::Eigen)

message (" Copying shaders")
file(GLOB MY_PUBLIC_HEADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/shader/*.spv"
)
file(COPY ${MY_PUBLIC_HEADERS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/vert.spv
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/frag.spv
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/densVert.spv
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/densFrag.spv
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/histVert.spv
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/histGeo.spv
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/histFrag.spv
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/rectVert.spv
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader/rectFrag.spv
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)
