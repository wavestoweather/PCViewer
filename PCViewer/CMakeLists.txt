# CMakeList.txt: CMake-Projekt für "PCViewer". Schließen Sie die Quelle ein, und definieren Sie
# projektspezifische Logik hier.
#
cmake_minimum_required (VERSION 3.7)

#message (" Eigen include dir ${Eigen3_DIR}")
add_subdirectory("imgui/glfw-3.3")

SET( Eigen3_DIR "$ENV{Eigen3_DIR}" )
IF( NOT Eigen3_DIR )
    message ("Eigen3 libary include path not set. Searching project subdirectories...")
    message ("checking for directory ${CMAKE_CURRENT_SOURCE_DIR}/eigen-master")
    IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/eigen-master")
        #add_subdirectory("eigen-master")
        #include_directories("eigen-master")
        message ("using subdirectory eigen-master for the Eigen3 library")
        SET( Eigen3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/eigen-master" )
    else()
        IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.3.7")
            #add_subdirectory("eigen-3.3.7")
            #include_directories("eigen-3.3.7")
            message ("using subdirectory eigen-3.3.7 for the Eigen3 library")
            SET( Eigen3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.3.7" )
        endif()
    endif()
endif()
message (" Eigen include dir ${Eigen3_DIR}")
IF( NOT Eigen3_DIR )
    MESSAGE( FATAL_ERROR "Please point the environment variable Eigen3_DIR to the include directory of your Eigen3 installation.")
ENDIF()
INCLUDE_DIRECTORIES ( "${Eigen3_DIR}" )




add_subdirectory("glm")

include_directories("imgui/glfw-3.3/include" ; ${Vulkan_INCLUDE_DIR} ; ${THRUST_INCLUDE_DIR})
#include_directories(${Eigen3_DIR})

# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
find_package(Vulkan REQUIRED)
#find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)





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

target_link_libraries(PCViewer Vulkan::Vulkan glfw glm)# Eigen3::Eigen

message (" Copying shaders")
file(GLOB MY_PUBLIC_HEADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/shader/*.spv"
)
file(COPY ${MY_PUBLIC_HEADERS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shader)

message (" Copying fonts")
file(GLOB FONTS
  "${CMAKE_CURRENT_SOURCE_DIR}/fonts/*.ttf"
)
file(COPY ${FONTS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/fonts)

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
