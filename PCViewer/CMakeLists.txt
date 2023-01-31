cmake_minimum_required (VERSION 3.7)

SET( Eigen3_DIR "eigen-3.3.7" )
IF( NOT Eigen3_DIR )
    MESSAGE( FATAL_ERROR "Please point the environment variable Eigen3_DIR to the include directory of your Eigen3 installation.")
ENDIF()
INCLUDE_DIRECTORIES ( "${Eigen3_DIR}" )

IF (WIN32)
    message("Using custom FindSDL2.cmake for windows")
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/PCViewer/cmake/Win/")
ENDIF ()

IF (UNIX AND CMAKE_BUILD_TYPE MATCHES Debug)
    message("Enabling gcc stl debug checks")
    add_compile_options(-D_GLIBCXX_DEBUG)
ENDIF()

IF (CMAKE_BUILD_TYPE MATCHES "Debug")
    message("Enabling vulkan validation layers")
    add_compile_definitions(USEVKVALIDATIONLAYER)
ENDIF()

IF (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mavx512f -mavx512dq -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl")
ENDIF()
message("${CMAKE_CXX_FLAGS}")

# enable jpeg and png support for CImg library
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dcimg_use_png")

add_subdirectory("glm")

# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
set(THREADS_PREFER_PTHREAD_FLAG ON)

# check if libraries can be found, tries to search specific libraries in a specific libraries folder
find_package(Vulkan QUIET)
if(NOT DEFINED Vulkan_FOUND)
    set(lib_path "usr/lib/x86_64-linux-gnu")
    find_library(Vulkan_LIBRARY
    NAMES vulkan
    PATHS ${lib_path})

    find_library(ZLIB_LIBRARY
    NAMES z
    PATHS ${lib_path})

    find_library(PNG_LIBRARY
    NAMES png
    PATHS ${lib_path})

    find_library(X11_X11_LIB
    NAMES X11
    PATHS ${lib_path})
endif()

find_package(Vulkan COMPONENTS glslang glslc REQUIRED)
find_package(SDL2 REQUIRED)
find_package(netCDF CONFIG REQUIRED)
find_package(PNG REQUIRED)
find_package(Threads REQUIRED)
find_package(X11 REQUIRED)
find_package(TBB REQUIRED)
#find_package(Eigen3 3.3 REQUIRED NO_MODULE)

#include_directories(${Eigen3_DIR})

file(GLOB pcviewer_SRC
    "*.cpp"
)
list(REMOVE_ITEM pcviewer_SRC ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
    
file(GLOB imgui_SRC
    "imgui/*.h"
    "imgui/*.cpp"
    "imgui/*.hpp"
)
    
file(GLOB cimg_SRC
    "cimg/*.h"
)
    
file(GLOB tsne_SRC
    "tsne/*.h"
    "tsne/*.cpp"
)
    
file(GLOB rtree_SRC
    "rTree/*.h"
)
    
file(GLOB compress_SRC
    "compression/*.cpp"
    "compression/cpuCompression/*.cpp"
    "compression/gpuCompression/*.cpp"
)
    
file(GLOB largeVis_SRC
    "largeVis/*.cpp"
)
    
file(GLOB roaring_SRC
    "roaringc/src/*.c"
    "roaringc/src/containers/*.c"
)
    
set(roaring_HDR
    "roaringc/include/"
    "roaringc/cpp/"
)

file(GLOB vkMemory_SRC
    "vkMemory/*.cpp"
)

file(GLOB derive_SRC
    "deriveData/*.cpp"
)

file(GLOB imgui_nodes_SRC
    "imgui_nodes/*.cpp"
    "imgui_nodes/utilities/*.cpp"
)

file(GLOB imgui_file_dialog_SRC
    "imgui_file_dialog/*.cpp"
)

set(structures_INCLUDE_DIR
    "structures/"
    "util/"
    "half/"
    "vulkanMemoryAllocator/include/"
    "robin_hood_map"
    "imgui"
    "deriveData"
    "pipelines"
    "workbenches"
    "fast_float"
    "config"
    "radix_sort"
    "radix"
    "colorbrewer"
)

file(GLOB structures_SRC
    "structures/*.cpp"
)

file(GLOB workbenches_SRC
    "workbenches/*.cpp"
)

file(GLOB pipelines_SRC
    "pipelines/*.cpp"
)

file(GLOB util_SRC
    "util/*.cpp"
)

file(GLOB addditional_vulkan_SRC
    "vulkan/*.cpp"
)
include_directories( ${Vulkan_INCLUDE_DIR}; ${SDL2_INCLUDE_DIRS} ; ${THRUST_INCLUDE_DIR}; ${netCDF_INCLUDE_DIR}; ${PNG_INCLUDE_DIR}; ${roaring_HDR}; ${structures_INCLUDE_DIR})
add_executable (PCViewer ${pcviewer_SRC} ${util_SRC} ${imgui_SRC} ${cimg_SRC} ${tsne_SRC} ${rtree_SRC} ${compress_SRC} ${largeVis_SRC} ${roaring_SRC} ${vkMemory_SRC} ${derive_SRC} ${imgui_nodes_SRC} ${imgui_file_dialog_SRC} ${structures_SRC} ${workbenches_SRC} ${pipelines_SRC} ${addditional_vulkan_SRC})
add_executable (PCViewer_new main.cpp ${util_SRC} ${structures_SRC} ${workbenches_SRC} ${pipelines_SRC} ${imgui_SRC} ${cimg_SRC} ${tsne_SRC} ${rtree_SRC} ${derive_SRC} ${imgui_nodes_SRC} ${imgui_file_dialog_SRC} ${addditional_vulkan_SRC})

if(WIN32)
set(netCDF_LIBRARIES netCDF::netcdf)
endif()

target_link_libraries(PCViewer PRIVATE Vulkan::Vulkan Vulkan::glslang ${SDL2_LIBRARIES} glm ${netCDF_LIBRARIES} ${PNG_LIBRARIES} Threads::Threads ${X11_LIBRARIES} TBB::tbb)# Eigen3::Eigen
target_link_libraries(PCViewer_new PRIVATE Vulkan::Vulkan Vulkan::glslang ${SDL2_LIBRARIES} glm ${netCDF_LIBRARIES} ${PNG_LIBRARIES} Threads::Threads ${X11_LIBRARIES} TBB::tbb)# Eigen3::Eigen

add_subdirectory(shader_new)

add_subdirectory(test)

message (" Copying fonts")
file(GLOB FONTS
"${CMAKE_CURRENT_SOURCE_DIR}/fonts/*.ttf"
)
file(COPY ${FONTS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/fonts)
    