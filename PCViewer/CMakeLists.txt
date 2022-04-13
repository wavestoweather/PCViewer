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

# enable jpeg and png support for CImg library
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dcimg_use_png")

add_subdirectory("glm")

# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Vulkan REQUIRED)
find_package(SDL2 REQUIRED)
find_package(netCDF CONFIG REQUIRED)
find_package(PNG REQUIRED)
find_package(Threads REQUIRED)
find_package(X11 REQUIRED)
#find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message("include dirs: ${netCDF_INCLUDE_DIR}")

#include_directories(${Eigen3_DIR})

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

include_directories( ${Vulkan_INCLUDE_DIR}; ${SDL2_INCLUDE_DIRS} ; ${THRUST_INCLUDE_DIR}; ${netCDF_INCLUDE_DIR}; ${PNG_INCLUDE_DIR}; ${roaring_HDR})
add_executable (PCViewer ${pcviewer_SRC} ${imgui_SRC} ${cimg_SRC} ${tsne_SRC} ${rtree_SRC} ${compress_SRC} ${largeVis_SRC} ${roaring_SRC})

if(WIN32)
set(netCDF_LIBRARIES netCDF::netcdf)
endif()

target_link_libraries(PCViewer PRIVATE Vulkan::Vulkan ${SDL2_LIBRARIES} glm ${netCDF_LIBRARIES} ${PNG_LIBRARIES} Threads::Threads ${X11_LIBRARIES})# Eigen3::Eigen

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
    