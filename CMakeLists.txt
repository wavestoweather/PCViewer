# CMakeList.txt: CMake-Projektdatei der obersten Ebene. Führen Sie hier 
# die globale Konfiguration aus,
# und schließen Sie Unterprojekte ein.
cmake_minimum_required (VERSION 3.7)
set (CMAKE_CXX_STANDARD 17)
project ("PCViewer")

# Schließen Sie Unterprojekte ein.
add_subdirectory ("PCViewer")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 /bigobj" )

if(COMPILER_SUPPORTS_CXX17)
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
     message(STATUS "YES, C++ 2017 found")
elseif(COMPILER_SUPPORTS_CXX0X)
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
         message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++17 
support. Please use a different C++ compiler.")
endif()
