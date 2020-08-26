Associated publication
https://arxiv.org/abs/2007.15446


This tool imports .csv files and displays them via Parallel Coordinates, Violin Plots and Isocontours.

# Compilation notes
The program uses the Vulkan API as render back end and the SDL2 API as platform independent window back end.
Both of these have to be installed and findable by cmake with `find_package()`
## Vulkan
For Linux Vulkan can be installed via
```
    sudo apt update
    sudo apt install vulkan-sdk
```

For Windows head over to the [Vulkan-Website](https://vulkan.lunarg.com/sdk/home), download the latest installer and execute the installer.
## SDL2
For Linux SDL2 can be installed via
```
    sudo apt install cmake libsdl2-dev g++
```

For windows head over to the [SDL-Website](https://www.libsdl.org/download-2.0.php) and download the developement package `SDL2-devel-xxx-VC.zip`
and extract it to some location on your hard disk. To build we recommend using CMake GUI.
On first try the CMake build will fail. To resolve simply point the `SDL2_DIR` variable to the location where SDL2 was extracted.
Before reconfiguring a file `sdl2-config.cmake` has to be created in the folder where the extracted devolepment librariers are put with the following content:
```cmake
set(SDL2_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include")

# Support both 32 and 64 bit builds
if (${CMAKE_SIZEOF_VOID_P} MATCHES 8)
  set(SDL2_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/lib/x64/SDL2.lib;${CMAKE_CURRENT_LIST_DIR}/lib/x64/SDL2main.lib")
else ()
  set(SDL2_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/lib/x86/SDL2.lib;${CMAKE_CURRENT_LIST_DIR}/lib/x86/SDL2main.lib")
endif ()

string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)
```
after this everything should be set to compile.
For an alternative solution using a `FindSDL2.cmake` visit [Trenki's Dev Blog](https://trenki2.github.io/blog/2017/06/02/using-sdl2-with-cmake/).

Compile notes:
Some things seem to work differently for Windows and Linux, which is why there are separate Branches for every OS.
The master branch should compile immediately under Windows, while the Linux branch is for Linux.

It might be necessary to set the variable "Eigen2_DIR" to the directory of the eigen library in the folder (currently eigen-3.3.7). Further, a link to the Vulkan-SDK has to be set. 



Windows:
For some cmake versions, the cmakes clash. The branch master_ak solves this clash by removing an uninstall line from the cmake of the eigen library. Not the best solution, but then it also works for those cmake versions...
Further, the /bigobj compile option is necessary, since the main .cpp file is a bit large...
To make it possible to render millions of lines, the TDR Timeout (https://docs.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys) has to be set to a larger number. The default is normally 2 seconds under windows. Be aware that chaning this means that you have to wait longer for feedback if a tool crashes because of a GPU code problem! It is advised to never set this value to infinity!
Be careful if you don't know what you are doing!

As this tool relies on the Vulkan API, make sure to have hardware capabale of running vulkan and all necessary drivers installed!


Comments to Linux version:
The drag&drop between Drawlists in the PCViewer and the Violin Plots only works when they are in the same window. If they are in separate windows, first move the Violin Plots over the main PCViewer window or alternatively dock it into this window. This seems to be a problem of the currently used ImGUI version. Updating the files could help.

License:
Upon publication of the paper, our part of the code is avaiable under a permissive license (to be determined).
Please be aware of the licenses of included 3rd party code!