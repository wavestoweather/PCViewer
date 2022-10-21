
# PCViewer
### A detailed user documentation can be found in the [documentation section](/doc/overview.md).

Associated publications:
https://arxiv.org/abs/2007.15446
GPU Accelerated Scalable Parallel Coordinates Plots (Link will be added shortly)

This tool imports .csv files and displays them via Parallel Coordinates, Violin Plots and Isocontours.
To cope with very large data, please follow the guide in the [large vis docs](/doc/large_vis.md).
In the guide a test dataset is linked for verification of the method.

The UI and backend are currently under rework (see branch backend_rework) and the final program will also be released on this branch. Expect some bugs!

# Compilation notes
The program uses the Vulkan API as render back end, the SDL2 API as platform independent window back end and NetCDF API to open .nc files.
All of these packages have to be installed before compiliation and have to be found by cmake with `find_package()`. We advise to use package managers: For Linux use the preinstalled managers, for windows vcpkg can be used and chained to the PCViewer cmake project via cmake toolchain. For more information head over to [install vcpkg](https://vcpkg.readthedocs.io/en/latest/examples/installing-and-using-packages/).
### Vulkan
On Debian based distributions Vulkan can be installed via
```
    sudo apt update
    sudo apt install vulkan-sdk
```

For Windows using vcpkg Vulkan can be installed via
```
    vcpkg install vulkan:x64-windows
```
If no package manager is used head over to the [Vulkan-Website](https://vulkan.lunarg.com/sdk/home), download the latest installer and execute the installer.
### SDL2
For Debian based distributions SDL2 can be installed via
```
    sudo apt install cmake libsdl2-dev g++
```

For windows use the package manager
```
    vcpkg install sdl2:x64-windows
```
or head over to the [SDL-Website](https://www.libsdl.org/download-2.0.php) and download the developement package `SDL2-devel-xxx-VC.zip`
and extract it to some location on your hard disk.
The so downloaded SDL2 library is readily compiled and has only to be found by the FindSDL2.cmake file which comes with this project and is automatically used if windows is active.
In order for the cmake module to find SDL2 one has to create an environment variable called SDL2DIR which holds the directory where SDL2 was extracted.

### NetCDF
For Debian based distributions do
```
    sudo apt install libnetcdf-dev
```

For Windows using vcpkg do
```
    vcpkg install netcdf-c:x64-windows
```
### Small problems

It might be necessary to set the variable "Eigen2_DIR" to the directory of the eigen library in the folder (currently eigen-3.3.7). Further, a link to the Vulkan-SDK has to be set.

Windows:
For some cmake versions, the cmakes clash. The branch master_ak solves this clash by removing an uninstall line from the cmake of the eigen library. Not the best solution, but then it also works for those cmake versions...
Further, the /bigobj compile option is necessary, since the main .cpp file is a bit large...
To make it possible to render millions of lines, the TDR Timeout (https://docs.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys) has to be set to a larger number. The default is normally 2 seconds under windows. Be aware that chaning this means that you have to wait longer for feedback if a tool crashes because of a GPU code problem! It is advised to never set this value to infinity!
Be careful if you don't know what you are doing!

As this tool relies on the Vulkan API, make sure to have hardware capabale of running vulkan and all necessary drivers installed!

# License
Upon publication of the paper, our part of the code is avaiable under a permissive license (to be determined).
Please be aware of the licenses of included 3rd party code!
