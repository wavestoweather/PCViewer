
# PCViewer
### A detailed user documentation can be found in the [documentation section](/doc/overview.md).

Associated publication
https://arxiv.org/abs/2007.15446

This tool imports .csv files and displays them via Parallel Coordinates, Violin Plots and Isocontours.

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

## Compilation on Windows with VSCode
To compile on windows 64 bit with VSCode follow these steps:
- Follow the installation process on the VSCode website to install VSCode
- Install cmake
- Install vcpkg
- Install Vulkan from the [Vulkan website](https://vulkan.lunarg.com/sdk/home)
- Open a terminal to install packages with vcpkg:
```
    vcpkg install sdl2:x64-windows
    vcpkg install netcdf-c:x64-windows
    vcpkg install libpng:x64-windows
    vcpkg install tbb:x64-windows
    vcpkg integrate install
```
- In the visual studio project folder create a folder .vscode with a file settings.json and add the following to this file (exchange Toolchain_path with the path printed by the last output of the vcpkg command)
```
{
    "cmake.configureSettings": {
      "CMAKE_TOOLCHAIN_FILE": "Toolchain_path"
    }
}
```
- Open VSCode and setup build environment, care to select x64 build. If x86 is selected Vulkan will not be found as it is only installed in x64 configuration.
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

# Replicability instructions

For the replicability stamp please follow these commands to install the correct version of the program on an Ubuntu 22.04 LTS OS.

To install all needed libraries run the following commands
```
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.231-jammy.list https://packages.lunarg.com/vulkan/1.3.231/lunarg-vulkan-1.3.231-jammy.list
sudo apt update
sudo apt upgrade
sudo apt install build-essential cmake libsdl2-dev vulkan-sdk libnetcdf-dev libpng-dev libtbb-dev
```
To get the correct version of the program and building it execute the following commands in the folder where the application should be located:

```
wget https://github.com/wavestoweather/PCViewer/archive/refs/tags/v0.2-alpha.zip
unzip v0.2-alpha.zip
cd PCViewer-0.2-alpha
mkdir build
cd build
cmake ..
make -j4
```
The final application now is located in build/PCViewer/PCViewer.

Instructions for replicating the runtime test in video format can be found in [this video](https://drive.google.com/file/d/1z2bmqoyFM5wo3hU4uH_Pnp9wxHxg5jy3/view?usp=sharing), the associated synthetic data is available via this [link](https://mediatum.ub.tum.de/1690342).