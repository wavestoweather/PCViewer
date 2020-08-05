Associated publication
https://arxiv.org/abs/2007.15446


This tool imports .csv files and displays them via Parallel Coordinates, Violin Plots and Isocontours.


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

The tool does not behave well with multiple screens under Linux, which might be fixable with a later ImGUI version as well. If you encounter device lost vulkan errors, it's probably one of the two causes: Not all windows were on the same display all the time or the watchdog killed the tool (TDR) since rendering took too long and the timeout was exceeded.


License:
Upon publication of the paper, our part of the code is avaiable under a permissive license (to be determined).
Please be aware of the licenses of included 3rd party code!