This tool imports .csv files and displays them via Parallel Coordinates, Violin Plots and Isocontours.


Compile notes:
Some things seem to work differently for Windows and Linux, which is why there are separate Branches for every OS.
The master branch should compile immediately under Windows, while the Linux branch is for Linux.

It might be necessary to set the variable "Eigen2_DIR" to the directory of the eigen library in the folder (currently eigen-3.3.7). Further, a link to the Vulkan-SDK has to be set. 



Windows:
For some cmake versions, the cmakes clash. The branch master_ak solves this clash by removing an uninstall line from the cmake of the eigen library. Not the best solution, but then it also works for those cmake versions...



Comments to Linux version:
The drag&drop between Drawlists in the PCViewer and the Violin Plots only works when they are in the same window. If they are in separate windows, first move the Violin Plots over the main PCViewer window or alternatively dock it into this window.

License:
Upon publication of the paper, our part of the code is avaiable under a permissive license (to be determined).
Please be aware of the licenses of included 3rd party code!