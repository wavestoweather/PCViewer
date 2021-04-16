#!/bin/sh

glslangValidator -V shader.vert
glslangValidator -V shader.geom
glslangValidator -V shader.frag
glslangValidator -V -o histVert.spv histo.vert
glslangValidator -V -o histFrag.spv histo.frag
glslangValidator -V -o histGeo.spv histo.geom
glslangValidator -V -o rectVert.spv rect.vert
glslangValidator -V -o rectFrag.spv rect.frag
glslangValidator -V -o densVert.spv dens.vert
glslangValidator -V -o densFrag.spv dens.frag
glslangValidator -V -o 3dVert.spv 3d.vert
glslangValidator -V -o 3dFrag.spv 3d.frag
glslangValidator -V -o 3dComp.spv 3d.comp
glslangValidator -V -o nodeVert.spv nodeViewer.vert
glslangValidator -V -o nodeGeom.spv node.geom
glslangValidator -V -o nodeFrag.spv nodeViewer.frag
glslangValidator -V -o isoSurfVert.spv isoSurf.vert
glslangValidator -V -o isoSurfFrag.spv isoSurf.frag
glslangValidator -V -o isoSurfDirectFrag.spv isoSurfDirect.frag
glslangValidator -V -o isoSurfComp.spv isoSurf.comp
glslangValidator -V -o isoSurfDirectComp.spv isoSurfDirect.comp
glslangValidator -V -o isoSurfActiveIndComp.spv isoSurfActiveInd.comp
glslangValidator -V -o isoSurfBinComp.spv isoSurfBin.comp
glslangValidator -V -o isoSurfSmooth.spv isoSurfSmooth.comp
glslangValidator -V -o isoSurfCopyOnes.spv isoSurfCopyOnes.comp
glslangValidator -V -o brushComp.spv brush.comp
glslangValidator -V -o brushFractureComp.spv brushFracture.comp
glslangValidator -V -o brushMultvarComp.spv brushMultvar.comp
glslangValidator -V -o histComp.spv hist.comp
glslangValidator -V -o indexComp.spv index.comp