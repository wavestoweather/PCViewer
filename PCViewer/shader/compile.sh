#!/bin/sh

glslangValidator -V shader.vert
glslangValidator -V shader.geom
glslangValidator -V shader.frag
glslangValidator -V -o fragUint.spv shaderUint.frag
glslangValidator -V -o pcResolve.comp.spv pcResolve.comp
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
glslangValidator -V -o brushLasso.spv brushLasso.comp
glslangValidator -V -o histComp.spv hist.comp
glslangValidator -V -o indexComp.spv index.comp
glslangValidator -V -o band.vert.spv band.vert
glslangValidator -V -o band.frag.spv band.frag
glslangValidator -V -o band.geom.spv band.geom
glslangValidator -V -o cluster_band.vert.spv cluster_band.vert
glslangValidator -V -o cluster_band.geom.spv cluster_band.geom
glslangValidator -V -o scatter.vert.spv scatter.vert
glslangValidator -V -o scatter.frag.spv scatter.frag
glslangValidator -V -o compr.vert.spv compr.vert
glslangValidator -V -o compr.geom.spv compr.geom
glslangValidator -V -o compr.frag.spv compr.frag
glslangValidator -V --target-env vulkan1.1 -o corrKendall.comp.spv corrKendall.comp
glslangValidator -V --target-env vulkan1.1 -o corrMean.comp.spv corrMean.comp
glslangValidator -V --target-env vulkan1.1 -o corrPearson.comp.spv corrPearson.comp
glslangValidator -V --target-env vulkan1.1 -o corrSpearman.comp.spv corrSpearman.comp

glslangValidator -V --target-env vulkan1.1 -o radixControl.comp.spv radixControl.comp
glslangValidator -V --target-env vulkan1.1 -o radixDispatch.comp.spv radixDispatch.comp
glslangValidator -V --target-env vulkan1.1 -o radixGlobalScan.comp.spv radixGlobalScan.comp
glslangValidator -V --target-env vulkan1.1 -o radixHistogram.comp.spv radixHistogram.comp
glslangValidator -V --target-env vulkan1.1 -o radixScatter.comp.spv radixScatter.comp
glslangValidator -V --target-env vulkan1.1 -o radixLocalSort.comp.spv radixLocalSort.comp
glslangValidator -V --target-env vulkan1.1 -o radixDsHistogram.comp.spv radixDsHistogram.comp
glslangValidator -V --target-env vulkan1.1 -o radixDsScatter.comp.spv radixDsScatter.comp
glslangValidator -V --target-env vulkan1.1 -o radixDsLocalSort.comp.spv radixDsLocalSort.comp

glslangValidator -V --target-env vulkan1.1 -o lineCount.comp.spv lineCount.comp
glslangValidator -V --target-env vulkan1.1 -o lineCountAll.comp.spv lineCountAll.comp
glslangValidator -V --target-env vulkan1.1 -o lineCount.vert.spv lineCount.vert
glslangValidator -V --target-env vulkan1.1 -o lineCount.frag.spv lineCount.frag

glslangValidator -V --target-env vulkan1.1 -o largeVis.vert.spv largeVis.vert
glslangValidator -V --target-env vulkan1.1 -o largeVis.geom.spv largeVis.geom
glslangValidator -V --target-env vulkan1.1 -o largeVis.frag.spv largeVis.frag

glslangValidator -V --target-env vulkan1.1 -o convertImageToUBuffer.comp.spv convertImageToUBuffer.comp

glslangValidator -V --target-env vulkan1.1 -o compressHuffman_decode.comp.spv compressHuffman_decode.comp
glslangValidator -V --target-env vulkan1.1 -o compress_decodeTranspose.comp.spv compress_decodeTranspose.comp
