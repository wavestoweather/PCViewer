#ifndef __TUM3D_CUDACOMPRESS__RICE_GOLOMB_CPU_H__
#define __TUM3D_CUDACOMPRESS__RICE_GOLOMB_CPU_H__


#include "global.h"

#include <vector>

#include "BitStream.h"
#include "EncodeCommon.h"


namespace cudaCompress {

bool golombRiceEncodeCPU(BitStream& bitStream, const std::vector<Symbol16>& symbolStream, uint kBlockSize, std::vector<uint>& offsets, uint codingBlockSizeInKBlocks);
bool golombRiceDecodeCPU(BitStreamReadOnly& bitStream, uint symbolCount, std::vector<Symbol16>& symbolStream, uint kBlockSize);

}


#endif
