#ifndef __TUM3D_CUDACOMPRESS__ARITHMETIC_CPU_H__
#define __TUM3D_CUDACOMPRESS__ARITHMETIC_CPU_H__


#include "global.h"

#include <vector>

#include "BitStream.h"
#include "EncodeCommon.h"


namespace cudaCompress {

bool arithmeticEncodeCPU(BitStream& bitStream, const std::vector<Symbol16>& symbolStream, const std::vector<uint>& symbolProbabilitiesCum, std::vector<uint>& offsets, uint codingBlockSize);
bool arithmeticDecodeCPU(BitStreamReadOnly& bitStream, uint symbolCount, std::vector<Symbol16>& symbolStream, const std::vector<uint>& symbolProbabilitiesCum, const std::vector<uint>& offsets, uint codingBlockSize);

}


#endif
