#pragma once

#include "../cpuCompression/global.h"

#include "../cpuCompression/BitStream.h"

#include "../cpuCompression/EncodeCommon.h"
#include "RLHuffData.hpp"


namespace vkCompress {

class GpuInstance;

size_t encodeGetRequiredMemory(const GpuInstance* pInstance);
bool encodeInit(GpuInstance* pInstance);
bool encodeShutdown(GpuInstance* pInstance);

// note: all decode* functions assume that the bitstream memory is already page-locked; performace might degrade slightly if it isn't
// note: for decodeRLHuff, the output arrays (pdpSymbolStreams) must be zeroed!

//FIXME: encodeHuff modifies pdpSymbolStreams (adds padding for histogram)
//FIXME: decodeHuff expects its output arrays (pdpSymbolStreams) to be padded to a multiple of WARP_SIZE*codingBlockSize

// single bitstream for all blocks
bool encodeRLHuff(GpuInstance* pInstance, BitStream& bitStream, const Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);
bool decodeRLHuff(GpuInstance* pInstance, BitStreamReadOnly& bitStream, Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);

bool encodeHuff(GpuInstance* pInstance, BitStream& bitStream, /*const*/ Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);
bool decodeHuff(GpuInstance* pInstance, BitStreamReadOnly& bitStream, Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);

bool encodeRLHuff(GpuInstance* pInstance, BitStream& bitStream, const Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);
bool decodeRLHuff(GpuInstance* pInstance, BitStreamReadOnly& bitStream, Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);

bool encodeHuff(GpuInstance* pInstance, BitStream& bitStream, /*const*/ Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);
bool decodeHuff(GpuInstance* pInstance, BitStreamReadOnly& bitStream, Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);

// vulkan decode rlhuff with only a single bit stream and also only a single block for decoding
bool decodeRLHuff(GpuInstance* pInstance, BitStreamReadOnly& bitStream, std::vector<Symbol16>& symbolStream);
bool decodeRLHuff(GpuInstance* pInstance, VkBuffer bitStreamBuffer, BitStream& currentBitStream, size_t curBitStreamOffset, VkBuffer symbolBuffer, uint symbolSize, VkCommandBuffer commands);
bool decodeRLHuff(GpuInstance* pInstance, const RLHuffDecodeDataCpu& decodeDataCpu, const RLHuffDecodeDataGpu& decodDataGpu, VkCommandBuffer commands);

// separate bitstream for each block (but may contain duplicates)
bool encodeRLHuff(GpuInstance* pInstance, BitStream* ppBitStreams[], const Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);
bool decodeRLHuff(GpuInstance* pInstance, BitStreamReadOnly* ppBitStreams[], Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);

bool encodeHuff(GpuInstance* pInstance, BitStream* ppBitStreams[], /*const*/ Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);
bool decodeHuff(GpuInstance* pInstance, BitStreamReadOnly* ppBitStreams[], Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);

bool encodeRLHuff(GpuInstance* pInstance, BitStream* ppBitStreams[], const Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);
bool decodeRLHuff(GpuInstance* pInstance, BitStreamReadOnly* ppBitStreams[], Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);

bool encodeHuff(GpuInstance* pInstance, BitStream* ppBitStreams[], /*const*/ Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);
bool decodeHuff(GpuInstance* pInstance, BitStreamReadOnly* ppBitStreams[], Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);


//void encodeResetBitCounts();
//void encodePrintBitCounts();

}