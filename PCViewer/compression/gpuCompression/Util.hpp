#pragma once
#include "RLHuffData.hpp" 
#include "../cpuCompression/BitStream.h"
#include "PackInc.hpp"

namespace vkCompress{
    // parse bytes to decode tables and rl data
    RLHuffDecodeDataCpu parseCpuRLHuffData(const GpuInstance* pInstance, const std::vector<uint>& data){
        BitStreamReadOnly bitStream(data.data(), data.size());
        // compacted symbols
        uint compactSymbolCount;
        bitStream.readAligned(&compactSymbolCount, 1);

        HuffmanDecodeTable decodeTable(pInstance);
        decodeTable.readFromBitStream(pInstance, bitStream);

        uint codewordBitsize;
        bitStream.readAligned(&codewordBitsize, 1);
        
        std::vector<uint8_t> codewordStream(codewordBitsize / 8);   // divide by 8 to get the byte size
        bitStream.readAligned(codewordStream.data(), codewordStream.size());

        uint offsetCount = (compactSymbolCount + pInstance->m_codingBlockSize - 1) / pInstance->m_codingBlockSize;
        std::vector<ushort> offsetsInc(offsetCount);
        bitStream.readAligned(offsetsInc.data(), offsetsInc.size());
        std::vector<uint> offsets(offsetsInc.size());
        unpackInc16CPU(offsets.data(), offsetsInc.data(), offsetsInc.size());

        // zero counts
        HuffmanDecodeTable zeroTable(pInstance);
        zeroTable.readFromBitStream(pInstance, bitStream);

        bitStream.readAligned(&codewordBitsize, 1);
        std::vector<uint8_t> zeroStream(codewordBitsize / 16);
        bitStream.readAligned(zeroStream.data(), zeroStream.size());

        std::vector<ushort> zeroOffsetsInc(offsetCount);
        bitStream.readAligned(zeroOffsetsInc.data(), zeroOffsetsInc.size());
        std::vector<uint> zeroOffsets(zeroOffsetsInc.size());
        unpackInc16CPU(zeroOffsets.data(), zeroOffsetsInc.data(), zeroOffsetsInc.size());

        return RLHuffDecodeDataCpu{
            compactSymbolCount, 
            std::move(offsets), 
            std::move(zeroOffsets), 
            std::move(decodeTable), 
            std::move(zeroTable), 
            std::move(codewordStream), 
            std::move(zeroStream)};
    }
}