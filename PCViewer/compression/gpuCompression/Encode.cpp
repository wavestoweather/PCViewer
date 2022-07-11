#include "Encode.hpp"

#include "GpuInstance.hpp"

#include "../cpuCompression/util.h"
#include "HuffmanTable.h"
#include "Huffman.hpp"
#include "PackInc.hpp"
#include "VkPointer.hpp"
#include "RunLength.hpp"

namespace vkCompress {
//using namespace cudaCompress;
//static uint g_bitsSymbolEncodeTables = 0;
//static uint g_bitsSymbolCodewords = 0;
//static uint g_bitsSymbolOffsets = 0;
//static uint g_bitsZeroCountEncodeTables = 0;
//static uint g_bitsZeroCountCodewords = 0;
//static uint g_bitsZeroCountOffsets = 0;
//static uint g_bitsTotal = 0;
//static uint g_totalEncodedCount = 0;


static uint getNumUintsForBits(uint bitsize)
{
    uint bitsPerUint = sizeof(uint) * 8;
    return (bitsize + bitsPerUint - 1) / bitsPerUint;
}

static uint getNumUintsForBytes(uint bytesize)
{
    uint bytesPerUint = sizeof(uint);
    return (bytesize + bytesPerUint - 1) / bytesPerUint;
}

static uint getNumOffsets(uint symbolCount, uint codingBlockSize)
{
    return (symbolCount + codingBlockSize - 1) / codingBlockSize;
}


size_t encodeGetRequiredMemory(const GpuInstance* pInstance)
{
    bool longSymbols = (pInstance->m_log2HuffmanDistinctSymbolCountMax > 16);
    uint symbolSize = longSymbols ? sizeof(Symbol32) : sizeof(Symbol16);

    uint streamCountMax = pInstance->m_streamCountMax;
    uint symbolCountPerStreamMax = pInstance->m_elemCountPerStreamMax;

    uint symbolStreamMaxBytes = symbolCountPerStreamMax * symbolSize;
    uint offsetStreamMaxBytes = getNumOffsets(symbolCountPerStreamMax, pInstance->m_codingBlockSize) * sizeof(uint);

    uint distinctSymbolCountMax = 1 << pInstance->m_log2HuffmanDistinctSymbolCountMax;

    size_t size = 0;

    // streamInfo.dpSymbolStream - compacted symbols
    size += streamCountMax * getAlignedSize(symbolStreamMaxBytes, 128);
    // streamInfo.dpSymbolStream - zero counts
    size += streamCountMax * getAlignedSize(symbolStreamMaxBytes, 128);

    // streamInfo.dpCodewordStream
    size += streamCountMax * getAlignedSize(symbolStreamMaxBytes, 128);
    // streamInfo.dpOffsets
    size += streamCountMax * getAlignedSize(offsetStreamMaxBytes, 128);

    // streamInfo.dpEncodeCodeWords
    size += streamCountMax * getAlignedSize(distinctSymbolCountMax * sizeof(uint), 128);
    // streamInfo.dpEncodeCodeWordLengths
    size += streamCountMax * getAlignedSize(distinctSymbolCountMax * sizeof(uint), 128);

    // streamInfo.dpDecodeTable
    size += streamCountMax * getAlignedSize(HuffmanDecodeTable::computeMaxGPUSize(pInstance), 128);

    return size;
}

bool encodeInit(GpuInstance* pInstance)
{
    // currently no encoding will be implemented
    return false;
    bool longSymbols = (pInstance->m_log2HuffmanDistinctSymbolCountMax > 16);
    uint symbolSize = longSymbols ? sizeof(Symbol32) : sizeof(Symbol16);

    uint streamCountMax = pInstance->m_streamCountMax;
    uint symbolCountPerStreamMax = pInstance->m_elemCountPerStreamMax;


    uint symbolStreamMaxBytes = symbolCountPerStreamMax * symbolSize;
    uint offsetStreamMaxBytes = getNumOffsets(symbolCountPerStreamMax, pInstance->m_codingBlockSize) * sizeof(uint);
    uint distinctSymbolCountMax = 1 << pInstance->m_log2HuffmanDistinctSymbolCountMax;

    uint symbolStreamMaxBytesAligned = (uint)getAlignedSize(symbolStreamMaxBytes, 128);
    uint offsetStreamMaxBytesAligned = (uint)getAlignedSize(offsetStreamMaxBytes, 128);

    uint symbolStreamMaxElemsAligned = uint(symbolStreamMaxBytesAligned / sizeof(uint));
    uint offsetStreamMaxElemsAligned = uint(offsetStreamMaxBytesAligned / sizeof(uint));

    uint distinctSymbolCountMaxBytesAligned = (uint)getAlignedSize(distinctSymbolCountMax * sizeof(uint), 128);

    /*
    cudaSafeCall(cudaMallocHost(&pInstance->Encode.pCodewordBuffer, 2 * streamCountMax * symbolStreamMaxBytesAligned));
    cudaSafeCall(cudaMallocHost(&pInstance->Encode.pOffsetBuffer,   2 * streamCountMax * offsetStreamMaxBytesAligned));

    cudaSafeCall(cudaMallocHost(&pInstance->Encode.pEncodeCodewords,       2 * streamCountMax * distinctSymbolCountMaxBytesAligned));
    cudaSafeCall(cudaMallocHost(&pInstance->Encode.pEncodeCodewordLengths, 2 * streamCountMax * distinctSymbolCountMaxBytesAligned));

    cudaSafeCall(cudaMallocHost(&pInstance->Encode.pEncodeSymbolStreamInfos, 2 * streamCountMax * sizeof(HuffmanGPUStreamInfo)));


    pInstance->Encode.symbolEncodeTables.reserve(streamCountMax);
    pInstance->Encode.zeroCountEncodeTables.reserve(streamCountMax);
    for(uint i = 0; i < streamCountMax; i++) {
        pInstance->Encode.symbolEncodeTables.push_back(HuffmanEncodeTable(pInstance));
        pInstance->Encode.zeroCountEncodeTables.push_back(HuffmanEncodeTable(pInstance));
    }

    cudaSafeCall(cudaEventCreate(&pInstance->Encode.encodeFinishedEvent, cudaEventDisableTiming));
    cudaSafeCall(cudaEventRecord(pInstance->Encode.encodeFinishedEvent));


    for(int i = 0; i < pInstance->Encode.ms_decodeResourcesCount; i++) {
        GpuInstance::EncodeResources::DecodeResources& res = pInstance->Encode.Decode[i];

        res.symbolDecodeTables.reserve(streamCountMax);
        res.zeroCountDecodeTables.reserve(streamCountMax);
        for(uint i = 0; i < streamCountMax; i++) {
            res.symbolDecodeTables.push_back(HuffmanDecodeTable(pInstance));
            res.zeroCountDecodeTables.push_back(HuffmanDecodeTable(pInstance));
        }

        size_t decodeTableSizeMax = getAlignedSize(HuffmanDecodeTable::computeMaxGPUSize(pInstance), 128);
        cudaSafeCall(cudaMallocHost(&res.pSymbolDecodeTablesBuffer, streamCountMax * decodeTableSizeMax));
        cudaSafeCall(cudaMallocHost(&res.pZeroCountDecodeTablesBuffer, streamCountMax * decodeTableSizeMax));

        // we read from these in huffmanDecode, so don't alloc as write combined
        cudaSafeCall(cudaMallocHost(&res.pSymbolStreamInfos, streamCountMax * sizeof(HuffmanGPUStreamInfo)));
        cudaSafeCall(cudaMallocHost(&res.pZeroCountStreamInfos, streamCountMax * sizeof(HuffmanGPUStreamInfo)));

        cudaSafeCall(cudaEventCreate(&res.syncEvent, cudaEventDisableTiming));
        cudaSafeCall(cudaEventRecord(res.syncEvent));

        cudaSafeCall(cudaMallocHost(&res.pCodewordStreams, streamCountMax * symbolStreamMaxBytes));
        cudaSafeCall(cudaMallocHost(&res.pSymbolOffsets, streamCountMax * offsetStreamMaxBytes));
        cudaSafeCall(cudaMallocHost(&res.pZeroCountOffsets, streamCountMax * offsetStreamMaxBytes));
    }

    return true;
    */
}

bool encodeShutdown(GpuInstance* pInstance)
{
    return false;
    /*
    for(int i = 0; i < pInstance->Encode.ms_decodeResourcesCount; i++) {
        GpuInstance::EncodeResources::DecodeResources& res = pInstance->Encode.Decode[i];

        cudaSafeCall(cudaFreeHost(res.pZeroCountOffsets));
        res.pZeroCountOffsets = nullptr;
        cudaSafeCall(cudaFreeHost(res.pSymbolOffsets));
        res.pSymbolOffsets = nullptr;
        cudaSafeCall(cudaFreeHost(res.pCodewordStreams));
        res.pCodewordStreams = nullptr;

        cudaSafeCall(cudaEventDestroy(res.syncEvent));
        res.syncEvent = 0;

        cudaSafeCall(cudaFreeHost(res.pZeroCountStreamInfos));
        res.pZeroCountStreamInfos = nullptr;
        cudaSafeCall(cudaFreeHost(res.pSymbolStreamInfos));
        res.pSymbolStreamInfos = nullptr;

        cudaSafeCall(cudaFreeHost(res.pZeroCountDecodeTablesBuffer));
        res.pZeroCountDecodeTablesBuffer = nullptr;
        cudaSafeCall(cudaFreeHost(res.pSymbolDecodeTablesBuffer));
        res.pSymbolDecodeTablesBuffer = nullptr;

        res.zeroCountDecodeTables.clear();
        res.symbolDecodeTables.clear();
    }


    cudaSafeCall(cudaEventDestroy(pInstance->Encode.encodeFinishedEvent));
    pInstance->Encode.encodeFinishedEvent = 0;

    pInstance->Encode.zeroCountEncodeTables.clear();
    pInstance->Encode.symbolEncodeTables.clear();

    cudaSafeCall(cudaFreeHost(pInstance->Encode.pEncodeSymbolStreamInfos));
    pInstance->Encode.pEncodeSymbolStreamInfos = nullptr;

    cudaSafeCall(cudaFreeHost(pInstance->Encode.pEncodeCodewords));
    pInstance->Encode.pEncodeCodewords = nullptr;
    cudaSafeCall(cudaFreeHost(pInstance->Encode.pEncodeCodewordLengths));
    pInstance->Encode.pEncodeCodewordLengths = nullptr;

    cudaSafeCall(cudaFreeHost(pInstance->Encode.pOffsetBuffer));
    pInstance->Encode.pOffsetBuffer = nullptr;
    cudaSafeCall(cudaFreeHost(pInstance->Encode.pCodewordBuffer));
    pInstance->Encode.pCodewordBuffer = nullptr;

    return true;
    */
}

template<typename Symbol>
bool encodeRLHuff(GpuInstance* pInstance, BitStream* ppBitStreams[], bool singleBitStream, const Symbol* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    // not yet implemented (currently only decoding)
    return false;
    /*
    uint symbolStreamMaxBytes = symbolCountPerStream * sizeof(Symbol);
    uint offsetStreamMaxBytes = getNumOffsets(symbolCountPerStream, pInstance->m_codingBlockSize) * sizeof(uint);

    uint distinctSymbolCountMax = 1 << pInstance->m_log2HuffmanDistinctSymbolCountMax;

    uint symbolStreamMaxBytesAligned = (uint)getAlignedSize(symbolStreamMaxBytes, 128);
    uint offsetStreamMaxBytesAligned = (uint)getAlignedSize(offsetStreamMaxBytes, 128);
    uint distinctSymbolCountMaxAligned = (uint)getAlignedSize(distinctSymbolCountMax, 128 / sizeof(uint));

    uint symbolStreamMaxElemsAligned = uint(symbolStreamMaxBytesAligned / sizeof(uint));
    uint offsetStreamMaxElemsAligned = uint(offsetStreamMaxBytesAligned / sizeof(uint));

    // get GPU buffers from pInstance
    std::vector<Symbol*> pdpSymbolStreamsCompacted(streamCount);
    std::vector<Symbol*> pdpZeroCounts(streamCount);
    std::vector<HuffmanGPUStreamInfo> pStreamInfos(streamCount);
    uint* dpCodewordStreams       = pInstance->getBuffer<uint>(streamCount * symbolStreamMaxElemsAligned);
    uint* dpOffsets               = pInstance->getBuffer<uint>(streamCount * offsetStreamMaxElemsAligned);
    uint* dpEncodeCodewords       = pInstance->getBuffer<uint>(streamCount * distinctSymbolCountMaxAligned);
    uint* dpEncodeCodewordLengths = pInstance->getBuffer<uint>(streamCount * distinctSymbolCountMaxAligned);
    for(uint block = 0; block < streamCount; block++) {
        pdpSymbolStreamsCompacted[block] = (Symbol*)pInstance->getBuffer<Symbol>(symbolCountPerStream);
        pdpZeroCounts[block]             = (Symbol*)pInstance->getBuffer<Symbol>(symbolCountPerStream);

        HuffmanGPUStreamInfo& streamInfo = pStreamInfos[block];

        streamInfo.dpCodewordStream = dpCodewordStreams + block * symbolStreamMaxElemsAligned;
        streamInfo.dpOffsets        = dpOffsets         + block * offsetStreamMaxElemsAligned;

        // dpEncodeCodewords and dpEncodeCodewordLengths will be filled later
    }

    cudaSafeCall(cudaMemsetAsync(dpCodewordStreams, 0, streamCount * symbolStreamMaxBytesAligned, pInstance->m_stream));

    cudaSafeCall(cudaEventSynchronize(pInstance->Encode.encodeFinishedEvent));

    util::CudaScopedTimer timerLow(pInstance->Encode.timerEncodeLowDetail);
    util::CudaScopedTimer timerHigh(pInstance->Encode.timerEncodeHighDetail);

    timerLow("Run Length Encode");

    // run length encode
    std::vector<uint> symbolCountsPerBlock(streamCount, symbolCountPerStream);
    std::vector<uint> symbolCountsCompact(streamCount);
    runLengthEncode(pInstance, pdpSymbolStreamsCompacted.data(), pdpZeroCounts.data(), (const Symbol**)pdpSymbolStreams, symbolCountsPerBlock.data(), streamCount, ZERO_COUNT_MAX, symbolCountsCompact.data());

    timerLow("Huffman Encode Symbols");

    timerHigh("Symbols:    Design Huffman Tables");

    for(uint block = 0; block < streamCount; block++) {
        // padding for histogram (which wants the element count to be a multiple of 8)
        histogramPadData(pInstance, pdpSymbolStreamsCompacted[block], symbolCountsCompact[block]);
        histogramPadData(pInstance, pdpZeroCounts[block],             symbolCountsCompact[block]);
    }

    // 1. compacted symbols
    // build encode tables
    std::vector<HuffmanEncodeTable>& symbolEncodeTables = pInstance->Encode.symbolEncodeTables;
    if(!HuffmanEncodeTable::design(pInstance, symbolEncodeTables.data(), streamCount, (const Symbol**)pdpSymbolStreamsCompacted.data(), symbolCountsCompact.data())) {
        pInstance->releaseBuffers(4 + 2 * streamCount);
        return false;
    }

    timerHigh("Symbols:    Upload Huffman Tables");

    // fill stream infos
    uint* dpEncodeCodewordsNext = dpEncodeCodewords;
    uint* dpEncodeCodewordLengthsNext = dpEncodeCodewordLengths;
    uint* pEncodeCodewordsNext = pInstance->Encode.pEncodeCodewords;
    uint* pEncodeCodewordLengthsNext = pInstance->Encode.pEncodeCodewordLengths;
    for(uint block = 0; block < streamCount; block++) {
        HuffmanGPUStreamInfo& streamInfo = pStreamInfos[block];

        streamInfo.dpSymbolStream = (byte*)pdpSymbolStreamsCompacted[block];
        streamInfo.symbolCount = symbolCountsCompact[block];

        streamInfo.dpEncodeCodewords       = dpEncodeCodewordsNext;
        streamInfo.dpEncodeCodewordLengths = dpEncodeCodewordLengthsNext;

        symbolEncodeTables[block].copyToBuffer(pEncodeCodewordsNext, pEncodeCodewordLengthsNext);

        size_t elems = symbolEncodeTables[block].getTableSize();
        pEncodeCodewordsNext        += elems;
        pEncodeCodewordLengthsNext  += elems;
        dpEncodeCodewordsNext       += elems;
        dpEncodeCodewordLengthsNext += elems;
    }

    // upload encode tables
    size_t encodeCodewordElems = pEncodeCodewordsNext - pInstance->Encode.pEncodeCodewords;
    cudaSafeCall(cudaMemcpyAsync(dpEncodeCodewords,       pInstance->Encode.pEncodeCodewords,       encodeCodewordElems * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));
    cudaSafeCall(cudaMemcpyAsync(dpEncodeCodewordLengths, pInstance->Encode.pEncodeCodewordLengths, encodeCodewordElems * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));

    timerHigh("Symbols:    Huffman Encode");

    // encode the symbols
    std::vector<uint> codewordBitsizeSymbols(streamCount);
    huffmanEncode(pInstance, pStreamInfos.data(), streamCount, pInstance->m_codingBlockSize, codewordBitsizeSymbols.data());

    timerHigh("Symbols:    Download");

    // download encoded symbols and offsets
    // (GPU buffers will be used again for the zero counts)
    // for small blocks: download everything in a single memcpy (but more memory traffic)
    //cudaSafeCall(cudaMemcpyAsync(pInstance->Encode.pCodewordBuffer, dpCodewordStreams, streamCount * symbolStreamMaxBytesAligned, cudaMemcpyDeviceToHost, pInstance->m_stream));
    // for large blocks: download only getNumUintsForBits(codewordBitsizeSymbols[block]) uints per block
    for(uint block = 0; block < streamCount; block++) {
        uint offset = block * symbolStreamMaxElemsAligned;
        uint numBytes = getNumUintsForBits(codewordBitsizeSymbols[block]) * sizeof(uint);
        cudaSafeCall(cudaMemcpyAsync(pInstance->Encode.pCodewordBuffer + offset, dpCodewordStreams + offset, numBytes, cudaMemcpyDeviceToHost, pInstance->m_stream));
    }
    // offsets are small, so always download everything in a single memcpy
    cudaSafeCall(cudaMemcpyAsync(pInstance->Encode.pOffsetBuffer, dpOffsets, streamCount * offsetStreamMaxBytesAligned, cudaMemcpyDeviceToHost, pInstance->m_stream));

    // clear codeword stream again for zero counts
    cudaSafeCall(cudaMemsetAsync(dpCodewordStreams, 0, streamCount * symbolStreamMaxBytesAligned, pInstance->m_stream));


    timerLow("Huffman Encode ZeroCounts");

    timerHigh("ZeroCounts: Design Huffman Tables");

    // 2. zero counts
    // build encode tables
    std::vector<HuffmanEncodeTable>& zeroCountEncodeTables = pInstance->Encode.zeroCountEncodeTables;
    if(!HuffmanEncodeTable::design(pInstance, zeroCountEncodeTables.data(), streamCount, (const Symbol**)pdpZeroCounts.data(), symbolCountsCompact.data())) {
        pInstance->releaseBuffers(4 + 2 * streamCount);
        return false;
    }

    timerHigh("ZeroCounts: Upload Huffman Tables");

    // fill stream infos
    cudaSafeCall(cudaDeviceSynchronize()); // sync before overwriting pStreamInfos
    dpEncodeCodewordsNext = dpEncodeCodewords;
    dpEncodeCodewordLengthsNext = dpEncodeCodewordLengths;
    pEncodeCodewordsNext       = pInstance->Encode.pEncodeCodewords       + streamCount * distinctSymbolCountMaxAligned;
    pEncodeCodewordLengthsNext = pInstance->Encode.pEncodeCodewordLengths + streamCount * distinctSymbolCountMaxAligned;
    for(uint block = 0; block < streamCount; block++) {
        HuffmanGPUStreamInfo& streamInfo = pStreamInfos[block];

        streamInfo.dpSymbolStream = (byte*)pdpZeroCounts[block];
        streamInfo.symbolCount = symbolCountsCompact[block];

        streamInfo.dpEncodeCodewords       = dpEncodeCodewordsNext;
        streamInfo.dpEncodeCodewordLengths = dpEncodeCodewordLengthsNext;

        zeroCountEncodeTables[block].copyToBuffer(pEncodeCodewordsNext, pEncodeCodewordLengthsNext);

        size_t elems = zeroCountEncodeTables[block].getTableSize();
        pEncodeCodewordsNext        += elems;
        pEncodeCodewordLengthsNext  += elems;
        dpEncodeCodewordsNext       += elems;
        dpEncodeCodewordLengthsNext += elems;
    }

    // upload encode tables
    encodeCodewordElems = pEncodeCodewordsNext - (pInstance->Encode.pEncodeCodewords + streamCount * distinctSymbolCountMaxAligned);
    cudaSafeCall(cudaMemcpyAsync(dpEncodeCodewords,       pInstance->Encode.pEncodeCodewords       + streamCount * distinctSymbolCountMaxAligned, encodeCodewordElems * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));
    cudaSafeCall(cudaMemcpyAsync(dpEncodeCodewordLengths, pInstance->Encode.pEncodeCodewordLengths + streamCount * distinctSymbolCountMaxAligned, encodeCodewordElems * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));

    timerHigh("ZeroCounts: Huffman Encode");

    // encode the zero counts
    std::vector<uint> codewordBitsizeZeroCounts(streamCount);
    huffmanEncode(pInstance, pStreamInfos.data(), streamCount, pInstance->m_codingBlockSize, codewordBitsizeZeroCounts.data());

    timerHigh("ZeroCounts: Download");

    // download zero count codeword stream and offsets
    // for small blocks: download everything in a single memcpy (but more memory traffic)
    //cudaSafeCall(cudaMemcpyAsync(pInstance->Encode.pCodewordBuffer + streamCount * symbolStreamMaxElemsAligned, dpCodewordStreams, streamCount * symbolStreamMaxBytesAligned, cudaMemcpyDeviceToHost, pInstance->m_stream));
    // for large blocks: download only getNumUintsForBits(codewordBitsizeZeroCounts[block]) uints per block
    uint* pCodewordBufferZeroCounts = pInstance->Encode.pCodewordBuffer + streamCount * symbolStreamMaxElemsAligned;
    for(uint block = 0; block < streamCount; block++) {
        uint offset = block * symbolStreamMaxElemsAligned;
        uint numBytes = getNumUintsForBits(codewordBitsizeZeroCounts[block]) * sizeof(uint);
        cudaSafeCall(cudaMemcpyAsync(pCodewordBufferZeroCounts + offset, dpCodewordStreams + offset, numBytes, cudaMemcpyDeviceToHost, pInstance->m_stream));
    }
    // offsets are small, so always download everything in a single memcpy
    uint* pOffsetBufferZeroCounts = pInstance->Encode.pOffsetBuffer + streamCount * offsetStreamMaxElemsAligned;
    cudaSafeCall(cudaMemcpyAsync(pOffsetBufferZeroCounts, dpOffsets, streamCount * offsetStreamMaxBytesAligned, cudaMemcpyDeviceToHost, pInstance->m_stream));

    cudaSafeCall(cudaDeviceSynchronize());

    timerLow();
    timerHigh();


    // write to bitstream
    //#pragma omp parallel for if(!singleBitStream) TODO: need to check that bitstreams are unique!
    for(int block = 0; block < int(streamCount); block++) {
        BitStream& bitStream = *ppBitStreams[singleBitStream ? 0 : block];
        //uint bitStreamPosStart = bitStream.getBitPosition();
        //uint bitStreamPos = bitStreamPosStart;

        // write compacted symbol count
        bitStream.writeAligned(&symbolCountsCompact[block], 1);

        // 1. compacted symbols
        // write encode table
        symbolEncodeTables[block].writeToBitStream(pInstance, bitStream);

        //g_bitsSymbolEncodeTables += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();

        // write symbol codeword stream
        bitStream.writeAligned(&codewordBitsizeSymbols[block], 1);
        uint codewordUints = getNumUintsForBits(codewordBitsizeSymbols[block]);
        uint* pCodewordBuffer = pInstance->Encode.pCodewordBuffer + block * symbolStreamMaxElemsAligned;
        //cudaSafeCall(cudaEventSynchronize(pInstance->Encode.pSyncEvents[block]));
        bitStream.writeAligned(pCodewordBuffer, codewordUints);

        //g_bitsSymbolCodewords += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();

        // make symbol offsets incremental and write
        uint numOffsets = getNumOffsets(symbolCountsCompact[block], pInstance->m_codingBlockSize);
        uint* pOffsetBuffer = pInstance->Encode.pOffsetBuffer + block * offsetStreamMaxElemsAligned;
        packInc16CPU(pOffsetBuffer, (ushort*)pOffsetBuffer, numOffsets);
        bitStream.writeAligned((ushort*)pOffsetBuffer, numOffsets);

        //g_bitsSymbolOffsets += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();


        // 2. zero counts
        // write encode table
        zeroCountEncodeTables[block].writeToBitStream(pInstance, bitStream);

        //g_bitsZeroCountEncodeTables += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();

        // write zero count codeword stream
        bitStream.writeAligned(&codewordBitsizeZeroCounts[block], 1);
        codewordUints = getNumUintsForBits(codewordBitsizeZeroCounts[block]);
        pCodewordBuffer = pInstance->Encode.pCodewordBuffer + (streamCount + block) * symbolStreamMaxElemsAligned;
        //cudaSafeCall(cudaEventSynchronize(pInstance->Encode.pSyncEvents[streamCount + block]));
        bitStream.writeAligned(pCodewordBuffer, codewordUints);

        //g_bitsZeroCountCodewords += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();

        // make zero count offsets incremental and write
        pOffsetBuffer = pInstance->Encode.pOffsetBuffer + (streamCount + block) * offsetStreamMaxElemsAligned;
        packInc16CPU(pOffsetBuffer, (ushort*)pOffsetBuffer, numOffsets);
        bitStream.writeAligned((ushort*)pOffsetBuffer, numOffsets);

        //g_bitsZeroCountOffsets += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();

        //g_bitsTotal += bitStreamPos - bitStreamPosStart;
    }
    //g_totalEncodedCount += streamCount * symbolCountPerStream;

    cudaSafeCall(cudaEventRecord(pInstance->Encode.encodeFinishedEvent, pInstance->m_stream));

    pInstance->releaseBuffers(4 + 2 * streamCount);

    return true;
    */
}

template<typename Symbol>
bool decodeRLHuff(GpuInstance* pInstance, BitStreamReadOnly* ppBitStreams[], bool singleBitStream, Symbol* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    //assert(streamCount == 1);   // more than 1 block is currently not supported
    //const uint warpSize = pInstance->m_warpSize;
    //uint pad = warpSize * pInstance->m_codingBlockSize;
    //uint symbolCountPerBlockPadded = (symbolCountPerStream + pad - 1) / pad * pad;
//
    //uint symbolStreamMaxBytes = symbolCountPerStream * sizeof(Symbol);
    //uint offsetStreamMaxBytes = getNumOffsets(symbolCountPerStream, pInstance->m_codingBlockSize) * sizeof(uint);
    //GpuInstance::EncodeResources::DecodeResources& resources = pInstance->Encode.GetDecodeResources();
    //check_vk_result(vkWaitForFences(pInstance->vkContext.device, 1, &resources.syncFence, VK_TRUE, 0));
    ////cudaSafeCall(cudaEventSynchronize(resources.syncEvent));
//
    ////Symbol* dpSymbolStreamCompacted = pInstance->getBuffer<Symbol>(streamCount * symbolCountPerBlockPadded);
    ////Symbol* dpZeroCounts            = pInstance->getBuffer<Symbol>(streamCount * symbolCountPerBlockPadded);
    ////uint* dpOffsets = (uint*)pInstance->getBuffer<byte>(streamCount * offsetStreamMaxBytes);
    //size_t decodeTableSizeMax = getAlignedSize(HuffmanDecodeTable::computeMaxGPUSize(pInstance), 128);
    ////byte* dpDecodeTables = pInstance->getBuffer<byte>(streamCount * decodeTableSizeMax);
    //
    //std::vector<VkDeviceSize> sizes{streamCount * symbolCountPerBlockPadded * sizeof(Symbol), streamCount * symbolCountPerBlockPadded * sizeof(Symbol), streamCount * offsetStreamMaxBytes, streamCount * decodeTableSizeMax, symbolStreamMaxBytes};
    //std::vector<VkBufferUsageFlags> usages(sizes.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    //auto [buffer, offsets, memory] = VkUtil::createMultiBufferBound(pInstance->vkContext, sizes, usages, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    //auto bSymbolStreamCompacted = buffer[0];
    //auto bZeroCounts = buffer[1];
    //auto bOffsets = buffer[2];
    //auto bDecodeTables = buffer[3];
    //auto dpCodewordStream = buffer[4];
//
    //for(uint block = 0; block < streamCount; block++) {
    //    // get GPU buffers
    //    //uint* dpCodewordStream = (uint*)pInstance->getBuffer<byte>(symbolStreamMaxBytes);
//
    //    // fill stream infos
    //    // all buffers except the symbol stream buffers are shared between compacted symbols and zero counts
    //    HuffmanGPUStreamInfo& streamInfoSymbols = resources.pSymbolStreamInfos[block];
    //    HuffmanGPUStreamInfo& streamInfoZeroCounts = resources.pZeroCountStreamInfos[block];
//
    //    streamInfoSymbols.dpSymbolStream = VkUtil::getBufferAddress(pInstance->vkContext.device, bSymbolStreamCompacted);//(byte*)(dpSymbolStreamCompacted + block * symbolCountPerBlockPadded);
    //    streamInfoZeroCounts.dpSymbolStream = VkUtil::getBufferAddress(pInstance->vkContext.device, bZeroCounts);//(byte*)(dpZeroCounts + block * symbolCountPerBlockPadded);
//
    //    streamInfoSymbols.dpCodewordStream = streamInfoZeroCounts.dpCodewordStream = VkUtil::getBufferAddress(pInstance->vkContext.device, dpCodewordStream);
    //    // streamInfo.dpOffsets will be filled later, with pointers into our single dpOffsets buffer (see above)
    //    // streamInfo.dpDecodeTable will be filled later, with pointers into our single dpDecodeTables buffer (see above)
    //}
//
    ////util::CudaScopedTimer timerLow(pInstance->Encode.timerDecodeLowDetail);
    ////util::CudaScopedTimer timerHigh(pInstance->Encode.timerDecodeHighDetail);
//
    ////timerLow("Huffman Decode Symbols");
//
    ////timerHigh("Symbols:    Upload (+read BitStream)");
//
    //VkPointer pSymbolOffsetsNext = {resources.pSymbolOffsets, resources.memory, 0, resources.symbolOffsetsOffset};
    //VkPointer dpOffsetsNext = {bOffsets, memory, 0, offsets[2]};//dpOffsets;
    //// read and upload decode tables, upload codeword streams and offsets, and fill stream info for symbols
    //std::vector<HuffmanDecodeTable>& symbolDecodeTables = resources.symbolDecodeTables;
    //size_t symbolDecodeTablesBufferOffset = 0;
    //std::vector<HuffmanDecodeTable>& zeroCountDecodeTables = resources.zeroCountDecodeTables;
    //std::vector<const uint*> pZeroCountCodewordStreams(streamCount);
    //std::vector<uint> zeroCountCodewordUintCounts(streamCount);
    //std::vector<const uint*> pZeroCountOffsets(streamCount);
    //for(uint block = 0; block < streamCount; block++) {
    //    BitStreamReadOnly& bitStream = *ppBitStreams[singleBitStream ? 0 : block];
//
    //    HuffmanGPUStreamInfo& streamInfo = resources.pSymbolStreamInfos[block];
//
    //    // read compacted symbol count
    //    bitStream.readAligned(&streamInfo.symbolCount, 1);
    //    resources.pZeroCountStreamInfos[block].symbolCount = streamInfo.symbolCount;
//
    //    // 1. compacted symbols
    //    // read symbol decode table
    //    symbolDecodeTables[block].readFromBitStream(pInstance, bitStream);
//
    //    // copy decode table into upload buffer
    //    //symbolDecodeTables[block].copyToBuffer(pInstance, resources.pSymbolDecodeTablesBuffer + symbolDecodeTablesBufferOffset);
    //    //streamInfo.dpDecodeTable = dpDecodeTables + symbolDecodeTablesBufferOffset;
    //    streamInfo.decodeSymbolTableSize = symbolDecodeTables[block].getSymbolTableSize();
    //    symbolDecodeTablesBufferOffset += getAlignedSize(symbolDecodeTables[block].computeGPUSize(pInstance), 128);
//
    //    // upload symbol codewords
    //    uint codewordBitsize;
    //    bitStream.readAligned(&codewordBitsize, 1);
    //    uint codewordUintCount = getNumUintsForBits(codewordBitsize);
    //    const uint* pCodewordStream = bitStream.getRaw() + bitStream.getBitPosition() / (sizeof(uint)*8);
    //    //cudaSafeCall(cudaMemcpyAsync(streamInfo.dpCodewordStream, pCodewordStream, codewordUintCount * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));
    //    bitStream.skipBits(codewordUintCount * sizeof(uint) * 8);
//
    //    // get symbol offsets pointer
    //    uint numOffsets = getNumOffsets(streamInfo.symbolCount, pInstance->m_codingBlockSize);
    //    bitStream.align<uint>();
    //    const uint* pOffsets = bitStream.getRaw() + bitStream.getBitPosition() / (sizeof(uint)*8);
    //    bitStream.skipAligned<ushort>(numOffsets);
//
    //    // make offsets absolute (prefix sum)
    //    //unpackInc16CPU(pSymbolOffsetsNext, (const ushort*)pOffsets, numOffsets);
    //    pSymbolOffsetsNext += numOffsets;
    //    streamInfo.dpOffsets = dpOffsetsNext;
    //    dpOffsetsNext += numOffsets;
//
    //    // 2. zero counts
    //    // read zero count decode table
    //    zeroCountDecodeTables[block].readFromBitStream(pInstance, bitStream);
//
    //    // read zero count codewords pointer
    //    bitStream.readAligned(&codewordBitsize, 1);
    //    zeroCountCodewordUintCounts[block] = getNumUintsForBits(codewordBitsize);
    //    pZeroCountCodewordStreams[block] = bitStream.getRaw() + bitStream.getBitPosition() / (sizeof(uint)*8);
    //    bitStream.skipBits(zeroCountCodewordUintCounts[block] * sizeof(uint) * 8);
//
    //    // read zero count offsets pointer
    //    bitStream.align<uint>();
    //    pZeroCountOffsets[block] = bitStream.getRaw() + bitStream.getBitPosition() / (sizeof(uint)*8);
    //    bitStream.skipAligned<ushort>(numOffsets);
    //}
//
    //// upload decode tables
    ////cudaSafeCall(cudaMemcpyAsync(dpDecodeTables, resources.pSymbolDecodeTablesBuffer, symbolDecodeTablesBufferOffset, cudaMemcpyHostToDevice, pInstance->m_stream));
//
    //// upload offsets
    //size_t offsetCountTotal = pSymbolOffsetsNext - resources.pSymbolOffsets;
    ////cudaSafeCall(cudaMemcpyAsync(dpOffsets, resources.pSymbolOffsets, offsetCountTotal * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));
//
   //// timerHigh("Symbols:    Huffman Decode");
//
    //// decode symbols
    ////huffmanDecode(pInstance, resources.pSymbolStreamInfos, streamCount, pInstance->m_codingBlockSize);
//
    ////timerLow("Huffman Decode ZeroCounts");
//
    ////timerHigh("ZeroCounts: Upload");
//
    //uint* pZeroCountOffsetsNext = resources.pZeroCountOffsets;
    //dpOffsetsNext = dpOffsets;
    //size_t zeroCountDecodeTablesBufferOffset = 0;
    //// upload decode tables, codeword streams and offsets, and fill stream infos for zero counts
    //for(uint block = 0; block < streamCount; block++) {
    //    HuffmanGPUStreamInfo& streamInfo = resources.pZeroCountStreamInfos[block];
//
    //    // copy decode table into upload buffer
    //    zeroCountDecodeTables[block].copyToBuffer(pInstance, resources.pZeroCountDecodeTablesBuffer + zeroCountDecodeTablesBufferOffset);
    //    streamInfo.dpDecodeTable = dpDecodeTables + zeroCountDecodeTablesBufferOffset;
    //    streamInfo.decodeSymbolTableSize = zeroCountDecodeTables[block].getSymbolTableSize();
    //    zeroCountDecodeTablesBufferOffset += getAlignedSize(zeroCountDecodeTables[block].computeGPUSize(pInstance), 128);
//
    //    // upload zero count codewords
    //    cudaSafeCall(cudaMemcpyAsync(streamInfo.dpCodewordStream, pZeroCountCodewordStreams[block], zeroCountCodewordUintCounts[block] * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));
//
    //    uint numOffsets = getNumOffsets(streamInfo.symbolCount, pInstance->m_codingBlockSize);
    //    unpackInc16CPU(pZeroCountOffsetsNext, (const ushort*)pZeroCountOffsets[block], numOffsets);
    //    pZeroCountOffsetsNext += numOffsets;
    //    streamInfo.dpOffsets = dpOffsetsNext;
    //    dpOffsetsNext += numOffsets;
    //}
//
    //// upload decode tables
    //cudaSafeCall(cudaMemcpyAsync(dpDecodeTables, resources.pZeroCountDecodeTablesBuffer, zeroCountDecodeTablesBufferOffset, cudaMemcpyHostToDevice, pInstance->m_stream));
//
    //// upload offsets
    //offsetCountTotal = pZeroCountOffsetsNext - resources.pZeroCountOffsets;
    //cudaSafeCall(cudaMemcpyAsync(dpOffsets, resources.pZeroCountOffsets, offsetCountTotal * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));
//
    //timerHigh("ZeroCounts: Huffman Decode");
//
    //// decode zero counts
    //huffmanDecode(pInstance, resources.pZeroCountStreamInfos, streamCount, pInstance->m_codingBlockSize);
//
    //timerHigh();
//
    //timerLow("Run Length Decode");
//
    //// run length decode
    //std::vector<uint> symbolCountsCompact(streamCount);
    //for(uint block = 0; block < streamCount; block++) {
    //    symbolCountsCompact[block] = resources.pSymbolStreamInfos[block].symbolCount;
    //}
    //runLengthDecode(pInstance, dpSymbolStreamCompacted, dpZeroCounts, symbolCountsCompact.data(), symbolCountPerBlockPadded, pdpSymbolStreams, symbolCountPerStream, streamCount);
//
    //timerLow();
//
    //cudaSafeCall(cudaEventRecord(resources.syncEvent, pInstance->m_stream));

    //pInstance->releaseBuffers(4 + 1 * streamCount);

    return true;
}

bool decodeRLHuff(GpuInstance* pInstance, BitStreamReadOnly& bitStream, std::vector<Symbol16>& symbolStream){
    return false;
}

bool decodeRLHuff(GpuInstance* pInstance, VkBuffer bitStreamBuffer, BitStream& currentBitStream, size_t decodeTableOffset, size_t codewordStreamOffset, VkBuffer symbolBuffer, uint symbolSize, VkCommandBuffer commands){
    const bool useGpuUnpackInc = false;
    const auto& context = pInstance->vkContext;
    
    // getting the symbol count from the bit stream
    auto &resources = pInstance->Encode.GetDecodeResources();
    HuffmanGPUStreamInfo& streamInfo = resources.pSymbolStreamInfos[0]; // we assume here to only have a single decoding block! -> index 0
    currentBitStream.readAligned(&streamInfo.symbolCount, 1);
    resources.pZeroCountStreamInfos[0].symbolCount = streamInfo.symbolCount;

    // getting the cpu huffman decode table
    HuffmanDecodeTable& decodeTable = resources.symbolDecodeTables[0];
    HuffmanDecodeTable& zeroDecodeTable = resources.zeroCountDecodeTables[0];
    decodeTable.readFromBitStream(pInstance, currentBitStream);

    // upload to gpu not needed, as already done in loading the compressed dataset, so binding data which is already on gpu
    streamInfo.dpDecodeTable = VkUtil::getBufferAddress(context.device, bitStreamBuffer) + decodeTableOffset; // The  //dpDecodeTables + symbolDecodeTablesBufferOffset;
    streamInfo.decodeSymbolTableSize = decodeTable.getSymbolTableSize();
    size_t symbolDecodeTablesBufferOffset = getAlignedSize(decodeTable.computeGPUSize(pInstance), 128);

    // set the codeword buffer address
    streamInfo.dpCodewordStream = VkUtil::getBufferAddress(context.device, bitStreamBuffer) + codewordStreamOffset;
    uint codewordBitsize;
    currentBitStream.readAligned(&codewordBitsize, 1);
    uint codewordUintCount = getNumUintsForBits(codewordBitsize);
    currentBitStream.skipBits(codewordUintCount * sizeof(uint) * 8);    // skipping the codeword stream in the bit stream

    // calculating the symbol offsets pointer (doing it currently on the cpu. Use const bool at the beginning of the function to change behaviour)
    if(useGpuUnpackInc){
        // TODO implement
    }
    else{
        // get symbol offsets pointer
        uint numOffsets = getNumOffsets(symbolSize, pInstance->m_codingBlockSize);
        currentBitStream.align<uint>();
        const uint* pOffsets = currentBitStream.getRaw() + currentBitStream.getBitPosition() / (sizeof(uint)*8);
        currentBitStream.skipAligned<ushort>(numOffsets);

        // make offsets absolute (prefix sum)
        std::vector<uint> symbolOffsets(numOffsets);
        unpackInc16CPU(symbolOffsets.data(), (const ushort*)pOffsets, numOffsets);
        // upload offsets data
        VkUtil::uploadData(context.device, resources.memory, resources.symbolOffsetsOffset, symbolOffsets.size() * sizeof(symbolOffsets[0]), symbolOffsets.data());
        streamInfo.dpOffsets = 0;// TODO missing
    }

    // adding decode command for rl encoded data (The zero counts are stored in a different array)
    //huffmanDecodeCommands(pInstance, commands, resources.pSymbolStreamInfos, 1, pInstance->m_codingBlockSize);

    // zero counts ? Whatever these are ....+
    // seems like they should already be on the gpu at this point
    return false;

}

// decoding with proper input data structures
// decodes only 16 bit symbols
// this is the same function as the orignial decodeRLHuff with easier setup, and pre setup of the data structures
bool decodeRLHuffHalf(GpuInstance* pInstance, const RLHuffDecodeDataCpu& decodeDataCpu, const RLHuffDecodeDataGpu& decodeDataGpu, VkDeviceAddress outSymbols, VkCommandBuffer commands){
    const auto& context = pInstance->vkContext;
    auto &resources = pInstance->Encode.GetDecodeResources();
    pInstance->Encode.Decode[0].pSymbolStreamInfos[0].symbolCount = decodeDataCpu.symbolCount;
    const uint pad = pInstance->m_subgroupSize * pInstance->m_codingBlockSize;
    const uint symbolCountPadded = (decodeDataCpu.symbolCount + pad - 1) / pad * pad;

    VkDeviceAddress compactSymbolsAddress = VkUtil::getBufferAddress(context.device, resources.symbolsZeroCounts) + resources.compactSymbolsOffset;
    VkDeviceAddress zeroCountsAddress = VkUtil::getBufferAddress(context.device, resources.symbolsZeroCounts) + resources.zeroCountsOffset;
    
    // decompressing the zero counts -------------------------------------------------
    HuffmanGPUStreamInfo& streamInfo = resources.pSymbolStreamInfos[0]; // we assume here to only have a single decoding block! -> index 0
    
    streamInfo.symbolCount = decodeDataCpu.symbolCount;

    streamInfo.dpDecodeTable = VkUtil::getBufferAddress(context.device, decodeDataGpu.buffer) + decodeDataGpu.symbolTableOffset;
    streamInfo.decodeSymbolTableSize = decodeDataCpu.symbolTable.getSymbolTableSize();

    streamInfo.dpCodewordStream = VkUtil::getBufferAddress(context.device, decodeDataGpu.buffer) + decodeDataGpu.symbolStreamOffset;

    streamInfo.dpOffsets = VkUtil::getBufferAddress(context.device, decodeDataGpu.buffer) + decodeDataGpu.symbolOffsetsOffset;
    streamInfo.dpSymbolStream = compactSymbolsAddress;

    // uploading the stream info and calling the decoding function for the rl encoded stream
    // note: the descriptor set resources.streamInfoSet has to be created and the streamInfos buffer has to be bound
    VkUtil::uploadData(context.device, resources.memory, resources.streamInfosOffset, sizeof(HuffmanGPUStreamInfo), &streamInfo);
    huffmanDecode(pInstance, commands, resources.streamInfoSet, 1u, pInstance->m_codingBlockSize);

    // decompressing the zero counts -------------------------------------------------
    HuffmanGPUStreamInfo& zeroStreamInfo = resources.pZeroCountStreamInfos[0];
    zeroStreamInfo.symbolCount = decodeDataCpu.symbolCount; // symbol count is equivalent to the normal decomrpession symbol count
    
    zeroStreamInfo.dpDecodeTable = VkUtil::getBufferAddress(context.device, decodeDataGpu.buffer) +  decodeDataGpu.zeroCountTableOffset;
    zeroStreamInfo.decodeSymbolTableSize = decodeDataCpu.zeroCountTable.getSymbolTableSize();

    zeroStreamInfo.dpCodewordStream = VkUtil::getBufferAddress(context.device, decodeDataGpu.buffer) + decodeDataGpu.zeroCountStreamOffset;

    zeroStreamInfo.dpOffsets = VkUtil::getBufferAddress(context.device, decodeDataGpu.buffer) + decodeDataGpu.zeroCountOffsetsOffset;
    zeroStreamInfo.dpSymbolStream = zeroCountsAddress;

    VkUtil::uploadData(context.device, resources.memory, resources.zeroInfosOffset, sizeof(zeroStreamInfo), &zeroStreamInfo);
    huffmanDecode(pInstance, commands, resources.zeroStreamInfoSet, 1u, pInstance->m_codingBlockSize);

    // decompressing the run length encoding ------------------------------------------
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, {}, 0, {}, 0, {});
    runLengthDecodeHalf(pInstance, commands, compactSymbolsAddress, zeroCountsAddress, {static_cast<uint>(decodeDataCpu.symbolCount)}, symbolCountPadded, outSymbols, {static_cast<uint>(decodeDataCpu.symbolCount)}, 1);
    
    return true;
}

template<typename Symbol>
bool encodeHuff(GpuInstance* pInstance, BitStream* ppBitStreams[], bool singleBitStream, /*const*/ Symbol* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    // not yet encoded
    return false;
    /*
    uint symbolStreamMaxBytes = symbolCountPerStream * sizeof(Symbol);
    uint offsetStreamMaxBytes = getNumOffsets(symbolCountPerStream, pInstance->m_codingBlockSize) * sizeof(uint);

    uint distinctSymbolCountMax = 1 << pInstance->m_log2HuffmanDistinctSymbolCountMax;

    uint symbolStreamMaxBytesAligned = (uint)getAlignedSize(symbolStreamMaxBytes, 128);
    uint offsetStreamMaxBytesAligned = (uint)getAlignedSize(offsetStreamMaxBytes, 128);
    uint distinctSymbolCountMaxAligned = (uint)getAlignedSize(distinctSymbolCountMax, 128 / sizeof(uint));

    uint symbolStreamMaxElemsAligned = uint(symbolStreamMaxBytesAligned / sizeof(uint));
    uint offsetStreamMaxElemsAligned = uint(offsetStreamMaxBytesAligned / sizeof(uint));

    // get GPU buffers from pInstance
    std::vector<HuffmanGPUStreamInfo> pStreamInfos(streamCount);
    uint* dpCodewordStreams       = pInstance->getBuffer<uint>(streamCount * symbolStreamMaxElemsAligned);
    uint* dpOffsets               = pInstance->getBuffer<uint>(streamCount * offsetStreamMaxElemsAligned);
    uint* dpEncodeCodewords       = pInstance->getBuffer<uint>(streamCount * distinctSymbolCountMaxAligned);
    uint* dpEncodeCodewordLengths = pInstance->getBuffer<uint>(streamCount * distinctSymbolCountMaxAligned);
    for(uint block = 0; block < streamCount; block++) {
        HuffmanGPUStreamInfo& streamInfo = pStreamInfos[block];

        streamInfo.dpCodewordStream = dpCodewordStreams + block * symbolStreamMaxElemsAligned;
        streamInfo.dpOffsets        = dpOffsets         + block * offsetStreamMaxElemsAligned;

        // dpEncodeCodewords and dpEncodeCodewordLengths will be filled later
    }

    cudaSafeCall(cudaMemsetAsync(dpCodewordStreams, 0, streamCount * symbolStreamMaxBytesAligned, pInstance->m_stream));

    cudaSafeCall(cudaEventSynchronize(pInstance->Encode.encodeFinishedEvent));

    util::CudaScopedTimer timerLow(pInstance->Encode.timerEncodeLowDetail);
    util::CudaScopedTimer timerHigh(pInstance->Encode.timerEncodeHighDetail);

    timerLow("Huffman Encode Symbols");

    timerHigh("Symbols:    Design Huffman Tables");

    for(uint block = 0; block < streamCount; block++) {
        // padding for histogram (which wants the element count to be a multiple of 8)
        histogramPadData(pInstance, pdpSymbolStreams[block], symbolCountPerStream);
    }

    // build encode tables
    std::vector<HuffmanEncodeTable>& symbolEncodeTables = pInstance->Encode.symbolEncodeTables;
    std::vector<uint> symbolCount(streamCount, symbolCountPerStream);
    if(!HuffmanEncodeTable::design(pInstance, symbolEncodeTables.data(), streamCount, (const Symbol**)pdpSymbolStreams, symbolCount.data())) {
        pInstance->releaseBuffers(4);
        return false;
    }

    timerHigh("Symbols:    Upload Huffman Tables");

    // fill stream infos
    uint* dpEncodeCodewordsNext = dpEncodeCodewords;
    uint* dpEncodeCodewordLengthsNext = dpEncodeCodewordLengths;
    uint* pEncodeCodewordsNext = pInstance->Encode.pEncodeCodewords;
    uint* pEncodeCodewordLengthsNext = pInstance->Encode.pEncodeCodewordLengths;
    for(uint block = 0; block < streamCount; block++) {
        HuffmanGPUStreamInfo& streamInfo = pStreamInfos[block];

        streamInfo.dpSymbolStream = (byte*)pdpSymbolStreams[block];
        streamInfo.symbolCount = symbolCount[block];

        streamInfo.dpEncodeCodewords       = dpEncodeCodewordsNext;
        streamInfo.dpEncodeCodewordLengths = dpEncodeCodewordLengthsNext;

        symbolEncodeTables[block].copyToBuffer(pEncodeCodewordsNext, pEncodeCodewordLengthsNext);

        size_t elems = symbolEncodeTables[block].getTableSize();
        pEncodeCodewordsNext        += elems;
        pEncodeCodewordLengthsNext  += elems;
        dpEncodeCodewordsNext       += elems;
        dpEncodeCodewordLengthsNext += elems;
    }

    // upload encode tables
    size_t encodeCodeWordElems = pEncodeCodewordsNext - pInstance->Encode.pEncodeCodewords;
    cudaSafeCall(cudaMemcpyAsync(dpEncodeCodewords, pInstance->Encode.pEncodeCodewords, encodeCodeWordElems * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));
    cudaSafeCall(cudaMemcpyAsync(dpEncodeCodewordLengths, pInstance->Encode.pEncodeCodewordLengths, encodeCodeWordElems * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));

    timerHigh("Symbols:    Huffman Encode");

    // encode the symbols
    std::vector<uint> codewordBitsizeSymbols(streamCount);
    huffmanEncode(pInstance, pStreamInfos.data(), streamCount, pInstance->m_codingBlockSize, codewordBitsizeSymbols.data());

    timerHigh("Symbols:    Download");

    // download encoded symbols and offsets
    //TODO for large blocks, download only getNumUintsForBits(codewordBitsizeSymbols[block]) uints per block?
    cudaSafeCall(cudaMemcpyAsync(pInstance->Encode.pCodewordBuffer, dpCodewordStreams, streamCount * symbolStreamMaxBytesAligned, cudaMemcpyDeviceToHost, pInstance->m_stream));
    cudaSafeCall(cudaMemcpyAsync(pInstance->Encode.pOffsetBuffer,   dpOffsets,         streamCount * offsetStreamMaxBytesAligned, cudaMemcpyDeviceToHost, pInstance->m_stream));
    cudaSafeCall(cudaDeviceSynchronize());

    timerLow();
    timerHigh();

    // write to bitstream
    //#pragma omp parallel for if(!singleBitStream) TODO: need to check that bitstreams are unique!
    for(int block = 0; block < int(streamCount); block++) {
        BitStream& bitStream = *ppBitStreams[singleBitStream ? 0 : block];
        //uint bitStreamPosStart = bitStream.getBitPosition();
        //uint bitStreamPos = bitStreamPosStart;

        // write encode table
        symbolEncodeTables[block].writeToBitStream(pInstance, bitStream);

        //g_bitsSymbolEncodeTables += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();

        // write codeword stream
        bitStream.writeAligned(&codewordBitsizeSymbols[block], 1);
        uint codewordUints = getNumUintsForBits(codewordBitsizeSymbols[block]);
        uint* pCodewordBuffer = pInstance->Encode.pCodewordBuffer + block * symbolStreamMaxElemsAligned;
        bitStream.writeAligned(pCodewordBuffer, codewordUints);

        //g_bitsSymbolCodewords += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();

        // make offsets incremental and write
        uint numOffsets = getNumOffsets(symbolCountPerStream, pInstance->m_codingBlockSize);
        uint* pOffsetBuffer = pInstance->Encode.pOffsetBuffer + block * offsetStreamMaxElemsAligned;
        packInc16CPU(pOffsetBuffer, (ushort*)pOffsetBuffer, numOffsets);
        bitStream.writeAligned((ushort*)pOffsetBuffer, numOffsets);

        //g_bitsSymbolOffsets += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();


        //g_bitsTotal += bitStreamPos - bitStreamPosStart;
    }
    //g_totalEncodedCount += streamCount * symbolCountPerStream;

    cudaSafeCall(cudaEventRecord(pInstance->Encode.encodeFinishedEvent, pInstance->m_stream));

    pInstance->releaseBuffers(4);

    return true;
    */
}

template<typename Symbol>
bool decodeHuff(GpuInstance* pInstance, BitStreamReadOnly* ppBitStreams[], bool singleBitStream, Symbol* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    //ScopedProfileSample sample0(pInstance->m_pProfiler, "decodeHuff");
//
    //uint symbolStreamMaxBytes = symbolCountPerStream * sizeof(Symbol);
    //uint offsetStreamMaxBytes = getNumOffsets(symbolCountPerStream, pInstance->m_codingBlockSize) * sizeof(uint);
//
    //GpuInstance::EncodeResources::DecodeResources& resources = pInstance->Encode.GetDecodeResources();
    //{ ScopedProfileSample sample0(pInstance->m_pProfiler, "sync");
    //    cudaSafeCall(cudaEventSynchronize(resources.syncEvent));
    //}
//
    ////TODO ? size_t symbolStreamMaxBytesPadded = getAlignedSize(symbolStreamMaxBytes, 128);
    //uint* dpCodewordStreams = (uint*)pInstance->getBuffer<byte>(streamCount * symbolStreamMaxBytes);
    //uint* dpOffsets = (uint*)pInstance->getBuffer<byte>(streamCount * offsetStreamMaxBytes);
    //size_t decodeTableSizeMax = getAlignedSize(HuffmanDecodeTable::computeMaxGPUSize(pInstance), 128);
    //byte* dpDecodeTables = pInstance->getBuffer<byte>(streamCount * decodeTableSizeMax);
    //for(uint block = 0; block < streamCount; block++) {
    //    HuffmanGPUStreamInfo& streamInfo = resources.pSymbolStreamInfos[block];
//
    //    streamInfo.dpSymbolStream = (byte*)pdpSymbolStreams[block];
    //    streamInfo.symbolCount = symbolCountPerStream;
//
    //    // streamInfo.dpCodewordStream, dpOffsets, dpDecodeTable will be filled later, with pointers into our contiguous buffers (see above)
    //}
//
    //util::CudaScopedTimer timerLow(pInstance->Encode.timerDecodeLowDetail);
    //util::CudaScopedTimer timerHigh(pInstance->Encode.timerDecodeHighDetail);
//
    //timerLow("Huffman Decode Symbols");
//
    //timerHigh("Symbols:    Upload (+read BitStream)");
//
    //std::vector<HuffmanDecodeTable>& symbolDecodeTables = resources.symbolDecodeTables;
    ////if(!singleBitStream) {
    ////    // read decode tables
    ////    ScopedProfileSample sample1(pInstance->m_pProfiler, "read decode table");
    ////    //#pragma omp parallel for
    ////    for(int block = 0; block < int(streamCount); block++) {
    ////        BitStreamReadOnly& bitStream = *ppBitStreams[block];
    ////        symbolDecodeTables[block].readFromBitStream(pInstance, bitStream);
    ////    }
    ////}
//
    //size_t symbolDecodeTablesBufferOffset = 0;
    //uint* pCodewordStreamsNext = resources.pCodewordStreams;
    //uint* dpCodewordStreamsNext = dpCodewordStreams;
    //uint* pOffsetsNext = resources.pSymbolOffsets;
    //uint* dpOffsetsNext = dpOffsets;
    //// read and upload decode tables, upload codeword streams and offsets, and fill stream info
    //for(uint block = 0; block < streamCount; block++) {
    //    BitStreamReadOnly& bitStream = *ppBitStreams[singleBitStream ? 0 : block];
//
    //    /*if(singleBitStream)*/ {
    //        // read decode table
    //        ScopedProfileSample sample1(pInstance->m_pProfiler, "read decode table");
    //        symbolDecodeTables[block].readFromBitStream(pInstance, bitStream);
    //    }
//
    //    HuffmanGPUStreamInfo& streamInfo = resources.pSymbolStreamInfos[block];
//
    //    // copy decode table into upload buffer
    //    { ScopedProfileSample sample1(pInstance->m_pProfiler, "copy decode table into upload buffer");
    //        symbolDecodeTables[block].copyToBuffer(pInstance, resources.pSymbolDecodeTablesBuffer + symbolDecodeTablesBufferOffset);
    //        streamInfo.dpDecodeTable = dpDecodeTables + symbolDecodeTablesBufferOffset;
    //        streamInfo.decodeSymbolTableSize = symbolDecodeTables[block].getSymbolTableSize();
    //        symbolDecodeTablesBufferOffset += getAlignedSize(symbolDecodeTables[block].computeGPUSize(pInstance), 128);
    //    }
//
    //    // copy codewords into pinned buffer
    //    { ScopedProfileSample sample1(pInstance->m_pProfiler, "copy codewords into pinned buffer");
    //        uint codewordBitsize;
    //        bitStream.readAligned(&codewordBitsize, 1);
    //        uint codewordUints = getNumUintsForBits(codewordBitsize);
    //        const uint* pCodewordStream = bitStream.getRaw() + bitStream.getBitPosition() / (sizeof(uint)*8);
    //        bitStream.skipBits(codewordUints * sizeof(uint) * 8);
    //        memcpy(pCodewordStreamsNext, pCodewordStream, codewordUints * sizeof(uint));
    //        streamInfo.dpCodewordStream = dpCodewordStreamsNext;
    //        //TODO align?
    //        pCodewordStreamsNext += codewordUints;
    //        dpCodewordStreamsNext += codewordUints;
    //    }
//
    //    // get offsets pointer
    //    uint numOffsets = getNumOffsets(symbolCountPerStream, pInstance->m_codingBlockSize);
    //    bitStream.align<uint>();
    //    const uint* pOffsets = bitStream.getRaw() + bitStream.getBitPosition() / (sizeof(uint)*8);
    //    bitStream.skipAligned<ushort>(numOffsets);
//
    //    // make offsets absolute (prefix sum)
    //    { ScopedProfileSample sample1(pInstance->m_pProfiler, "make offsets absolute (prefix sum)");
    //        unpackInc16CPU(pOffsetsNext, (const ushort*)pOffsets, numOffsets);
    //        streamInfo.dpOffsets = dpOffsetsNext;
    //        //TODO align?
    //        pOffsetsNext += numOffsets;
    //        dpOffsetsNext += numOffsets;
    //    }
    //}
//
    //// upload decode tables
    //cudaSafeCall(cudaMemcpyAsync(dpDecodeTables, resources.pSymbolDecodeTablesBuffer, symbolDecodeTablesBufferOffset, cudaMemcpyHostToDevice, pInstance->m_stream));
//
    //// upload codewords
    //size_t codewordUintsTotal = pCodewordStreamsNext - resources.pCodewordStreams;
    //cudaSafeCall(cudaMemcpyAsync(dpCodewordStreams, resources.pCodewordStreams, codewordUintsTotal * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));
//
    //// upload offsets
    //size_t offsetCountTotal = pOffsetsNext - resources.pSymbolOffsets;
    //cudaSafeCall(cudaMemcpyAsync(dpOffsets, resources.pSymbolOffsets, offsetCountTotal * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));
//
    //timerHigh("Symbols:    Huffman Decode");
//
    //// decode symbols
    ////FIXME huffmanDecode requires the symbol streams to be padded, which we can't guarantee here..
    //huffmanDecode(pInstance, resources.pSymbolStreamInfos, streamCount, pInstance->m_codingBlockSize);
//
    //timerLow();
    //timerHigh();
//
    //cudaSafeCall(cudaEventRecord(resources.syncEvent, pInstance->m_stream));
//
    //pInstance->releaseBuffers(3);

    return true;
}



bool encodeRLHuff(GpuInstance* pInstance, BitStream* ppBitStreams[], bool singleBitStream, const Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return encodeRLHuff<Symbol16>(pInstance, ppBitStreams, singleBitStream, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeRLHuff(GpuInstance* pInstance, BitStreamReadOnly* ppBitStreams[], bool singleBitStream, Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return decodeRLHuff<Symbol16>(pInstance, ppBitStreams, singleBitStream, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeHuff(GpuInstance* pInstance, BitStream* ppBitStreams[], bool singleBitStream, /*const*/ Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return encodeHuff<Symbol16>(pInstance, ppBitStreams, singleBitStream, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeHuff(GpuInstance* pInstance, BitStreamReadOnly* ppBitStreams[], bool singleBitStream, Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return decodeHuff<Symbol16>(pInstance, ppBitStreams, singleBitStream, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeRLHuff(GpuInstance* pInstance, BitStream* ppBitStreams[], bool singleBitStream, const Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return encodeRLHuff<Symbol32>(pInstance, ppBitStreams, singleBitStream, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeRLHuff(GpuInstance* pInstance, BitStreamReadOnly* ppBitStreams[], bool singleBitStream, Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return decodeRLHuff<Symbol32>(pInstance, ppBitStreams, singleBitStream, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeHuff(GpuInstance* pInstance, BitStream* ppBitStreams[], bool singleBitStream, /*const*/ Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return encodeHuff<Symbol32>(pInstance, ppBitStreams, singleBitStream, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeHuff(GpuInstance* pInstance, BitStreamReadOnly* ppBitStreams[], bool singleBitStream, Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return decodeHuff<Symbol32>(pInstance, ppBitStreams, singleBitStream, pdpSymbolStreams, streamCount, symbolCountPerStream);
}


// INTERFACE FUNCTIONS

// single bitstream for all blocks
bool encodeRLHuff(GpuInstance* pInstance, BitStream& bitStream, const Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    BitStream* pBitStream = &bitStream;
    return encodeRLHuff(pInstance, &pBitStream, true, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeRLHuff(GpuInstance* pInstance, BitStreamReadOnly& bitStream, Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    BitStreamReadOnly* pBitStream = &bitStream;
    return decodeRLHuff(pInstance, &pBitStream, true, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeHuff(GpuInstance* pInstance, BitStream& bitStream, /*const*/ Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    BitStream* pBitStream = &bitStream;
    return encodeHuff(pInstance, &pBitStream, true, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeHuff(GpuInstance* pInstance, BitStreamReadOnly& bitStream, Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    BitStreamReadOnly* pBitStream = &bitStream;
    return decodeHuff(pInstance, &pBitStream, true, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeRLHuff(GpuInstance* pInstance, BitStream& bitStream, const Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    BitStream* pBitStream = &bitStream;
    return encodeRLHuff(pInstance, &pBitStream, true, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeRLHuff(GpuInstance* pInstance, BitStreamReadOnly& bitStream, Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    BitStreamReadOnly* pBitStream = &bitStream;
    return decodeRLHuff(pInstance, &pBitStream, true, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeHuff(GpuInstance* pInstance, BitStream& bitStream, /*const*/ Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    BitStream* pBitStream = &bitStream;
    return encodeHuff(pInstance, &pBitStream, true, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeHuff(GpuInstance* pInstance, BitStreamReadOnly& bitStream, Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    BitStreamReadOnly* pBitStream = &bitStream;
    return decodeHuff(pInstance, &pBitStream, true, pdpSymbolStreams, streamCount, symbolCountPerStream);
}


// separate bitstream for each block (but may contain duplicates)
bool encodeRLHuff(GpuInstance* pInstance, BitStream* ppBitStreams[], const Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return encodeRLHuff(pInstance, ppBitStreams, false, pdpSymbolStreams, streamCount, symbolCountPerStream);

}
bool decodeRLHuff(GpuInstance* pInstance, BitStreamReadOnly* ppBitStreams[], Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return decodeRLHuff(pInstance, ppBitStreams, false, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeHuff(GpuInstance* pInstance, BitStream* ppBitStreams[], /*const*/ Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return encodeHuff(pInstance, ppBitStreams, false, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeHuff(GpuInstance* pInstance, BitStreamReadOnly* ppBitStreams[], Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return decodeHuff(pInstance, ppBitStreams, false, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeRLHuff(GpuInstance* pInstance, BitStream* ppBitStreams[], const Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return encodeRLHuff(pInstance, ppBitStreams, false, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeRLHuff(GpuInstance* pInstance, BitStreamReadOnly* ppBitStreams[], Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return decodeRLHuff(pInstance, ppBitStreams, false, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeHuff(GpuInstance* pInstance, BitStream* ppBitStreams[], /*const*/ Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return encodeHuff(pInstance, ppBitStreams, false, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeHuff(GpuInstance* pInstance, BitStreamReadOnly* ppBitStreams[], Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return decodeHuff(pInstance, ppBitStreams, false, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

}
