#include "Huffman.hpp"
#include "GpuInstance.hpp"
#include "../cpuCompression/util.h"

namespace vkCompress
{
    const uint COMPACTIFY_ELEM_PER_THREAD = 8;
    auto getPrefixCount = [](uint symbolCount) { return (symbolCount + COMPACTIFY_ELEM_PER_THREAD - 1) / COMPACTIFY_ELEM_PER_THREAD; };

    size_t huffmanGetRequiredMemory(const GpuInstance* pInstance)
    {
        uint streamCountMax = pInstance->m_streamCountMax;
        uint symbolCountPerStreamMax = pInstance->m_elemCountPerStreamMax;

        size_t sizeEncode = 0;
        size_t sizeDecode = 0;
    
        // encode: dpStreamInfos
        sizeEncode += getAlignedSize(sizeof(HuffmanGPUStreamInfo) * streamCountMax, 128);
    
        // encode: dpScratch
        uint prefixCountMax = getPrefixCount(symbolCountPerStreamMax);
        uint scratchBytes = (uint)getAlignedSize((prefixCountMax + 1) * sizeof(uint), 128);
        sizeEncode += streamCountMax * getAlignedSize(scratchBytes, 128);
        // encode: dppScratch
        sizeEncode += getAlignedSize(streamCountMax * sizeof(uint*), 128);
    
        // encode: dpScanTotal
        sizeEncode += getAlignedSize(streamCountMax * sizeof(uint), 128);
    
        // decode: dpStreamInfos
        sizeDecode += getAlignedSize(sizeof(HuffmanGPUStreamInfo) * streamCountMax, 128);
    
        return max(sizeEncode, sizeDecode);
    }

    bool huffmanInit(GpuInstance* pInstance)
    {
        uint streamCountMax = pInstance->m_streamCountMax;

        // TODO: vulkan allocation
        //cudaSafeCall(cudaMallocHost(&pInstance->Huffman.pReadback, streamCountMax * sizeof(uint)));
    
        //cudaSafeCall(cudaEventCreateWithFlags(&pInstance->Huffman.syncEventReadback, cudaEventDisableTiming));

        return true;
    }

    bool huffmanShutdown(GpuInstance* pInstance) 
    {
        //TODO free vulkan memory...
        //cudaSafeCall(cudaEventDestroy(pInstance->Huffman.syncEventReadback));
        pInstance->Huffman.syncEventReadback = 0;

        //cudaSafeCall(cudaFreeHost(pInstance->Huffman.pReadback));
        pInstance->Huffman.pReadback = nullptr;

        return true;
    }
    
    bool huffmanEncode(GpuInstance* pInstance, const HuffmanGPUStreamInfo* pStreamInfos, uint streamCount, uint codingBlockSize, uint* pCompressedSizeBits) 
    {
        // currently empty. All encoding has to be done via the cpu
    }
    
    bool huffmanDecode(GpuInstance* pInstance, const HuffmanGPUStreamInfo* pStreamInfos, uint streamCount, uint codingBlockSize) 
    {
        assert(streamCount <= pInstance->m_streamCountMax);

        bool longSymbols = (pInstance->m_log2HuffmanDistinctSymbolCountMax > 16);
    
        HuffmanGPUStreamInfo* dpStreamInfos = pInstance->getBuffer<HuffmanGPUStreamInfo>(streamCount);
    
        util::CudaScopedTimer timer(pInstance->Huffman.timerDecode);
    
        timer("Upload Info");
    
        // upload stream infos
        cudaSafeCall(cudaMemcpyAsync(dpStreamInfos, pStreamInfos, sizeof(HuffmanGPUStreamInfo) * streamCount, cudaMemcpyHostToDevice, pInstance->m_stream));
        // note: we don't sync on this upload - we trust that the caller won't overwrite/delete the array...
    
        timer("Decode");
    
        // get max number of symbols
        uint symbolCountPerStreamMax = 0;
        for(uint i = 0; i < streamCount; i++)
            symbolCountPerStreamMax = max(symbolCountPerStreamMax, pStreamInfos[i].symbolCount);
    
        if(symbolCountPerStreamMax == 0) {
            pInstance->releaseBuffer();
            return true;
        }
    
        // launch decode kernel
        uint threadCountPerStream = (symbolCountPerStreamMax + codingBlockSize - 1) / codingBlockSize;
        uint blockSize = min(192u, threadCountPerStream);
        blockSize = max(blockSize, HUFFMAN_LOOKUP_SIZE);
        assert(blockSize >= HUFFMAN_LOOKUP_SIZE);
        dim3 blockCount((threadCountPerStream + blockSize - 1) / blockSize, streamCount);
    
        if(longSymbols) {
            huffmanDecodeKernel<Symbol32><<<blockCount, blockSize, 0, pInstance->m_stream>>>(dpStreamInfos, codingBlockSize);
        } else {
            huffmanDecodeKernel<Symbol16><<<blockCount, blockSize, 0, pInstance->m_stream>>>(dpStreamInfos, codingBlockSize);
        }
        cudaCheckMsg("huffmanDecodeKernel execution failed");
    
        timer("Transpose");
    
        // launch transpose kernel
        dim3 blockSizeTranspose(TRANSPOSE_BLOCKDIM_X, TRANSPOSE_BLOCKDIM_Y);
        dim3 blockCountTranspose((symbolCountPerStreamMax + WARP_SIZE * codingBlockSize - 1) / (WARP_SIZE * codingBlockSize), streamCount);
    
        if(longSymbols) {
            switch(codingBlockSize) {
                case 32:
                    huffmanDecodeTransposeKernel<Symbol32, 32><<<blockCountTranspose, blockSizeTranspose, 0, pInstance->m_stream>>>(dpStreamInfos);
                    break;
                case 64:
                    huffmanDecodeTransposeKernel<Symbol32, 64><<<blockCountTranspose, blockSizeTranspose, 0, pInstance->m_stream>>>(dpStreamInfos);
                    break;
                case 128:
                    huffmanDecodeTransposeKernel<Symbol32, 128><<<blockCountTranspose, blockSizeTranspose, 0, pInstance->m_stream>>>(dpStreamInfos);
                    break;
                case 256:
                    huffmanDecodeTransposeKernel<Symbol32, 256><<<blockCountTranspose, blockSizeTranspose, 0, pInstance->m_stream>>>(dpStreamInfos);
                    break;
                default:
                    assert(false);
            }
        } else {
            switch(codingBlockSize) {
                case 32:
                    huffmanDecodeTransposeKernel<Symbol16, 32><<<blockCountTranspose, blockSizeTranspose, 0, pInstance->m_stream>>>(dpStreamInfos);
                    break;
                case 64:
                    huffmanDecodeTransposeKernel<Symbol16, 64><<<blockCountTranspose, blockSizeTranspose, 0, pInstance->m_stream>>>(dpStreamInfos);
                    break;
                case 128:
                    huffmanDecodeTransposeKernel<Symbol16, 128><<<blockCountTranspose, blockSizeTranspose, 0, pInstance->m_stream>>>(dpStreamInfos);
                    break;
                case 256:
                    huffmanDecodeTransposeKernel<Symbol16, 256><<<blockCountTranspose, blockSizeTranspose, 0, pInstance->m_stream>>>(dpStreamInfos);
                    break;
                default:
                    assert(false);
            }
        }
        cudaCheckMsg("huffmanDecodeTransposeKernel execution failed");
    
        timer();
    
        pInstance->releaseBuffer();
    
        return true;
        }
}