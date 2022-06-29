#include "Test.hpp"
#include "largeVis/LineCounter.hpp"
#include "largeVis/RenderLineCounter.hpp"
#include "largeVis/CpuLineCounter.hpp"
#include "largeVis/RoaringCounter.hpp"
#include "largeVis/Renderer.hpp"
#include "range.hpp"
#include "PCUtil.h"
#include "compression/gpuCompression/Encode.hpp"
#include "compression/gpuCompression/Huffman.hpp"
#include "compression/cpuCompression/EncodeCPU.h"
#include <iostream>
#include "compression/gpuCompression/Util.hpp"

void TEST(const VkUtil::Context& context, const TestInfo& testInfo){
    // range testing ------------------------------
    //for(int i: range(0, 100)){
    //    std::cout << i << " ";
    //}
    //std::cout << std::endl;
    //for(int i: range(100, 50, -10)){
    //    std::cout << i << " ";
    //}
    //std::cout << std::endl;
    //range speedtest -----------------------------
    //const int end = 1 << 27;
    //const int inc = 2;
    //std::max<int>(1, 2);
    //{
    //    PCUtil::Stopwatch watch(std::cout, "range for");
    //    volatile int max = 0;
    //    for(int i: range(0, end, inc)){
    //        max = std::max(int(max), i);
    //    }
    //    assert(max = end - 1);
    //}
    //{
    //    PCUtil::Stopwatch watch(std::cout, "static range for");
    //    volatile int max = 0;
    //    for(int i: static_range<0, end>()){
    //        max = std::max(int(max), i);
    //    }
    //    assert(max = end - 1);
    //}
    //{
    //    PCUtil::Stopwatch watch(std::cout, "normal for");
    //    volatile int max = 0;
    //    for(int i = 0; i < end; i += inc){
    //        max = std::max(int(max), i);
    //    }
    //    assert(max = end - 1);
    //}

    // line counting tests -------------------------------------
    //RenderLineCounter::tests(RenderLineCounter::CreateInfo{VkUtil::Context{{0,0}, g_PhysicalDevice, g_Device, g_DescriptorPool, g_PcPlotCommandPool, g_Queue}});
	//LineCounter::tests(LineCounter::CreateInfo{context});
	//compression::testCounting();
	//compression::testRoaringCounting();
    //compression::testRoaringRealWorld();

    // testing the rendering pipeline creation ----------------------------
    //auto renderer = compression::Renderer::acquireReference({context, testInfo.pcNoClearPass, testInfo.pcFramebuffer});

    // testing gpu decompression
    //vkCompress::decodeRLHuff({}, {}, (vkCompress::Symbol16**){}, {}, {});
    return;
    vkCompress::GpuInstance gpu(context, 1, 1 << 20, 0, 0);
    const uint symbolsSize = 1 << 20;
    std::vector<uint16_t> symbols(symbolsSize);
    for(auto& s: symbols)
        s = rand() & 0xff;
    cudaCompress::BitStream bitStream;
    cudaCompress::BitStream* arr[]{&bitStream};
    std::vector<cudaCompress::Symbol16>* sArr[]{&symbols};
    cudaCompress::encodeRLHuffCPU(arr, sArr, 1, gpu.m_codingBlockSize);
    cudaCompress::BitStreamReadOnly readStream(bitStream.getRaw(), bitStream.getBitSize());
    cudaCompress::BitStreamReadOnly* bArr[]{&readStream};
    cudaCompress::decodeRLHuffCPU(bArr, sArr, symbolsSize, 1, gpu.m_codingBlockSize);
    uint bitStreamSize = bitStream.getRawSizeBytes();
    uint originalSize = symbols.size() * sizeof(symbols[0]);
    auto cpuData = vkCompress::parseCpuRLHuffData(&gpu, bitStream.getVector(), gpu.m_codingBlockSize);
    RLHuffDecodeDataGpu gpuData(&gpu, cpuData);
    //vkCompress::decodeHuff()

    VkCommandBuffer commands;
    VkUtil::createCommandBuffer(context.device, context.commandPool, &commands);
    auto &resources = gpu.Encode.Decode[0];
    auto& streamInfo = resources.pSymbolStreamInfos[0]; // we assume here to only have a single decoding block! -> index 0
    
    streamInfo.symbolCount = symbolsSize;

    streamInfo.dpDecodeTable = VkUtil::getBufferAddress(context.device, gpuData.buffer) + gpuData.symbolTableOffset;
    streamInfo.decodeSymbolTableSize = cpuData.symbolTable.getSymbolTableSize();

    streamInfo.dpCodewordStream = VkUtil::getBufferAddress(context.device, gpuData.buffer) + gpuData.symbolStreamOffset;

    streamInfo.dpOffsets = VkUtil::getBufferAddress(context.device, gpuData.buffer) + gpuData.zeroCountOffsetsOffset;
    VkUtil::uploadData(context.device, resources.memory, resources.streamInfosOffset, sizeof(vkCompress::HuffmanGPUStreamInfo), &streamInfo);
    vkCompress::huffmanDecode(&gpu, commands, resources.streamInfoSet, 1, gpu.m_codingBlockSize);

    VkUtil::commitCommandBuffer(context.queue, commands);
    check_vk_result(vkQueueWaitIdle(context.queue));
}