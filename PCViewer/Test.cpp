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
    vkCompress::GpuInstance gpu(context, 1, 1 << 20, 0, 0);
    const uint symbolsSize = 1 << 20;
    std::vector<uint16_t> symbols(symbolsSize);
    for(auto& s: symbols)
        s = rand() & std::numeric_limits<uint16_t>::max();
    cudaCompress::BitStream bitStream;
    cudaCompress::BitStream* arr[]{&bitStream};
    std::vector<cudaCompress::Symbol16>* sArr[]{&symbols};
    cudaCompress::encodeHuffCPU(arr, sArr, 1, symbols.size());
    uint bitStreamSize = bitStream.getRawSizeBytes();
    uint originalSize = symbols.size() * sizeof(symbols[0]);
    //vkCompress::decodeHuff()
}