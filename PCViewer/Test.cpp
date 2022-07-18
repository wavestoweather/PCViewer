#define NOSTATICS
#include "Test.hpp"
#include "largeVis/LineCounter.hpp"
#include "largeVis/RenderLineCounter.hpp"
#include "largeVis/CpuLineCounter.hpp"
#include "largeVis/RoaringCounter.hpp"
#include "largeVis/Renderer.hpp"
#include "range.hpp"
#include "PCUtil.h"
#include "VkUtil.h"
#include "compression/gpuCompression/Encode.hpp"
#include "compression/gpuCompression/Huffman.hpp"
#include "compression/cpuCompression/EncodeCPU.h"
#include <iostream>
#include "compression/gpuCompression/Util.hpp"
#include "compression/cpuCompression/EncodeCPU.h"
#include "compression/cpuCompression/DWTCpu.h"
#include "compression/gpuCompression/Scan.hpp"
#include "compression/gpuCompression/Quantize.hpp"
#include "compression/gpuCompression/DWT.hpp"
#include "largeVis/OpenCompressedDataset.hpp"
#include "largeVis/DecompressManager.hpp"
#include "compression/cpuCompression/HuffmanCPU.h"

// note: src vector is changed!
static void compressVector(std::vector<float>& src, float quantizationStep, /*out*/ cudaCompress::BitStream& bitStream, uint32_t& symbolsSize){
    //compressing the data with 2 dwts, followed by run-length and huffman encoding of quantized symbols
    //padding to size % 4 size
    uint32_t originalLength = src.size();
    uint32_t paddedLength = ((4 - (src.size() & 0b11)) & 0b11) + src.size();
    src.resize(paddedLength); 
    std::vector<float> tmp(paddedLength);
    cudaCompress::util::dwtFloatForwardCPU(tmp.data(), src.data(), src.size(), 0, 0);
    std::copy(tmp.begin(), tmp.begin() + paddedLength / 2, src.begin());
    cudaCompress::util::dwtFloatForwardCPU(src.data(), tmp.data(), tmp.size() / 2, tmp.size() / 2, tmp.size() / 2);
    std::vector<cudaCompress::Symbol16> symbols(src.size());
    cudaCompress::util::quantizeToSymbols(symbols.data(), src.data(), src.size(), quantizationStep);
	cudaCompress::BitStream* arr[]{&bitStream};
    std::vector<cudaCompress::Symbol16>* sArr[]{&symbols};
    cudaCompress::encodeRLHuffCPU(arr, sArr, 1, 128);//symbols.size());
    symbolsSize = symbols.size();
}

// note: src vector is changed!
static std::pair<cudaCompress::BitStream, uint32_t> compressVector(std::vector<float>& src, float quantizationStep){
    std::pair<cudaCompress::BitStream, uint32_t> t;
    compressVector(src, quantizationStep, t.first, t.second);
    return t;
}

static void decompressVector(std::vector<uint32_t> src, float quantizationStep, uint32_t symbolsSize, /*out*/ std::vector<float>& data){
    cudaCompress::BitStreamReadOnly bs(src.data(), src.size() * sizeof(src[0]) * 8);
	cudaCompress::BitStreamReadOnly* dec[]{&bs};
	std::vector<cudaCompress::Symbol16> nS(symbolsSize);
	std::vector<cudaCompress::Symbol16>* ss[]{&nS};
	cudaCompress::decodeRLHuffCPU(dec, ss, symbolsSize, 1, 128);//symbolsSize);
	std::vector<float> result2(symbolsSize);
    data.resize(symbolsSize);
	cudaCompress::util::unquantizeFromSymbols(data.data(), nS.data(), nS.size(), quantizationStep);
	result2 = data;
	cudaCompress::util::dwtFloatInverseCPU(result2.data(), data.data(), data.size() / 2, data.size() / 2, data.size() / 2);
	cudaCompress::util::dwtFloatInverseCPU(data.data(), result2.data(), data.size());
}

static std::vector<float> vkDecompress(const VkUtil::Context& context, vkCompress::GpuInstance& gpu, const RLHuffDecodeDataCpu& cpuData, const RLHuffDecodeDataGpu& gpuData, float quantizationStep, uint32_t symbolsSize){
    // creating buffer for the symbol table
    uint pad = gpu.m_subgroupSize * gpu.m_codingBlockSize * sizeof(uint16_t);
    uint paddedSymbols = (symbolsSize * sizeof(uint16_t) + pad - 1) / pad * pad;
    // symbolBuffer holds enough memory to store all intermediate data as well: this means that we need 2 * float vector containing all data
    uint flags  = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    auto [symbolBuffer, offs, mem] = VkUtil::createMultiBufferBound(context, {paddedSymbols * 2, paddedSymbols * 2}, {flags, flags}, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkCommandBuffer commands;
    VkUtil::createCommandBuffer(context.device, context.commandPool, &commands);
    
    VkDeviceAddress srcA = VkUtil::getBufferAddress(context.device, symbolBuffer[0]); // the symbol buffer containes the quantized values
    VkDeviceAddress dstA = VkUtil::getBufferAddress(context.device, symbolBuffer[1]); // is the padded offset * 2 as the offset is for halves

    vkCompress::decodeRLHuffHalf(&gpu, cpuData, gpuData, srcA, commands);
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, {}, 0, {}, 0, {});
    
    vkCompress::unquantizeFromSymbols(&gpu, commands, dstA, srcA, symbolsSize, quantizationStep);
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, {}, 0, {}, 0, {});

    vkCompress::dwtFloatInverse(&gpu, commands, srcA, dstA, symbolsSize / 2, symbolsSize / 2, symbolsSize / 2);
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, {}, 0, {}, 0, {});
    VkBufferCopy cpy{};
    cpy.dstOffset = 0;
    cpy.srcOffset = 0;
    cpy.size = symbolsSize / 2 * sizeof(float);
    vkCmdCopyBuffer(commands, symbolBuffer[1], symbolBuffer[0], 1, &cpy);
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, {}, 0, {}, 0, {});
    vkCompress::dwtFloatToHalfInverse(&gpu, commands, dstA, srcA, symbolsSize);

    VkUtil::commitCommandBuffer(context.queue, commands);
    auto err = vkQueueWaitIdle(context.queue); check_vk_result(err);
    std::vector<half> final(symbolsSize);
    //VkUtil::downloadData(context.device, mem, offs[1], symbolsSize * sizeof(final[0]), final.data());
    VkUtil::downloadDataIndirect(context, symbolBuffer[1], symbolsSize * sizeof(final[0]), final.data());
    return std::vector<float>(final.begin(), final.end());
}

static std::vector<float> vkDecompress(const VkUtil::Context& context, std::vector<uint32_t> src, float quantizationStep, uint32_t symbolsSize){
    vkCompress::GpuInstance gpu(context, 1, symbolsSize, 0, 0);
    auto cpuData = vkCompress::parseCpuRLHuffData(&gpu, src, gpu.m_codingBlockSize);
    RLHuffDecodeDataGpu gpuData(&gpu, cpuData);

    return vkDecompress(context, gpu, cpuData, gpuData, quantizationStep, symbolsSize);    
}

static std::vector<float> vkDecompressBenchmark(const VkUtil::Context& context, vkCompress::GpuInstance& gpu, const RLHuffDecodeDataCpu& cpuData, const RLHuffDecodeDataGpu& gpuData, float quantizationStep, uint32_t symbolsSize){
    VkQueryPool timings{};
    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = 9;
    vkCreateQueryPool(context.device, &info, nullptr, &timings);

    // creating buffer for the symbol table
    uint pad = gpu.m_subgroupSize * gpu.m_codingBlockSize * sizeof(uint16_t);
    uint paddedSymbols = (symbolsSize * sizeof(uint16_t) + pad - 1) / pad * pad;
    // symbolBuffer holds enough memory to store all intermediate data as well: this means that we need 2 * float vector containing all data
    uint flags  = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    auto [symbolBuffer, offs, mem] = VkUtil::createMultiBufferBound(context, {paddedSymbols * 2, paddedSymbols * 2}, {flags, flags}, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkCommandBuffer commands;
    VkUtil::createCommandBuffer(context.device, context.commandPool, &commands);
    
    VkDeviceAddress srcA = VkUtil::getBufferAddress(context.device, symbolBuffer[0]); // the symbol buffer containes the quantized values
    VkDeviceAddress dstA = VkUtil::getBufferAddress(context.device, symbolBuffer[1]); // is the padded offset * 2 as the offset is for halves

    vkCmdResetQueryPool(commands, timings, 0, info.queryCount);
    //vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timings, 0);
    vkCompress::decodeRLHuffHalf(&gpu, cpuData, gpuData, srcA, commands, {timings, 0, 5});
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, {}, 0, {}, 0, {});
    //vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timings, 1);

    vkCompress::unquantizeFromSymbols(&gpu, commands, dstA, srcA, symbolsSize, quantizationStep);
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timings, 5);
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, {}, 0, {}, 0, {});

    vkCompress::dwtFloatInverse(&gpu, commands, srcA, dstA, symbolsSize / 2, symbolsSize / 2, symbolsSize / 2);
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timings, 6);
    //vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, {}, 0, {}, 0, {});
    VkBufferCopy cpy{};
    cpy.dstOffset = 0;
    cpy.srcOffset = 0;
    cpy.size = symbolsSize / 2 * sizeof(float);
    vkCmdCopyBuffer(commands, symbolBuffer[1], symbolBuffer[0], 1, &cpy);
    //vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, {}, 0, {}, 0, {});
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timings, 7);
    vkCompress::dwtFloatToHalfInverse(&gpu, commands, dstA, srcA, symbolsSize);
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timings, 8);

    std::vector<uint32_t> timestamps(info.queryCount);
    {
    PCUtil::Stopwatch cpuWatch(std::cout, "GpuDecompressionTime");
    VkUtil::commitCommandBuffer(context.queue, commands);
    check_vk_result(vkGetQueryPoolResults(context.device, timings, 0, info.queryCount, timestamps.size() * sizeof(uint32_t), timestamps.data(), sizeof(uint32_t), VK_QUERY_RESULT_WAIT_BIT));
    check_vk_result(vkQueueWaitIdle(context.queue));
    }

    std::vector<std::string_view> timingNames{"Huff Symbol", "Huff Zeros", "RLScan", "RLScatter", "Unquantize", "DWT Inverse", "DWT Copy", "DWT Inverse Full"};
    for(int i: irange(timingNames)){
        std::cout << std::left << "[timing] " << std::setw(17) << timingNames[i] << " : " << (timestamps[i + 1] - timestamps[i]) * 1e-6 << " ms" << std::endl;
    }

    std::vector<half> final(symbolsSize);
    //VkUtil::downloadData(context.device, mem, offs[1], symbolsSize * sizeof(final[0]), final.data());
    VkUtil::downloadDataIndirect(context, symbolBuffer[1], symbolsSize * sizeof(final[0]), final.data());
    vkDestroyBuffer(context.device, symbolBuffer[0], nullptr);
    vkDestroyBuffer(context.device, symbolBuffer[1], nullptr);
    vkDestroyQueryPool(context.device, timings, nullptr);
    vkFreeMemory(context.device, mem, nullptr);
    return std::vector<float>(final.begin(), final.end());
}

static std::vector<float> vkDecompressBenchmark(const VkUtil::Context& context, std::vector<uint32_t> src, float quantizationStep, uint32_t symbolsSize){
    vkCompress::GpuInstance gpu(context, 1, symbolsSize, 0, 0);
    auto cpuData = vkCompress::parseCpuRLHuffData(&gpu, src, gpu.m_codingBlockSize);
    RLHuffDecodeDataGpu gpuData(&gpu, cpuData);

    return vkDecompressBenchmark(context, gpu, cpuData, gpuData, quantizationStep, symbolsSize);
}

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
    const bool testDecomp = false;
    const bool testExclusiveScan = false;
    const bool testUnquanzite = false;
    const bool testDWTInverse = false;
    const bool testDWTInverseToHalf = false;
    const bool testFullDecomp = false;
    const bool testDecompressManager = false;
    const bool testRealWorldDataCompression = false;
    const bool testRealWorldHuffmanDetail = false;
    const bool testUnquantizePerformance = false;
    const bool testUPloadSpeed = true;
    if(testDecomp){
        vkCompress::GpuInstance gpu(context, 1, 1 << 20, 0, 0);
        const uint symbolsSize = 1 << 20;
        std::vector<uint16_t> symbols(symbolsSize), symbolsCpu(symbolsSize);
        srand(10);  //seeding
        for(auto& s: symbols)
            s = rand() & 0xff;
        cudaCompress::BitStream bitStream;
        cudaCompress::BitStream* arr[]{&bitStream};
        std::vector<cudaCompress::Symbol16>* sArr[]{&symbols};
        {
        PCUtil::Stopwatch encodeWatch(std::cout, "Encoding Time");
        cudaCompress::encodeRLHuffCPU(arr, sArr, 1, gpu.m_codingBlockSize);
        //cudaCompress::encodeHuffCPU(arr, sArr, 1, gpu.m_codingBlockSize);
        }
        cudaCompress::BitStreamReadOnly readStream(bitStream.getRaw(), bitStream.getBitSize());
        cudaCompress::BitStreamReadOnly* bArr[]{&readStream};
        sArr[0] = &symbolsCpu;
        cudaCompress::decodeRLHuffCPU(bArr, sArr, symbolsSize, 1, gpu.m_codingBlockSize);
        //cudaCompress::decodeHuffCPU(bArr, sArr, symbolsSize, 1, gpu.m_codingBlockSize);
        uint bitStreamSize = bitStream.getRawSizeBytes();
        uint originalSize = symbols.size() * sizeof(symbols[0]);
        auto cpuData = vkCompress::parseCpuRLHuffData(&gpu, bitStream.getVector(), gpu.m_codingBlockSize);
        std::cout << cpuData.symbolOffsets.size() << std::endl;
        //cpuData.symbolOffsets[1] -= 32;
        RLHuffDecodeDataGpu gpuData(&gpu, cpuData);
        //vkCompress::decodeHuff()

        // creating buffer for the symbol table
        uint pad = gpu.m_subgroupSize * gpu.m_codingBlockSize * sizeof(uint16_t);
        uint paddedSymbols = (symbolsSize * sizeof(uint16_t) + pad - 1) / pad * pad;
        auto [symbolBuffer, offs, mem] = VkUtil::createMultiBufferBound(context, {2 * paddedSymbols}, {VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT}, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        //std::vector<uint16_t> sorted(symbolsSize); uint16_t count{};
        //for(auto& i: sorted) i = count++;
        //VkUtil::uploadData(context.device, mem, 0, symbolsSize * sizeof(uint16_t), sorted.data());

        VkCommandBuffer commands;
        VkUtil::createCommandBuffer(context.device, context.commandPool, &commands);

        VkQueryPool times;
        VkQueryPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        createInfo.queryCount = 2;
        createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        vkCreateQueryPool(context.device, &createInfo, nullptr, &times);
        vkCmdResetQueryPool(commands, times, 0, 2);
        vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, times, 0);
        
        // old debuggin code to test inner workings of huffman decoding --------------------------------------
        //auto &resources = gpu.Encode.Decode[0];
        //auto& streamInfo = resources.pSymbolStreamInfos[0]; // we assume here to only have a single decoding block! -> index 0
        //
        //streamInfo.symbolCount = cpuData.symbolCount;
//
        //streamInfo.dpDecodeTable = VkUtil::getBufferAddress(context.device, gpuData.buffer) + gpuData.symbolTableOffset;
        //streamInfo.decodeSymbolTableSize = cpuData.symbolTable.getSymbolTableSize();
//
        //streamInfo.dpCodewordStream = VkUtil::getBufferAddress(context.device, gpuData.buffer) + gpuData.symbolStreamOffset;
//
        //streamInfo.dpOffsets = VkUtil::getBufferAddress(context.device, gpuData.buffer) + gpuData.symbolOffsetsOffset;
        //streamInfo.dpSymbolStream = VkUtil::getBufferAddress(context.device, symbolBuffer[0]);
        //VkUtil::uploadData(context.device, resources.memory, resources.streamInfosOffset, sizeof(vkCompress::HuffmanGPUStreamInfo), &streamInfo);
        //
        //vkCompress::huffmanDecode(&gpu, commands, resources.streamInfoSet, 1, gpu.m_codingBlockSize);
        //
        //auto& zeroStreamInfo = resources.pZeroCountStreamInfos[0];
        //zeroStreamInfo.symbolCount = cpuData.symbolCount; // symbol count is equivalent to the normal decomrpession symbol count
//
        //zeroStreamInfo.dpDecodeTable = VkUtil::getBufferAddress(context.device, gpuData.buffer) +  gpuData.zeroCountTableOffset;
        //zeroStreamInfo.decodeSymbolTableSize = cpuData.zeroCountTable.getSymbolTableSize();
//
        //zeroStreamInfo.dpCodewordStream = VkUtil::getBufferAddress(context.device, gpuData.buffer) + gpuData.zeroCountStreamOffset;
//
        //zeroStreamInfo.dpOffsets = VkUtil::getBufferAddress(context.device, gpuData.buffer) + gpuData.zeroCountOffsetsOffset;
        //zeroStreamInfo.dpSymbolStream = VkUtil::getBufferAddress(context.device, symbolBuffer[0]) + paddedSymbols;
        //VkUtil::uploadData(context.device, resources.memory, resources.zeroInfosOffset, sizeof(zeroStreamInfo), &zeroStreamInfo);
        //vkCompress::huffmanDecode(&gpu, commands, resources.zeroStreamInfoSet, 1u, gpu.m_codingBlockSize);
        
        // new debugging code for testing complete huffman decoding
        vkCompress::decodeRLHuffHalf(&gpu, cpuData, gpuData, VkUtil::getBufferAddress(context.device, symbolBuffer[0]), commands);

        vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, times, 1);
        uint32_t t[2];
        {
        PCUtil::Stopwatch decodeWatch(std::cout, "Decoding time Gpu");
        VkUtil::commitCommandBuffer(context.queue, commands);
        vkGetQueryPoolResults(context.device, times, 0, 2, sizeof(uint32_t) * 2, t, sizeof(uint32_t), VK_QUERY_RESULT_WAIT_BIT);
        }
        check_vk_result(vkQueueWaitIdle(context.queue)); // have to wait for buffer freeing
        std::cout << (t[1] - t[0]) * 1e-6 << " ms?" << std::endl;

        vkDestroyQueryPool(context.device, times, nullptr);
        std::vector<uint16_t> downloadedData(symbolsSize);
        VkUtil::downloadData(context.device, mem, 0, symbolsSize * sizeof(uint16_t), downloadedData.data());
        downloadedData = std::vector<uint16_t>(downloadedData.end() - 1000, downloadedData.end());
        symbols = std::vector<uint16_t>(symbols.end() - 1000, symbols.end());
        //std::set<uint16_t> block1Orig(symbols.begin(), symbols.begin() + 4096);
        //std::set<uint16_t> block1Gpu(downloadedData.begin(), downloadedData.begin() + 4096);
        //bool equ = block1Orig == block1Gpu;
        vkDestroyBuffer(context.device, symbolBuffer[0], nullptr);
        vkFreeMemory(context.device, mem, nullptr);
        bool test = true;
    }
    if(testExclusiveScan){
        const uint scanSize = 1 << 11;
        vkCompress::GpuInstance gpu(context, 1, scanSize, 0, 0);
        VkCommandBuffer commands;
        VkUtil::createCommandBuffer(context.device, context.commandPool, &commands);
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        auto [buffer, offsets, mem] = VkUtil::createMultiBufferBound(context, {scanSize * 2, scanSize * 4, scanSize * 4}, {usage, usage, usage}, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

        VkDeviceAddress src = VkUtil::getBufferAddress(context.device, buffer[0]);
        VkDeviceAddress dst = VkUtil::getBufferAddress(context.device, buffer[1]);
        VkDeviceAddress dstEx = VkUtil::getBufferAddress(context.device, buffer[2]);
        std::vector<short> init(scanSize, 1);
        //short s = 0; for(auto& i: init) i = s++;
        VkUtil::uploadData(context.device, mem, offsets[0], sizeof(init[0]) * init.size(), init.data());
        vkCompress::scanArray<false>(&gpu, commands, dst, src, init.size(), gpu.m_pScanPlan); //testing inclusive scan
        vkCompress::scanArray<true>(&gpu, commands, dstEx, src, init.size(), gpu.m_pScanPlan); //testing exclusive scan
        VkUtil::commitCommandBuffer(context.queue, commands);
        auto err = vkQueueWaitIdle(context.queue); check_vk_result(err);
        
        std::vector<uint32_t> final(init.size());
        VkUtil::downloadData(context.device, mem, offsets[1], final.size() * sizeof(final[0]), final.data());
        std::vector<uint32_t> upper(final.begin() + 1000, final.end());
        VkUtil::downloadData(context.device, gpu.m_pScanPlan->m_blockSumsMemory, 0, sizeof(uint) * 3, final.data());
        for(int i = final.size() - 1; i > 0; --i){
            final[i] -= final[i - 1];
        }
        VkUtil::downloadData(context.device, mem, offsets[2], final.size() * sizeof(final[0]), final.data());
        upper = std::vector<uint32_t>(final.begin() + 1000, final.end());

        bool letssee = true;
    }
    if(testUnquanzite){
        const uint quantSize = 1 << 20;
        const float quantStep = .001f;
        vkCompress::GpuInstance gpu(context, 1, quantSize, 0, 0);
        VkCommandBuffer commands;
        VkUtil::createCommandBuffer(context.device, context.commandPool, &commands);

        std::vector<float> orig(quantSize);
        srand(10);
        for(auto& f: orig)
            f = random() / float(1u << 31);
        
        std::vector<uint16_t> symbols(quantSize);
        cudaCompress::util::quantizeToSymbols(symbols.data(), orig.data(), quantSize, quantStep);

        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        auto [buffer, offsets, mem] = VkUtil::createMultiBufferBound(context, {quantSize * 2, quantSize * 4}, {usage, usage}, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    
        VkUtil::uploadData(context.device, mem, 0, quantSize * 2, symbols.data());
        auto dst = VkUtil::getBufferAddress(context.device, buffer[1]);
        auto src = VkUtil::getBufferAddress(context.device, buffer[0]);
        vkCompress::unquantizeFromSymbols(&gpu, commands, dst, src, quantSize, quantStep);
        {
        PCUtil::Stopwatch unquantWatch(std::cout, "Unquantizationtime");
        VkUtil::commitCommandBuffer(context.queue, commands);
        auto err = vkQueueWaitIdle(context.queue); check_vk_result(err);
        }

        std::vector<float> res(quantSize);
        VkUtil::downloadData(context.device, mem, offsets[1], quantSize * 4, res.data());
        std::vector<float> ref(quantSize);
        cudaCompress::util::unquantizeFromSymbols(ref.data(), symbols.data(), symbols.size(), quantStep);
        res = std::vector<float>(res.end() - 1000, res.end());
        ref = std::vector<float>(ref.end() - 1000, ref.end());
        bool heyho = true;
    }
    if(testDWTInverse){
        const uint size = 1 << 20;
        vkCompress::GpuInstance gpu(context, 1, size, 0, 0);
        VkCommandBuffer commands;
        VkUtil::createCommandBuffer(context.device, context.commandPool, &commands);

        std::vector<float> orig(size);
        srand(10);
        for(auto& f: orig)
            f = random() / float(1u << 31);
        std::vector<float> dst(size);
        cudaCompress::util::dwtFloatForwardCPU(dst.data(), orig.data(), size);

        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        auto [buffer, offsets, mem] = VkUtil::createMultiBufferBound(context, {size * 4, size * 4}, {usage, usage}, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

        VkUtil::uploadData(context.device, mem, 0, size * 4, dst.data());
        auto dstA = VkUtil::getBufferAddress(context.device, buffer[1]);
        auto srcA = VkUtil::getBufferAddress(context.device, buffer[0]); 
        vkCompress::dwtFloatInverse(&gpu, commands, dstA, srcA, size, 0, 0);

        {
        PCUtil::Stopwatch unquantWatch(std::cout, "dwt inverse time");
        VkUtil::commitCommandBuffer(context.queue, commands);
        auto err = vkQueueWaitIdle(context.queue); check_vk_result(err);
        }

        std::vector<float> res(size);
        VkUtil::downloadData(context.device, mem, offsets[1], size * 4, res.data());
        std::vector<float> ref(size);
        cudaCompress::util::dwtFloatInverseCPU(ref.data(), dst.data(), size, 0, 0);
        res = std::vector<float>(res.end() - 1000, res.end());
        ref = std::vector<float>(ref.end() - 1000, ref.end());
        bool heyho = true;
    }
    if(testDWTInverseToHalf){
        const uint size = 1 << 20;
        vkCompress::GpuInstance gpu(context, 1, size, 0, 0);
        VkCommandBuffer commands;
        VkUtil::createCommandBuffer(context.device, context.commandPool, &commands);

        std::vector<float> orig(size);
        srand(10);
        for(auto& f: orig)
            f = random() / float(1u << 31);
        std::vector<float> dst(size);
        cudaCompress::util::dwtFloatForwardCPU(dst.data(), orig.data(), size / 2, size / 2, size / 2);

        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        auto [buffer, offsets, mem] = VkUtil::createMultiBufferBound(context, {size * 4, size * 2}, {usage, usage}, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

        VkUtil::uploadData(context.device, mem, 0, size * 4, dst.data());
        auto dstA = VkUtil::getBufferAddress(context.device, buffer[1]);
        auto srcA = VkUtil::getBufferAddress(context.device, buffer[0]); 
        vkCompress::dwtFloatToHalfInverse(&gpu, commands, dstA, srcA, size /  2, 0, size / 2);

        {
        PCUtil::Stopwatch unquantWatch(std::cout, "dwt inverse time");
        VkUtil::commitCommandBuffer(context.queue, commands);
        auto err = vkQueueWaitIdle(context.queue); check_vk_result(err);
        }

        orig = std::vector<float>(orig.begin() + size / 2, orig.end());

        std::vector<half> res(size);
        VkUtil::downloadData(context.device, mem, offsets[1], size * 2, res.data());
        std::vector<float> conv(res.begin(), res.end());
        bool heyho = true;
    }
    if(testFullDecomp){
        const uint size = 1 << 20;
        const float quantStep = .0001;
        std::vector<float> orig(size);
        srand(10);
        for(auto& f: orig)
            f = random() / float(1u << 31);

        auto copy = orig;
        auto [bitstream, s] = compressVector(orig, quantStep);
        std::vector<float> cpuDecomp;
        decompressVector(bitstream.getVector(), quantStep, size, cpuDecomp);
        auto decomp = vkDecompress(context, bitstream.getVector(), quantStep, size);
        bool letssee = true;
    }

    // testing fucking decompressing blockwise stored data
    {
        //std::ifstream columnFile(hierarchyFolder + "/" + std::to_string(i) + ".comp", std::ios_base::binary);
        //assert(columnFile);
        //struct{uint64_t streamSize; uint32_t symbolSize;}sizes{};
        //while(columnFile.read(reinterpret_cast<char*>(&sizes), sizeof(sizes))){
        //    // streamSize is in bytes, while symbolSize is the resulting size of the decompressed vector
        //    dataVec.resize(sizes.streamSize / sizeof(dataVec[0]));
        //    columnFile.read(reinterpret_cast<char*>(dataVec.data()), sizes.streamSize);
        //    columnData[i].compressedRLHuffCpu.emplace_back(vkCompress::parseCpuRLHuffData(gpuInstance.get(), dataVec));
        //    columnData[i].compressedRLHuffGpu.emplace_back(gpuInstance.get(), columnData[i].compressedRLHuffCpu.back());
        //}
    }

    // testing full decompression on real world data
    if (testDecompressManager){
        const uint32_t cols = 10;
        const uint32_t colSize = 1 << 20;
        const float quantStep = .02f;
        std::vector<std::vector<float>> original(cols, std::vector<float>(colSize));
        srand(20);
        for(int i: irange(cols)){
            for(int j: irange(colSize))
                original[i][j] = random() / float(1u << 31);
        }
        std::vector<cudaCompress::BitStream> compressedStreams(cols);
        for(int i: irange(cols)){
            uint32_t symbolSize;
            auto cpy = original[i];
            compressVector(cpy, quantStep, compressedStreams[i], symbolSize);
        }
        // decompressing the first blocks with the cpu
        std::vector<std::vector<float>> cpuDecodedColumns(cols);
        for(int column: irange(cols)){
            // huffman + rl decoding
            decompressVector(compressedStreams[column].getVector(), quantStep, colSize, cpuDecodedColumns[column]);
        }

        // pgu decoding
        std::vector<std::vector<float>> gpuDecodedColumns(cols);
        for(int column: irange(cols)){
            gpuDecodedColumns[column] = vkDecompress(context, compressedStreams[column].getVector(), quantStep, colSize);
        }

        std::vector<std::vector<float>> gpuDecodeManagerColumns(cols);
        std::vector<RLHuffDecodeDataCpu> cpuDatas;
        std::vector<RLHuffDecodeDataGpu> gpuDatas;
        vkCompress::GpuInstance gpu(context, 1, colSize);
        auto cpuD = vkCompress::parseCpuRLHuffData(&gpu, compressedStreams[0].getVector(), gpu.m_codingBlockSize);
        RLHuffDecodeDataGpu gpuD(&gpu, cpuD);
        DecompressManager decompManager(colSize, gpu, {&cpuD}, {&gpuD});
        std::vector<uint8_t> zeros(gpu.m_pScanPlan->m_memorySize, 0);
        for(int column: irange(cols)){
            cpuDatas.emplace_back(vkCompress::parseCpuRLHuffData(&gpu, compressedStreams[column].getVector(), gpu.m_codingBlockSize));
            gpuDatas.emplace_back(&gpu, cpuDatas[column]);
            //DecompressManager::CpuColumns cpuCols{&cpuData};
            decompManager.executeBlockDecompression(colSize, gpu, {&cpuDatas[column]}, {&gpuDatas[column]}, quantStep);
            auto err = vkQueueWaitIdle(context.queue); check_vk_result(err);
            std::vector<half> final(colSize);
            VkUtil::downloadDataIndirect(context, decompManager.buffers[0], final.size() * sizeof(final[0]), final.data());

            gpuDecodeManagerColumns[column] = std::vector<float>(final.begin(), final.end());
        }

        // differences
        std::vector<float> maxDiffsCpu(cols);
        std::vector<float> maxDiffsGpu(cols);
        std::vector<float> maxDiffsGpuMan(cols);
        std::vector<double> avgDiffsCpu(cols);
        std::vector<double> avgDiffsGpu(cols);
        std::vector<double> avgDiffsGpuMan(cols);
        for(int c: irange(cols)){
            for(int e: irange(colSize)){
                float orig = original[c][e];
                float cpu = std::abs(orig - cpuDecodedColumns[c][e]);
                float gpu = std::abs(orig - gpuDecodedColumns[c][e]);
                float gpuMan = std::abs(orig - gpuDecodeManagerColumns[c][e]);
                maxDiffsCpu[c] = std::max(maxDiffsCpu[c], cpu);
                maxDiffsGpu[c] = std::max(maxDiffsGpu[c], gpu);
                maxDiffsGpuMan[c] = std::max(maxDiffsGpuMan[c], gpuMan);
                avgDiffsCpu[c] += cpu / colSize;
                avgDiffsGpu[c] += gpu / colSize;
                avgDiffsGpuMan[c] += gpuMan / colSize;
            }
        }
        bool test = true;
    }

    if(testRealWorldDataCompression){
        // testing encoding times for real world data
        std::vector<std::string_view> filenames{"/run/media/lachei/3d02119e-bc93-4969-9fc5-523f06321708/w2w/takumi/scripts/tp.bin", "/run/media/lachei/3d02119e-bc93-4969-9fc5-523f06321708/w2w/takumi/scripts/q.bin", "/run/media/lachei/3d02119e-bc93-4969-9fc5-523f06321708/w2w/takumi/scripts/NCCLOUD.bin"};
        std::vector<float> quants{0.001f, .01f, .1f};
        for(auto file: filenames){
            for(float q: quants){
                uint32_t s;
                std::vector<uint32_t> bits;
                size_t bytesize;
                {
                    // encoding
                    std::ifstream f(file.data(), std::ios_base::binary);
                    //assert(f);
                    std::vector<float> tpVals(1024 * 1024);
 
                    f.read(reinterpret_cast<char*>(tpVals.data()), tpVals.size() * sizeof(tpVals[0]));

                    PCUtil::Stopwatch encode(std::cout, "Encode " + std::to_string(q) + std::string(file));
                    auto [stream, size] = compressVector(tpVals, q);
                    bytesize = stream.getRawSizeBytes();
                    bits = std::move(stream.getVector());
                    s = size;
                }
                {
                    // decoding
                    PCUtil::Stopwatch decode(std::cout, "Decode " + std::to_string(q) + std::string(file));
                    std::vector<float> symb = vkDecompressBenchmark(context, bits, q, s);
                    bool test = true;
                }
                std::cout << "Compression Ratio: 1 : " << std::to_string(float(1024 * 1024 * 4) / bytesize) << std::endl;
            }
        }
    }
    if constexpr(testRealWorldHuffmanDetail){

        auto ds = util::openCompressedDataset(context, "/run/media/lachei/TOSHIBA EXT/PCViewer_LargeVis/takumiReduced300mio/comp_only/100mio");
        for(int comp: irange(ds.compressedData.columnData)){
            PCUtil::Stopwatch decode(std::cout, "Decode " + std::to_string(comp));
            std::vector<float> symb = vkDecompressBenchmark(context, *ds.compressedData.gpuInstance, ds.compressedData.columnData[comp].compressedRLHuffCpu[0], ds.compressedData.columnData[comp].compressedRLHuffGpu[0], ds.compressedData.quantizationStep, ds.compressedData.columnData[comp].compressedSymbolSize[0]);
        }
    }
    if constexpr(testUnquantizePerformance){
        const uint32_t size = 100000000;
        vkCompress::GpuInstance gpu(context, 1, size, 0, 0);
        VkBufferUsageFlags flags = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        auto [buffers, offsets, mem] = VkUtil::createMultiBufferBound(context, {size * sizeof(float), size * sizeof(uint16_t)}, {flags, flags}, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        std::vector<uint16_t> symbols(size);
        //srand(10);
        //for(auto& i: symbols)
        //    i = rand() & 0xff;
        //VkUtil::uploadData(context.device, mem, offsets[1], size * sizeof(uint16_t), symbols.data());
        
        VkDeviceAddress symbolAddress = VkUtil::getBufferAddress(context.device, buffers[1]);
        VkDeviceAddress floatAddress = VkUtil::getBufferAddress(context.device, buffers[0]);

        VkCommandBuffer commands;
        VkUtil::createCommandBuffer(context.device, context.commandPool, &commands);
        vkCompress::unquantizeFromSymbols(&gpu, commands, floatAddress, symbolAddress, size, .2f);

        {
            PCUtil::Stopwatch timer(std::cout, "Unquant Timer");
            VkUtil::commitCommandBuffer(context.queue, commands);
            check_vk_result(vkQueueWaitIdle(context.queue));
        }
    }
    if constexpr(testUPloadSpeed){
        uint32_t byteSize = 1<<30; // 1 gigabytes
        auto [buffer, offset, mem] = VkUtil::createMultiBufferBound(context, {byteSize}, {VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        PCUtil::Stopwatch upload(std::cout, "Upload Speed");
        std::vector<uint8_t> data(byteSize, 6);
        VkUtil::uploadData(context.device, mem, 0, byteSize, data.data());
    }
}