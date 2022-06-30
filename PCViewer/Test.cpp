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
#include "compression/cpuCompression/EncodeCPU.h"
#include "compression/cpuCompression/DWTCpu.h"

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

static std::pair<cudaCompress::BitStream, uint32_t> compressVector(std::vector<float>& src, float quantizationStep){
    std::pair<cudaCompress::BitStream, uint32_t> t;
    cudaCompress::BitStream stream;
    uint32_t symbolSize;
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

    // testing encoding times for real world data
    //std::vector<std::string_view> filenames{"/run/media/lachei/3d02119e-bc93-4969-9fc5-523f06321708/w2w/takumi/scripts/tp.bin", "/run/media/lachei/3d02119e-bc93-4969-9fc5-523f06321708/w2w/takumi/scripts/q.bin", "/run/media/lachei/3d02119e-bc93-4969-9fc5-523f06321708/w2w/takumi/scripts/NCCLOUD.bin"};
    //std::vector<float> quants{0.001f, .01f, .1f};
    //for(auto file: filenames){
    //    for(float q: quants){
    //        uint32_t s;
    //        std::vector<uint32_t> bits;
    //        size_t bytesize;
    //        {
    //            // encoding
    //            std::ifstream f(file.data(), std::ios_base::binary);
    //            std::vector<float> tpVals(1024 * 1024);
////
    //            f.read(reinterpret_cast<char*>(tpVals.data()), tpVals.size() * sizeof(tpVals[0]));
    //            
    //            PCUtil::Stopwatch encode(std::cout, "Encode " + std::to_string(q) + std::string(file));
    //            auto [stream, size] = compressVector(tpVals, q);
    //            bytesize = stream.getRawSizeBytes();
    //            bits = std::move(stream.getVector());
    //            s = size;
    //        }
    //        {
    //            // decoding
    //            std::vector<float> symb(s);
    //            PCUtil::Stopwatch decode(std::cout, "Decode " + std::to_string(q) + std::string(file));
    //            decompressVector(bits, q, s, symb);
    //            bool test = true;
    //        }
    //        std::cout << "Compression Ratio: 1 : " << std::to_string(float(1024 * 1024 * 4) / bytesize) << std::endl;
    //    }
    //}

    // testing gpu decompression
    //vkCompress::decodeRLHuff({}, {}, (vkCompress::Symbol16**){}, {}, {});
    
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
    RLHuffDecodeDataGpu gpuData(&gpu, cpuData);
    //vkCompress::decodeHuff()

    // creating buffer for the symbol table
    uint pad = gpu.m_subgroupSize * gpu.m_codingBlockSize * sizeof(uint16_t);
    uint paddedSymbols = (symbolsSize * sizeof(uint16_t) + pad - 1) / pad * pad;
    auto [symbolBuffer, offs, mem] = VkUtil::createMultiBufferBound(context, {paddedSymbols}, {VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT}, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    std::vector<uint16_t> sorted(symbolsSize); uint16_t count{};
    for(auto& i: sorted) i = count++;
    VkUtil::uploadData(context.device, mem, 0, symbolsSize * sizeof(uint16_t), sorted.data());
    uint8_t* ddd = (uint8_t*)cpuData.codewordStream.data();

    VkCommandBuffer commands;
    VkUtil::createCommandBuffer(context.device, context.commandPool, &commands);
    auto &resources = gpu.Encode.Decode[0];
    auto& streamInfo = resources.pSymbolStreamInfos[0]; // we assume here to only have a single decoding block! -> index 0
    
    streamInfo.symbolCount = symbolsSize;

    streamInfo.dpDecodeTable = VkUtil::getBufferAddress(context.device, gpuData.buffer) + gpuData.symbolTableOffset;
    streamInfo.decodeSymbolTableSize = cpuData.symbolTable.getSymbolTableSize();

    streamInfo.dpCodewordStream = VkUtil::getBufferAddress(context.device, gpuData.buffer) + gpuData.symbolStreamOffset;

    streamInfo.dpOffsets = VkUtil::getBufferAddress(context.device, gpuData.buffer) + gpuData.zeroCountOffsetsOffset;
    streamInfo.dpSymbolStream = VkUtil::getBufferAddress(context.device, symbolBuffer[0]);
    VkUtil::uploadData(context.device, resources.memory, resources.streamInfosOffset, sizeof(vkCompress::HuffmanGPUStreamInfo), &streamInfo);
    vkCompress::huffmanDecode(&gpu, commands, resources.streamInfoSet, 1, gpu.m_codingBlockSize);

    VkUtil::commitCommandBuffer(context.queue, commands);
    check_vk_result(vkQueueWaitIdle(context.queue));

    std::vector<uint16_t> downloadedData(symbolsSize);
    VkUtil::downloadData(context.device, mem, 0, symbolsSize * sizeof(uint16_t), downloadedData.data());
    //std::set<uint16_t> block1Orig(symbols.begin(), symbols.begin() + 4096);
    //std::set<uint16_t> block1Gpu(downloadedData.begin(), downloadedData.begin() + 4096);
    //bool equ = block1Orig == block1Gpu;
    bool test = true;
}