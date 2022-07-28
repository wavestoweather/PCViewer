#pragma once

#define NOSTATICS
#include "../Structures.hpp"
#include "../VkUtil.h"
#include "../compression/HirarchyCreation.hpp"
#include "../compression/gpuCompression/Util.hpp"
#include "../compression/Constants.hpp"
#include "UVecHasher.hpp"
#include <filesystem>

#ifdef None
#undef None
#endif

namespace util{
    static DataSet openCompressedDataset(const VkUtil::Context& context, std::string_view folder, uint32_t transferQueueIndex = (uint32_t)-1){
        std::string hierarchyFolder(folder);

        // --------------------------------------------------------------------------------
        // determine kind of stored data (used to convert internally to a unified represantation for further processing)
        // --------------------------------------------------------------------------------
        std::ifstream dataInfo(hierarchyFolder + "/data.info", std::ios::binary);
        compression::DataStorageBits dataBits;
        uint32_t dataBlockSize;
        size_t dataSize;
        float quantizationStep;
        dataInfo >> dataSize;
        dataInfo >> dataBits;
        dataInfo >> quantizationStep;
        dataInfo >> dataBlockSize;  // block size for compressed data
        dataInfo.close();

        // --------------------------------------------------------------------------------
        // attribute infos, cluster infos
        // --------------------------------------------------------------------------------
        std::vector<Attribute> attributes;
        std::ifstream attributeInfos(hierarchyFolder + "/attr.info", std::ios_base::binary);
        uint binsMaxCenterAmt;
        attributeInfos >> binsMaxCenterAmt; // first line contains the maximum amt of centers/bins
        std::string a; float aMin, aMax;
        while(attributeInfos >> a >> aMin >> aMax){
            attributes.push_back({a, a, {}, {}, aMin, aMax});
        }
        attributeInfos.close();

        // --------------------------------------------------------------------------------
        // center infos
        // --------------------------------------------------------------------------------
        std::vector<std::vector<compression::IndexCenterFileData>> _attributeCenters;
        std::ifstream attributeCenterFile(hierarchyFolder + "/attr.ac", std::ios_base::binary);
        std::vector<compression::ByteOffsetSize> offsetSizes(attributes.size());
        attributeCenterFile.read(reinterpret_cast<char*>(offsetSizes.data()), offsetSizes.size() * sizeof(offsetSizes[0]));
        _attributeCenters.resize(attributes.size());
        for(int i = 0; i < attributes.size(); ++i){
            assert(!attributeCenterFile || attributeCenterFile.tellg() == offsetSizes[i].offset);
            _attributeCenters[i].resize(offsetSizes[i].size / sizeof(_attributeCenters[0][0]));
            attributeCenterFile.read(reinterpret_cast<char*>(_attributeCenters[i].data()), offsetSizes[i].size);
        }

        // --------------------------------------------------------------------------------
        // 1d index data either compressed or not (automatic conversion if not compressed)
        // --------------------------------------------------------------------------------
        robin_hood::unordered_map<std::vector<uint32_t>, std::vector<roaring::Roaring64Map>, UVecHash> ndBuckets;
        if((dataBits & compression::DataStorageBits::RawAttributeBins) != compression::DataStorageBits::None){
            // reading index data
            std::cout << "[import] Loading indexdata..." << std::endl;
            std::vector<std::vector<uint32_t>> attributeIndices;
            attributeIndices.resize(attributes.size());
            for(int i = 0; i < attributes.size(); ++i){
                std::ifstream indicesData(hierarchyFolder + "/" + std::to_string(i) + ".ids", std::ios_base::binary);
                uint32_t indicesSize = _attributeCenters[i].back().offset + _attributeCenters[i].back().size;
                if(indicesSize > dataSize)
                    dataSize = indicesSize;
                attributeIndices[i].resize(indicesSize);
                indicesData.read(reinterpret_cast<char*>(attributeIndices[i].data()), indicesSize * sizeof(attributeIndices[0][0]));
            }
            // compressing index data
            std::cout << "[import] Compressing indexdata..."  << std::endl;
            for(uint32_t compInd: irange(_attributeCenters)){
                ndBuckets[{compInd}].resize(_attributeCenters[compInd].size());
                size_t indexlistSize{attributeIndices[compInd].size() * sizeof(uint32_t)}, compressedSize{};
                for(uint32_t bin: irange(ndBuckets[{compInd}])){
                    ndBuckets[{compInd}][bin] = roaring::Roaring64Map(_attributeCenters[compInd][bin].size, attributeIndices[compInd].data() + _attributeCenters[compInd][bin].offset);
                    ndBuckets[{compInd}][bin].runOptimize();
                    ndBuckets[{compInd}][bin].shrinkToFit();
                    compressedSize += ndBuckets[{compInd}][bin].getSizeInBytes();
                }
                std::cout << "Attribute " << attributes[compInd].name << ": Uncompressed Indices take " << indexlistSize / float(1 << 20) << " MByte vs " << compressedSize / float(1 << 20) << " MByte compressed." << "Compression rate 1:" << indexlistSize / float(compressedSize) << std::endl;
            }
        }
        else if((dataBits & compression::DataStorageBits::RoaringAttributeBins) != compression::DataStorageBits::None){
            // compressed indices, can be read out directly
            std::cout << "[import] Loading compressed indexdata..." << std::endl;
            for(uint32_t i: irange(attributes)){
                std::ifstream indicesData(hierarchyFolder + "/" + std::to_string(i) + ".ids", std::ios_base::binary);
                uint32_t indicesSize = _attributeCenters[i].back().offset + _attributeCenters[i].back().size;   // size is given in bytes
                std::vector<char> indices(indicesSize);
                indicesData.read(indices.data(), indicesSize * sizeof(indices[0]));
                // parse into roaring bitmaps
                // filling only the attribute centers that are available, the other centers are empty
                ndBuckets[{i}].resize(_attributeCenters[i].size());
                size_t curSize{};
                for(uint32_t bin: irange(_attributeCenters[i])){
                    ndBuckets[{i}][bin] = roaring::Roaring64Map::readSafe(indices.data() + _attributeCenters[i][bin].offset, _attributeCenters[i][bin].size * sizeof(indices[0]));
                    curSize += ndBuckets[{i}][bin].cardinality();
                }
                //std::cout << "Idex size attribute " << attributes[i].name << ": " << curSize << std::endl;
                if(curSize > dataSize)
                    dataSize = curSize;
            }
        }

        // --------------------------------------------------------------------------------
        // 1d data either compressed or not (automatic conversion if not compressed, stored currently as 16bit float vec)
        // --------------------------------------------------------------------------------
        std::vector<CompressedColumnData> columnData(attributes.size());
        std::unique_ptr<vkCompress::GpuInstance> gpuInstance;
        if((dataBits & compression::DataStorageBits::RawColumnData) != compression::DataStorageBits::None){
            // convert normalized float data automatically to half data
            std::cout << "[import] Loading float column data" << std::endl;
            for(uint32_t i: irange(attributes)){
                std::ifstream data(hierarchyFolder + "/" + std::to_string(i) + ".col", std::ios_base::binary);
                std::vector<float> dVec(dataSize);
                data.read(reinterpret_cast<char*>(dVec.data()), dVec.size() * sizeof(dVec[0]));
                columnData[i].cpuData = std::vector<half>(dVec.begin(), dVec.end());  // automatic conversion to half via range constructor
            }
        }
        else if((dataBits & compression::DataStorageBits::HalfColumnData) != compression::DataStorageBits::None){
            // directly parse
            std::cout << "[import] Loading half column data" << std::endl;
            if(dataSize == 0){  //getting the data size from the file size
                dataSize = std::filesystem::file_size(hierarchyFolder + "/0.col") / 2;
                std::cout << "Data size from col file: " << dataSize << std::endl;
            }
            PCUtil::Stopwatch importWatch(std::cout, "Half import time for " + std::to_string(dataSize * attributes.size() * sizeof(half) / double(1<<30)) + " GByte");
            for(uint32_t i: irange(attributes)){
                std::cout << "[import] Loading half data for attribute " << attributes[i].name << std::endl;
                PCUtil::CIFile data(hierarchyFolder + "/" + std::to_string(i) + ".col");
                auto& dVec = columnData[i].cpuData;
                dVec.resize(dataSize);
                data.read(reinterpret_cast<char*>(dVec.data()), dVec.size() * sizeof(dVec[0]));
            }
        }
        if((dataBits & compression::DataStorageBits::CuComColumnData) != compression::DataStorageBits::None){
            // directly parse if same compression block size
            gpuInstance = std::make_unique<vkCompress::GpuInstance>(context, 1, dataBlockSize, 0,0);
            std::vector<uint32_t> dataVec;
            for(uint32_t i: irange(attributes)){
                std::cout << "[import] Loading compressed data for attribute " << attributes[i].name << std::endl;
                std::ifstream columnFile(hierarchyFolder + "/" + std::to_string(i) + ".comp", std::ios_base::binary);
                assert(columnFile);
                struct{uint64_t streamSize; uint32_t symbolSize;}sizes{};
                while(columnFile.read(reinterpret_cast<char*>(&sizes), sizeof(sizes))){
                    // streamSize is in bytes, while symbolSize is the resulting size of the decompressed vector
                    dataVec.resize(sizes.streamSize / sizeof(dataVec[0]));
                    columnFile.read(reinterpret_cast<char*>(dataVec.data()), sizes.streamSize);
                    columnData[i].compressedRLHuffCpu.emplace_back(vkCompress::parseCpuRLHuffData(gpuInstance.get(), dataVec));
                    columnData[i].compressedRLHuffGpu.emplace_back(gpuInstance.get(), columnData[i].compressedRLHuffCpu.back());
                    columnData[i].compressedSymbolSize.push_back(sizes.symbolSize);
                }
            }
        }
        // only upload half data uncompressed if no cuda compressed data is available
        std::unique_ptr<UploadManager> uploadManager;
        if(columnData[0].cpuData.size()){
            // either uploading half data directly, or setting everything up to upload the half data on the fly via an uploadmanager

            // getting memory information
            VkPhysicalDeviceMemoryProperties memProps{};
            vkGetPhysicalDeviceMemoryProperties(context.physicalDevice, &memProps);
            size_t deviceLocalSize{};
            for(int i: irange(memProps.memoryHeapCount)){
                if(memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT){
                    deviceLocalSize = memProps.memoryHeaps[i].size;
                    break;
                }
            }
            std::cout << "[import] Found " << deviceLocalSize / double(1 << 30) << " GB of device local memory." << std::endl;
            size_t dataByteSize = dataSize * columnData.size() * sizeof(half);
            if(dataByteSize > .9 * deviceLocalSize){
                std::cout << "[import] More than 90\% of GPU Memory is used -> using streaming upload to gpu." << std::endl;

                // calculating all buffer sizes to occupy 80% of GPU memory
                const uint32_t amtStagingBuffers{2};
                size_t bufferByteSize = deviceLocalSize * .8 / (columnData.size() + amtStagingBuffers);
                bufferByteSize = PCUtil::alignedSize(bufferByteSize, 0x40);
                VkBufferUsageFlags usages = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                for(auto& d: columnData){
                    auto [b, o, m] = VkUtil::createMultiBufferBound(context, {bufferByteSize}, {usages}, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                    d.gpuHalfData = b[0];
                    d.gpuMemory = m;
                }

                uploadManager = std::make_unique<UploadManager>(context, transferQueueIndex, amtStagingBuffers, bufferByteSize);
            }
            else{
                uint32_t alignedSize = PCUtil::alignedSize(columnData[0].cpuData.size() * sizeof(columnData[0].cpuData[0]), 0x40);
                for(auto& d: columnData){
                    //std::cout << "Creating vulkan buffer" << std::endl;
                    // creating the vulkan resources and uploading the data to them
                    VkUtil::createBuffer(context.device, alignedSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &d.gpuHalfData);
                    VkMemoryRequirements memReq{};
                    vkGetBufferMemoryRequirements(context.device, d.gpuHalfData, &memReq);
                    VkMemoryAllocateInfo allocInfo{};
                    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                    allocInfo.allocationSize = memReq.size;
                    allocInfo.memoryTypeIndex = VkUtil::findMemoryType(context.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                    VkMemoryAllocateFlagsInfo allocFlags{};
                    allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
                    allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
                    allocInfo.pNext = &allocFlags;
                    check_vk_result(vkAllocateMemory(context.device, &allocInfo, nullptr, &d.gpuMemory));
                    vkBindBufferMemory(context.device, d.gpuHalfData, d.gpuMemory, 0);
                    PCUtil::Stopwatch uploadWatch(std::cout, "Indirect upload with staging buffer creation");
                    VkUtil::uploadDataIndirect(context, d.gpuHalfData, d.cpuData.size() * sizeof(d.cpuData[0]), d.cpuData.data());
                }
            }
        }

        // creating the decompress manager if needed
        std::unique_ptr<DecompressManager> decompressManager;
        if(gpuInstance){
	    	DecompressManager::GpuColumns gpuColumns(columnData.size());
	    	DecompressManager::CpuColumns cpuColumns(gpuColumns.size());
	    	for(int i : irange(gpuColumns)){
	    		gpuColumns[i] = columnData[i].compressedRLHuffGpu.data();
	    		cpuColumns[i] = columnData[i].compressedRLHuffCpu.data();
	    	}
	    	decompressManager = std::make_unique<DecompressManager>(dataBlockSize, *gpuInstance, cpuColumns, gpuColumns);
	    }

        std::cout << "[import] Loaded " << dataSize << " datapoints" << std::endl;

        return DataSet{
            std::string(folder.substr(folder.find_last_of("/\\") + 1)),
            {},
            {},
            {},
            1,
            DataType::Compressed,
            std::vector<uint8_t>(hierarchyFolder.begin(), hierarchyFolder.end()),
            {   // compressedData
                dataSize,
                std::move(columnData),
                std::move(attributes),
                dataBlockSize,
                quantizationStep,
                std::move(gpuInstance),
                std::move(decompressManager),
                std::move(uploadManager)
            }
        };
    }
}

