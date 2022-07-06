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
    DataSet openCompressedDataset(const VkUtil::Context& context, std::string_view folder){
        std::string hierarchyFolder(folder);

        // --------------------------------------------------------------------------------
        // determine kind of stored data (used to convert internally to a unified represantation for further processing)
        // --------------------------------------------------------------------------------
        std::ifstream dataInfo(hierarchyFolder + "/data.info", std::ios::binary);
        compression::DataStorageBits dataBits;
        uint32_t dataBlockSize;
        dataInfo >> dataBits;
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
        size_t dataSize = 0;
        if((dataBits & compression::DataStorageBits::RawAttributeBins) != compression::DataStorageBits::None){
            // reading index data
            std::cout << "Loading indexdata..." << std::endl;
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
            std::cout << "Compressing indexdata..."  << std::endl;
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
            std::cout << "Loading compressed indexdata..." << std::endl;
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
            std::cout << "Loading float column data" << std::endl;
            for(uint32_t i: irange(attributes)){
                std::ifstream data(hierarchyFolder + "/" + std::to_string(i) + ".col", std::ios_base::binary);
                std::vector<float> dVec(dataSize);
                data.read(reinterpret_cast<char*>(dVec.data()), dVec.size() * sizeof(dVec[0]));
                columnData[i].cpuData = std::vector<half>(dVec.begin(), dVec.end());  // automatic conversion to half via range constructor
            }
        }
        else if((dataBits & compression::DataStorageBits::HalfColumnData) != compression::DataStorageBits::None){
            // directly parse
            std::cout << "Loading half column data" << std::endl;
            if(dataSize == 0){  //getting the data size from the file size
                dataSize = std::filesystem::file_size(hierarchyFolder + "/0.col") / 2;
                std::cout << "Data size from col file: " << dataSize << std::endl;
            }
            for(uint32_t i: irange(attributes)){
                std::cout << "Loading half data for attribute " << attributes[i].name << std::endl;
                std::ifstream data(hierarchyFolder + "/" + std::to_string(i) + ".col", std::ios_base::binary);
                auto& dVec = columnData[i].cpuData;
                dVec.resize(dataSize);
                data.read(reinterpret_cast<char*>(dVec.data()), dVec.size() * sizeof(dVec[0]));
            }
        }
        else if((dataBits & compression::DataStorageBits::CuComColumnData) != compression::DataStorageBits::None){
            // directly parse if same compression block size
            gpuInstance = std::make_unique<vkCompress::GpuInstance>(context, 1, dataBlockSize, 0,0);
            std::vector<uint32_t> dataVec;
            for(uint32_t i: irange(attributes)){
                std::ifstream columnFile(hierarchyFolder + "/" + std::to_string(i) + ".comp", std::ios_base::binary);
                struct{uint64_t streamSize; uint32_t symbolSize;}sizes{};
                while(columnFile.read(reinterpret_cast<char*>(&sizes), sizeof(sizes))){
                    // streamSize is in bytes, while symbolSize is the resulting size of the decompressed vector
                    dataVec.resize(sizes.streamSize / sizeof(dataVec[0]));
                    columnFile.read(reinterpret_cast<char*>(dataVec.data()), sizes.streamSize);
                    columnData[i].compressedRLHuffCpu.emplace_back(vkCompress::parseCpuRLHuffData(gpuInstance.get(), dataVec));
                    columnData[i].compressedRLHuffGpu.emplace_back(gpuInstance.get(), columnData[i].compressedRLHuffCpu.back());
                }
            }
        }
        // only upload half data uncompressed if no cuda compressed data is available
        if(columnData[0].cpuData.size()){
            // currently the uncompressed 16 bit vectors are uploaded. Has to be changed to compressed vectors
            for(auto& d: columnData){
                //std::cout << "Creating vulkan buffer" << std::endl;
                // creating the vulkan resources and uploading the data to them
                VkUtil::createBuffer(context.device, d.cpuData.size() * sizeof(d.cpuData[0]), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, &d.gpuHalfData);
                VkMemoryRequirements memReq{};
                vkGetBufferMemoryRequirements(context.device, d.gpuHalfData, &memReq);
                VkMemoryAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                allocInfo.allocationSize = memReq.size;
                allocInfo.memoryTypeIndex = VkUtil::findMemoryType(context.physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                VkMemoryAllocateFlagsInfo allocFlags{};
                allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
                allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
                allocInfo.pNext = &allocFlags;
                vkAllocateMemory(context.device, &allocInfo, nullptr, &d.gpuMemory);
                vkBindBufferMemory(context.device, d.gpuHalfData, d.gpuMemory, 0);
                VkUtil::uploadData(context.device, d.gpuMemory, 0, d.cpuData.size() * sizeof(d.cpuData[0]), d.cpuData.data());
            }
        }

        std::cout << "Loaded " << dataSize << " datapoints" << std::endl;

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
                std::move(gpuInstance)
            }
        };
    }
}

