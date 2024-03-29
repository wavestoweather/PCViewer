#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <list>
#include "Data.hpp"
#include "LineBundles.hpp"
#include "ClusterBundles.hpp"
#include "TemplateList.hpp"
#include "Attribute.hpp"
#include "largeVis/IndBinManager.hpp"
#include "largeVis/DecompressManager.hpp"
#include "half/half.hpp"
#include "compression/gpuCompression/Encode.hpp"
#include "vkMemory/UploadManager.hpp"
#include <memory>

//forward declaration
class IndBinManager;

struct Vec4 {
    float x;
    float y;
    float z;
    float w;

    bool operator==(Vec4& other) {
        return this->x == other.x && this->y == other.y && this->z == other.z && this->w == other.w;
    }

    bool operator!=(Vec4& other) {
        return !(*this == other);
    }
};

struct Vec2 {
    float u;
    float v;
};

struct Buffer {
    VkBuffer buffer;
    VkBuffer uboBuffer;
    VkDeviceMemory memory;
    VkDescriptorSet descriptorSet;

    bool operator==(const Buffer& other) {
        return this->buffer == other.buffer && this->memory == other.memory;
    }
};

enum class DataType{
    Continuous,
    ContinuousDlf,
    Hierarchichal,
    Compressed
};

struct CompressedColumnData{
    std::vector<half> cpuData{};// half compressed data
    VkBuffer gpuHalfData{};        // half compredded data on Gpu
    VkDeviceMemory gpuMemory{};    // memory for half compressed gpu data
    std::vector<RLHuffDecodeDataCpu> compressedRLHuffCpu{};
    std::vector<RLHuffDecodeDataGpu> compressedRLHuffGpu{};
    std::vector<uint32_t> compressedSymbolSize{};    // this compressed symbol size is taken from the loaded data and is already 4 aligned
};

struct CompressedData{
    size_t dataSize{};
    std::vector<CompressedColumnData> columnData{};
    std::vector<Attribute> attributes{};
    uint32_t compressedBlockSize{};
    float quantizationStep;
    std::unique_ptr<vkCompress::GpuInstance> gpuInstance{};
    std::unique_ptr<DecompressManager> decompressManager{}; 
    std::unique_ptr<UploadManager> uploadManager{};
};

struct DataSet {
    std::string name{};
    Buffer buffer{};
    Data data{};
    std::list<TemplateList> drawLists{};
    int reducedDataSetSize{};                    //size of the reduced dataset(when clustering was applied). This is set to data.size() on creation.
    DataType dataType{};
    std::vector<uint8_t> additionalData{};        //byte vector for additional data. For Hierarchical data this is where teh hierarchy folder is stored
    CompressedData compressedData{};
    std::vector<Attribute> attributes{};
    uint32_t originalAttributeSize{};
    // herer should go space for roaring bins

    bool operator==(const DataSet& other) const {
        return this->name == other.name;
    }
};

static DataSet& getDataset(std::list<DataSet>& datasets, std::string_view datasetId){
    return *std::find_if(datasets.begin(), datasets.end(), [&](const DataSet& ds){return ds.name == datasetId;});
}

struct Brush {
    int id;
    std::pair<float, float> minMax;
};

enum class InheritanceFlags{
    dlf = 1,
    hierarchical = 1 << 1,
    compressed = 1 << 2
};

enum class AlphaMappingTypes: uint32_t{
    MappingMultiplicative,
    MappingBound01,
    MappingConstAlpha,
    MappingAlphaAdoption,
    MappingExp,
    MappingSqrt,
    MappingLog,
    MappingComp
};

static std::vector<std::string_view> alphaMappingNames{
    "MappingMultiplicative",
    "MappingBound01",
    "MappingConstAlpha",
    "MappingAlphaAdoption",
    "MappingExp",
    "MappingSqrt",
    "MappingLog",
    "MappingComp"
};

// struct holding the information for a drawable instance of a TemplateList
//
// The id of the Drawlist is its name!
//
// The inheritedFlags field contains important information inherited from the dataset and template list this drawlistis created from
// Such inheritance bits can be:
//    - Hierarchical: Instead of creating buffers which are sized to hold the data information, the buffers have a size to be able to hold as much lines as set in maxHierarchyLines
//
struct DrawList {
    std::string name;
    std::string parentDataSet;
    TemplateList* parentTemplateList;
    const Data* data;
    const std::vector<Attribute>* attributes;
    InheritanceFlags inheritanceFlags;
    Vec4 color;
    Vec4 prefColor;
    bool show;
    bool showHistogramm;
    std::vector<float> brushedRatioToParent;         // Stores the ratio of points of this data set and points going through the same 1D brushes of the parent.
    bool immuneToGlobalBrushes;
    VkBuffer buffer;                                // vulkan data buffer
    VkDescriptorSet dataDescriptorSet;                //is relesed when dataset is removed
    VkBuffer indexBuffer;                            //indexbuffer for line rendering!!!
    uint32_t indexBufferOffset;
    VkBuffer ubo;
    //VkBuffer histogramIndBuffer;
    //uint32_t histIndexBufferOffset;
    std::vector<VkBuffer> histogramUbos;
    VkBuffer medianBuffer;
    VkDescriptorSet medianBufferSet;                //has to be created/released in drawlist creation
    VkBuffer medianUbo;
    uint32_t priorityColorBufferOffset;
    VkBuffer priorityColorBuffer;
    uint32_t activeIndicesBufferOffset;
    VkBuffer activeIndicesBuffer;                    //bool buffer of length n with n being the amount of data in the parent dataset
    uint32_t indicesBufferOffset;
    VkBuffer indicesBuffer;                            //graphics buffer with all indices which are in this drawlist
    VkBufferView activeIndicesBufferView;            //buffer view to address the active indices buffer bytewise
    int medianUboOffset;
    VkDescriptorSet medianUboDescSet;
    uint32_t medianBufferOffset;
    Vec4 medianColor;
    int activeMedian;
    std::vector<uint32_t> histogramUbosOffsets;
    std::vector<VkDescriptorSet> histogrammDescSets;
    VkDeviceMemory dlMem;
    VkDeviceMemory indexBufferMemory;
    VkDescriptorSet uboDescSet;
    std::vector<uint32_t> indices;
    //std::vector<uint32_t> activeInd;
    std::vector<std::vector<Brush>> brushes;        //the pair contains first min and then max for the brush
    LineBundles* lineBundles;                        // line bundles are bundles from 1d clustered (canopy clustering) clustering
    ClusterBundles* clusterBundles;                    // cluster bundles are bundles lines from a dimensional clustering
    bool renderBundles, renderClusterBundles;
    uint32_t activeLinesAmt;                        //contains the amount of lines after brushing has been applied
    std::shared_ptr<IndBinManager> indBinManager;    //optional import manger for large vis files
    AlphaMappingTypes alphaMappingType;
};

struct DrawlistDragDropInfo{
    std::list<DrawList>* drawlists;
    std::vector<int>* selected;
};

struct UniformBufferObject {
    float alpha;
    uint32_t amtOfVerts;
    uint32_t amtOfAttributes;
    float padding;
    uint32_t dataFlags, plotWidth, plotHeight, alphaMapping;
    Vec4 color;
    std::vector<Vec4> vertTransformations;
    //Vec4 VertexTransormations[];            //is now a variable length array at the end of the UBO
    uint32_t size(){
        return sizeof(UniformBufferObject) - sizeof(vertTransformations) + sizeof(vertTransformations[0]) * vertTransformations.size();
    }
};

template<typename Infos, typename TArray>
struct ArrayStruct: public Infos{
    std::vector<TArray> array;
    uint32_t size(){
        return sizeof(ArrayStruct) - sizeof(array) + sizeof(array[0]) * array.size();
    }
    std::vector<uint8_t> toByteArray(){
        std::vector<uint8_t> bytes(size());
        const uint32_t infoSize = sizeof(ArrayStruct) - sizeof(array);
        std::memcpy(bytes.data(), this, infoSize);
        std::memcpy(bytes.data() + infoSize, array.data(), sizeof(array[0] * array.size()));
        return bytes;
    }
};

struct VertexBufferCreateInfo{
    DataType dataType;
    uint32_t maxLines;
    uint32_t additionalAttributeStorage;
};