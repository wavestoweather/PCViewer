#pragma once
#include "../VkUtil.h"
#include "TimingInfo.hpp"
#include "../Brushing.hpp"
#include <map>

// holds a single vulkan compute pipeline instance for counting active lines in cluster
class LineCounter{
public:
    // struct holding information needed to create the vulkan pipeline
    struct CreateInfo{
        VkUtil::Context context;
    };
    struct CountLinesInfo{
        VkBuffer aValues, bValues;
        uint32_t dataSize;
        std::vector<float> aBounds, bBounds;
    };

    enum ReductionTypes: uint32_t{
        ReductionAdd,
        ReductionAddNonAtomic,
        ReductionAddPartitionNonAtomic,
        ReductionSubgroupAdd ,
        ReductionSubgroupAllAdd,
        ReductionMin,
        ReductionSubgroupMin,
        ReductionMax,           // currently unused
        ReductionSubgroupMax,   // currently unused
        ReductionAddPartitionGeneric,
        ReductionEnumMax
    };

    // compression renderer can only be created internally, can not be moved, copied or destoryed
    LineCounter() = delete;
    LineCounter(const LineCounter&) = delete;
    LineCounter& operator=(const LineCounter&) = delete;

    static LineCounter* acquireReference(const CreateInfo& info); // acquire a reference (automatically creates renderer if not yet existing)
    static void tests(const CreateInfo& info);
    void release();                                 // has to be called to notify destruction before vulkan resources are destroyed
    void countLines(VkCommandBuffer commands, const CountLinesInfo& info);  // test function
    void countLinesPair(size_t dataSize, VkBuffer aData, VkBuffer bData, uint32_t aIndices, uint32_t bIndices, VkBuffer counts, VkBuffer indexActivation, bool clearCounts = false, ReductionTypes reductionType = ReductionAdd);
    VkSemaphore countLinesAll(size_t dataSize, const std::vector<VkBuffer>& data, uint32_t binAmt, const std::vector<VkBuffer>& counts, const std::vector<uint32_t>& activeIndices, VkBuffer indexActivation, size_t indexOffset = 0, bool clearCounts = false, ReductionTypes reductionType = ReductionAdd, VkSemaphore prevPipeSemaphore = {}, TimingInfo timingInfo = {});

    VkSemaphore countBrushLinesAll(size_t dataSize, const std::vector<VkBuffer>& data, uint32_t binAmt, const std::vector<VkBuffer>& counts, const std::vector<uint32_t>& activeIndices, const brushing::RangeBrushes& rangeBrushes, const Polygons& lassoBrushes, bool andBrushes = true, bool clearCounts = false, ReductionTypes reductionType = ReductionAdd, VkSemaphore prevPipeSemaphore = {}, TimingInfo timingIfo = {});

    static const uint32_t maxAttributes{30};
private:
    struct PairInfos{
        uint32_t amtofDataPoints, aBins, bBins, indexOffset, allAmtOfPairs, attributeActive, countActive, padding;
    };

    struct BPair{
        VkBuffer a, b;
        bool operator<(const BPair& o) const{return a < o.a || (a == o.a && b < o.b);};
    };

    LineCounter(const CreateInfo& info);
    ~LineCounter();

    static LineCounter* _singleton;
    int _refCount{};

    // vulkan resources that are destroyed externally
    VkUtil::Context _vkContext;
    VkDescriptorSet _descSet{}; //only here for test purposes
    VkDescriptorSet _pairSet{}, _allSet{}, _allBrushSet{};
    uint32_t _brushBufferSize{};
    VkBuffer _pairUniform{}, _brushBuffer{};
    VkDeviceMemory _pairUniformMem{}, _brushMem{};
    std::map<BPair, VkDescriptorSet> _pairSets{};
    std::map<BPair, VkSemaphore> _pairSemaphores{};
    VkSemaphore _allSemaphore{}, _allBrushSemaphore{};
    VkFence _allFence{}, _allBrushFence{};
    VkCommandBuffer _allCommands{}, _allBrushCommands{};

    // vulkan resources that have to be destroyed
    //VkUtil::PipelineInfo _countPipeInfo{}, _countSubgroupAllInfo{}, _countPartitionedPipeInfo{}, _minPipeInfo{}, _countAllPipeInfo{}, _countAllSubgroupAllInfo{}, _countAllPartitionedInfo{};
    std::map<ReductionTypes, VkUtil::PipelineInfo> _pairInfos;
    std::map<ReductionTypes, VkUtil::PipelineInfo> _fullInfos;
    std::map<ReductionTypes, VkUtil::PipelineInfo> _brushFullInfos;

    const std::string_view _computeShader = "shader/lineCount.comp.spv";
    const std::string_view _computeAllShader = "shader/lineCountAll.comp.spv";
    const std::string_view _computeAllBrushingShader = "shader/lineCountAllBrush.comp.spv";

    VkDeviceMemory _bins;
    size_t _binsSize;
};