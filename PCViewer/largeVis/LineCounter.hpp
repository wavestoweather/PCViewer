#pragma once
#include "../VkUtil.h"
#include "TimingInfo.hpp"
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
        ReductionSubgroupAdd ,
        ReductionSubgroupAllAdd,
        ReductionMin,
        ReductionSubgroupMin,
        ReductionMax,           // currently unused
        ReductionSubgroupMax,   // currently unused
    };

    // compression renderer can only be created internally, can not be moved, copied or destoryed
    LineCounter() = delete;
    LineCounter(const LineCounter&) = delete;
    LineCounter& operator=(const LineCounter&) = delete;

    static LineCounter* acquireReference(const CreateInfo& info); // acquire a reference (automatically creates renderer if not yet existing)
    static void tests(const CreateInfo& info);
    void release();                                 // has to be called to notify destruction before vulkan resources are destroyed
    void countLines(VkCommandBuffer commands, const CountLinesInfo& info);  // test function
    void countLinesPair(size_t dataSize, VkBuffer aData, VkBuffer bData, uint32_t aIndices, uint32_t bIndices, VkBuffer counts, VkBuffer indexActivation, bool clearCounts = false, ReductionTypes reductionType = ReductionAdd) const;
    VkEvent countLinesAll(size_t dataSize, const std::vector<VkBuffer>& data, uint32_t binAmt, const std::vector<VkBuffer>& counts, const std::vector<uint32_t>& activeIndices, VkBuffer indexActivation, bool clearCounts = false, ReductionTypes reductionType = ReductionAdd, VkEvent prevPipeEvent = {}, TimingInfo timingInfo = {});

    const uint32_t maxAttributes{30};
private:
    struct PairInfos{
        uint32_t amtofDataPoints, aBins, bBins, indexOffset, allAmtOfPairs, pa,dd,ing;
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
    VkDescriptorSet _pairSet{}, _allSet{};
    VkBuffer _pairUniform{};
    VkDeviceMemory _pairUniformMem{};
    std::map<BPair, VkDescriptorSet> _pairSets{};
    std::map<BPair, VkEvent> _pairEvents{};
    VkEvent _allEvent{};
    VkCommandBuffer _allCommands{};

    // vulkan resources that have to be destroyed
    VkUtil::PipelineInfo _countPipeInfo{}, _countSubgroupAllInfo{}, _countPartitionedPipeInfo{}, _minPipeInfo{}, _countAllPipeInfo{}, _countAllSubgroupAllInfo{}, _countAllPartitionedInfo{};

    const std::string _computeShader = "shader/lineCount.comp.spv";
    const std::string _computeAllShader = "shader/lineCountAll.comp.spv";

    VkDeviceMemory _bins;
    size_t _binsSize;
};