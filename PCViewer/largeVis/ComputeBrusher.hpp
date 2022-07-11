#pragma once

#include "../VkUtil.h"
#include "../Brushing.hpp"
#include "TimingInfo.hpp"

// executes index brushing. Every GPU thread will do 32 datapoints and stores the brushing result in a uint vector
// like this, every GPU thread can write a single 32 bit integer.
// To enable vectorized loading, al threads in a thread block will read consecutive datapoints, calculate its activation and store the activations via subgroup balloting
class ComputeBrusher{
public:
    // struct holding information needed to create the vulkan pipeline
    struct CreateInfo{
        VkUtil::Context context;
    };

    // compression renderer can only be created internally, can not be moved, copied or destoryed
    ComputeBrusher() = delete;
    ComputeBrusher(const ComputeBrusher&) = delete;
    ComputeBrusher& operator=(const ComputeBrusher&) = delete;

    static ComputeBrusher* acquireReference(const CreateInfo& info); // acquire a reference (automatically creates renderer if not yet existing)
    void release();                                                 // has to be called to notify destruction before vulkan resources are destroyed
    VkEvent updateActiveIndices(size_t amtDatapoints, const std::vector<brushing::RangeBrush>& brushes, const Polygons& lassoBrushes, const std::vector<VkBuffer>& dataBuffer, VkBuffer indexActivations, size_t indexOffset = 0, bool andBrushes = false, VkEvent prevPipeEvent = {}, TimingInfo timingInfo = {});

private:
    struct BrushInfos{
        uint32_t amtofDataPoints, amtOfAttributes, amtOfBrushes, andBrushes, offsetLassoBrushes, outputOffset, pa,dding;    // outputOffset is already divided by 32
        // float list of brush infos ...coming soon :)
    };

    ComputeBrusher(const CreateInfo& info);
    ~ComputeBrusher();

    static ComputeBrusher* _singleton;
    int _refCount{};

    // vulkan resources that are destroyed externally
    VkUtil::Context _vkContext;
    VkDescriptorSet _descSet{};
    uint32_t _infoByteSize{};   // used to check if the current info buffer is large enough to hold all infos
    VkBuffer _infoBuffer{};
    uint32_t _dataOffset{};
    uint32_t _dataSize{};       // size in amtOfAttributes
    VkBuffer _dataBuffer{};     // have to be in 2 different buffers to be able to have to variable length storage buffers
    VkDeviceMemory _infoMemory{};

    // vulkan resources that have to be destroyed
    VkUtil::PipelineInfo _brushPipelineInfo{};
    VkEvent _brushEvent{};
    VkFence _brushFence{};
    VkCommandBuffer _commands{};

    const std::string _computeShader = "shader/largeVisBrush.comp.spv";

    void createOrUpdateBuffer(uint32_t amtOfAttributes, uint32_t infoByteSize);
};