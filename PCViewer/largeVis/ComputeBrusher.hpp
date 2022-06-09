#pragma once

#include "../VkUtil.h"
#include "../Brushing.hpp"

// holds a single vulkan compute pipeline instance for counting active lines in cluster

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
    void updateActiveIndices(const std::vector<brushing::RangeBrush>& brushes, const Polygons& lassoBrushes, const std::vector<VkBuffer>& dataBuffer, VkBuffer indexActivations, bool andBrushes = false);

private:
    struct BrushInfos{
        uint32_t amtofDataPoints, amtOfAttributes, amtOfBrushes, padding;
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

    const std::string _computeShader = "shader/largeVisBrush.comp.spv";

    void createOrUpdateBuffer(uint32_t amtOfAttributes, uint32_t infoByteSize);
};