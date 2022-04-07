#pragma once
#include "../VkUtil.h"

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

    // compression renderer can only be created internally, can not be moved, copied or destoryed
    LineCounter() = delete;
    LineCounter(const LineCounter&) = delete;
    LineCounter& operator=(const LineCounter&) = delete;

    static LineCounter* acquireReference(const CreateInfo& info); // acquire a reference (automatically creates renderer if not yet existing)
    static void tests(const CreateInfo& info);
    void release();                                 // has to be called to notify destruction before vulkan resources are destroyed
    void countLines(VkCommandBuffer commands, const CountLinesInfo& info);
private:
    LineCounter(const CreateInfo& info);
    ~LineCounter();

    static LineCounter* _singleton;
    int _refCount{};

    // vulkan resources that are destroyed externally
    VkUtil::Context _vkContext;
    VkRenderPass _renderPass;
    VkDescriptorSet _descSet; //only here for test purposes

    // vulkan resources that have to be destroyed
    VkUtil::PipelineInfo _countPipeInfo{};

    const std::string _computeShader = "shader/lineCount.comp.spv";
};