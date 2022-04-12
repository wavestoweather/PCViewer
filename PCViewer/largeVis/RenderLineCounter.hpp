#pragma once

#include "../VkUtil.h"

// holds a single vulkan compute pipeline instance for counting active lines by using rendering pipeline

class RenderLineCounter{
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
    RenderLineCounter() = delete;
    RenderLineCounter(const RenderLineCounter&) = delete;
    RenderLineCounter& operator=(const RenderLineCounter&) = delete;

    static RenderLineCounter* acquireReference(const CreateInfo& info); // acquire a reference (automatically creates renderer if not yet existing)
    static void tests(const CreateInfo& info);
    void release();                                 // has to be called to notify destruction before vulkan resources are destroyed
    void countLines(VkCommandBuffer commands, const CountLinesInfo& info);
private:
    RenderLineCounter(const CreateInfo& info);
    ~RenderLineCounter();

    static RenderLineCounter* _singleton;
    int _refCount{};

    // vulkan resources that are destroyed externally
    VkUtil::Context _vkContext;
    VkRenderPass _renderPass;
    VkFramebuffer _framebuffer;
    VkDescriptorSet _descSet{}; //only here for test purposes
    VkImage _countImage{};
    VkImageView _countImageView{};
    VkDeviceMemory _countImageMem{};
    uint32_t _aBins, _bBins;

    // vulkan resources that have to be destroyed
    VkUtil::PipelineInfo _countPipeInfo{};

    const std::string _vertexShader = "shader/lineCount.vert.spv";
    const std::string _fragmentShader = "shader/lineCount.frag.spv";
};