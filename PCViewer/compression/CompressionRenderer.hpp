#pragma once
#include <vulkan/vulkan.h>
#include "../VkUtil.h"
#include "HierarchyLoadManager.hpp"

class CompressionRenderer{
public:
    // struct holding information needed to create the vulkan pipeline
    struct CreateInfo{
        VkUtil::Context context;
        VkRenderPass renderPass;
        VkSampleCountFlagBits sampleCount;
        VkDescriptorSetLayout dataLayout;
        VkDescriptorSetLayout uniformLayout;
    };

    // compression renderer can only be created internally, can not be moved, copied or destoryed
    CompressionRenderer() = delete;
    CompressionRenderer(const CompressionRenderer&) = delete;
    CompressionRenderer & operator=(const CompressionRenderer&) = delete;

    static CompressionRenderer* acquireReference(const CreateInfo& info); // acquire a reference (automatically creates renderer if not yet existing)
    void release();                                 // has to be called to notify destruction before vulkan resources are destroyed
    void render(VkCommandBuffer commands, std::shared_ptr<HierarchyLoadManager> loadManager);
private:
    CompressionRenderer(const CreateInfo& info);
    ~CompressionRenderer();

    static CompressionRenderer* _singleton;
    int _refCount{0};

    // vulkan resources that are destroyed externally
    VkUtil::Context _vkContext;
    VkRenderPass _renderPass;
    VkDescriptorSetLayout _dataLayout;
    VkDescriptorSetLayout _uniformLayout;

    // vulkan resources that have to be destroyed
    VkUtil::PipelineInfo _polyPipeInfo{}, _splinePipeInfo{};

    const std::string _vertexShader = "shader/compr.vert.spv";
    const std::string _geometryShader = "shader/compr.geom.spv";
    const std::string _fragmentShader = "shader/compr.frag.spv";
};