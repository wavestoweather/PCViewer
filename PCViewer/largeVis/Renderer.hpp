#pragma once
#include <string>
#include <vulkan/vulkan.hpp>
#include "../VkUtil.h"
#include <map>
#include <../Attribute.hpp>

namespace compression{
class Renderer{
public:
    enum class RenderType{
        Polyline,
        Spline,
        PriorityPolyline,
        PrioritySpline
    };
    // struct holding information needed to create the vulkan pipeline
    struct CreateInfo{
        VkUtil::Context context;
        VkRenderPass renderPass;
        VkFramebuffer framebuffer;
    };
    struct RenderInfo{
        VkCommandBuffer renderCommands;                     // the command buffer from the main command buffer
        std::string drawListId;                             // drawlist id used for internal resource management
        RenderType renderType;
        std::vector<VkBuffer>& counts;                      // the count buffer are expected to be in row major order (same order as for images)
        std::vector<std::pair<uint32_t, uint32_t>>& axes;   // contains for each counts buffer the axes
        uint32_t countSizes;                                // contains for each counts buffer the amount of values in it
        std::vector<int>& order;
        std::vector<Attribute>& attributes;
        bool* attributeActive;
        uint32_t attributeAxisSizes;                        //contains for each axis how many bins exist.
        VkBuffer attributeInformation;                      // contains mapping information for axis scaling, axis positioning and padding
        bool clear;                                         // indicates if the framebuffer should be cleared before rendered to
    };

    // compression renderer can only be created internally, can not be moved, copied or destoryed
    Renderer() = delete;
    Renderer(const Renderer&) = delete;
    Renderer & operator=(const Renderer&) = delete;

    static Renderer* acquireReference(const CreateInfo& info);  // acquire a reference (automatically creates renderer if not yet existing)
    void release();                                             // has to be called to notify destruction before vulkan resources are destroyed
    void updateFramebuffer(VkFramebuffer framebuffer, uint32_t newWidth, uint32_t newHeight);   // used to update the framebuffer of the renderer incase of image resizing(future use)
    void render(const RenderInfo& renderInfo);
private:
    struct PushConstants{
        uint32_t aAxis, bAxis, aSize, bSize;
    };

    Renderer(const CreateInfo& info);
    ~Renderer();

    static Renderer* _singleton;
    int _refCount{0};

    // vulkan resources that are destroyed externally
    VkUtil::Context _vkContext;
    VkRenderPass _renderPass;       // holds a non clear renderpass
    VkFramebuffer _framebuffer;     // holds the framebuffer for teh pcviewer image

    // vulkan resources that have to be destroyed
    VkUtil::PipelineInfo _polyPipeInfo{}, _splinePipeInfo{};

    std::map<std::string, VkDescriptorSet> _infoDescSets{};  // contains for each drawlist a descriptor set as multiple descriptor sets might be required

    const std::string _vertexShader = "shader/largeVis.vert.spv";
    const std::string _geometryShader = "shader/largeVis.geom.spv";
    const std::string _fragmentShader = "shader/largeVis.frag.spv";
};
}