#pragma once
#include "imgui/imgui.h"
#include <vulkan/vulkan.h>
#include "VkUtil.h" 
#include "imgui/imgui_impl_vulkan.h"

#define TRANSFERFUNCTIONSIZE 256

class TransferFunctionEditor {
public:
    enum ColorMap {
        standard
    };

    TransferFunctionEditor(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool);
    ~TransferFunctionEditor();

    //activates the showing of the editor
    void show();
    //has to be called to draw the editor
    void draw();
    
    std::vector<uint8_t> getColorMap(ColorMap map);
    VkImageView getTransferImageView();
    VkDescriptorSet getTransferDescriptorSet();
    void setNextEditorPos(const ImVec2& pos, const ImVec2& pivot);

    uint32_t editorWidth, editorHeight, previewHeight;
private:
    //vulkan resources which are not allocated here
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkCommandPool commandPool;
    VkQueue queue;
    VkDescriptorPool descriptorPool;

    //vulkan resources which have to be deleted
    VkImage         transferImage;
    VkImageView     transferImageView;
    VkDeviceMemory  transferMemory;
    VkSampler       transferSampler;
    VkDescriptorSet transferDescriptorSet;

    VkFormat transferFormat;

    std::vector<uint8_t> currentTransferFunction;

    bool active;
    bool changed;
    uint32_t activeChannel;

    ImVec2 pos;
    ImVec2 pivot;

    void updateGpuImage();
};