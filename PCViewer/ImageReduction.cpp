#include "ImageReduction.h"

void ImageReduction::init(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool) 
{
    Internals::device = device;
    Internals::physicalDevice = physicalDevice;
    Internals::commandPool = commandPool;
    Internals::queue = queue;
    Internals::descriptorPool = descriptorPool;

    // init reduce y compute
    {
        using namespace Internals;
        using namespace MaxY;
        VkShaderModule binaryModule = VkUtil::createShaderModule(device, PCUtil::readByteFile(shaderFile));
        std::vector<VkDescriptorSetLayoutBinding> bindings;

        VkDescriptorSetLayoutBinding binding;
        binding.binding = 0;
        binding.descriptorCount = 1;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings.push_back(binding);

        binding.binding = 1;
        bindings.push_back(binding);

        binding.binding = 2;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings.push_back(binding);

        VkUtil::createDescriptorSetLayout(device, bindings, &descriptorSetLayout);
        std::vector<VkDescriptorSetLayout> layouts{ descriptorSetLayout };

        VkUtil::createComputePipeline(device, binaryModule, layouts, &pipelineLayout, &pipeline);

        VkUtil::createBuffer(device, sizeof(UBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &uboBuffer);
        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device, uboBuffer, &memReq);
        
        VkMemoryAllocateInfo memAlloc = {};
        memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memAlloc.allocationSize = memReq.size;
        //To be continued
    }
}

void ImageReduction::cleanup()
{
    if (Internals::MaxY::pipeline) vkDestroyPipeline(Internals::device, Internals::MaxY::pipeline, nullptr);
    if (Internals::MaxY::pipelineLayout) vkDestroyPipelineLayout(Internals::device, Internals::MaxY::pipelineLayout, nullptr);
    if (Internals::MaxY::descriptorSetLayout) vkDestroyDescriptorSetLayout(Internals::device, Internals::MaxY::descriptorSetLayout, nullptr);
}
