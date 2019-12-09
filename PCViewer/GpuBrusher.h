#pragma once

#include <vulkan/vulkan.h>
#include <string.h>
#include <vector>
#include <map>

#include "VkUtil.h"
#include "PCUtil.h"

#define LOCALSIZE 64
#define SHADERPATH "shader/brush.spv"

class GpuBrusher {
private:
	struct BrushInfo {
		uint32_t axis;
		uint32_t brushArrayOffset;
		uint32_t amtOfBrushes;
	};

	struct UBOinfo {
		uint32_t amtOfAttributes;
		uint32_t* indicesOffsets;			//index of brush axis, offset in brushes array and amount of brushes: ind1 offesetBrushes1 amtOfBrushes1 PADDING ind2 offsetBrushes2 amtOfBrushes2... 
	} informations;

	struct UBObrushes {
		uint32_t* brushes:			//min max PADDING PADDING min max PADDING PADDING min....
	} brushRanges;

	struct UBOdata {
		float* data;				//data array is without padding!
	} data;

	struct UBOindices {
		uint32_t amtOfIndices;
		uint32_t* indices;			//indices which should be brushed
	} indices;

	struct UBOstorage {
		uint32_t counter;
		uint32_t* indices;			//indices remaining after brush
	} storage;

	//stored vulkan resources to create temporary gpu resources
	VkDevice device;
	VkPhysicalDevice physicalDevice;
	VkCommandPool commandPool;
	VkQueue queue;
	VkDescriptorPool descriptorPool;

	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSets[4];
	VkBuffer uboBuffers[4];
	uint32_t uboOffsets[4];
	VkDeviceMemory uboMemory;
public:
	GpuBrusher(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool) : queue(queue), device(device), physicalDevice(physicalDevice), commandPool(commandPool), descriptorPool(descriptorPool) {
		VkResult err;

		VkShaderModule module = VkUtil::createShaderModule(device, PCUtil::readByteFile(std::string(SHADERPATH)));

		std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
		VkDescriptorSetLayoutBinding binding = {};
		binding.binding = 0;
		binding.descriptorCount = 1;		//informations
		binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		layoutBindings.push_back(binding);

		binding.binding = 1;				//brush ranges
		binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		layoutBindings.push_back(binding);

		binding.binding = 2;				//data buffer
		binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		layoutBindings.push_back(binding);

		binding.binding = 3;				//output indices
		binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings.push_back(binding);

		VkUtil::createDescriptorSetLayout(device, layoutBindings, &descriptorSetLayout);

		std::vector<VkDescriptorSetLayout> layouts;
		layouts.push_back(descriptorSetLayout);
		VkUtil::createComputePipeline(device, module, layouts, &pipelineLayout, &pipeline);
	};

	~GpuBrusher() {
		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyBuffer(device, uboBuffer, nullptr);
		vkFreeMemory(device, uboMemory, nullptr);
	};

	std::vector<int> brushIndices(std::map<int, std::vector<std::pair<float, float>>> brushes, VkBuffer data, std::vector<int> indices) {
		//allocating all ubos and collection iformation about amount of brushes etc.
		uint32_t infoBytesSize = sizeof(uint32_t) * 4 + sizeof(uint32_t) * 4 * brushes.size();
		UBOinfo* informations = (UBOinfo*)malloc(infoBytesSize);
		
		std::vector<BrushInfo> brushInfos;
		uint32_t off = 0;
		for (auto axis : brushes) {
			brushInfos.push_back({ axis.first,off,axis.second.size() });
			off += axis.second.size();
		}
		uint32_t brushesSize = sizeof(uint32_t) * 4 * off;
		UBObrushes* brushes = (UBObrushes*)malloc(brushesSize);
	};
};