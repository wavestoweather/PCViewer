#pragma once
#include <vector>
#include <vulkan/vulkan.h>
#include <string>
#include <unordered_map>

class HistogramManager {
public:
	void computeHistogramm(std::string name, std::vector<uint32_t> indices, VkBuffer data);
	void setNumberOfBins(uint32_t n);

private:
	struct histogram {
		std::string namme;
		std::vector<float> bins;
	};

	VkDevice device;
	VkPhysicalDevice physicalDevice;
	VkCommandPool commandPool;
	VkQueue queue;
	VkDescriptorPool descriptorPool;

	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
	VkDescriptorSetLayout descriptorSetLayout;
	VkBuffer uboBuffers[5];
	uint32_t uboOffsets[5];
	VkDeviceMemory uboMemory;

	uint32_t numOfBins;
};