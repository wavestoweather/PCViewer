#pragma once
#include <vector>
#include <vulkan/vulkan.h>
#include <string>
#include <map>
#include "VkUtil.h"
#include "PCUtil.h"

#define SHADERPATH "shader/histComp.spv"
#define LOCALSIZE 256

class HistogramManager {
public:
	struct Histogram {
		std::vector<float> maxCount;					//maximum histogramm value for each attribute
		std::vector<std::pair<float, float>> ranges;	//the value ranges for each attribute
		std::vector<std::vector<float>> bins;			//histogramm values for each attribute
	};


	HistogramManager(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool);
	~HistogramManager();

	void computeHistogramm(std::string& name, std::vector<uint32_t>& indices, std::vector<std::pair<float,float>>& minMax, VkBuffer data, uint32_t amtOfData);
	Histogram& getHistogram(std::string name);
	bool containsHistogram(std::string& name);
	void setNumberOfBins(uint32_t n);

private:
	VkDevice device;
	VkPhysicalDevice physicalDevice;
	VkCommandPool commandPool;
	VkQueue queue;
	VkDescriptorPool descriptorPool;

	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
	VkDescriptorSetLayout descriptorSetLayout;
	// 3 ubo buffers for
	// informations:
	// numOfBins numOfAttributes float[min,max]
	// indices:
	// simply all indices
	// bins:
	// array for all bins
	VkBuffer uboBuffers[3];
	uint32_t uboOffsets[3];
	VkDeviceMemory uboMemory;

	uint32_t numOfBins;
	std::map<std::string, Histogram> histograms;
};