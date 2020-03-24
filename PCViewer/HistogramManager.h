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
		float maxGlobalCount;							//maximung value accross all attributes
		std::vector<float> area;						//area when scaling equals 1
		std::vector<float> maxCount;					//maximum histogramm value for each attribute
		std::vector<std::pair<float, float>> ranges;	//the value ranges for each attribute
		std::vector<std::vector<float>> originalBins;	//histogramm values for each attribute before smoothing was applied
		std::vector<std::vector<float>> bins;			//histogramm values for each attribute
		std::vector<unsigned int> side;					//stores on which side each attribute is rendered.
		std::vector<unsigned int> attributeColorOrderIdx;//stores indices in the order in which colors have to be assigned to them. In combination with side, the right colors can be chosen from the colorpalette "Dark2ExtendedReorder", such that the colors are on the right side. 
	};


	HistogramManager(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool, uint32_t binsAmount);
	~HistogramManager();

	void computeHistogramm(std::string& name, std::vector<std::pair<float,float>>& minMax, VkBuffer data, uint32_t amtOfData, VkBuffer indices, uint32_t amtOfIndices, VkBufferView indicesActivations);
	Histogram& getHistogram(std::string name);
	bool containsHistogram(std::string& name);
	void setNumberOfBins(uint32_t n);
	//setting stdDev to a negative number leads to automatic choose of kernel size
	void setSmoothingKernelSize(float stdDev);
	void updateSmoothedValues();

	void determineSideHist(Histogram& hist, bool **active = nullptr);

	bool ignoreZeroValues;
	bool ignoreZeroBins;
	bool* logScale;

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
	// numOfBins, numOfAttributes, amtOfIndices, ignoreZeroValues, float[min,max]
	// bins:
	// array for all bins
	VkBuffer uboBuffers[2];
	uint32_t uboOffsets[2];
	VkDeviceMemory uboMemory;

	uint32_t numOfBins;
	float stdDev;
	std::map<std::string, Histogram> histograms;

	void updateSmoothedValues(Histogram& hist);

	
};