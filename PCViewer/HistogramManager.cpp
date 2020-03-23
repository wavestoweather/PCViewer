#include "HistogramManager.h"
#include <cmath>

HistogramManager::HistogramManager(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool, uint32_t binsAmount) : device(device), physicalDevice(physicalDevice), commandPool(commandPool), queue(queue), descriptorPool(descriptorPool), numOfBins(binsAmount)
{
	VkShaderModule module = VkUtil::createShaderModule(device, PCUtil::readByteFile(std::string(SHADERPATH)));

	std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
	VkDescriptorSetLayoutBinding binding = {};
	binding.binding = 0;
	binding.descriptorCount = 1;		//informations
	binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	layoutBindings.push_back(binding);

	binding.binding = 1;				//indices
	binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	layoutBindings.push_back(binding);

	binding.binding = 2;				//data
	binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	layoutBindings.push_back(binding);

	binding.binding = 3;				//activations
	binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
	layoutBindings.push_back(binding);

	binding.binding = 4;				//bins
	binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	layoutBindings.push_back(binding);

	VkUtil::createDescriptorSetLayout(device, layoutBindings, &descriptorSetLayout);

	std::vector<VkDescriptorSetLayout> layouts;
	layouts.push_back(descriptorSetLayout);
	VkUtil::createComputePipeline(device, module, layouts, &pipelineLayout, &pipeline);

	stdDev = -1;

	ignoreZeroValues = true;
	ignoreZeroBins = true;
	logScale = nullptr;
}

HistogramManager::~HistogramManager()
{
	if (pipeline) {
		vkDestroyPipeline(device, pipeline, nullptr);
	}
	if (pipelineLayout) {
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
	}
	if (descriptorSetLayout) {
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
	}
	if (logScale) {
		delete[] logScale;
	}
}

void HistogramManager::computeHistogramm(std::string& name, std::vector<std::pair<float, float>>& minMax, VkBuffer data, uint32_t amtOfData, VkBuffer indices, uint32_t amtOfIndices, VkBufferView indicesActivations)
{
	if (!logScale) {
		logScale = new bool[minMax.size()];
		for (int i = 0; i < minMax.size(); ++i) logScale[i] = false;
	}

	VkResult err;

	uint32_t infosByteSize = (4 + minMax.size() * 2) * sizeof(float);
	char* infosBytes = new char[infosByteSize];
	uint32_t* inf = (uint32_t*)infosBytes;
	inf[0] = numOfBins;
	inf[1] = minMax.size();
	inf[2] = amtOfIndices;
	inf[3] = ignoreZeroValues;
#ifdef _DEBUG
	std::cout << "Bins: " << numOfBins << std::endl << "Amount of attributes: " << minMax.size() << std::endl << "Amount of indices: " << amtOfIndices << std::endl;
#endif
	float* infos = (float*)infosBytes;
	infos += 4;
	for (int i = 0; i < minMax.size(); ++i) {
		infos[2 * i] = minMax[i].first;
		infos[2 * i + 1] = minMax[i].second;
#ifdef _DEBUG
		std::cout << infos[2 * i] << "|" << infos[2 * i + 1] << std::endl;
#endif
	}

	uint32_t binsByteSize = minMax.size() * numOfBins * sizeof(uint32_t);
	char* binsBytes = new char[binsByteSize];
	for (int i = 0; i < minMax.size() * numOfBins; ++i) ((float*)binsBytes)[i] = 0;

	//buffer allocations
	VkUtil::createBuffer(device, infosByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, uboBuffers);
	VkUtil::createBuffer(device, binsByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &uboBuffers[1]);
	uboOffsets[0] = 0;
	uint32_t memType = 0;
	VkMemoryRequirements memReq;
	vkGetBufferMemoryRequirements(device, uboBuffers[0], &memReq);
	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize += memReq.size;
	memType |= memReq.memoryTypeBits;

	uboOffsets[1] = allocInfo.allocationSize;
	vkGetBufferMemoryRequirements(device, uboBuffers[1], &memReq);
	allocInfo.allocationSize += memReq.size;
	memType |= memReq.memoryTypeBits;
	allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memType, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	vkAllocateMemory(device, &allocInfo, nullptr, &uboMemory);

	vkBindBufferMemory(device, uboBuffers[0], uboMemory, uboOffsets[0]);
	vkBindBufferMemory(device, uboBuffers[1], uboMemory, uboOffsets[1]);

	//upload of data
	VkUtil::uploadData(device, uboMemory, 0, infosByteSize, infosBytes);
	VkUtil::uploadData(device, uboMemory, uboOffsets[1], binsByteSize, binsBytes);

	//creation of the descriptor set
	VkDescriptorSet descSet;
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.push_back(descriptorSetLayout);
	VkUtil::createDescriptorSets(device, layouts, descriptorPool, &descSet);
	VkUtil::updateDescriptorSet(device, uboBuffers[0], infosByteSize, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
	VkUtil::updateDescriptorSet(device, indices, amtOfIndices * sizeof(uint32_t), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
	VkUtil::updateDescriptorSet(device, data, amtOfData * minMax.size() * sizeof(float), 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
	VkUtil::updateTexelBufferDescriptorSet(device, indicesActivations, 3, descSet);
	VkUtil::updateDescriptorSet(device, uboBuffers[1], binsByteSize, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);

	//running the compute pipeline
	VkCommandBuffer commands;
	VkUtil::createCommandBuffer(device, commandPool, &commands);
	vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, {});
	vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	int patchAmount = amtOfIndices / LOCALSIZE;
	patchAmount += (amtOfIndices % LOCALSIZE) ? 1 : 0;
	vkCmdDispatch(commands, patchAmount, 1, 1);
	VkUtil::commitCommandBuffer(queue, commands);
	err = vkQueueWaitIdle(queue);
	check_vk_result(err);

	//downloading results, analysing and saving them
	VkUtil::downloadData(device, uboMemory, uboOffsets[1], binsByteSize, binsBytes);
	uint32_t* bins = (uint32_t*)binsBytes;
	Histogram histogram = {};
	for (int i = 0; i < minMax.size(); ++i) {
		histogram.bins.push_back({ });			//push back empty vector
		histogram.originalBins.push_back({ });
		histogram.maxCount.push_back({ });
		histogram.area.push_back(0);
		for (int j = 0; j < numOfBins; ++j) {
			float curVal = 0;
			//int div = 0;
			//int h = .2f * numOfBins;
			//for (int k = -h>>1; k <= h>>1; k += 1) {	//applying a box cernel according to chamers et al.
			//	if (j + k >= 0 && j + k < numOfBins) {
			//		curVal += bins[i * numOfBins + j + k];
			//		div++;
			//	}
			//}
			//curVal /= div;
			//if (curVal > maxVal)maxVal = curVal;
			histogram.bins.back().push_back(curVal);
			histogram.originalBins.back().push_back(bins[i * numOfBins + j]);
			histogram.area.back() += histogram.originalBins.back().back();
		}
	}
	histogram.ranges = minMax;
	updateSmoothedValues(histogram);

	histograms[name] = histogram;

	vkFreeCommandBuffers(device, commandPool, 1, &commands);
	vkFreeDescriptorSets(device, descriptorPool, 1, &descSet);
	vkDestroyBuffer(device, uboBuffers[0], nullptr);
	vkDestroyBuffer(device, uboBuffers[1], nullptr);
	vkFreeMemory(device, uboMemory, nullptr);
	delete[] infosBytes;
	delete[] binsBytes;
}

HistogramManager::Histogram& HistogramManager::getHistogram(std::string name)
{
	return histograms[name];
}

bool HistogramManager::containsHistogram(std::string& name)
{
	return histograms.find(name) != histograms.end();
}

void HistogramManager::setNumberOfBins(uint32_t n)
{
	numOfBins = n;
}

void HistogramManager::setSmoothingKernelSize(float stdDev)
{
	this->stdDev = stdDev;
	
	for (auto& hist : histograms) {
		updateSmoothedValues(hist.second);
	}
}

void HistogramManager::updateSmoothedValues()
{
	for (auto& hist : histograms) {
		updateSmoothedValues(hist.second);
	}
}

void HistogramManager::updateSmoothedValues(Histogram& hist)
{
	float stdDevSq = 2 * stdDev * stdDev;
	int kSize = (stdDev < 0) ? 0.2 * numOfBins + 1 : stdDev * 3 + 1;	//the plus 1 is there to realise the ceiling function

	//integrated is to 3 sigma standard deviation
	hist.maxGlobalCount = 0;
	float maxVal = 0;
	hist.area = std::vector<float>(hist.area.size(), 0);
	int att = 0;
	for (auto& attribute : hist.originalBins) {
		for (int bin = 0; bin < attribute.size(); ++bin) {
			float divisor = 0;
			float divider = 0;

			for (int k = -kSize; k <= kSize; ++k) {
				if (bin + k >= ((ignoreZeroBins)? 1:0) && bin + k < attribute.size()) {
					float factor = std::exp(-(k * k) / stdDevSq);
					divisor += attribute[bin + k] * factor;
					divider += factor;
				}
			}

			hist.bins[att][bin] = divisor / divider;
			if (logScale[att]) hist.bins[att][bin] = log(hist.bins[att][bin] + 1);
			hist.area[att] += hist.bins[att][bin];
			if (hist.bins[att][bin] > maxVal) maxVal = hist.bins[att][bin];
		}
		hist.maxCount[att] = maxVal;
		if (maxVal > hist.maxGlobalCount) {
			hist.maxGlobalCount = maxVal;
		}
		maxVal = 0;
		++att;
	}
}
