#include "HistogramManager.h"

HistogramManager::HistogramManager(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool) : device(device), physicalDevice(physicalDevice), commandPool(commandPool), queue(queue), descriptorPool(descriptorPool)
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

	binding.binding = 3;				//bins
	binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	layoutBindings.push_back(binding);

	VkUtil::createDescriptorSetLayout(device, layoutBindings, &descriptorSetLayout);

	std::vector<VkDescriptorSetLayout> layouts;
	layouts.push_back(descriptorSetLayout);
	VkUtil::createComputePipeline(device, module, layouts, &pipelineLayout, &pipeline);
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
}

void HistogramManager::computeHistogramm(std::string& name, std::vector<uint32_t>& indices, std::vector<std::pair<float, float>>& minMax, VkBuffer data, uint32_t amtOfData)
{
	VkResult err;

	uint32_t infosByteSize = (3 + minMax.size() * 2) * sizeof(float);
	char* infosBytes = new char[infosByteSize];
	uint32_t* inf = (uint32_t*)infosBytes;
	inf[0] = numOfBins;
	inf[1] = minMax.size();
	inf[2] = indices.size();
	float* infos = (float*)infosBytes;
	infos += 3;
	for (int i = 0; i < minMax.size(); ++i) {
		infos[2 * i] = minMax[i].first;
		infos[2 * i + 1] = minMax[i].second;
	}

	uint32_t binsByteSize = minMax.size() * numOfBins * sizeof(uint32_t);
	char* binsBytes = new char[binsByteSize];
	for (int i = 0; i < minMax.size() * numOfBins; ++i) ((float*)binsBytes)[i] = 0;

	//buffer allocations
	VkUtil::createBuffer(device, infosByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, uboBuffers);
	VkUtil::createBuffer(device, indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &uboBuffers[1]);
	VkUtil::createBuffer(device, binsByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &uboBuffers[2]);
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

	uboOffsets[2] = allocInfo.allocationSize;
	vkGetBufferMemoryRequirements(device, uboBuffers[2], &memReq);
	allocInfo.allocationSize += memReq.size;
	memType |= memReq.memoryTypeBits;
	allocInfo.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memType, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	vkAllocateMemory(device, &allocInfo, nullptr, &uboMemory);

	vkBindBufferMemory(device, uboBuffers[0], uboMemory, uboOffsets[0]);
	vkBindBufferMemory(device, uboBuffers[1], uboMemory, uboOffsets[1]);
	vkBindBufferMemory(device, uboBuffers[2], uboMemory, uboOffsets[2]);

	//upload of data
	VkUtil::uploadData(device, uboMemory, 0, infosByteSize, infosBytes);
	VkUtil::uploadData(device, uboMemory, uboOffsets[1], indices.size() * sizeof(uint32_t), indices.data());
	VkUtil::uploadData(device, uboMemory, uboOffsets[2], binsByteSize, binsBytes);

	//creation of the descriptor set
	VkDescriptorSet descSet;
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.push_back(descriptorSetLayout);
	VkUtil::createDescriptorSets(device, layouts, descriptorPool, &descSet);
	VkUtil::updateDescriptorSet(device, uboBuffers[0], infosByteSize, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
	VkUtil::updateDescriptorSet(device, uboBuffers[1], indices.size() * sizeof(uint32_t), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
	VkUtil::updateDescriptorSet(device, data, amtOfData * minMax.size() * sizeof(float), 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);
	VkUtil::updateDescriptorSet(device, uboBuffers[2], binsByteSize, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descSet);

	//running the compute pipeline
	VkCommandBuffer commands;
	VkUtil::createCommandBuffer(device, commandPool, &commands);
	vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, {});
	vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	int patchAmount = indices.size() / LOCALSIZE;
	patchAmount += (indices.size() % LOCALSIZE) ? 1 : 0;
	vkCmdDispatch(commands, patchAmount, 1, 1);
	VkUtil::commitCommandBuffer(queue, commands);
	err = vkQueueWaitIdle(queue);
	check_vk_result(err);

	//downloading results, analysing and saving them
	VkUtil::downloadData(device, uboMemory, uboOffsets[2], binsByteSize, binsBytes);
	uint32_t* bins = (uint32_t*)binsBytes;
	Histogram histogram;
	for (int i = 0; i < minMax.size(); ++i) {
		uint32_t maxVal = 0;
		histogram.bins.push_back({ }); //push back empty vector
		for (int j = 0; j < numOfBins; ++j) {
			uint32_t curVal = bins[i * minMax.size() + j];
			if (curVal > maxVal)maxVal = curVal;
			histogram.bins.back().push_back(curVal);
		}
		histogram.maxCount.push_back(maxVal);
	}

	histograms[name] = histogram;

	vkFreeCommandBuffers(device, commandPool, 1, &commands);
	vkFreeDescriptorSets(device, descriptorPool, 1, &descSet);
	vkDestroyBuffer(device, uboBuffers[0], nullptr);
	vkDestroyBuffer(device, uboBuffers[1], nullptr);
	vkDestroyBuffer(device, uboBuffers[2], nullptr);
	vkFreeMemory(device, uboMemory, nullptr);
	delete[] infosBytes;
	delete[] binsBytes;
}

HistogramManager::Histogram& HistogramManager::getHistogram(std::string name)
{
	return histograms[name];
	
}

void HistogramManager::setNumberOfBins(uint32_t n)
{
	numOfBins = n;
}
