#pragma once

#include <vulkan/vulkan.h>
#include <string.h>
#include <vector>
#include <map>
#include <set>

#include "VkUtil.h"
#include "PCUtil.h"

#define LOCALSIZE 256
#define SHADERPATH "shader/brushComp.spv"

class GpuBrusher {
private:
	struct BrushInfo {
		uint32_t axis;
		uint32_t brushArrayOffset;
		uint32_t amtOfBrushes;
		uint32_t padding;
	};

	struct UBOinfo {
		uint32_t amtOfAttributes;
		uint32_t amtOfBrushAxes;
		uint32_t amtOfIndices;
		uint32_t lineCount;					//count for active lines would only one brush be applied
		int globalLineCount;				//count for actually active lines, if -1 this should not be counted
		uint32_t first;
		uint32_t andOr;
		uint32_t padding;
		uint32_t* indicesOffsets;			//index of brush axis, offset in brushes array and amount of brushes: ind1 offesetBrushes1 amtOfBrushes1 PADDING ind2 offsetBrushes2 amtOfBrushes2... 
	};

	struct UBObrushes {
		uint32_t* brushes;			//min max PADDING PADDING min max PADDING PADDING min....
	};

	struct UBOdata {
		float* data;				//data array is without padding!
	};

	struct UBOindices {
		uint32_t amtOfIndices;
		uint32_t* indices;			//indices which should be brushed
	};

	struct UBOstorage {
		uint32_t counter;
		uint32_t* indices;			//indices remaining after brush
	};

	//stored vulkan resources to create temporary gpu resources
	VkDevice device;
	VkPhysicalDevice physicalDevice;
	VkCommandPool commandPool;
	VkQueue queue;
	VkDescriptorPool descriptorPool;

	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
	VkDescriptorSetLayout descriptorSetLayout;
	VkBuffer uboBuffers[2];
	uint32_t uboOffsets[2];
	VkDeviceMemory uboMemory;
public:
	GpuBrusher(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool) : device(device), physicalDevice(physicalDevice), commandPool(commandPool), queue(queue), descriptorPool(descriptorPool) {
		VkResult err;

		VkShaderModule module = VkUtil::createShaderModule(device, PCUtil::readByteFile(std::string(SHADERPATH)));

		std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
		VkDescriptorSetLayoutBinding binding = {};
		binding.binding = 0;
		binding.descriptorCount = 1;		//informations
		binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings.push_back(binding);

		binding.binding = 1;				//brush ranges
		binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings.push_back(binding);

		binding.binding = 2;				//data buffer
		binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings.push_back(binding);

		binding.binding = 3;				//input indices
		binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings.push_back(binding);

		binding.binding = 4;				//output active indices
		binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
		layoutBindings.push_back(binding);

		VkUtil::createDescriptorSetLayout(device, layoutBindings, &descriptorSetLayout);

		std::vector<VkDescriptorSetLayout> layouts;
		layouts.push_back(descriptorSetLayout);
		VkUtil::createComputePipeline(device, module, layouts, &pipelineLayout, &pipeline);
	};

	~GpuBrusher() {
		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
	};

	//returns a pair containing the number of lines which would be active, would only one brush be applied, and the number of lines that really are still active
	std::pair<uint32_t,int> brushIndices(std::map<int, std::vector<std::pair<float, float>>>& brushes, uint32_t dataSize, VkBuffer data, VkBuffer indices, uint32_t indicesSize, VkBufferView activeIndices, uint32_t amtOfAttributes, bool first, bool andy, bool lastBrush) {
		//allocating all ubos and collection iformation about amount of brushes etc.
		uint32_t infoBytesSize = sizeof(UBOinfo) - sizeof(uint32_t) + sizeof(uint32_t) * 4 * brushes.size();
		UBOinfo* informations = (UBOinfo*)malloc(infoBytesSize);
		
		std::vector<BrushInfo> brushInfos;
		uint32_t off = 0;
		for (auto axis : brushes) {
			brushInfos.push_back({ (uint32_t)axis.first,off,(uint32_t)axis.second.size() });
			off += axis.second.size();
		}
		uint32_t brushesByteSize = sizeof(uint32_t) * 4 * off;
		UBObrushes* gpuBrushes = (UBObrushes*)malloc(brushesByteSize);

		UBOstorage* result;

		//allocating buffers and memory for ubos
		uboOffsets[0] = 0;
		VkUtil::createBuffer(device, infoBytesSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &uboBuffers[0]);
		VkUtil::createBuffer(device, brushesByteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &uboBuffers[1]);

		VkResult err;
		VkMemoryRequirements memReq;
		uint32_t memTypeBits;

		uboOffsets[0] = 0;
		vkGetBufferMemoryRequirements(device, uboBuffers[0], &memReq);
		VkMemoryAllocateInfo memalloc = {};
		memalloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memalloc.allocationSize = memReq.size;
		memTypeBits = memReq.memoryTypeBits;

		uboOffsets[1] = memalloc.allocationSize;
		vkGetBufferMemoryRequirements(device, uboBuffers[1], &memReq);
		memalloc.allocationSize += memReq.size;
		memTypeBits |= memReq.memoryTypeBits;

		memalloc.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		err = vkAllocateMemory(device, &memalloc, nullptr, &uboMemory);
		check_vk_result(err);

		//binding buffers to memory
		vkBindBufferMemory(device, uboBuffers[0], uboMemory, uboOffsets[0]);
		vkBindBufferMemory(device, uboBuffers[1], uboMemory, uboOffsets[1]);

		//creating the descriptor set and binding all buffers to the corrsponding bindings
		VkDescriptorSet descriptorSet;
		std::vector<VkDescriptorSetLayout> layouts;
		layouts.push_back(descriptorSetLayout);
		VkUtil::createDescriptorSets(device, layouts, descriptorPool, &descriptorSet);

		VkUtil::updateDescriptorSet(device, uboBuffers[0], infoBytesSize, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorSet);
		VkUtil::updateDescriptorSet(device, uboBuffers[1], brushesByteSize, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorSet);
		VkUtil::updateDescriptorSet(device, data, dataSize * amtOfAttributes * sizeof(float), 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorSet);
		VkUtil::updateDescriptorSet(device, indices, indicesSize * sizeof(uint32_t), 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorSet);
		VkUtil::updateTexelBufferDescriptorSet(device, activeIndices, 4, descriptorSet);

		//uploading data for brushing
		void* d;
		informations->amtOfBrushAxes = brushes.size();
		informations->amtOfAttributes = amtOfAttributes;
		informations->amtOfIndices = indicesSize;
		informations->lineCount = 0;
		informations->globalLineCount = (lastBrush) ? 0 : -1;
		informations->first = first;
		informations->andOr = andy;
		uint32_t* point = (uint32_t*)informations +((sizeof(UBOinfo)-sizeof(uint32_t))/sizeof(uint32_t)) - 1;		//strangeley the size of UBOinfo is too big
		uint32_t offset = 0;
		for (BrushInfo bi : brushInfos) {
			memcpy(point + offset, &bi, sizeof(uint32_t) * 4);
			offset +=  4;
		}
//#ifdef _DEBUG
//		std::cout << "Brush informations:" << std::endl;
//		std::cout << "amtOfBrushAxes" << informations->amtOfBrushAxes << std::endl;
//		std::cout << "amtOfAttributes" << informations->amtOfAttributes << std::endl;
//		std::cout << "amtOfIndices" << informations->amtOfIndices << std::endl;
//		std::cout << "lineCount" << informations->lineCount << std::endl;
//		std::cout << "first" << informations->first << std::endl;
//		std::cout << "andOr" << informations->andOr << std::endl;
//		for (int i = 0; i < brushInfos.size(); ++i) {
//			std::cout << point[i * 4] << " " << point[i * 4 + 1] << " " << point[i * 4 + 2] << " " << point[i * 4 + 3] << std::endl;
//		}
//#endif
		vkMapMemory(device, uboMemory, uboOffsets[0], infoBytesSize, 0, &d);
		memcpy(d, informations, infoBytesSize);
		vkUnmapMemory(device, uboMemory);

		offset = 0;
		float* bru = (float*)gpuBrushes;
		for (auto& axis : brushes) {
			for (auto& range : axis.second) {
				bru[offset++] = range.first;
				bru[offset++] = range.second;
				offset += 2;
			}
		}
		vkMapMemory(device, uboMemory, uboOffsets[1], brushesByteSize, 0, &d);
		memcpy(d, bru, brushesByteSize);
		vkUnmapMemory(device, uboMemory);

		//dispatching the command buffer to calculate active indices
		VkCommandBuffer command;
		VkUtil::createCommandBuffer(device, commandPool, &command);

		vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, {});
		vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
		int patchAmount = indicesSize / LOCALSIZE;
		patchAmount += (indicesSize % LOCALSIZE) ? 1 : 0;
		vkCmdDispatch(command, patchAmount, 1, 1);
		VkUtil::commitCommandBuffer(queue, command);
		err = vkQueueWaitIdle(queue);
		check_vk_result(err);

		//getting the amount of remaining lines(if only this brush would have been applied)
		VkUtil::downloadData(device, uboMemory, 0, sizeof(UBOinfo), informations);
		std::pair<uint32_t,int> res(informations->lineCount,informations->globalLineCount);
		
		vkFreeCommandBuffers(device, commandPool, 1, &command);
		vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
		vkFreeMemory(device, uboMemory, nullptr);
		vkDestroyBuffer(device, uboBuffers[0], nullptr);
		vkDestroyBuffer(device, uboBuffers[1], nullptr);

		free(informations);
		free(gpuBrushes);

		return res;
	};
};