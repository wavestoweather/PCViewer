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
		uint32_t padding;
	};

	struct UBOinfo {
		uint32_t amtOfAttributes;
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
	VkBuffer uboBuffers[5];
	uint32_t uboOffsets[5];
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

		binding.binding = 3;				//input indices
		binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		layoutBindings.push_back(binding);

		binding.binding = 4;				//output indices
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
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
	};

	std::vector<int> brushIndices(std::map<int, std::vector<std::pair<float, float>>>& brushes, uint32_t dataByteSize, VkBuffer data, std::vector<int>& indices) {
		//allocating all ubos and collection iformation about amount of brushes etc.
		uint32_t infoBytesSize = sizeof(uint32_t) * 4 + sizeof(uint32_t) * 4 * brushes.size();
		UBOinfo* informations = (UBOinfo*)malloc(infoBytesSize);
		
		std::vector<BrushInfo> brushInfos;
		uint32_t off = 0;
		for (auto axis : brushes) {
			brushInfos.push_back({ axis.first,off,axis.second.size() });
			off += axis.second.size();
		}
		uint32_t brushesByteSize = sizeof(uint32_t) * 4 * off;
		UBObrushes* gpuBrushes = (UBObrushes*)malloc(brushesByteSize);

		UBOstorage* result;

		//allocating buffers and memory for ubos
		uboOffsets[0] = 0;
		VkUtil::createBuffer(device, infoBytesSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &uboBuffers[0]);
		VkUtil::createBuffer(device, brushesByteSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &uboBuffers[1]);
		VkUtil::createBuffer(device, indices.size() * sizeof(uint32_t) * 4, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &uboBuffers[3]);
		VkUtil::createBuffer(device, indices.size() * sizeof(uint32_t) * 4 + sizeof(uint32_t) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &uboBuffers[4]);

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

		uboOffsets[3] = memalloc.allocationSize;
		vkGetBufferMemoryRequirements(device, uboBuffers[3], &memReq);
		memalloc.allocationSize += memReq.size;
		memTypeBits |= memReq.memoryTypeBits;

		uboOffsets[4] = memalloc.allocationSize;
		vkGetBufferMemoryRequirements(device, uboBuffers[4], &memReq);
		memalloc.allocationSize += memReq.size;
		memTypeBits |= memReq.memoryTypeBits;

		memalloc.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		err = vkAllocateMemory(device, &memalloc, nullptr, &uboMemory);
		check_vk_result(err);

		//binding buffers to memory
		vkBindBufferMemory(device, uboBuffers[0], uboMemory, uboOffsets[0]);
		vkBindBufferMemory(device, uboBuffers[1], uboMemory, uboOffsets[1]);
		vkBindBufferMemory(device, uboBuffers[3], uboMemory, uboOffsets[3]);
		vkBindBufferMemory(device, uboBuffers[4], uboMemory, uboOffsets[4]);

		//creating the descriptor set and binding all buffers to the corrsponding bindings
		VkDescriptorSet descriptorSet;
		std::vector<VkDescriptorSetLayout> layouts;
		layouts.push_back(descriptorSetLayout);
		VkUtil::createDescriptorSets(device, layouts, descriptorPool, &descriptorSet);

		VkUtil::updateDescriptorSet(device, uboBuffers[0], infoBytesSize, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorSet);
		VkUtil::updateDescriptorSet(device, uboBuffers[1], brushesByteSize, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorSet);
		VkUtil::updateDescriptorSet(device, data, dataByteSize, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorSet);
		VkUtil::updateDescriptorSet(device, uboBuffers[3], indices.size() * sizeof(uint32_t) * 4, 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorSet);
		VkUtil::updateDescriptorSet(device, uboBuffers[4], indices.size() * sizeof(uint32_t) * 4, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorSet);

		//uploading data for brushing
		void* d;
		informations->amtOfAttributes = brushes.size();
		informations->indicesOffsets = ((uint32_t*)informations) + 4;
		uint32_t offset = 0;
		for (BrushInfo bi : brushInfos) {
			memcpy(informations->indicesOffsets + offset, &bi, sizeof(uint32_t) * 4);
			offset += sizeof(uint32_t) * 4;
		}
		vkMapMemory(device, uboMemory, uboOffsets[0], infoBytesSize, 0, &d);
		memcpy(d, informations, infoBytesSize);
		vkUnmapMemory(device, uboMemory);

		offset = 0;
		int32_t* bru = (int32_t*)gpuBrushes;
		for (auto& axis : brushes) {
			for (auto& range : axis.second) {
				bru[offset++] = range.first;
				bru[offset++] = range.second;
				offset += 2;
			}
		}
		vkMapMemory(device, uboMemory, uboOffsets[1], brushesByteSize, 0, &d);
		memcpy(d, gpuBrushes, brushesByteSize);
		vkUnmapMemory(device, uboMemory);

		uint32_t* gpuInd = (uint32_t*)malloc(indices.size() * 4 * sizeof(uint32_t));
		offset = 0;
		for (int i : indices) {
			gpuInd[offset] = i;
			offset += 4;
		}

		vkMapMemory(device, uboMemory, uboOffsets[3], indices.size() * 4 * sizeof(uint32_t), 0, &d);
		memcpy(d, gpuInd, indices.size() * 4 * sizeof(uint32_t));
		vkUnmapMemory(device, uboMemory);

		//dispatching the command buffer to calculate active indices
		VkCommandBuffer command;
		VkUtil::createCommandBuffer(device, commandPool, &command);

		vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, {});
		vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
		int patchAmount = indices.size() / LOCALSIZE;
		patchAmount += (indices.size() % LOCALSIZE) ? 1 : 0;
		vkCmdDispatch(command, patchAmount, 1, 1);
		VkUtil::commitCommandBuffer(queue, command);
		vkQueueWaitIdle(queue);

		//pulling the result from the gpu and filling the result vector
		uint32_t* brushInd = (uint32_t*)malloc(indices.size() * 4 * sizeof(uint32_t));
		vkMapMemory(device, uboMemory, uboOffsets[4], indices.size() * 4 * sizeof(uint32_t) + sizeof(uint32_t) * 4, 0, &d);
		memcpy(brushInd, d, indices.size() * 4 * sizeof(uint32_t) + sizeof(uint32_t) * 4);
		vkUnmapMemory(device, uboMemory);
		std::vector<int> res;
		for (int i = 0; i < brushInd[0]; i++) {
			res.push_back(brushInd[4 + i*4]);
		}
		
		vkFreeCommandBuffers(device, commandPool, 1, &command);
		vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
		vkFreeMemory(device, uboMemory, nullptr);
		vkDestroyBuffer(device, uboBuffers[0], nullptr);
		vkDestroyBuffer(device, uboBuffers[1], nullptr);
		vkDestroyBuffer(device, uboBuffers[3], nullptr);
		vkDestroyBuffer(device, uboBuffers[4], nullptr);

		delete[] informations;
		delete[] gpuBrushes;
		delete[] gpuInd;
		delete[] brushInd;

		return res;
	};
};