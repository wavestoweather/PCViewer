#ifndef PriorityColorUpdater_H
#define PriorityColorUpdater_H
#include <vulkan/vulkan.h>
#include <map>
#include <string.h>
#include <vector>

#include "VkUtil.h"
#include "PCUtil.h"

#define LOCALSIZE 64

class PriorityColorUpdater {
private:
	struct UBO {
		uint32_t amtOfAttributes;
		uint32_t priorityAttribute;
		float priorityCenter;
		float denominator;
		uint32_t amtOfData;
	} ubo;

	VkDevice device;
	VkPhysicalDevice physicalDevice;
	VkCommandPool commandPool;
	VkQueue queue;
	VkDescriptorPool descriptorPool;

	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
	VkDescriptorSetLayout descriptorSetLayout;
	std::map<std::string, VkDescriptorSet> descriptorSets;
	VkBuffer uboBuffer;
	VkDeviceMemory uboMemory;

	const char shaderPath[23];

public:
	PriorityColorUpdater(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool): queue(queue), device(device), physicalDevice(physicalDevice), commandPool(commandPool), descriptorPool(descriptorPool), shaderPath("shader/colorUpdate.spv") {
		VkResult err;
		
		VkShaderModule module = VkUtil::createShaderModule(device, PCUtil::readByteFile(std::string(shaderPath)));

		std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
		VkDescriptorSetLayoutBinding binding = {};
		binding.binding = 0;
		binding.descriptorCount = 1;		//ubo
		binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		layoutBindings.push_back(binding);
		
		binding.binding = 1;				//indexBuffer
		binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		layoutBindings.push_back(binding);

		binding.binding = 2;				//data buffer
		binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		layoutBindings.push_back(binding);

		binding.binding = 3;				//priority color
		binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings.push_back(binding);

		VkUtil::createDescriptorSetLayout(device, layoutBindings, &descriptorSetLayout);

		std::vector<VkDescriptorSetLayout> layouts;
		layouts.push_back(descriptorSetLayout);
		VkUtil::createComputePipeline(device, module, layouts, &pipelineLayout, &pipeline);

		VkUtil::createBuffer(device, sizeof(UBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &uboBuffer);
		VkMemoryRequirements memReq;
		vkGetBufferMemoryRequirements(device, uboBuffer, &memReq);
		
		VkMemoryAllocateInfo memalloc = {};
		memalloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memalloc.allocationSize = memReq.size;
		memalloc.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		err = vkAllocateMemory(device, &memalloc, nullptr, &uboMemory);
		check_vk_result(err);

		vkBindBufferMemory(device, uboBuffer, uboMemory, 0);
	};

	~PriorityColorUpdater() {
		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyBuffer(device, uboBuffer, nullptr);
		vkFreeMemory(device, uboMemory, nullptr);
	};

	void updatePriorityColor(uint32_t dataSize, uint32_t amtOfAttributes, uint32_t priorityAttribute, float priorityCenter, float denominator, VkBuffer data, VkBuffer priorityColorBuffer,std::vector<int>& indices) {
		VkResult err;
		VkDescriptorSet descSet;
		
		std::vector<VkDescriptorSetLayout> layouts;
		layouts.push_back(descriptorSetLayout);
		VkUtil::createDescriptorSets(device, layouts, descriptorPool, &descSet);

		//update ubo and upload
		ubo.amtOfAttributes = amtOfAttributes;
		ubo.priorityAttribute = priorityAttribute;
		ubo.priorityCenter = priorityCenter;
		ubo.denominator = denominator;
		ubo.amtOfData = indices.size();
		void* d;
		vkMapMemory(device, uboMemory, 0, sizeof(UBO), 0, &d);
		memcpy(d, &ubo, sizeof(UBO));
		vkUnmapMemory(device, uboMemory);
		VkUtil::updateDescriptorSet(device, uboBuffer, sizeof(UBO), 0, descSet);
		
		VkBuffer indexBuffer;
		uint32_t indexByteSize = indices.size() * sizeof(uint32_t);
		VkUtil::createBuffer(device, indexByteSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &indexBuffer);
		VkMemoryRequirements memReq;
		vkGetBufferMemoryRequirements(device, indexBuffer, &memReq);

		VkMemoryAllocateInfo memalloc = {};
		memalloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memalloc.allocationSize = memReq.size;
		memalloc.memoryTypeIndex = VkUtil::findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		VkDeviceMemory indexMemory;
		err = vkAllocateMemory(device, &memalloc, nullptr, &indexMemory);
		check_vk_result(err);

		vkBindBufferMemory(device, indexBuffer, indexMemory, 0);

		vkMapMemory(device, indexMemory, 0, indexByteSize, 0, &d);
		memcpy(d, indices.data(), indexByteSize);
		vkUnmapMemory(device, indexMemory);

		VkUtil::updateDescriptorSet(device, indexBuffer, indexByteSize, 1, descSet);
		VkUtil::updateDescriptorSet(device, data, dataSize * amtOfAttributes * sizeof(float), 2, descSet);
		VkUtil::updateDescriptorSet(device, priorityColorBuffer, dataSize * sizeof(float), 3, descSet);

		VkCommandBuffer command;
		VkUtil::createCommandBuffer(device, commandPool, &command);
		
		vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, {});
		vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
		int patchAmount = indices.size() / LOCALSIZE;
		patchAmount += (indices.size() % LOCALSIZE) ? 1 : 0;
		vkCmdDispatch(command, indices.size() / LOCALSIZE + 1, 1, 1);


		vkDeviceWaitIdle(device);
		vkFreeCommandBuffers(device, commandPool, 1, &command);
		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeDescriptorSets(device, descriptorPool, 1, &descSet);
		vkFreeMemory(device, indexMemory, nullptr);
	};
};

#endif