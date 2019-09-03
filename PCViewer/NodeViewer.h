#pragma once
#ifndef NodeViewer_H
#define NodeViewer_H

#include "Sphere.h"
#include "Cylinder.h"
#include "VkUtil.h"
#include "PCUtil.h"
#include <vulkan/vulkan.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include "glm/gtc/matrix_transform.hpp"
#include "glm/glm.hpp"

#define VERTICALROTSPEED .01f
#define HORIZONTALROTSPEED .01f
#define ZOOOMSPEED .03f

class NodeViewer {
public:
	NodeViewer(uint32_t width, uint32_t height, VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool);

	~NodeViewer();

	void resizeImage(uint32_t width, uint32_t height);
	void addSphere(float r, glm::vec4 color, glm::vec3 pos);
	void addCylinder(float r, float length, glm::vec4 color, glm::vec3 pos);
	void render();
	void updateCameraPos(float* mouseMovement);		//mouse movement must have following format: {x-velocity,y-velocity,mousewheel-velocity}
	VkSampler getImageSampler();
	VkImageView getImageView();
	void setImageDescSet(VkDescriptorSet desc);
	VkDescriptorSet getImageDescSet();

private:
	struct Ubo {
		glm::vec3 cameraPos;
		alignas(16) glm::vec4 color;
		alignas(16) glm::mat4 mvp;
		alignas(16) glm::mat4 worldNormals;
	};

	struct gSphere {
		float radius;
		glm::vec4 color;
		glm::vec3 pos;
		VkDescriptorSet descSet;
	};

	struct gCylinder {
		float radius;
		float length;
		glm::vec4 color;
		glm::vec3 pos;
		VkDescriptorSet descSet;
	};

	//vulkan resources which do not have to be deleted/released
	VkPhysicalDevice	physicalDevice;
	VkDevice			device;
	VkDescriptorPool	descriptorPool;
	VkQueue				queue;
	VkCommandPool		commandPool;
	VkCommandBuffer		renderCommands;

	//vulkan resources which have to be deleted
	VkDeviceMemory		imageMemory;
	VkImage				image;
	VkDeviceMemory		depthImageMemory;			//depth image and color image need different memory types
	VkImage				depthImage;
	VkImageView			imageView;
	VkImageView			depthImageView;
	VkFramebuffer		framebuffer;
	VkSampler			imageSampler;
	VkDescriptorSet		imageDescSet;
	VkPipeline			pipeline;
	VkPipelineLayout	pipelineLayout;
	VkRenderPass		renderPass;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDeviceMemory		bufferMemory;
	VkBuffer			sphereVertexBuffer;
	VkBuffer			sphereIndexBuffer;
	VkBuffer			cylinderVertexBuffer;
	VkBuffer			cylinderIndexBuffer;

	//here a few debug resources are
	VkBuffer			ubo;
	VkDeviceMemory		uboMem;
	VkDescriptorSet		uboSet;

	//resources
	static char vertPath[];
	static char fragPath[];

	uint32_t imageWidth;
	uint32_t imageHeight;

	glm::vec3 cameraPos;

	std::vector<gSphere> spheres;
	uint32_t amtOfIdxSphere;
	std::vector<gCylinder> cylinders;
	uint32_t amtOfIdxCylinder;

	//mehtods
	void setupRenderPipeline();
	void setupBuffer();
	void recordRenderCommands();
	void setupUbo();
};

#endif