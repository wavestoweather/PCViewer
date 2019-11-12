/*
This class is the 3d view for the datasets. It is programmed, such that it takes a 3d texture, and renders it via effective raymarching.
Effective raymarching is accomplished via rendering a cube with its local coordinates as uv-coordinates, and then in the fragment shader only fragmetns which are occupied by
the backside of the cube are raymarched.
*/
#pragma once
#ifndef View3d_H
#define	View3d_H

#include "VkUtil.h"
#include "PCUtil.h"
#include <vulkan/vulkan.h>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include <limits.h>
#include <string.h>

#define VERTICALPANSPEED .01f
#define HORIZONTALPANSPEED .01f
#define ZOOMSPEED .03f

#define IDX3D(x,y,z,width,height) ((x)+((y)*width)+((z)*width*height))

class View3d {
public:
	View3d(uint32_t height, uint32_t width, VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool);
	~View3d();

	void resize(uint32_t width, uint32_t height);
	void resizeBox(float width, float height, float depth);
	void update3dImage(uint32_t width, uint32_t height, uint32_t depth, float* data);
	void updateCameraPos(float* mouseMovement);		//mouse movement must have following format: {x-velocity,y-velocity,mousewheel-velocity}
	void render();
	void setImageDescriptorSet(VkDescriptorSet descriptor);
	VkDescriptorSet getImageDescriptorSet();
	VkSampler getImageSampler();
	VkImageView getImageView();
private:
	struct UniformBuffer {
		glm::vec3 camPos;	//cameraPosition in model space
		alignas(16) glm::vec3 faces;//face positions for intersection tests
		alignas(16) glm::vec3 lightDir;
		alignas(16) glm::mat4 mvp;	//modelViewProjection Matrix
	};
	
	//shaderpaths
	static char vertPath[];
	static char fragPath[];

	//general information about the 3d view
	uint32_t imageHeight;
	uint32_t imageWidth;

	//information about 3d texture
	uint32_t image3dHeight;
	uint32_t image3dWidth;
	uint32_t image3dDepth;

	//information about the rendered 3dBox
	float boxHeight = 1;
	float boxWidth = 1;
	float boxDepth = 1;

	//Vulkan member variables
	VkPhysicalDevice	physicalDevice;
	VkDevice			device;
	VkCommandPool		commandPool;
	VkQueue				queue;
	VkCommandBuffer		commandBuffer;
	VkCommandBuffer		prepareImageCommand;
	VkDeviceMemory		imageMemory;
	VkImage				image;
	VkImageView			imageView;
	VkFramebuffer		frameBuffer;
	VkSampler			sampler;
	VkDescriptorSet		imageDescriptorSet;
	VkDeviceMemory		image3dMemory;
	VkImage				image3d;
	VkImageView			image3dView;
	VkSampler			image3dSampler;			//sampler seems to be not needed
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorPool	descriptorPool;
	VkDescriptorSet		descriptorSet;
	VkPipeline			pipeline;
	VkRenderPass		renderPass;
	VkPipelineLayout	pipelineLayout;
	VkDeviceMemory		constantMemory;
	VkBuffer			vertexBuffer;
	VkBuffer			indexBuffer;
	VkBuffer			uniformBuffer;
	uint32_t			uniformBufferOffset;

	//camera variables
	glm::vec3 camPos;		//camera position
	glm::vec3 lightDir;

	//methods to instatiate vulkan resources
	void createPrepareImageCommandBuffer();
	void createImageResources();
	void createBuffer();
	void createPipeline();
	void createDescriptorSets();
	void updateCommandBuffer();
};
#endif 