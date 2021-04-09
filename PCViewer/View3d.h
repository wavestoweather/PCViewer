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
#define LOCALSIZE 256

#define IDX3D(x,y,z,width,height) ((x)+((y)*width)+((z)*width*height))

class View3d {
public:
	View3d(uint32_t height, uint32_t width, VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool);
	~View3d();

	void resize(uint32_t width, uint32_t height);
	void resizeBox(float width, float height, float depth);
	void update3dImage(const std::vector<float>& xDim, const std::vector<float>& yDim, const std::vector<float>& zDim, bool linAxis[3], const uint32_t posIndices[3], uint32_t densityIndex, const float minMax[2], VkBuffer data, uint32_t dataByteSize, VkBuffer indices, uint32_t indicesSize, uint32_t amtOfAttributes);
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
		uint32_t linearAxes;
		uint32_t padding[3];
	};

	struct ComputeUBO {
		uint32_t posIndices[3];
		uint32_t linearAxes;
		uint32_t densityAttribute;
		uint32_t amtOfIndices;
		uint32_t amtOfAttributes;
		float xMin;
		float xMax;
		float yMin;
		float yMax;
		float zMin;
		float zMax;
		uint32_t dimX;
		uint32_t dimY;
		uint32_t dimZ;
		float minValue;
		float maxValue;
	};

	//constants
	const int			dimensionCorrectionSize = 256;
	const VkFormat		dimensionCorrectionFormat = VK_FORMAT_R32_SFLOAT;
	
	//shaderpaths
	static char vertPath[];
	static char fragPath[];
	static char computePath[];

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
	std::vector<float>	dimensionCorrectionArrays[3];
	bool				dimensionCorrectionLinearDim[3];
	VkDeviceMemory		dimensionCorrectionMemory;
	VkImage				dimensionCorrectionImages[3];
	std::vector<VkImageView> dimensionCorrectionViews;

	//compute resources
	VkPipeline			densityFillPipeline;
	VkPipelineLayout	densityFillPipelineLayout;
	VkDescriptorSetLayout densityFillDescriptorLayout;

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
	bool updateDimensionImages(const std::vector<float>& xDim, const std::vector<float>& yDim, const std::vector<float>& zDim);
};
#endif 