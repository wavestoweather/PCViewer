/*
This class is the 3d view for the datasets. It is programmed, such that it takes a 3d texture, and renders it via effective raymarching.
Effective raymarching is accomplished via rendering a cube with its local coordinates as uv-coordinates, and then in the fragment shader only fragmetns which are occupied by
the backside of the cube are raymarched.
*/
#pragma once
#ifndef IsoSurfRenderer_H
#define	IsoSurfRenderer_H

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

//#define AMTOF3DTEXTURES 7			//amount of textures used for the density values. The total amount of density attributes is AMTOF3DTEXTURES * 4 (4 density channels per image) NOTE: found out that it is easily possible to write an array of textures to a binding
#define MAXAMTOF3DTEXTURES 30
#define LOCALSIZE 256

#define IDX3D(x,y,z,width,height) ((x)+((y)*width)+((z)*width*height))

class IsoSurfRenderer {
public:
	// height :		height of the rendered image
	// width :		width of the rendered image
	// device, pyhsicalDevice ... : valid vulkan instanes of the respective type
	IsoSurfRenderer(uint32_t height, uint32_t width, VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool);
	~IsoSurfRenderer();

	void resize(uint32_t width, uint32_t height);
	void resizeBox(float width, float height, float depth);
	void update3dDensities(uint32_t width, uint32_t height, uint32_t depth, uint32_t amtOfAttributes, std::vector<uint32_t>& densityAttributes, std::vector<std::pair<float, float>>& densityAttributesMinMax, glm::uvec3& positionIndices, uint32_t amtOfIndices, VkBuffer indices, uint32_t amtOfData, VkBuffer data);
	void updateCameraPos(float* mouseMovement);		//mouse movement must have following format: {x-velocity,y-velocity,mousewheel-velocity}
	void render();
	void setImageDescriptorSet(VkDescriptorSet descriptor);
	VkDescriptorSet getImageDescriptorSet();
	VkSampler getImageSampler();
	VkImageView getImageView();
private:
	struct UniformBuffer {
		glm::vec3 camPos;				//cameraPosition in model space
		alignas(16) glm::vec3 faces;	//face positions for intersection tests
		alignas(16) glm::vec3 lightDir;
		alignas(16) glm::mat4 mvp;		//modelViewProjection Matrix
	};

	struct ComputeInfos {
		uint32_t amtOfAttributes;		//amount of attributes in the dataset
		uint32_t amtOfDensityAttributes;//amount of attributes for which the density maps should be created
		uint32_t amtOfIndices;
		uint32_t dimX;
		uint32_t dimY;
		uint32_t dimZ;
		uint32_t xInd;
		uint32_t yInd;
		uint32_t zInd;
		float xMin;
		float xMax;
		float yMin;
		float yMax;
		float zMin;
		float zMax;
		uint32_t padding;
		//float array containing attribute infos:
		//index attr 1,
		//index attr 2,
		//...
	};
	
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
	//vulkan resources for the generated image
	VkImage				image;
	VkImageView			imageView;
	VkFramebuffer		frameBuffer;
	VkSampler			sampler;
	//vulkan resources for the 3d density images
	VkDescriptorSet		imageDescriptorSet;
	VkDeviceMemory		image3dMemory;
	std::vector<uint32_t>	image3dOffsets;
	std::vector<VkImage>	image3d;
	std::vector<VkImageView>image3dView;
	VkSampler			image3dSampler;
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
	//vulkan resources for the compute pipeline
	VkPipeline			computePipeline;
	VkPipelineLayout	computePipelineLayout;
	VkDescriptorSetLayout computeDescriptorSetLayout;

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