#pragma once
#ifndef BubblePlotter_H
#define BubblePlotter_H

#include "Sphere.h"
#include "Cylinder.h"
#include "VkUtil.h"
#include "PCUtil.h"
#include <vulkan/vulkan.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include "glm/gtc/matrix_transform.hpp"
#include "glm/glm.hpp"
#include "glm/gtx/euler_angles.hpp"
#include "CameraNav.hpp"
#include "Color.h"
#include <string.h>
#include <algorithm>
#include <random>

class BubblePlotter {
public:
	enum Scale {
		Scale_Normal,
		Scale_Logarithmic,
		Scale_Squareroot
	};

	BubblePlotter(uint32_t width, uint32_t height, VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool);

	~BubblePlotter();

	void resizeImage(uint32_t width, uint32_t height);
	void addSphere(float r, glm::vec4 color, glm::vec3 pos);
	void addCylinder(float r, float length, glm::vec4 color, glm::vec3 pos);

	//pos:				a 3 component vector containing the indeces for the 3 dimensions
	//indices:			Indices of data which should be shown 
	//attributeNames:	vector with all attribute names
	//attributesMinMax:	min max values for the attributes
	//gData:			vulkan buffer with all data
	//activeAttributes: Buffer view for the bool buffer with the activations
	//amtOfAttributes:	amount of attributes
	//amtOfData:		amount of data
	void setBubbleData(glm::uvec3& pos,std::vector<uint32_t> indices, std::vector<std::string>& attributeNames, std::vector<std::pair<float, float>> attributesMinMax, std::vector<float*>& data, VkBuffer gData,VkBufferView activeData, uint32_t amtOfAttributes, uint32_t amtOfData);
	void render();
	void updateCameraPos(CamNav::NavigationInput input, float deltaT);
	void setPointScale(Scale scale);
	VkSampler getImageSampler();
	VkImageView getImageView();
	void setImageDescSet(VkDescriptorSet desc);
	VkDescriptorSet getImageDescSet();

	//public Attributes
	glm::uvec3 posIndices;
	float maxPointSize;
	float farClip;
	float Fov;
	float fovSpeed;
	float flySpeed;
	float fastFlyMultiplier;
	float rotationSpeed;
	float alphaMultiplier;
	bool clipping;
	bool normalization;
	float grey[4];
	Scale scale;
	bool scalePointsOnZoom;
	float layerSpacing;
	glm::vec3 boundingRectMin;
	glm::vec3 boundingRectMax;
	glm::vec3 clippingRectMin;
	glm::vec3 clippingRectMax;
	bool* attributeActivations;
	glm::vec4* attributeColors;
	float* attributeTopOffsets;				//The offsets for each attribute are given in %. 0 means that the point lies in its original layer, 1 is the last space before the next layer
	std::vector<std::pair<float,float>> attributeMinMaxValues;
	std::vector<Scale> attributeScales;
	std::vector<std::string> attributeNames;

private:
	struct Ubo {
		float alphaMultiplier;
		uint32_t clipNormalize;					//is interpreted as bool. If 0, then all "unactive" data points will be shown as transparent grey discs, else they are discarded
		uint32_t amtOfAttributes;
		float offset;							//total space between two layers
		float FoV;
		uint32_t relative;						//bool to indicate if the points should be scaled on zooming away
		uint32_t pad;
		uint32_t padding;
		alignas(16) glm::vec4 cameraPos;		//contains the maximum piont size in w
		alignas(16) glm::uvec4 posIndices;		//indices of the position attributes
		alignas(16) glm::vec4 grey;
		alignas(16) glm::vec3 boundingRectMin;	//used to scale the 3d coordinates
		alignas(16) glm::vec3 boundingRectMax;
		alignas(16) glm::vec3 clippingRectMin;	//clippingRects min values for the 3d coordinates
		alignas(16) glm::vec3 clippingRectMax;	//clippingRects max values for the 3d coordinates
		alignas(16) glm::mat4 mvp;
		//float Array:
		//Color_0(4 floats), TopOffset_0(in Percent  as float), min_0, max_0, scale_0 (0->Normal, 1->logarithmic, 2->squareroot),
		//Color_1[4], TopOffset_1[1], Color_2[4], TopOffset_2, ..
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

	struct Bubble {
		uint32_t attributeIndex;	//with this index things like the attribute name ... can be found out
		uint32_t id;
		bool active;
	};

	struct gBubble {
		uint32_t attributeIndex;
		uint32_t dataIndex;
		uint32_t active;		//information if the current bubble is an active datum
	};

	//vulkan resources which do not have to be deleted/released
	VkPhysicalDevice	physicalDevice;
	VkDevice			device;
	VkDescriptorPool	descriptorPool;
	VkQueue				queue;
	VkCommandPool		commandPool;
	VkCommandBuffer		renderCommands;
	VkCommandBuffer		inverseRenderCommands;

	//vulkan resources which are temporarily saved
	VkBufferView		dataActivations;

	//vulkan resources which have to be deleted
	VkDeviceMemory		imageMemory;
	VkImage				image;
	VkDeviceMemory		depthImageMemory;			//depth image and color image need different memory types and thus cannot be stored in the same device memory
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
	VkDeviceMemory		bubbleIndexMemory;		//instances have an extra memory block, as the size of the instances might change and thus rallocation is necessary
	VkBuffer			bubbleIndexBuffer;
	uint32_t			bubbleIndexSize;
	VkBuffer			dataBuffer;					//data buffer for the bubbles

	//here a few debug resources
	VkBuffer			ubo;
	VkDeviceMemory		uboMem;
	VkDescriptorSet		uboSet;

	//resources
	static char vertPath[];
	static char geomPath[];
	static char fragPath[];

	uint32_t imageWidth;
	uint32_t imageHeight;

	glm::vec3 cameraPos;
	glm::vec3 cameraRot;
	uint32_t amtOfAttributes;
	uint32_t amtOfDatapoints;

	std::uniform_int_distribution<int> distribution;
	std::default_random_engine randomEngine;

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