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
#include <string.h>
#include <algorithm>

#define VERTICALROTSPEED .01f
#define HORIZONTALROTSPEED .01f
#define ZOOOMSPEED .03f

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
	void addBubbles(std::vector<uint32_t>& attributeIndex, std::vector<glm::vec3>& pos, std::vector<std::string>& attributeName, std::vector<uint32_t>& id, std::vector<bool>& active, std::vector<float*>& data, VkBuffer gData);
	void render();
	void updateCameraPos(float* mouseMovement);		//mouse movement must have following format: {x-velocity,y-velocity,mousewheel-velocity}
	void setPointScale(Scale scale);
	float& pointSize();
	VkSampler getImageSampler();
	VkImageView getImageView();
	void setImageDescSet(VkDescriptorSet desc);
	VkDescriptorSet getImageDescSet();

private:
	struct Ubo {
		float alphaMultiplier;
		uint32_t clipNormalize;					//is interpreted as bool. If 0, then all "unactive" data points will be shown as transparent grey discs, else they are discarded
		uint32_t amtOfAttributes;
		float offset;
		uint32_t scale;							//scale for the points: 0->Normal, 1->logarithmic, 2->squareroot
		float FoV;
		uint32_t relative;						//bool to indicate if the points should be scaled on zooming away
		alignas(16) glm::vec4 cameraPos;		//contains the maximum piont size i w
		alignas(16) glm::vec4 grey;
		alignas(16) glm::vec3 boundingRectMin;	//used to scale the 3d coordinates
		alignas(16) glm::vec3 boundingRectMax;
		alignas(16) glm::vec3 clippingRectMin;	//clippingRects min values for the 3d coordinates
		alignas(16) glm::vec3 clippingRectMax;	//clippingRects max values for the 3d coordinates
		alignas(16) glm::mat4 mvp;
		//float Array:
		//Color_0(4 floats), TopOffset_0(in Percent  as float), min_0, max_0, 
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
		uint32_t attributeIndex;
		glm::vec3 pos;
		std::string attributeName;
		uint32_t id;
		bool active;
	};

	struct gBubble {
		uint32_t attributeIndex;
		uint32_t dataIndex;
		glm::uvec3 posIndices;
		bool active;		//information if the current bubble is an active datum
	};

	//vulkan resources which do not have to be deleted/released
	VkPhysicalDevice	physicalDevice;
	VkDevice			device;
	VkDescriptorPool	descriptorPool;
	VkQueue				queue;
	VkCommandPool		commandPool;
	VkCommandBuffer		renderCommands;
	VkCommandBuffer		inverseRenderCommands;

	//vulkan resources which have to be deleted
	VkDeviceMemory		imageMemory;
	VkImage				image;
	VkDeviceMemory		depthImageMemory;			//depth image and color image need different memory typesn and thus cannot be stored in the same device memory
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
	VkDeviceMemory		bubbleInstancesMemory;		//instances have an extra memory block, as the size of the instances might change and thus rallocation is necessary
	VkBuffer			bubbleInstancesBuffer;
	uint32_t			bubbleInstancesSize;
	VkBuffer			dataBuffer;					//data buffer for the bubbles

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
	glm::vec3 cameraRot;
	float maxPointSize;
	float Fov;

	std::vector<gSphere> spheres;
	uint32_t amtOfIdxSphere;
	std::vector<gCylinder> cylinders;
	uint32_t amtOfIdxCylinder;
	std::vector<Bubble> bubbleInstances;

	//mehtods
	void setupRenderPipeline();
	void setupBuffer();
	void recordRenderCommands();
	void setupUbo();
};

#endif