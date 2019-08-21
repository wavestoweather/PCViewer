#ifndef View3d_H
#define	View3d_H

#include "VkUtil.h"
#include <vulkan/vulkan.h>
#include "glm/glm/glm.hpp"

class View3d {
public:
	View3d();
	~View3d();

	void render();
	void setDescriptorSet(VkDescriptorSet descriptor);
private:
	VkDeviceMemory		graphicsMemory;
	VkImage				iamge;
	VkImageView			imageView;
	VkDescriptorSet		descriptorSet;
	VkPipeline			pipeline;
	VkRenderPass		renderPass;
	VkPipelineLayout	pipelineLayout;
	VkBuffer			vertexBuffer;
	VkBuffer			indexBuffer;
};

#endif 