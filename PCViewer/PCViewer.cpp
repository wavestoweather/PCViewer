// PCViewer.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "PCViewer.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_vulkan.h"

#include <stdio.h>          // printf, fprintf
#include <stdlib.h>         // abort
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <vector>
#include <limits>


// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

//#define IMGUI_UNLIMITED_FRAME_RATE
#ifdef _DEBUG
#define IMGUI_VULKAN_DEBUG_REPORT
#endif

static VkAllocationCallbacks* g_Allocator = NULL;
static VkInstance               g_Instance = VK_NULL_HANDLE;
static VkPhysicalDevice         g_PhysicalDevice = VK_NULL_HANDLE;
static VkDevice                 g_Device = VK_NULL_HANDLE;
static uint32_t                 g_QueueFamily = (uint32_t)-1;
static VkQueue                  g_Queue = VK_NULL_HANDLE;
static VkDebugReportCallbackEXT g_DebugReport = VK_NULL_HANDLE;
static VkPipelineCache          g_PipelineCache = VK_NULL_HANDLE;
static VkDescriptorPool         g_DescriptorPool = VK_NULL_HANDLE;

static ImGui_ImplVulkanH_Window g_MainWindowData;
static int                      g_MinImageCount = 2;
static bool                     g_SwapChainRebuild = false;
static int                      g_SwapChainResizeWidth = 0;
static int                      g_SwapChainResizeHeight = 0;

static VkDeviceMemory			g_PcPlotMem = VK_NULL_HANDLE;
static VkImage					g_PcPlot = VK_NULL_HANDLE;
static VkImageView				g_PcPlotView = VK_NULL_HANDLE;
static VkSampler				g_PcPlotSampler = VK_NULL_HANDLE;
static VkDescriptorSet			g_PcPlotImageDescriptorSet = VK_NULL_HANDLE;
static VkRenderPass				g_PcPlotRenderPass = VK_NULL_HANDLE;		//contains the render pass for the pc
static VkDescriptorSetLayout	g_PcPlotDescriptorLayout = VK_NULL_HANDLE;
static VkDescriptorPool			g_PcPlotDescriptorPool = VK_NULL_HANDLE;
static VkDescriptorSet			g_PcPlotDescriptorSet = VK_NULL_HANDLE;
static VkPipelineLayout			g_PcPlotPipelineLayout = VK_NULL_HANDLE;	//contains the pipeline which is used to assign global shader variables
static VkPipeline				g_PcPlotPipeline = VK_NULL_HANDLE;			//contains the graphics pipeline for the pc
static VkFramebuffer			g_PcPlotFramebuffer = VK_NULL_HANDLE;
static VkCommandPool			g_PcPlotCommandPool = VK_NULL_HANDLE;
static VkCommandBuffer			g_PcPlotCommandBuffer = VK_NULL_HANDLE;
static VkBuffer					g_PcPlotVertexBuffer = VK_NULL_HANDLE;
static VkDeviceMemory			g_PcPlotVertexBufferMemory = VK_NULL_HANDLE;
static VkBuffer					g_PcPlotIndexBuffer = VK_NULL_HANDLE;
static VkDeviceMemory			g_PcPlotIndexBufferMemory = VK_NULL_HANDLE;
static uint32_t					g_PcPlotWidth = 100;
static uint32_t					g_PcPlotHeight = 100;
static char						g_fragShaderPath[] = "shader/frag.spv";
static char						g_vertShaderPath[] = "shader/vert.spv";

struct Attribute {
	std::string name;
	float min;			//min value of all values
	float max;			//max value of all values
};

struct Vertex {			//currently holds just the y coordinate. The x computed in the vertex shader via the index
	float y;
};

struct UniformBufferObject {
	float alpha;
	uint32_t amtOfVerts;
	uint32_t order[20];			//IMPORTANT: the length of this array should be the same length it is in the shader
};

static void check_vk_result(VkResult err)
{
	if (err == 0) return;
	printf("VkResult %d\n", err);
	if (err < 0)
		abort();
}

#ifdef IMGUI_VULKAN_DEBUG_REPORT
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_report(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType, uint64_t object, size_t location, int32_t messageCode, const char* pLayerPrefix, const char* pMessage, void* pUserData)
{
	(void)flags; (void)object; (void)location; (void)messageCode; (void)pUserData; (void)pLayerPrefix; // Unused arguments
	fprintf(stderr, "[vulkan] ObjectType: %i\nMessage: %s\n\n", objectType, pMessage);
	return VK_FALSE;
}
#endif // IMGUI_VULKAN_DEBUG_REPORT

static uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memProps;
	vkGetPhysicalDeviceMemoryProperties(g_PhysicalDevice, &memProps);
	uint32_t typeIndex = memProps.memoryTypeCount;
	for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
		if ((typeFilter & (1 << i))&&(memProps.memoryTypes[i].propertyFlags&properties)==properties) {
			return i;
		}
	}
	//safety call to see whther a valid type Index was found
	__debugbreak();
}

static void createPcPlotImageView() {
	VkResult err;

	//creating the VkImage for the PcPlot
	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = static_cast<uint32_t>(g_PcPlotWidth);
	imageInfo.extent.height = static_cast<uint32_t>(g_PcPlotHeight);
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

	err = vkCreateImage(g_Device, &imageInfo, nullptr, &g_PcPlot);
	check_vk_result(err);

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(g_Device, g_PcPlot, &memRequirements);

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, 0);

	err = vkAllocateMemory(g_Device, &allocInfo, nullptr, &g_PcPlotMem);
	check_vk_result(err);

	vkBindImageMemory(g_Device, g_PcPlot, g_PcPlotMem, 0);

	VkImageViewCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	createInfo.image = g_PcPlot;
	createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	createInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
	createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	createInfo.subresourceRange.baseMipLevel = 0;
	createInfo.subresourceRange.levelCount = 1;
	createInfo.subresourceRange.baseArrayLayer = 0;
	createInfo.subresourceRange.layerCount = 1;

	err = vkCreateImageView(g_Device, &createInfo, nullptr, &g_PcPlotView);
	check_vk_result(err);
	//the image view is now nearly ready set up as render target
}

static void cleanupPcPlotImageView() {
	vkDestroyImageView(g_Device, g_PcPlotView, nullptr);
	vkDestroyImage(g_Device, g_PcPlot, nullptr);
	vkFreeMemory(g_Device, g_PcPlotMem, nullptr);
}

static std::vector<char> readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		std::cout << "failed to open file!" << std::endl;
		__debugbreak();
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	return buffer;
}

static VkShaderModule createShaderModule(const std::vector<char>& code) {
	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
	createInfo.codeSize = code.size();

	VkShaderModule shaderModule;
	VkResult err = vkCreateShaderModule(g_Device, &createInfo, nullptr, &shaderModule);
	check_vk_result(err);
	
	return shaderModule;
}

static void createPcPlotPipeline() {
	VkResult err;

	auto vertShader = readFile(g_vertShaderPath);
	auto fragShader = readFile(g_fragShaderPath);

	VkShaderModule vertShaderModule = createShaderModule(vertShader);
	VkShaderModule fragShaderModule = createShaderModule(fragShader);

	VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
	vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertShaderStageInfo.module = vertShaderModule;
	vertShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
	fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragShaderStageInfo.module = fragShaderModule;
	fragShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo,fragShaderStageInfo };

	VkVertexInputBindingDescription bindingDescripiton = {};		//describes how big the vertex data is and how to read the data
	bindingDescripiton.binding = 0;
	bindingDescripiton.stride = sizeof(Vertex);
	bindingDescripiton.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attributeDescription = {};	//describes the attribute of the vertex. If more than 1 attribute is used this has to be an array
	attributeDescription.binding = 0;
	attributeDescription.location = 0;
	attributeDescription.format = VK_FORMAT_R32G32_SFLOAT;
	attributeDescription.offset = offsetof(Vertex, y);

	VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescripiton;
	vertexInputInfo.vertexAttributeDescriptionCount = 1;
	vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	VkViewport viewport = {};					//description for our viewport for transfomation operation after rasterization
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)g_PcPlotWidth;
	viewport.height = (float)g_PcPlotHeight;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor = {};						//description for cutting the rendered result if wanted
	scissor.offset = { 0, 0 };
	scissor.extent = { g_PcPlotWidth,g_PcPlotHeight };

	VkPipelineViewportStateCreateInfo viewportState= {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	VkPipelineRasterizationStateCreateInfo rasterizer = {};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = VK_CULL_MODE_NONE;
	rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;
	rasterizer.depthBiasClamp = 0.0f;
	rasterizer.depthBiasConstantFactor = 0.0f;
	rasterizer.depthBiasSlopeFactor = 0.0f;

	VkPipelineMultisampleStateCreateInfo multisampling = {};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampling.minSampleShading = 1.0f;
	multisampling.pSampleMask = nullptr;
	multisampling.alphaToCoverageEnable = VK_FALSE;
	multisampling.alphaToOneEnable = VK_FALSE;

	VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_TRUE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

	VkPipelineColorBlendStateCreateInfo colorBlending = {};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f;
	colorBlending.blendConstants[1] = 0.0f;
	colorBlending.blendConstants[2] = 0.0f;
	colorBlending.blendConstants[3] = 0.0f;

	VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_LINE_WIDTH };

	VkPipelineDynamicStateCreateInfo dynamicState = {};			//enables change of the linewidth at runtime
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = 1;
	dynamicState.pDynamicStates = dynamicStates;

	VkDescriptorSetLayoutBinding uboLayoutBinding = {};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;

	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 1;
	layoutInfo.pBindings = &uboLayoutBinding;
	
	err = vkCreateDescriptorSetLayout(g_Device, &layoutInfo, nullptr, &g_PcPlotDescriptorLayout);
	check_vk_result(err);

	VkDescriptorPoolSize poolSize = {};
	poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSize.descriptorCount = 1;

	VkDescriptorPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	poolInfo.maxSets = 1;
	
	err = vkCreateDescriptorPool(g_Device, &poolInfo, nullptr, &g_PcPlotDescriptorPool);
	check_vk_result(err);

	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = g_PcPlotDescriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &g_PcPlotDescriptorLayout;

	err = vkAllocateDescriptorSets(g_Device, &allocInfo, &g_PcPlotDescriptorSet);
	check_vk_result(err);

	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &g_PcPlotDescriptorLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 0;
	pipelineLayoutInfo.pPushConstantRanges = nullptr;

	err = vkCreatePipelineLayout(g_Device, &pipelineLayoutInfo, nullptr, &g_PcPlotPipelineLayout);
	check_vk_result(err);

	VkGraphicsPipelineCreateInfo pipelineInfo = {};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shaderStages;
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pDepthStencilState = nullptr;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = g_PcPlotPipelineLayout;
	pipelineInfo.renderPass = g_PcPlotRenderPass;
	pipelineInfo.subpass = 0;

	err = vkCreateGraphicsPipelines(g_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &g_PcPlotPipeline);
	check_vk_result(err);

	vkDestroyShaderModule(g_Device, fragShaderModule, nullptr);
	vkDestroyShaderModule(g_Device, vertShaderModule, nullptr);
}

static void cleanupPcPlotPipeline() {
	vkDestroyDescriptorPool(g_Device, g_PcPlotDescriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(g_Device, g_PcPlotDescriptorLayout, nullptr);
	vkDestroyPipelineLayout(g_Device, g_PcPlotPipelineLayout, nullptr);
	vkDestroyPipeline(g_Device, g_PcPlotPipeline, nullptr);
}

static void createPcPlotRenderPass() {
	VkResult err;

	VkAttachmentDescription colorAttachment = {};
	colorAttachment.format = VK_FORMAT_R8G8B8A8_UNORM;
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference colorAttachmentRef = {};
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;

	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &colorAttachment;
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;

	err = vkCreateRenderPass(g_Device, &renderPassInfo, nullptr, &g_PcPlotRenderPass);
	check_vk_result(err);
}

static void cleanupPcPlotRenderPass() {
	vkDestroyRenderPass(g_Device, g_PcPlotRenderPass, nullptr);
}

static void createPcPlotFramebuffer() {
	VkResult err;

	VkFramebufferCreateInfo framebufferInfo = {};
	framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	framebufferInfo.renderPass = g_PcPlotRenderPass;
	framebufferInfo.attachmentCount = 1;
	framebufferInfo.pAttachments = &g_PcPlotView;
	framebufferInfo.width = g_PcPlotWidth;
	framebufferInfo.height = g_PcPlotHeight;
	framebufferInfo.layers = 1;
	
	err = vkCreateFramebuffer(g_Device, &framebufferInfo, nullptr, &g_PcPlotFramebuffer);
	check_vk_result(err);

}

static void cleanupPcPlotFramebuffer() {
	vkDestroyFramebuffer(g_Device, g_PcPlotFramebuffer, nullptr);
}

static void createPcPlotCommandPool() {
	VkResult err;

	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = g_QueueFamily;

	err = vkCreateCommandPool(g_Device, &poolInfo, nullptr, &g_PcPlotCommandPool);
	check_vk_result(err);
}

static void cleanupPcPlotCommandPool() {
	vkDestroyCommandPool(g_Device, g_PcPlotCommandPool, nullptr);
}

static void createPcPlotVertexBuffer(uint32_t amtOfVertices, uint32_t amtOfAttributes, const std::vector<float*>& data) {
	VkResult err;

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = sizeof(Vertex) * amtOfVertices;
	bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &g_PcPlotVertexBuffer);
	check_vk_result(err);

	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(g_Device, g_PcPlotVertexBuffer, &memRequirements);

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	err = vkAllocateMemory(g_Device, &allocInfo, nullptr, &g_PcPlotVertexBufferMemory);
	check_vk_result(err);
	
	vkBindBufferMemory(g_Device, g_PcPlotVertexBuffer, g_PcPlotVertexBufferMemory, 0);

	VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(g_PcPlotCommandBuffer, 0, 1, &g_PcPlotVertexBuffer, offsets);

	//creating a 1-D array with all the Attributes
	float* d = new float[amtOfVertices * amtOfAttributes];
	uint32_t i = 0;
	for (float* p : data) {
		memcpy(&d[i], p, sizeof(float) * amtOfAttributes);
		i += amtOfAttributes;
	}

	//filling the Vertex Buffer with all Datapoints
	void* mem;
	vkMapMemory(g_Device, g_PcPlotVertexBufferMemory, 0, sizeof(Vertex) * amtOfVertices, 0, &mem);
	memcpy(d, mem, amtOfVertices * amtOfAttributes);
	vkUnmapMemory(g_Device, g_PcPlotVertexBufferMemory);
	
	//creating the index buffer
	//index buffer also contains the uniform bufffer memory
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = sizeof(uint16_t) * amtOfAttributes + sizeof(UniformBufferObject);
	bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &g_PcPlotIndexBuffer);
	check_vk_result(err);

	vkGetBufferMemoryRequirements(g_Device, g_PcPlotIndexBuffer, &memRequirements);

	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	err = vkAllocateMemory(g_Device, &allocInfo, nullptr, &g_PcPlotIndexBufferMemory);
	check_vk_result(err);

	vkBindBufferMemory(g_Device, g_PcPlotVertexBuffer, g_PcPlotVertexBufferMemory, 0);
	vkCmdBindIndexBuffer(g_PcPlotCommandBuffer, g_PcPlotIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

	//specifying the uniform buffer location
	VkDescriptorBufferInfo desBufferInfo = {};
	desBufferInfo.buffer = g_PcPlotIndexBuffer;
	desBufferInfo.offset = sizeof(uint16_t)*amtOfAttributes;
	desBufferInfo.range = sizeof(UniformBufferObject);

	VkWriteDescriptorSet descriptorWrite = {};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = g_PcPlotDescriptorSet;
	descriptorWrite.dstBinding = 0;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pBufferInfo = &desBufferInfo;

	vkUpdateDescriptorSets(g_Device, 1, &descriptorWrite, 0, nullptr);
}

static void cleanupPcPlotVertexBuffer() {
	if(g_PcPlotVertexBuffer)
		vkDestroyBuffer(g_Device, g_PcPlotVertexBuffer, nullptr);
	if (g_PcPlotVertexBufferMemory)
		vkFreeMemory(g_Device, g_PcPlotVertexBufferMemory, nullptr);
	if (g_PcPlotIndexBuffer)
		vkDestroyBuffer(g_Device, g_PcPlotVertexBuffer, nullptr);
	if (g_PcPlotIndexBufferMemory)
		vkFreeMemory(g_Device, g_PcPlotVertexBufferMemory, nullptr);
}

static void createPcPlotCommandBuffer() {
	VkResult err;

	VkCommandBufferAllocateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	bufferInfo.commandPool = g_PcPlotCommandPool;
	bufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	bufferInfo.commandBufferCount = 1;

	err = vkAllocateCommandBuffers(g_Device, &bufferInfo, &g_PcPlotCommandBuffer);
	check_vk_result(err);

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
	beginInfo.pInheritanceInfo = nullptr;

	err = vkBeginCommandBuffer(g_PcPlotCommandBuffer, &beginInfo);
	check_vk_result(err);

	VkRenderPassBeginInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.renderPass = g_PcPlotRenderPass;
	renderPassInfo.framebuffer = g_PcPlotFramebuffer;
	renderPassInfo.renderArea.offset = { 0,0 };
	renderPassInfo.renderArea.extent = { g_PcPlotWidth,g_PcPlotHeight };

	VkClearValue clearColor = { 0.0f,0.0f,0.0f,1.0f };
	
	renderPassInfo.clearValueCount = 1;
	renderPassInfo.pClearValues = &clearColor;

	vkCmdBeginRenderPass(g_PcPlotCommandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotPipeline);
}

static void cleanupPcPlotCommandBuffer() {
	VkResult err;
	vkCmdEndRenderPass(g_PcPlotCommandBuffer);
	
	err = vkEndCommandBuffer(g_PcPlotCommandBuffer);
	check_vk_result(err);

	vkFreeCommandBuffers(g_Device, g_PcPlotCommandPool, 1, &g_PcPlotCommandBuffer);
}

static void drawPcPlot(const std::vector<Attribute>& attributes, const std::vector<int>& attributeOrder, const bool* attributeEnabled, const std::vector<float*>& data, float alpha) {
	VkResult err;

	//drawing via copying the indeces into the index buffer
	//the indeces have to have just the right ordering for the vertices
	int amtOfIndeces = 0;
	for (int i = 0; i < attributes.size(); i++) {
		if (attributeEnabled[i])
			amtOfIndeces++;
	}

	void* d;
	vkMapMemory(g_Device, g_PcPlotIndexBufferMemory, 0, sizeof(uint16_t) * attributes.size() + sizeof(UniformBufferObject), 0, &d);

	//filling the indexbuffer with the used indeces
	uint16_t* ind = new uint16_t[amtOfIndeces];			//contains all indeces to copy
	int j = 0;											//current index in the ind array
	for (int i = 0; i < attributes.size(); i++) {
		if (attributeEnabled[i]) {
			ind[j++] = attributeOrder[i];
		}
	}

#ifdef _DEBUG
	if (j != amtOfIndeces)
		__debugbreak();
#endif

	//copying the indexbuffer
	memcpy(d, ind, amtOfIndeces * sizeof(uint16_t));
	

	//filling the uniform buffer and copying it into the end of the indexbuffer
	UniformBufferObject ubo;
	ubo.alpha = alpha;
	ubo.amtOfVerts = amtOfIndeces;
	int c = 0;
	for (int i : attributeOrder) {
		ubo.order[i] = c;
		if (attributeEnabled[i])
			c++;
	}

#ifdef _DEBUG
	if (c != amtOfIndeces)
		__debugbreak();
#endif

	//copying the uniform buffer to the end of the indexbuffer
	d = (uint16_t*)d + attributes.size();
	memcpy(d, &ubo, sizeof(UniformBufferObject));

	vkUnmapMemory(g_Device, g_PcPlotIndexBufferMemory);

	//binding the uniformbuffer
	vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotPipelineLayout, 0, 1, &g_PcPlotDescriptorSet, 0, nullptr);

	//ready to draw with draw indexed
	uint32_t vertOffset = 0;
	for (int i = 0; i < data.size(); i++) {
		vkCmdDrawIndexed(g_PcPlotCommandBuffer, amtOfIndeces, 1, 0, vertOffset, 0);
		vertOffset += attributes.size();
	}

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.signalSemaphoreCount = 0;
	submitInfo.waitSemaphoreCount = 0;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &g_PcPlotCommandBuffer;

	err = vkQueueSubmit(g_Queue, 1, &submitInfo, VK_NULL_HANDLE);
	check_vk_result(err);
}

static void SetupVulkan(const char** extensions, uint32_t extensions_count)
{
	VkResult err;

	// Create Vulkan Instance
	{
		VkInstanceCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		create_info.enabledExtensionCount = extensions_count;
		create_info.ppEnabledExtensionNames = extensions;

#ifdef IMGUI_VULKAN_DEBUG_REPORT
		// Enabling multiple validation layers grouped as LunarG standard validation
		const char* layers[] = { "VK_LAYER_LUNARG_standard_validation" };
		create_info.enabledLayerCount = 1;
		create_info.ppEnabledLayerNames = layers;

		// Enable debug report extension (we need additional storage, so we duplicate the user array to add our new extension to it)
		const char** extensions_ext = (const char**)malloc(sizeof(const char*) * (extensions_count + 1));
		memcpy(extensions_ext, extensions, extensions_count * sizeof(const char*));
		extensions_ext[extensions_count] = "VK_EXT_debug_report";
		create_info.enabledExtensionCount = extensions_count + 1;
		create_info.ppEnabledExtensionNames = extensions_ext;

		// Create Vulkan Instance
		err = vkCreateInstance(&create_info, g_Allocator, &g_Instance);
		check_vk_result(err);
		free(extensions_ext);

		// Get the function pointer (required for any extensions)
		auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(g_Instance, "vkCreateDebugReportCallbackEXT");
		IM_ASSERT(vkCreateDebugReportCallbackEXT != NULL);

		// Setup the debug report callback
		VkDebugReportCallbackCreateInfoEXT debug_report_ci = {};
		debug_report_ci.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
		debug_report_ci.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
		debug_report_ci.pfnCallback = debug_report;
		debug_report_ci.pUserData = NULL;
		err = vkCreateDebugReportCallbackEXT(g_Instance, &debug_report_ci, g_Allocator, &g_DebugReport);
		check_vk_result(err);
#else
		// Create Vulkan Instance without any debug feature
		err = vkCreateInstance(&create_info, g_Allocator, &g_Instance);
		check_vk_result(err);
		IM_UNUSED(g_DebugReport);
#endif
	}

	// Select GPU
	{
		uint32_t gpu_count;
		err = vkEnumeratePhysicalDevices(g_Instance, &gpu_count, NULL);
		check_vk_result(err);
		IM_ASSERT(gpu_count > 0);

		VkPhysicalDevice* gpus = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * gpu_count);
		err = vkEnumeratePhysicalDevices(g_Instance, &gpu_count, gpus);
		check_vk_result(err);

		// If a number >1 of GPUs got reported, you should find the best fit GPU for your purpose
		// e.g. VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU if available, or with the greatest memory available, etc.
		// for sake of simplicity we'll just take the first one, assuming it has a graphics queue family.
		g_PhysicalDevice = gpus[0];
		free(gpus);
	}

	// Select graphics queue family
	{
		uint32_t count;
		vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count, NULL);
		VkQueueFamilyProperties* queues = (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties) * count);
		vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count, queues);
		for (uint32_t i = 0; i < count; i++)
			if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				g_QueueFamily = i;
				break;
			}
		free(queues);
		IM_ASSERT(g_QueueFamily != (uint32_t)-1);
	}

	// Create Logical Device (with 1 queue)
	{
		int device_extension_count = 1;
		const char* device_extensions[] = { "VK_KHR_swapchain" };
		const float queue_priority[] = { 1.0f };
		VkDeviceQueueCreateInfo queue_info[1] = {};
		queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queue_info[0].queueFamilyIndex = g_QueueFamily;
		queue_info[0].queueCount = 1;
		queue_info[0].pQueuePriorities = queue_priority;
		VkDeviceCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		create_info.queueCreateInfoCount = sizeof(queue_info) / sizeof(queue_info[0]);
		create_info.pQueueCreateInfos = queue_info;
		create_info.enabledExtensionCount = device_extension_count;
		create_info.ppEnabledExtensionNames = device_extensions;
		err = vkCreateDevice(g_PhysicalDevice, &create_info, g_Allocator, &g_Device);
		check_vk_result(err);
		vkGetDeviceQueue(g_Device, g_QueueFamily, 0, &g_Queue);
	}

	// Create Descriptor Pool
	{
		VkDescriptorPoolSize pool_sizes[] =
		{
			{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
			{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
			{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
		};
		VkDescriptorPoolCreateInfo pool_info = {};
		pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
		pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
		pool_info.pPoolSizes = pool_sizes;
		err = vkCreateDescriptorPool(g_Device, &pool_info, g_Allocator, &g_DescriptorPool);
		check_vk_result(err);
	}
}

// All the ImGui_ImplVulkanH_XXX structures/functions are optional helpers used by the demo. 
// Your real engine/app may not use them.
static void SetupVulkanWindow(ImGui_ImplVulkanH_Window * wd, VkSurfaceKHR surface, int width, int height)
{
	wd->Surface = surface;

	// Check for WSI support
	VkBool32 res;
	vkGetPhysicalDeviceSurfaceSupportKHR(g_PhysicalDevice, g_QueueFamily, wd->Surface, &res);
	if (res != VK_TRUE)
	{
		fprintf(stderr, "Error no WSI support on physical device 0\n");
		exit(-1);
	}

	// Select Surface Format
	const VkFormat requestSurfaceImageFormat[] = { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM };
	const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
	wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(g_PhysicalDevice, wd->Surface, requestSurfaceImageFormat, (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat), requestSurfaceColorSpace);

	// Select Present Mode
#ifdef IMGUI_UNLIMITED_FRAME_RATE
	VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR };
#else
	VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_FIFO_KHR };
#endif
	wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(g_PhysicalDevice, wd->Surface, &present_modes[0], IM_ARRAYSIZE(present_modes));
	//printf("[vulkan] Selected PresentMode = %d\n", wd->PresentMode);

	// Create SwapChain, RenderPass, Framebuffer, etc.
	IM_ASSERT(g_MinImageCount >= 2);
	ImGui_ImplVulkanH_CreateWindow(g_Instance, g_PhysicalDevice, g_Device, wd, g_QueueFamily, g_Allocator, width, height, g_MinImageCount);
}

static void CleanupVulkan()
{
	vkDestroyDescriptorPool(g_Device, g_DescriptorPool, g_Allocator);

#ifdef IMGUI_VULKAN_DEBUG_REPORT
	// Remove the debug report callback
	auto vkDestroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(g_Instance, "vkDestroyDebugReportCallbackEXT");
	vkDestroyDebugReportCallbackEXT(g_Instance, g_DebugReport, g_Allocator);
#endif // IMGUI_VULKAN_DEBUG_REPORT

	vkDestroyDevice(g_Device, g_Allocator);
	vkDestroyInstance(g_Instance, g_Allocator);
}

static void CleanupVulkanWindow()
{
	ImGui_ImplVulkanH_DestroyWindow(g_Instance, g_Device, &g_MainWindowData, g_Allocator);
}

static void FrameRender(ImGui_ImplVulkanH_Window * wd)
{
	VkResult err;

	VkSemaphore image_acquired_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
	VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
	err = vkAcquireNextImageKHR(g_Device, wd->Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &wd->FrameIndex);
	check_vk_result(err);

	ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
	{
		err = vkWaitForFences(g_Device, 1, &fd->Fence, VK_TRUE, UINT64_MAX);    // wait indefinitely instead of periodically checking
		check_vk_result(err);

		err = vkResetFences(g_Device, 1, &fd->Fence);
		check_vk_result(err);
	}
	{
		err = vkResetCommandPool(g_Device, fd->CommandPool, 0);
		check_vk_result(err);
		VkCommandBufferBeginInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		err = vkBeginCommandBuffer(fd->CommandBuffer, &info);
		check_vk_result(err);
	}
	{
		VkRenderPassBeginInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		info.renderPass = wd->RenderPass;
		info.framebuffer = fd->Framebuffer;
		info.renderArea.extent.width = wd->Width;
		info.renderArea.extent.height = wd->Height;
		info.clearValueCount = 1;
		info.pClearValues = &wd->ClearValue;
		vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	// Record Imgui Draw Data and draw funcs into command buffer
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), fd->CommandBuffer);

	// Submit command buffer
	vkCmdEndRenderPass(fd->CommandBuffer);
	{
		VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSubmitInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		info.waitSemaphoreCount = 1;
		info.pWaitSemaphores = &image_acquired_semaphore;
		info.pWaitDstStageMask = &wait_stage;
		info.commandBufferCount = 1;
		info.pCommandBuffers = &fd->CommandBuffer;
		info.signalSemaphoreCount = 1;
		info.pSignalSemaphores = &render_complete_semaphore;

		err = vkEndCommandBuffer(fd->CommandBuffer);
		check_vk_result(err);
		err = vkQueueSubmit(g_Queue, 1, &info, fd->Fence);
		check_vk_result(err);
	}
}

static void FramePresent(ImGui_ImplVulkanH_Window * wd)
{
	VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
	VkPresentInfoKHR info = {};
	info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	info.waitSemaphoreCount = 1;
	info.pWaitSemaphores = &render_complete_semaphore;
	info.swapchainCount = 1;
	info.pSwapchains = &wd->Swapchain;
	info.pImageIndices = &wd->FrameIndex;
	VkResult err = vkQueuePresentKHR(g_Queue, &info);
	check_vk_result(err);
	wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->ImageCount; // Now we can use the next set of semaphores
}

static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

static void glfw_resize_callback(GLFWwindow*, int w, int h)
{
	g_SwapChainRebuild = true;
	g_SwapChainResizeWidth = w;
	g_SwapChainResizeHeight = h;
}

int main(int, char**)
{
	//Section for variables
	float pcLinesAlpha = 1.0f;
	char pcFilePath[100] = {};

	
	bool* pcAttributeEnabled = NULL;										//Contains whether a specific attribute is enabled
	
	std::vector<Attribute> pcAttributes = std::vector<Attribute>();			//Contains the attributes and its bounds
	
	std::vector<int> pcAttrOrd = std::vector<int>();						//Contains the ordering of the attributes

	//Contains all data
	std::vector<float*> pcData = std::vector<float*>();

	// Setup GLFW window
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		return 1;

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	GLFWwindow* window = glfwCreateWindow(1280, 720, "Parallel Coordinates Viewer", NULL, NULL);

	// Setup Vulkan
	if (!glfwVulkanSupported())
	{
		printf("GLFW: Vulkan Not Supported\n");
		return 1;
	}
	uint32_t extensions_count = 0;
	const char** extensions = glfwGetRequiredInstanceExtensions(&extensions_count);
	SetupVulkan(extensions, extensions_count);

	// Create Window Surface
	VkSurfaceKHR surface;
	VkResult err = glfwCreateWindowSurface(g_Instance, window, g_Allocator, &surface);
	check_vk_result(err);

	// Create Framebuffers
	int w, h;
	glfwGetFramebufferSize(window, &w, &h);
	glfwSetFramebufferSizeCallback(window, glfw_resize_callback);
	ImGui_ImplVulkanH_Window* wd = &g_MainWindowData;
	SetupVulkanWindow(wd, surface, w, h);

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// Setup Platform/Renderer bindings
	ImGui_ImplGlfw_InitForVulkan(window, true);
	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = g_Instance;
	init_info.PhysicalDevice = g_PhysicalDevice;
	init_info.Device = g_Device;
	init_info.QueueFamily = g_QueueFamily;
	init_info.Queue = g_Queue;
	init_info.PipelineCache = g_PipelineCache;
	init_info.DescriptorPool = g_DescriptorPool;
	init_info.Allocator = g_Allocator;
	init_info.MinImageCount = g_MinImageCount;
	init_info.ImageCount = wd->ImageCount;
	init_info.CheckVkResultFn = check_vk_result;
	ImGui_ImplVulkan_Init(&init_info, wd->RenderPass);

	// Load Fonts
	// - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
	// - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
	// - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
	// - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
	// - Read 'misc/fonts/README.txt' for more instructions and details.
	// - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
	//io.Fonts->AddFontDefault();
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
	//ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
	//IM_ASSERT(font != NULL);

	// Upload Fonts
	{
		// Use any command queue
		VkCommandPool command_pool = wd->Frames[wd->FrameIndex].CommandPool;
		VkCommandBuffer command_buffer = wd->Frames[wd->FrameIndex].CommandBuffer;

		err = vkResetCommandPool(g_Device, command_pool, 0);
		check_vk_result(err);
		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		err = vkBeginCommandBuffer(command_buffer, &begin_info);
		check_vk_result(err);

		ImGui_ImplVulkan_CreateFontsTexture(command_buffer, g_Device, g_DescriptorPool);

		VkSubmitInfo end_info = {};
		end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		end_info.commandBufferCount = 1;
		end_info.pCommandBuffers = &command_buffer;
		err = vkEndCommandBuffer(command_buffer);
		check_vk_result(err);
		err = vkQueueSubmit(g_Queue, 1, &end_info, VK_NULL_HANDLE);
		check_vk_result(err);

		err = vkDeviceWaitIdle(g_Device);
		check_vk_result(err);
		ImGui_ImplVulkan_DestroyFontUploadObjects();
	}

	{//Section to initialize the pcPlot graphics queue
		createPcPlotImageView();
		createPcPlotRenderPass();
		createPcPlotPipeline();
		createPcPlotFramebuffer();
		createPcPlotCommandPool();
		createPcPlotCommandBuffer();

		//before being able to add the image to imgui the sampler has to be created
		VkSamplerCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		info.magFilter = VK_FILTER_LINEAR;
		info.minFilter = VK_FILTER_LINEAR;
		info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		info.minLod = -1000;
		info.maxLod = 1000;
		info.maxAnisotropy = 1.0f;
		err = vkCreateSampler(g_Device, &info, nullptr, &g_PcPlotSampler);
		check_vk_result(err);

		g_PcPlotImageDescriptorSet =(VkDescriptorSet) ImGui_ImplVulkan_AddTexture(g_PcPlotSampler, g_PcPlotView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, g_Device, g_DescriptorPool);
	}

	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	// Main loop
	while (!glfwWindowShouldClose(window))
	{
		// Poll and handle events (inputs, window resize, etc.)
		// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
		// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
		// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
		// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
		glfwPollEvents();

		if (g_SwapChainRebuild)
		{
			g_SwapChainRebuild = false;
			ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
			ImGui_ImplVulkanH_CreateWindow(g_Instance, g_PhysicalDevice, g_Device, &g_MainWindowData, g_QueueFamily, g_Allocator, g_SwapChainResizeWidth, g_SwapChainResizeHeight, g_MinImageCount);
			g_MainWindowData.FrameIndex = 0;
		}

		// Start the Dear ImGui frame
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();


		//draw the picture of the plotted pc coordinates
		if (ImGui::Begin("Plot", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
		{
			ImTextureID my_tex_id = (ImTextureID)g_PcPlotImageDescriptorSet;
			float my_tex_w = (float)io.Fonts->TexWidth;
			float my_tex_h = (float)io.Fonts->TexHeight;

			ImVec2 pos = ImGui::GetCursorScreenPos();

			ImGui::Image(my_tex_id, ImVec2(my_tex_w, my_tex_h), ImVec2(0, 0), ImVec2(1, 1), ImColor(255, 255, 255, 255), ImColor(255, 255, 255, 128));
			if (ImGui::Button("Render")) {

			}
		}
		ImGui::End();

		// Labels for the titels of the attributes
		// Position calculation for each of the Label
		size_t amtOfLabels = 0;
		for (int i = 0; i < pcAttributes.size(); i++)
			if (pcAttributeEnabled[i])
				amtOfLabels++;

		size_t paddingSide = 50;			//padding from left and right screen border
		size_t paddingTop = 50;				//padding from the top of the screen
		size_t gap = (io.DisplaySize.x - 2 * paddingSide) / (amtOfLabels - 1);

		//creating one Label for each Attribute
		ImVec2 window_pos = ImVec2(paddingSide, paddingTop);
		for (auto i:pcAttrOrd) {
			//not creating a label for unused Attributes
			if (!pcAttributeEnabled[i])
				continue;

			std::string a = pcAttributes[i].name;

			ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, ImVec2(.5f,0));
			ImGui::SetNextWindowBgAlpha(0.3f); // Transparent background

			if (ImGui::Begin(a.c_str(), NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
			{
				ImGui::Text(a.c_str());
			}
			ImGui::End();

			//setting up the position for the next label
			window_pos.x += gap;
		}

		//Settings section
		window_pos = ImVec2(100, 500);
		ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
		if (ImGui::Begin("Example: Simple overlay", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing))
		{
			ImGui::Text("Settings");
			ImGui::SliderFloat("Alpha value", &pcLinesAlpha, 0.0f, 1.0f);

			for (int i = 0; i < pcAttributes.size(); i++) {
				ImGui::Checkbox(pcAttributes[i].name.c_str(), &pcAttributeEnabled[i]);
			}

			ImGui::InputText("Directory Path", pcFilePath, 100);
			ImGui::SameLine();

			//Opening a new Dataset into the Viewer
			if (ImGui::Button("Open")) {
				//resetting all vectors
				pcAttributes.clear();
				for (auto& d : pcData)
					if (d)
						delete[] d;
				pcData.clear();
				if(pcAttributeEnabled)
					delete[] pcAttributeEnabled;
				pcAttrOrd.clear();

				std::ifstream input(pcFilePath);

				bool firstLine = true;

				for (std::string line; std::getline(input, line); )
				{
					std::string delimiter = ",";
					size_t pos = 0;
					std::string cur;

					//parsing the attributes in the first line
					if (firstLine) {
						while ((pos = line.find(delimiter)) != std::string::npos) {
							cur = line.substr(0, pos);
							line.erase(0, pos + delimiter.length());
							pcAttributes.push_back({ cur,std::numeric_limits<float>::max(),std::numeric_limits<float>::min() });
						}
						//adding the last item which wasn't recognized
						pcAttributes.push_back({ line,std::numeric_limits<float>::max(),std::numeric_limits<float>::min() });
						//setting up the boolarray and setting all the attributes to true
						pcAttributeEnabled =new bool[pcAttributes.size()];
						for (int i = 0; i < pcAttributes.size(); i++) {
							pcAttributeEnabled[i] = true;
							pcAttrOrd.push_back(i);
						}

						firstLine = false;
					}

					//parsing the data which follows the attribute declaration
					else {
						pcData.push_back(new float[pcAttributes.size()]);
						size_t attr = 0;
						float curF = 0;
						while ((pos = line.find(delimiter)) != std::string::npos) {
							cur = line.substr(0, pos);
							line.erase(0, pos + delimiter.length());
							
							//checking for an overrunning attribute counter
							if (attr == pcAttributes.size())
								__debugbreak();

							curF = std::stof(cur);

							//updating the bounds if a new highest value was found in the current data.
							if (curF > pcAttributes[attr].max)
								pcAttributes[attr].max = curF;
							if (curF < pcAttributes[attr].min)
								pcAttributes[attr].min = curF;

							pcData.back()[attr++] = curF;
						}
						if (attr == pcAttributes.size())
							__debugbreak();
						
						//adding the last item which wasn't recognized
						curF = std::stof(line);

						//updating the bounds if a new highest value was found in the current data.
						if (curF > pcAttributes[attr].max)
							pcAttributes[attr].max = curF;
						if (curF < pcAttributes[attr].min)
							pcAttributes[attr].min = curF;
						pcData.back()[attr] = curF;
					}
				}

				//vertexBufferRecreation
				cleanupPcPlotVertexBuffer();
				createPcPlotVertexBuffer(pcAttributes.size()*pcData.size(),pcAttributes.size(),pcData);

				//printing out the loaded attributes for debug reasons
				std::cout << "Attributes: " << std::endl;
				for (auto attribute : pcAttributes) {
					std::cout << attribute.name << ", MinVal: " << attribute.min << ", MaxVal: " << attribute.max << std::endl;
				}

				int dc = 0;
				std::cout << std::endl << "Data:" << std::endl;
				for (auto d : pcData) {
					for (int i = 0; i < pcAttributes.size(); i++) {
						std::cout << d[i] << " , ";
					}
					std::cout << std::endl;
					if (dc++ > 10)
						break;
				}
			}
			
		}
		ImGui::End();

		ImGui::ShowDemoWindow(NULL);

		// Rendering
		ImGui::Render();
		memcpy(&wd->ClearValue.color.float32[0], &clear_color, 4 * sizeof(float));
		FrameRender(wd);

		FramePresent(wd);
	}

	// Cleanup
	if (pcAttributeEnabled)
		delete[] pcAttributeEnabled;
	for (auto& d : pcData) {
		if(d)
			delete[] d;
	}

	err = vkDeviceWaitIdle(g_Device);
	check_vk_result(err);

	{//section to cleanup pcPlot
		vkDestroySampler(g_Device, g_PcPlotSampler, nullptr);
		cleanupPcPlotCommandBuffer();
		cleanupPcPlotCommandPool();
		cleanupPcPlotFramebuffer();
		cleanupPcPlotPipeline();
		cleanupPcPlotRenderPass();
		cleanupPcPlotImageView();
		cleanupPcPlotVertexBuffer();
	}

	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	CleanupVulkanWindow();
	CleanupVulkan();

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}

/*
using namespace std;

int main()
{
	cout << "Hello CMake." << endl;
	return 0;
}
*/