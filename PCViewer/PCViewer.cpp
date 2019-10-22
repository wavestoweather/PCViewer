/*
This program is written and maintained by Josef Stumpfegger(josefstumpfegger@outlook.de).
For everyone who wants to read and understand this code. I'm sorry. 
This code was written under time pressure and was not intended to be published in the first place.
Well if you still want to find out how this precious application works, go ahead.
Should you find errors, problems, speed up ideas or anything else, dont be shy and contact me!
Other than that, i wish you a beautiful day and a lot of fun with this program.
*/

//memory leak detection
#ifdef _DEBUG
#define DETECTMEMLEAK
#endif

//uncomment this to build the pcViewer with 3d view
//#define RENDER3D
//uncomment this to build hte pcViewer with the node viewer
//#define NODEVIEW

#ifdef DETECTMEMLEAK
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#pragma once
#include "PCViewer.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_vulkan.h"
#include "Color.h"
#include "VkUtil.h"
#include "PCUtil.h"
#include "NodeViewer.h"
#include "View3d.h"
#include "SpacialData.h"
#include "SettingsManager.h"

#include <stdio.h>          // printf, fprintf
#include <stdlib.h>         // abort
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <vector>
#include <limits>
#include <list>
#include <algorithm>
#include <functional>
#include <chrono>
#include <random>
#include <map>
#include <set>

#ifdef DETECTMEMLEAK
#define new new( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#endif

//defines for the medians
#define MEDIANCOUNT 3
#define MEDIAN 0
#define ARITHMEDIAN 1
#define GOEMEDIAN 2

#define BRUSHWIDTH 20
#define EDGEHOVERDIST 5
#define DRAGTHRESH .02f

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

//#define IMGUI_UNLIMITED_FRAME_RATE
//enable this define to print the time needed to render the pc Plot
#define PRINTRENDERTIME
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

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b) {
	assert(a.size() == b.size());
	
	std::vector<T> result;
	std::transform(a.begin(), a.end(), b.begin(), std::back_inserter(result), std::plus<T>());
	return result;
}

template <typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b) {
	assert(a.size() == b.size());

	std::vector<T> result;
	std::transform(a.begin(), a.end(), b.begin(), std::back_inserter(result), std::minus<T>());
	return result;
}

template <typename T>
std::vector<T> operator/(const std::vector<T>& a, const T b) {
	std::vector<T> result(a);
	for (int i = 0; i < result.size(); i++) {
		result[i] /= b;
	}
	return result;
}

template <typename T>
T squareDist(const std::vector<T>& a, const std::vector<T>& b) {
	assert(a.size() == b.size());

	float result = 0;
	float c;
	for (int i = 0; i < a.size(); i++) {
		c = a[i] - b[i];
		result += std::powf(c, 2);
	}
	return result;
}

template <typename T>
T eucDist(const std::vector<T>& a) {
	float result = 0;
	for (int i = 0; i < a.size(); i++) {
		result += std::powf(a[i], 2);
	}
	return std::sqrt(result);
}

struct Vec4 {
	float x;
	float y;
	float z;
	float w;

	bool operator==(Vec4& other) {
		return this->x == other.x && this->y == other.y && this->z == other.z && this->w == other.w;
	}

	bool operator!=(Vec4& other) {
		return !(*this == other);
	}
};

struct Vec2 {
	float u;
	float v;
};

struct Vertex {			//currently holds just the y coordinate. The x computed in the vertex shader via the index
	float y;
};

struct RectVertex {		//struct which describes the vertecies for the rects
	Vec4 pos;
	Vec4 col;
};

struct UniformBufferObject {
	float alpha;
	uint32_t amtOfVerts;
	uint32_t amtOfAttributes;
	float padding;
	Vec4 color;
	Vec4 VertexTransormations[50];			//IMPORTANT: the length of this array should be the same length it is in the shader. To be the same length, due to padding this array has to be 4 times the length and just evvery 4th entry is used
};

//uniform Buffer for the histogramms
struct HistogramUniformBuffer {
	float x;
	float width;
	float maxVal;
	float minVal;
	Vec4 color;
};

struct DensityUniformBuffer {
	bool enableMapping;
	float gaussRange;
	uint32_t imageHeight;
	uint32_t padding;
};

struct Brush {
	int id;
	std::pair<float, float> minMax;
};

struct GlobalBrush {
	bool active;										//global brushes can be activated and deactivated
	std::string name;									//the name of a global brush describes the template list it was created from and more...
	std::map<std::string, float> lineRatios;			//contains the ratio of still active lines per drawlist
	std::map<int, std::pair<unsigned int,std::pair<float, float>>> brushes;	//for every brush that exists, one entry in this map exists, where the key is the index of the Attribute in the pcAttributes vector and the pair describes the minMax values
};

struct TemplateBrush {
	std::string name;									//identifier for the template brush
	std::map<int,std::pair<float, float>> brushes;
};

struct DrawList {
	std::string name;
	std::string parentDataSet;
	Vec4 color;
	Vec4 prefColor;
	bool show;
	bool showHistogramm;
	VkBuffer buffer;
	VkBuffer indexBuffer;
	uint32_t indexBufferOffset;
	VkBuffer ubo;
	VkBuffer histogramIndBuffer;
	std::vector<VkBuffer> histogramUbos;
	VkBuffer medianBuffer;
	VkBuffer medianUbo;
	int medianUboOffset;
	VkDescriptorSet medianUboDescSet;
	uint32_t medianBufferOffset;
	Vec4 medianColor;
	int activeMedian;
	std::vector<uint32_t> histogramUbosOffsets;
	std::vector<VkDescriptorSet> histogrammDescSets;
	VkDeviceMemory dlMem;
	VkDescriptorSet uboDescSet;
	std::vector<int> indices;
	std::vector<int> activeInd;
	uint32_t histIndexBufferOffset;
	std::vector<std::vector<Brush>> brushes;		//the pair contains first min and then max for the brush
};

struct TemplateList {
	std::string name;
	VkBuffer buffer;
	std::vector<int> indices;
	std::vector<std::pair<float, float>> minMax;
};

struct Buffer {
	VkBuffer buffer;
	VkBuffer uboBuffer;
	VkDeviceMemory memory;

	bool operator==(const Buffer& other) {
		return this->buffer == other.buffer && this->memory == other.memory;
	}
};

struct DataSet {
	std::string name;
	Buffer buffer;
	std::vector<float*> data;
	std::list<TemplateList> drawLists;
	bool oneData = false;			//if is set to true, all data in data is in one continous float* array. -> on deletion only delete[] the float* of data[0]

	bool operator==(const DataSet& other) {
		return this->name == other.name;
	}
};


static VkDeviceMemory			g_PcPlotMem = VK_NULL_HANDLE;
static VkImage					g_PcPlot = VK_NULL_HANDLE;
static VkImageView				g_PcPlotView = VK_NULL_HANDLE;
static VkSampler				g_PcPlotSampler = VK_NULL_HANDLE;
static VkDescriptorSet			g_PcPlotImageDescriptorSet = VK_NULL_HANDLE;
static VkRenderPass				g_PcPlotRenderPass = VK_NULL_HANDLE;		//contains the render pass for the pc
static VkDescriptorSetLayout	g_PcPlotDescriptorLayout = VK_NULL_HANDLE;
static VkDescriptorPool			g_PcPlotDescriptorPool = VK_NULL_HANDLE;
static VkDescriptorSet			g_PcPlotDescriptorSet = VK_NULL_HANDLE;
static VkBuffer					g_PcPlotDescriptorBuffer = VK_NULL_HANDLE;
static VkDeviceMemory			g_PcPlotDescriptorBufferMemory = VK_NULL_HANDLE;
static VkPipelineLayout			g_PcPlotPipelineLayout = VK_NULL_HANDLE;	//contains the pipeline which is used to assign global shader variables
static VkPipeline				g_PcPlotPipeline = VK_NULL_HANDLE;			//contains the graphics pipeline for the pc
//variables for spline pipeline
static VkPipelineLayout			g_PcPlotSplinePipelineLayout = VK_NULL_HANDLE;
static VkPipeline				g_PcPlotSplinePipeline = VK_NULL_HANDLE;
static bool						g_RenderSplines = false;
//variables for the histogramm pipeline
static VkPipelineLayout			g_PcPlotHistoPipelineLayout = VK_NULL_HANDLE;
static VkPipeline				g_PcPlotHistoPipeline = VK_NULL_HANDLE;
static VkRenderPass				g_PcPlotHistoRenderPass = VK_NULL_HANDLE;
static VkDescriptorSetLayout	g_PcPlotHistoDescriptorSetLayout = VK_NULL_HANDLE;
static VkPipelineLayout			g_PcPlotRectPipelineLayout = VK_NULL_HANDLE;
static VkPipeline				g_PcPlotRectPipeline = VK_NULL_HANDLE;

//variables for the density pipeline
static VkPipelineLayout			g_PcPlotDensityPipelineLayout = VK_NULL_HANDLE;
static VkPipeline				g_PcPlotDensityPipeline = VK_NULL_HANDLE;
static VkDescriptorSetLayout	g_PcPlotDensityDescriptorSetLayout = VK_NULL_HANDLE;
static VkImage					g_PcPlotDensityImageCopy = VK_NULL_HANDLE;
static VkImageView				g_PcPlotDensityImageView = VK_NULL_HANDLE;
static VkSampler				g_PcPlotDensityImageSampler = VK_NULL_HANDLE;
static VkImage					g_PcPlotDensityIronMap = VK_NULL_HANDLE;
static uint32_t					g_PcPlotDensityIronMapOffset = 0;
static VkImageView				g_PcPLotDensityIronMapView = VK_NULL_HANDLE;
static VkSampler				g_PcPlotDensityIronMapSampler = VK_NULL_HANDLE;
static VkDescriptorSet			g_PcPlotDensityDescriptorSet = VK_NULL_HANDLE;
static VkBuffer					g_PcPlotDensityRectBuffer = VK_NULL_HANDLE;
static uint32_t					g_PcPlotDensityRectBufferOffset = 0;
static VkBuffer					g_PcPlotDensityUbo = VK_NULL_HANDLE;
static uint32_t					g_PcPLotDensityUboOffset = 0;
static VkRenderPass				g_PcPlotDensityRenderPass = VK_NULL_HANDLE;
static VkFramebuffer			g_PcPlotDensityFrameBuffer = VK_NULL_HANDLE;

static VkFramebuffer			g_PcPlotFramebuffer = VK_NULL_HANDLE;
static VkCommandPool			g_PcPlotCommandPool = VK_NULL_HANDLE;
static VkCommandBuffer			g_PcPlotCommandBuffer = VK_NULL_HANDLE;
static std::list<Buffer>		g_PcPlotVertexBuffers;
static std::list<DataSet>		g_PcPlotDataSets;
static std::list<DrawList>		g_PcPlotDrawLists;
static VkBuffer					g_PcPlotHistogrammRect = VK_NULL_HANDLE;
static uint32_t					g_PcPlotHistogrammRectOffset = 0;
static VkBuffer					g_PcPlotHistogrammIndex = VK_NULL_HANDLE;
static uint32_t					g_PcPlotHistogrammIndexOffset = 0;

//Indexbuffermemory also contaings the histogramm rect buffer and histogramm index buffer
static VkBuffer					g_PcPlotIndexBuffer = VK_NULL_HANDLE;
static VkDeviceMemory			g_PcPlotIndexBufferMemory = VK_NULL_HANDLE;
static uint32_t					windowWidth = 1920;
static uint32_t					windowHeight = 1080;
static uint32_t					g_PcPlotWidth = 2060;
static uint32_t					g_PcPlotHeight = 450;
static char						g_fragShaderPath[] = "shader/frag.spv";
static char						g_geomShaderPath[] = "shader/geom.spv";
static char						g_vertShaderPath[] = "shader/vert.spv";
static char						g_histFragPath[] = "shader/histFrag.spv";
static char						g_histVertPath[] = "shader/histVert.spv";
static char						g_histGeoPath[] = "shader/histGeo.spv";
static char						g_rectFragPath[] = "shader/rectFrag.spv";
static char						g_rectVertPath[] = "shader/rectVert.spv";
static char						g_densFragPath[] = "shader/densFrag.spv";
static char						g_densVertPath[] = "shader/densVert.spv";

//color palette for density
static const unsigned char		colorPalette[] = { 0, 0, 0		,255
												,0, 0, 36		,255
												,0, 0, 51		,255
												,0, 0, 66		,255
												,0, 0, 81		,255
												,2, 0, 90		,255
												,4, 0, 99		,255
												,7, 0, 106		,255
												,11, 0, 115		,255
												,14, 0, 119		,255
												,20, 0, 123		,255
												,27, 0, 128		,255
												,33, 0, 133		,255
												,41, 0, 137		,255
												,48, 0, 140		,255
												,55, 0, 143		,255
												,61, 0, 146		,255
												,66, 0, 149		,255
												,72, 0, 150		,255
												,78, 0, 151		,255
												,84, 0, 152		,255
												,91, 0, 153		,255
												,97, 0, 155		,255
												,104, 0, 155	,255
												,110, 0, 156	,255
												,115, 0, 157	,255
												,122, 0, 157	,255
												,128, 0, 157	,255
												,134, 0, 157	,255
												,139, 0, 157	,255
												,146, 0, 156	,255
												,152, 0, 155	,255
												,157, 0, 155	,255
												,162, 0, 155	,255
												,167, 0, 154	,255
												,171, 0, 153	,255
												,175, 1, 152	,255
												,178, 1, 151	,255
												,182, 2, 149	,255
												,185, 4, 149	,255
												,188, 5, 147	,255
												,191, 6, 146	,255
												,193, 8, 144	,255
												,195, 11, 142	,255
												,198, 13, 139	,255
												,201, 17, 135	,255
												,203, 20, 132	,255
												,206, 23, 127	,255
												,208, 26, 121	,255
												,210, 29, 116	,255
												,212, 33, 111	,255
												,214, 37, 103	,255
												,217, 41, 97	,255
												,219, 46, 89	,255
												,221, 49, 78	,255
												,223, 53, 66	,255
												,224, 56, 54	,255
												,226, 60, 42	,255
												,228, 64, 30	,255
												,229, 68, 25	,255
												,231, 72, 20	,255
												,232, 76, 16	,255
												,234, 78, 12	,255
												,235, 82, 10	,255
												,236, 86, 8		,255
												,237, 90, 7		,255
												,238, 93, 5		,255
												,239, 96, 4		,255
												,240, 100, 3	,255
												,241, 103, 3	,255
												,241, 106, 2	,255
												,242, 109, 1	,255
												,243, 113, 1	,255
												,244, 116, 0	,255
												,244, 120, 0	,255
												,245, 125, 0	,255
												,246, 129, 0	,255
												,247, 133, 0	,255
												,248, 136, 0	,255
												,248, 139, 0	,255
												,249, 142, 0	,255
												,249, 145, 0	,255
												,250, 149, 0	,255
												,251, 154, 0	,255
												,252, 159, 0	,255
												,253, 163, 0	,255
												,253, 168, 0	,255
												,253, 172, 0	,255
												,254, 176, 0	,255
												,254, 179, 0	,255
												,254, 184, 0	,255
												,254, 187, 0	,255
												,254, 191, 0	,255
												,254, 195, 0	,255
												,254, 199, 0	,255
												,254, 202, 1	,255
												,254, 205, 2	,255
												,254, 208, 5	,255
												,254, 212, 9	,255
												,254, 216, 12	,255
												,255, 219, 15	,255
												,255, 221, 23	,255
												,255, 224, 32	,255
												,255, 227, 39	,255
												,255, 229, 50	,255
												,255, 232, 63	,255
												,255, 235, 75	,255
												,255, 238, 88	,255
												,255, 239, 102	,255
												,255, 241, 116	,255
												,255, 242, 134	,255
												,255, 244, 149	,255
												,255, 245, 164	,255
												,255, 247, 179	,255
												,255, 248, 192	,255
												,255, 249, 203	,255
												,255, 251, 216	,255
												,255, 253, 228	,255
												,255, 254, 239	,255
												,255, 255, 249 ,255};

struct Attribute {
	std::string name;
	float min;			//min value of all values
	float max;			//max value of all values
};

static bool* pcAttributeEnabled = NULL;											//Contains whether a specific attribute is enabled
static std::vector<Attribute> pcAttributes = std::vector<Attribute>();			//Contains the attributes and its bounds	
static std::vector<int> pcAttrOrd = std::vector<int>();							//Contains the ordering of the attributes	
static std::vector<std::string> droppedPaths = std::vector<std::string>();
static bool* createDLForDrop = NULL;
static bool pathDropped = false;
static std::default_random_engine engine;
static std::uniform_int_distribution<int> distribution(0, 35);
static float alphaDrawLists = .5f;
static Vec4 PcPlotBackCol = { 0,0,0,1 };
static bool enableAxisLines = true;
 
//variables for the histogramm
static float histogrammWidth = .1f;
static bool drawHistogramm = false;
static bool histogrammDensity = false;
static bool pcPlotDensity = false;
static float densityRadius = .05f;
static bool enableDensityMapping = false;
static bool calculateMedians = false;
static bool mapDensity = true;
static Vec4 histogrammBackCol = { .2f,.2f,.2,1 };
static Vec4 densityBackCol = { 0,0,0,1 };
static float medianLineWidth = 1.0f;
static bool enableBrushing = false;

//variables for brush templates
static bool brushTemplatesEnabled = true;
static bool* brushTemplateAttrEnabled = NULL;
static int selectedTemplateBrush = -1;
static std::vector<TemplateBrush> templateBrushes;

//variables for global brushes
static int selectedGlobalBrush = -1;			//The global brushes are shown in a list where each brush is clickable to then be adaptable.
static std::vector<GlobalBrush> globalBrushes;
static int brushDragId = -1;
//information about the dragmode
//0 -> brush dragged
//1 -> top edge dragged
//2 -> bottom edge dragged
static int brushDragMode = 0;
static unsigned int currentBrushId = 0;
static bool* activeBrushAttributes = nullptr;
static bool toggleGlobalBrushes = true;
static int brushCombination = 0;				//How global brushes should be combined. 0->OR, 1->AND

//variables for the 3d views
static View3d * view3d;
static NodeViewer* nodeViewer;

static SettingsManager* settingsManager;

//method declarations
static void updateDrawListIndexBuffer(DrawList& dl);
static void updateActiveIndices(DrawList& dl);
/*static void check_vk_result(VkResult err)
{
	if (err == 0) return;
	printf("VkResult %d\n", err);
	if (err < 0)
		abort();
}*/

#ifdef IMGUI_VULKAN_DEBUG_REPORT
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_report(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType, uint64_t object, size_t location, int32_t messageCode, const char* pLayerPrefix, const char* pMessage, void* pUserData)
{
	(void)flags; (void)object; (void)location; (void)messageCode; (void)pUserData; (void)pLayerPrefix; // Unused arguments
	fprintf(stderr, "[vulkan] ObjectType: %i\nMessage: %s\n\n", objectType, pMessage);
	return VK_FALSE;
}
#endif // IMGUI_VULKAN_DEBUG_REPORT

static void createPcPlotHistoPipeline() {

	VkShaderModule shaderModules[5] = {};
	//the vertex shader for the pipeline
	std::vector<char> vertexBytes = PCUtil::readByteFile(g_histVertPath);
	shaderModules[0] = VkUtil::createShaderModule( g_Device, vertexBytes);
	//the geometry shader for the pipeline
	std::vector<char> geometryBytes = PCUtil::readByteFile(g_histGeoPath);
	shaderModules[3] = VkUtil::createShaderModule(g_Device, geometryBytes);
	//the fragment shader for the pipeline
	std::vector<char> fragmentBytes = PCUtil::readByteFile(g_histFragPath);
	shaderModules[4] = VkUtil::createShaderModule( g_Device, fragmentBytes);


	//Description for the incoming vertex attributes
	VkVertexInputBindingDescription bindingDescripiton = {};		//describes how big the vertex data is and how to read the data
	bindingDescripiton.binding = 0;
	bindingDescripiton.stride = sizeof(Vertex);
	bindingDescripiton.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attributeDescription = {};	//describes the attribute of the vertex. If more than 1 attribute is used this has to be an array
	attributeDescription.binding = 0;
	attributeDescription.location = 0;
	attributeDescription.format = VK_FORMAT_R32_SFLOAT;
	attributeDescription.offset = offsetof(Vertex, y);

	VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescripiton;
	vertexInputInfo.vertexAttributeDescriptionCount = 1;
	vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

	//vector with the dynamic states
	std::vector<VkDynamicState> dynamicStates;

	//Rasterizer Info
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

	//multisampling info
	VkPipelineMultisampleStateCreateInfo multisampling = {};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampling.minSampleShading = 1.0f;
	multisampling.pSampleMask = nullptr;
	multisampling.alphaToCoverageEnable = VK_FALSE;
	multisampling.alphaToOneEnable = VK_FALSE;

	//blendInfo
	VkUtil::BlendInfo blendInfo;
	
	VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_TRUE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
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
	
	blendInfo.blendAttachment = colorBlendAttachment;
	blendInfo.createInfo = colorBlending;

	//creating the descriptor set layout
	VkDescriptorSetLayoutBinding uboLayoutBinding = {};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;
	std::vector<VkDescriptorSetLayoutBinding> bindings;
	bindings.push_back(uboLayoutBinding);

	VkUtil::createDescriptorSetLayout(g_Device, bindings, &g_PcPlotHistoDescriptorSetLayout);
	std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
	descriptorSetLayouts.push_back(g_PcPlotHistoDescriptorSetLayout);

	VkUtil::createPipeline(g_Device, &vertexInputInfo, g_PcPlotWidth, g_PcPlotHeight, dynamicStates, shaderModules, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &g_PcPlotRenderPass, &g_PcPlotHistoPipelineLayout, &g_PcPlotHistoPipeline);

	shaderModules[3] = nullptr;
	//----------------------------------------------------------------------------------------------
	//creating the pipeline for the rect rendering
	//----------------------------------------------------------------------------------------------
	vertexBytes = PCUtil::readByteFile(g_rectVertPath);
	shaderModules[0] = VkUtil::createShaderModule(g_Device, vertexBytes);
	fragmentBytes = PCUtil::readByteFile(g_rectFragPath);
	shaderModules[4] = VkUtil::createShaderModule(g_Device, fragmentBytes);

	//describes how big the vertex data is and how to read the data
	bindingDescripiton.binding = 0;
	bindingDescripiton.stride = sizeof(RectVertex);
	bindingDescripiton.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attributeDescriptions[2];	//describes the attribute of the vertex. If more than 1 attribute is used this has to be an array
	attributeDescriptions[0].binding = 0;
	attributeDescriptions[0].location = 0;
	attributeDescriptions[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
	attributeDescriptions[0].offset = 0;

	attributeDescriptions[1].binding = 0;
	attributeDescriptions[1].location = 1;
	attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
	attributeDescriptions[1].offset = sizeof(Vec4);

	vertexInputInfo = {};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescripiton;
	vertexInputInfo.vertexAttributeDescriptionCount = 2;
	vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions;

	descriptorSetLayouts.clear();

	VkUtil::createPipeline(g_Device, &vertexInputInfo, g_PcPlotWidth, g_PcPlotHeight, dynamicStates, shaderModules, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &g_PcPlotRenderPass, &g_PcPlotRectPipelineLayout, &g_PcPlotRectPipeline);

	//----------------------------------------------------------------------------------------------
	//creating the pipeline for the density rendering
	//----------------------------------------------------------------------------------------------
	vertexBytes = PCUtil::readByteFile(g_densVertPath);
	shaderModules[0] = VkUtil::createShaderModule(g_Device, vertexBytes);
	fragmentBytes = PCUtil::readByteFile(g_densFragPath);
	shaderModules[4] = VkUtil::createShaderModule(g_Device, fragmentBytes);
	
	//describes how big the vertex data is and how to read the data
	bindingDescripiton.binding = 0;
	bindingDescripiton.stride = sizeof(Vec4);
	bindingDescripiton.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	attributeDescription = {};
	attributeDescription.binding = 0;
	attributeDescription.location = 0;
	attributeDescription.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	attributeDescription.offset = 0;

	vertexInputInfo = {};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescripiton;
	vertexInputInfo.vertexAttributeDescriptionCount = 1;
	vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

	//creating the descriptor set layout
	uboLayoutBinding = {};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	
	bindings.clear();
	bindings.push_back(uboLayoutBinding);

	uboLayoutBinding.binding = 2;
	bindings.push_back(uboLayoutBinding);

	uboLayoutBinding.binding = 1;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	
	bindings.push_back(uboLayoutBinding);

	VkUtil::createDescriptorSetLayout(g_Device, bindings, &g_PcPlotDensityDescriptorSetLayout);

	descriptorSetLayouts.clear();
	descriptorSetLayouts.push_back(g_PcPlotDensityDescriptorSetLayout);

	VkUtil::createPcPlotRenderPass(g_Device, VkUtil::PASS_TYPE_COLOR16_OFFLINE_NO_CLEAR, &g_PcPlotDensityRenderPass);

	VkUtil::createPipeline(g_Device, &vertexInputInfo, g_PcPlotWidth, g_PcPlotHeight, dynamicStates, shaderModules, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &g_PcPlotDensityRenderPass, &g_PcPlotDensityPipelineLayout, &g_PcPlotDensityPipeline);
}

static void cleanupPcPlotHistoPipeline() {
	vkDestroyDescriptorSetLayout(g_Device, g_PcPlotHistoDescriptorSetLayout, nullptr);
	vkDestroyPipelineLayout(g_Device, g_PcPlotHistoPipelineLayout, nullptr);
	vkDestroyPipeline(g_Device, g_PcPlotHistoPipeline, nullptr);
	vkDestroyPipelineLayout(g_Device, g_PcPlotRectPipelineLayout, nullptr);
	vkDestroyPipeline(g_Device, g_PcPlotRectPipeline, nullptr);
	vkDestroyPipelineLayout(g_Device, g_PcPlotDensityPipelineLayout, nullptr);
	vkDestroyDescriptorSetLayout(g_Device, g_PcPlotDensityDescriptorSetLayout, nullptr);
	vkDestroyRenderPass(g_Device, g_PcPlotDensityRenderPass, nullptr);
	vkDestroyPipeline(g_Device, g_PcPlotDensityPipeline, nullptr);
}

static uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memProps;
	vkGetPhysicalDeviceMemoryProperties(g_PhysicalDevice, &memProps);

	for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}
	//safety call to see whther a valid type Index was found
#ifdef _DEBUG
	std::cerr << "The memory type which is needed is not available!" << std::endl;
	exit(-1);
#endif
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
	imageInfo.format = VK_FORMAT_R16G16B16A16_UNORM;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
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

	//creating the Image and imageview for the density pipeline
	VkUtil::createImage(g_Device, g_PcPlotWidth, g_PcPlotHeight, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &g_PcPlotDensityImageCopy);

	uint32_t imageOffset = allocInfo.allocationSize;
	vkGetImageMemoryRequirements(g_Device, g_PcPlotDensityImageCopy, &memRequirements);
	allocInfo.allocationSize += memRequirements.size;

	//creating the Image and imageview for the iron map
	VkUtil::createImage(g_Device, sizeof(colorPalette) / sizeof(*colorPalette) / 4, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &g_PcPlotDensityIronMap);
	g_PcPlotDensityIronMapOffset = allocInfo.allocationSize;
	vkGetImageMemoryRequirements(g_Device, g_PcPlotDensityIronMap, &memRequirements);
	allocInfo.allocationSize += memRequirements.size;

	err = vkAllocateMemory(g_Device, &allocInfo, nullptr, &g_PcPlotMem);
	check_vk_result(err);

	vkBindImageMemory(g_Device, g_PcPlot, g_PcPlotMem, 0);
	vkBindImageMemory(g_Device, g_PcPlotDensityImageCopy, g_PcPlotMem, imageOffset);
	vkBindImageMemory(g_Device, g_PcPlotDensityIronMap, g_PcPlotMem, g_PcPlotDensityIronMapOffset);

	VkUtil::createImageView(g_Device, g_PcPlotDensityImageCopy, VK_FORMAT_R16G16B16A16_UNORM, 1, VK_IMAGE_ASPECT_COLOR_BIT,&g_PcPlotDensityImageView);
	VkUtil::createImageView(g_Device, g_PcPlotDensityIronMap, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_ASPECT_COLOR_BIT,&g_PcPLotDensityIronMapView);
	
	//creating the smapler for the density image
	VkUtil::createImageSampler(g_Device, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_FILTER_LINEAR, 1, 1, &g_PcPlotDensityImageSampler);
	VkUtil::createImageSampler(g_Device, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FILTER_LINEAR, 1, 1, &g_PcPlotDensityIronMapSampler);

	//creating the descriptorSet and updating the descriptorSetLayout
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.push_back(g_PcPlotDensityDescriptorSetLayout);
	VkUtil::createDescriptorSets(g_Device, layouts, g_DescriptorPool, &g_PcPlotDensityDescriptorSet);

	VkUtil::updateImageDescriptorSet(g_Device, g_PcPlotDensityImageSampler, g_PcPlotDensityImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, g_PcPlotDensityDescriptorSet);
	VkUtil::updateImageDescriptorSet(g_Device, g_PcPlotDensityIronMapSampler, g_PcPLotDensityIronMapView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2, g_PcPlotDensityDescriptorSet);

	//uploading the iron map via a staging buffer
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	VkUtil::createBuffer(g_Device, sizeof(colorPalette), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &stagingBuffer);

	VkMemoryRequirements memReq = {};
	vkGetBufferMemoryRequirements(g_Device, stagingBuffer, &memReq);

	VkMemoryAllocateInfo memalloc = {};
	memalloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memalloc.allocationSize = memReq.size;
	memalloc.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	
	err = vkAllocateMemory(g_Device, &memalloc, nullptr, &stagingBufferMemory);
	check_vk_result(err);

	vkBindBufferMemory(g_Device, stagingBuffer, stagingBufferMemory, 0);
	void* d;
	vkMapMemory(g_Device, stagingBufferMemory, 0, sizeof(colorPalette), 0, &d);
	memcpy(d, colorPalette, sizeof(colorPalette));
	vkUnmapMemory(g_Device, stagingBufferMemory);
	
	VkCommandBuffer stagingCommandBuffer;
	VkUtil::createCommandBuffer(g_Device, g_PcPlotCommandPool, &stagingCommandBuffer);
	
	VkUtil::transitionImageLayout(stagingCommandBuffer, g_PcPlotDensityIronMap, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	VkUtil::copyBufferToImage(stagingCommandBuffer, stagingBuffer, g_PcPlotDensityIronMap, sizeof(colorPalette) / sizeof(*colorPalette) / 4, 1);
	VkUtil::transitionImageLayout(stagingCommandBuffer, g_PcPlotDensityIronMap, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	VkUtil::commitCommandBuffer(g_Queue, stagingCommandBuffer);
	
	err = vkDeviceWaitIdle(g_Device);
	check_vk_result(err);
	vkDestroyBuffer(g_Device, stagingBuffer, nullptr);
	vkFreeMemory(g_Device, stagingBufferMemory, nullptr);

	VkImageViewCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	createInfo.image = g_PcPlot;
	createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	createInfo.format = VK_FORMAT_R16G16B16A16_UNORM;
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
	vkDestroyImageView(g_Device, g_PcPlotDensityImageView, nullptr);
	vkDestroyImage(g_Device, g_PcPlotDensityImageCopy, nullptr);
	vkDestroySampler(g_Device, g_PcPlotDensityImageSampler, nullptr);
	vkDestroyImageView(g_Device, g_PcPLotDensityIronMapView, nullptr);
	vkDestroyImage(g_Device, g_PcPlotDensityIronMap, nullptr);
	vkDestroySampler(g_Device, g_PcPlotDensityIronMapSampler, nullptr);
	vkFreeMemory(g_Device, g_PcPlotMem, nullptr);
}

static std::vector<char> readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		std::cerr << "failed to open file!" << std::endl;
		exit(-1);
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
	attributeDescription.format = VK_FORMAT_R32_SFLOAT;
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
	inputAssembly.primitiveRestartEnable = VK_TRUE;

	VkViewport viewport = {};					//description for our viewport for transformation operation after rasterization
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
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
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


	//----------------------------------------------------------------------------------------------
	//creating the pipeline for spline rendering
	//----------------------------------------------------------------------------------------------
	VkShaderModule shaderModules[5] = {};
	std::vector<char> vertexBytes = PCUtil::readByteFile(g_vertShaderPath);
	shaderModules[0] = VkUtil::createShaderModule(g_Device, vertexBytes);
	std::vector<char> geometryBytes = PCUtil::readByteFile(g_geomShaderPath);
	shaderModules[3] = VkUtil::createShaderModule(g_Device, geometryBytes);
	std::vector<char> fragmentBytes = PCUtil::readByteFile(g_fragShaderPath);
	shaderModules[4] = VkUtil::createShaderModule(g_Device, fragmentBytes);

	//describes how big the vertex data is and how to read the data
	bindingDescripiton.binding = 0;
	bindingDescripiton.stride = sizeof(float);
	bindingDescripiton.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	attributeDescription = {};
	attributeDescription.binding = 0;
	attributeDescription.location = 0;
	attributeDescription.format = VK_FORMAT_R32_SFLOAT;
	attributeDescription.offset = offsetof(Vertex, y);

	vertexInputInfo = {};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescripiton;
	vertexInputInfo.vertexAttributeDescriptionCount = 1;
	vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

	uboLayoutBinding = {};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;

	VkUtil::BlendInfo blendInfo;
	blendInfo.blendAttachment = colorBlendAttachment;
	blendInfo.createInfo = colorBlending;

	std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
	descriptorSetLayouts.push_back(g_PcPlotDescriptorLayout);

	std::vector<VkDynamicState> dynamicStateVec;
	dynamicStateVec.push_back(VK_DYNAMIC_STATE_LINE_WIDTH);

	VkUtil::createPipeline(g_Device, &vertexInputInfo, g_PcPlotWidth, g_PcPlotHeight, dynamicStateVec, shaderModules, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &g_PcPlotRenderPass, &g_PcPlotSplinePipelineLayout, &g_PcPlotSplinePipeline);
}

static void cleanupPcPlotPipeline() {
	vkDestroyDescriptorPool(g_Device, g_PcPlotDescriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(g_Device, g_PcPlotDescriptorLayout, nullptr);
	vkDestroyPipelineLayout(g_Device, g_PcPlotPipelineLayout, nullptr);
	vkDestroyPipeline(g_Device, g_PcPlotPipeline, nullptr);
	vkDestroyPipelineLayout(g_Device, g_PcPlotSplinePipelineLayout, nullptr);
	vkDestroyPipeline(g_Device, g_PcPlotSplinePipeline, nullptr);
}

static void createPcPlotRenderPass() {
	VkResult err;

	VkAttachmentDescription colorAttachment = {};
	colorAttachment.format = VK_FORMAT_R16G16B16A16_UNORM;
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

	//creating the Framebuffer for the density pass
	std::vector<VkImageView> attachments;
	attachments.push_back(g_PcPlotView);
	VkUtil::createFrameBuffer(g_Device, g_PcPlotDensityRenderPass, attachments, g_PcPlotWidth, g_PcPlotHeight, &g_PcPlotDensityFrameBuffer);
}

static void cleanupPcPlotFramebuffer() {
	vkDestroyFramebuffer(g_Device, g_PcPlotFramebuffer, nullptr);
	vkDestroyFramebuffer(g_Device, g_PcPlotDensityFrameBuffer, nullptr);
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

static void createPcPlotVertexBuffer( const std::vector<Attribute>& Attributes, const std::vector<float*>& data) {
	VkResult err;

	//creating the command buffer as its needed to do all the operations in here
	//createPcPlotCommandBuffer();

	Buffer vertexBuffer;

	uint32_t amtOfVertices = Attributes.size() * data.size();

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = sizeof(Vertex) * Attributes.size() * data.size();
	bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &vertexBuffer.buffer);
	check_vk_result(err);

	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(g_Device, vertexBuffer.buffer, &memRequirements);

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	err = vkAllocateMemory(g_Device, &allocInfo, nullptr, &vertexBuffer.memory);
	check_vk_result(err);
	
	vkBindBufferMemory(g_Device, vertexBuffer.buffer, vertexBuffer.memory, 0);

	//VkDeviceSize offsets[] = { 0 };
	//vkCmdBindVertexBuffers(g_PcPlotCommandBuffer, 0, 1, &g_PcPlotVertexBuffer, offsets);

	//creating a 1-D array with all the Attributes
	float* d = new float[data.size() * Attributes.size()];
	uint32_t i = 0;
	for (float* p : data) {
		for (int j = 0; j < Attributes.size(); j++) {
			d[i++] = p[j];
		}
	}

	//filling the Vertex Buffer with all Datapoints
	void* mem;
	vkMapMemory(g_Device, vertexBuffer.memory, 0, sizeof(Vertex) * amtOfVertices, 0, &mem);
	memcpy(mem, d, amtOfVertices * sizeof(Vertex));
	vkUnmapMemory(g_Device, vertexBuffer.memory);

	delete[] d;

	g_PcPlotVertexBuffers.push_back(vertexBuffer);
	
	if (g_PcPlotIndexBuffer)
		return;

	//creating the index buffer
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = sizeof(uint16_t) * Attributes.size();
	bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &g_PcPlotIndexBuffer);
	check_vk_result(err);

	vkGetBufferMemoryRequirements(g_Device, g_PcPlotIndexBuffer, &memRequirements);

	int memTypeBits = 0;

	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	memTypeBits = memRequirements.memoryTypeBits;
	g_PcPlotHistogrammRectOffset = allocInfo.allocationSize;
	
	//creating the histogramm rect buffers
	bufferInfo.size = sizeof(RectVertex) * 4 * pcAttributes.size();
	bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &g_PcPlotHistogrammRect);
	check_vk_result(err);

	vkGetBufferMemoryRequirements(g_Device, g_PcPlotHistogrammRect, &memRequirements);
	allocInfo.allocationSize += memRequirements.size;
	memTypeBits |= memRequirements.memoryTypeBits;
	g_PcPlotHistogrammIndexOffset = allocInfo.allocationSize;

	//creating the histogram index buffer
	bufferInfo.size = sizeof(uint16_t) * 6 * pcAttributes.size();
	bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &g_PcPlotHistogrammIndex);
	check_vk_result(err);

	vkGetBufferMemoryRequirements(g_Device, g_PcPlotHistogrammIndex, &memRequirements);
	allocInfo.allocationSize += memRequirements.size;
	memTypeBits |= memRequirements.memoryTypeBits;

	//creating the density rect buffer
	g_PcPlotDensityRectBufferOffset = allocInfo.allocationSize;
	bufferInfo.size = sizeof(Vec4) * 4 * pcAttributes.size() + 1;
	bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &g_PcPlotDensityRectBuffer);
	check_vk_result(err);

	vkGetBufferMemoryRequirements(g_Device, g_PcPlotDensityRectBuffer, &memRequirements);
	allocInfo.allocationSize += memRequirements.size;

	//creating the density uniform buffer
	g_PcPLotDensityUboOffset = allocInfo.allocationSize;
	bufferInfo.size = sizeof(DensityUniformBuffer);
	bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &g_PcPlotDensityUbo);
	check_vk_result(err);

	vkGetBufferMemoryRequirements(g_Device, g_PcPlotDensityUbo, &memRequirements);
	allocInfo.allocationSize += memRequirements.size;

	//allocating the memory
	allocInfo.memoryTypeIndex = findMemoryType(memTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	err = vkAllocateMemory(g_Device, &allocInfo, nullptr, &g_PcPlotIndexBufferMemory);
	check_vk_result(err);

	//binding the pcPlot index buffer
	vkBindBufferMemory(g_Device, g_PcPlotIndexBuffer, g_PcPlotIndexBufferMemory, 0);

	//binding the histogramm rect buffer
	vkBindBufferMemory(g_Device, g_PcPlotHistogrammRect, g_PcPlotIndexBufferMemory, g_PcPlotHistogrammRectOffset);

	//binding the histogramm index buffer
	vkBindBufferMemory(g_Device, g_PcPlotHistogrammIndex, g_PcPlotIndexBufferMemory, g_PcPlotHistogrammIndexOffset);

	//binding the density vertex buffer
	vkBindBufferMemory(g_Device, g_PcPlotDensityRectBuffer, g_PcPlotIndexBufferMemory, g_PcPlotDensityRectBufferOffset);

	//binding the uboBuffer and adding the buffer to the density descriptor set
	vkBindBufferMemory(g_Device, g_PcPlotDensityUbo, g_PcPlotIndexBufferMemory, g_PcPLotDensityUboOffset);
	VkUtil::updateDescriptorSet(g_Device, g_PcPlotDensityUbo, sizeof(DensityUniformBuffer), 1,g_PcPlotDensityDescriptorSet);
	uploadDensityUiformBuffer();

	//filling the histogramm index buffer
	//Vertex Arrangment:
	//0		3
	//|		|
	//1-----2
	uint16_t* indexBuffer = new uint16_t[pcAttributes.size() * 6];
	for (int i = 0; i < pcAttributes.size(); i++) {
		indexBuffer[i * 6] = i * 4;
		indexBuffer[i * 6 + 1] = i * 4 + 2;
		indexBuffer[i * 6 + 2] = i * 4 + 1;
		indexBuffer[i * 6 + 3] = i * 4;
		indexBuffer[i * 6 + 4] = i * 4 + 3;
		indexBuffer[i * 6 + 5] = i * 4 + 2;
	}
	void* ind;
	vkMapMemory(g_Device, g_PcPlotIndexBufferMemory, g_PcPlotHistogrammIndexOffset, sizeof(uint16_t) * 6 * pcAttributes.size(), 0, &ind);
	memcpy(ind, indexBuffer, sizeof(uint16_t) * 6 * pcAttributes.size());
	vkUnmapMemory(g_Device, g_PcPlotIndexBufferMemory);

	//filling the densityRectBuffer with the rect for full screen
	Vec4 rect[4] = { {-1,1,0,1},{-1,-1,0,1},{1,-1,0,1},{1,1,0,1} };
	vkMapMemory(g_Device, g_PcPlotIndexBufferMemory, g_PcPlotDensityRectBufferOffset, sizeof(Vec4)*4, 0, &ind);
	memcpy(ind, rect, sizeof(Vec4)*4);
	vkUnmapMemory(g_Device, g_PcPlotIndexBufferMemory);

	delete[] indexBuffer;

	//creating the bool array for brushtemplates
	brushTemplateAttrEnabled = new bool[pcAttributes.size()];
	for (int i = 0; i < pcAttributes.size(); i++) {
		brushTemplateAttrEnabled[i] = false;
	}
}

static void cleanupPcPlotVertexBuffer() {
	for (Buffer& b : g_PcPlotVertexBuffers) {
		if (b.buffer) {
			vkDestroyBuffer(g_Device, b.buffer, nullptr);
			b.buffer = VK_NULL_HANDLE;
		}
		if (b.memory) {
			vkFreeMemory(g_Device, b.memory, nullptr);
			b.memory = VK_NULL_HANDLE;
		}
	}
	if (g_PcPlotIndexBuffer) {
		vkDestroyBuffer(g_Device, g_PcPlotIndexBuffer, nullptr);
		g_PcPlotIndexBuffer = VK_NULL_HANDLE;
	}
	if (g_PcPlotHistogrammIndex) {
		vkDestroyBuffer(g_Device, g_PcPlotHistogrammIndex, nullptr);
		g_PcPlotHistogrammIndex = VK_NULL_HANDLE;
	}
	if (g_PcPlotHistogrammRect) {
		vkDestroyBuffer(g_Device, g_PcPlotHistogrammRect, nullptr);
		g_PcPlotHistogrammRect = VK_NULL_HANDLE;
	}
	if (g_PcPlotDensityRectBuffer) {
		vkDestroyBuffer(g_Device, g_PcPlotDensityRectBuffer, nullptr);
		g_PcPlotDensityRectBuffer = VK_NULL_HANDLE;
	}
	if (g_PcPlotDensityUbo) {
		vkDestroyBuffer(g_Device, g_PcPlotDensityUbo, nullptr);
		g_PcPlotDensityUbo = VK_NULL_HANDLE;
	}
	if (g_PcPlotIndexBufferMemory) {
		vkFreeMemory(g_Device, g_PcPlotIndexBufferMemory, nullptr);
		g_PcPlotIndexBufferMemory = VK_NULL_HANDLE;
	}

	/*
	if (g_PcPlotDescriptorBuffer) {
		vkDestroyBuffer(g_Device, g_PcPlotDescriptorBuffer, nullptr);
		g_PcPlotDescriptorBuffer = VK_NULL_HANDLE;
	}
	if (g_PcPlotDescriptorBufferMemory) {
		vkFreeMemory(g_Device, g_PcPlotDescriptorBufferMemory, nullptr);
		g_PcPlotDescriptorBufferMemory = VK_NULL_HANDLE;
	}
	*/
}

static void destroyPcPlotVertexBuffer(Buffer& buffer) {
	auto it = g_PcPlotVertexBuffers.begin();
	for (; it != g_PcPlotVertexBuffers.end(); ++it) {
		if (*it == buffer) {
			break;
		}
	}
	
	if (it == g_PcPlotVertexBuffers.end()) {
		std::cout << "Buffer to be destroyed not found" << std::endl;
		return;
	}

	if (buffer.buffer) {
		vkDestroyBuffer(g_Device, buffer.buffer, nullptr);
		buffer.buffer = VK_NULL_HANDLE;
	}
	if (buffer.memory) {
		vkFreeMemory(g_Device, buffer.memory, nullptr);
		buffer.memory = VK_NULL_HANDLE;
	}

	g_PcPlotVertexBuffers.erase(it);
}

static void removePcPlotDrawLists(DataSet dataSet) {
	for (auto it = g_PcPlotDrawLists.begin(); it != g_PcPlotDrawLists.end(); ) {
		if (it->parentDataSet == dataSet.name) {
			it->indices.clear();
			if (it->dlMem) {
				vkFreeMemory(g_Device, it->dlMem, nullptr);
				it->dlMem = VK_NULL_HANDLE;
			}
			if (it->ubo) {
				vkDestroyBuffer(g_Device, it->ubo, nullptr);
				it->ubo = VK_NULL_HANDLE;
			}
			if (it->histogramIndBuffer) {
				vkDestroyBuffer(g_Device, it->histogramIndBuffer, nullptr);
				it->histogramIndBuffer = VK_NULL_HANDLE;
			}
			if (it->medianBuffer) {
				vkDestroyBuffer(g_Device, it->medianBuffer, nullptr);
				it->medianBuffer = VK_NULL_HANDLE;
			}
			if (it->medianUbo) {
				vkDestroyBuffer(g_Device, it->medianUbo, nullptr);
				it->medianUbo = VK_NULL_HANDLE;
			}
			if (it->indexBuffer) {
				vkDestroyBuffer(g_Device, it->indexBuffer, nullptr);
				it->indexBuffer = VK_NULL_HANDLE;
			}
			for (int i = 0; i < it->histogramUbos.size(); i++) {
				vkDestroyBuffer(g_Device, it->histogramUbos[i], nullptr);
			}
			for (GlobalBrush& brush : globalBrushes) {
				brush.lineRatios.erase(it->name);
			}
			g_PcPlotDrawLists.erase(it++);
		}
		else{
			it++;
		}
	}
}

static void createPcPlotDrawList(const TemplateList& tl,const DataSet& ds,const char* listName) {
	VkResult err;

	DrawList dl = {};

	//uniformBuffer for pcPlot Drawing
	Buffer uboBuffer;

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = sizeof(UniformBufferObject);
	bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &dl.ubo);
	check_vk_result(err);

	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(g_Device, dl.ubo, &memRequirements);
	memRequirements.size = (memRequirements.size % memRequirements.alignment) ? memRequirements.size + (memRequirements.alignment - (memRequirements.size % memRequirements.alignment)) : memRequirements.size; //alining the memory

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	uint32_t memTypeBits = memRequirements.memoryTypeBits;
	
	//uniformBuffer for Histogramms
	bufferInfo.size = sizeof(HistogramUniformBuffer);
	for (int i = 0; i < pcAttributes.size(); i++) {
		dl.histogramUbos.push_back({});
		err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &dl.histogramUbos.back());
		check_vk_result(err);

		vkGetBufferMemoryRequirements(g_Device, dl.histogramUbos.back(), &memRequirements);
		memRequirements.size = (memRequirements.size % memRequirements.alignment) ? memRequirements.size + (memRequirements.alignment - (memRequirements.size % memRequirements.alignment)) : memRequirements.size; //alining the memory
		allocInfo.allocationSize += memRequirements.size;
		memTypeBits |= memRequirements.memoryTypeBits;
	}

	//Median line uniform buffer
	bufferInfo.size = sizeof(UniformBufferObject);
	bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	
	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &dl.medianUbo);
	check_vk_result(err);

	dl.medianUboOffset = allocInfo.allocationSize;

	vkGetBufferMemoryRequirements(g_Device, dl.medianUbo, &memRequirements);
	memRequirements.size = (memRequirements.size % memRequirements.alignment) ? memRequirements.size + (memRequirements.alignment - (memRequirements.size % memRequirements.alignment)) : memRequirements.size;
	allocInfo.allocationSize += memRequirements.size;

	//IndexBuffer for Histogramms
	bufferInfo.size = 2 * tl.indices.size() * sizeof(uint32_t);
	bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &dl.histogramIndBuffer);
	check_vk_result(err);

	vkGetBufferMemoryRequirements(g_Device, dl.histogramIndBuffer, &memRequirements);
	allocInfo.allocationSize += memRequirements.size;
	memTypeBits |= memRequirements.memoryTypeBits;

	//Median Buffer for Median Lines
	bufferInfo.size = MEDIANCOUNT * pcAttributes.size() * sizeof(float);
	bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &dl.medianBuffer);
	check_vk_result(err);

	dl.medianBufferOffset = allocInfo.allocationSize;

	vkGetBufferMemoryRequirements(g_Device, dl.medianBuffer, &memRequirements);
	allocInfo.allocationSize += memRequirements.size;
	memTypeBits |= memRequirements.memoryTypeBits;

	//Indexbuffer
	bufferInfo.size = tl.indices.size() * (pcAttributes.size() + 3) * sizeof(uint32_t);
	bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &dl.indexBuffer);
	check_vk_result(err);

	dl.indexBufferOffset = allocInfo.allocationSize;
	vkGetBufferMemoryRequirements(g_Device, dl.indexBuffer, &memRequirements);
	allocInfo.allocationSize += memRequirements.size;
	memTypeBits |= memRequirements.memoryTypeBits;

	//allocating the Memory for all draw list data
	allocInfo.memoryTypeIndex = findMemoryType(memTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	err = vkAllocateMemory(g_Device, &allocInfo, nullptr, &dl.dlMem);
	check_vk_result(err);

	//Binding Uniform Buffer
	vkBindBufferMemory(g_Device, dl.ubo, dl.dlMem, 0);

	//Binding histogram uniform buffers
	uint32_t offset = sizeof(UniformBufferObject);
	offset = (offset % memRequirements.alignment) ? offset + (memRequirements.alignment - (offset % memRequirements.alignment)) : offset; //alining the memory
	dl.histogramUbosOffsets.push_back(offset);
	for (int i = 0; i < dl.histogramUbos.size(); i++) {
		vkBindBufferMemory(g_Device, dl.histogramUbos[i], dl.dlMem, offset);
		offset += sizeof(HistogramUniformBuffer);
		offset = (offset % memRequirements.alignment) ? offset + (memRequirements.alignment - (offset % memRequirements.alignment)) : offset; //alining the memory
		dl.histogramUbosOffsets.push_back(offset);
	}
	dl.histogramUbosOffsets.pop_back();

	//creating the Descriptor sets for the histogramm uniform buffers
	std::vector<VkDescriptorSetLayout> layouts(dl.histogramUbos.size());
	for (auto& l : layouts) {
		l = g_PcPlotHistoDescriptorSetLayout;
	}

	dl.histogrammDescSets = std::vector<VkDescriptorSet>(layouts.size());
	VkUtil::createDescriptorSets(g_Device, layouts, g_DescriptorPool, dl.histogrammDescSets.data());

	//updating the descriptor sets
	for (int i = 0; i < layouts.size(); i++) {
		VkUtil::updateDescriptorSet(g_Device, dl.histogramUbos[i], sizeof(HistogramUniformBuffer), 0, dl.histogrammDescSets[i]);
	}

	//Binding the median uniform Buffer
	vkBindBufferMemory(g_Device, dl.medianUbo, dl.dlMem, dl.medianUboOffset);

	//creating the Descriptor set for the median uniform buffer
	layouts.clear();
	layouts.push_back(g_PcPlotDescriptorLayout);
	VkUtil::createDescriptorSets(g_Device, layouts, g_DescriptorPool, &dl.medianUboDescSet);
	VkUtil::updateDescriptorSet(g_Device, dl.medianUbo, sizeof(UniformBufferObject), 0, dl.medianUboDescSet);

	//Binding the histogram index buffer
	offset += sizeof(UniformBufferObject);
	offset = (offset % memRequirements.alignment) ? offset + (memRequirements.alignment - (offset % memRequirements.alignment)) : offset; //alining the memory
	vkBindBufferMemory(g_Device, dl.histogramIndBuffer, dl.dlMem, offset);
	dl.histIndexBufferOffset = offset;

	//creating and uploading the indexbuffer data
	uint32_t* indBuffer = new uint32_t[tl.indices.size()*2];
	for (int i = 0; i < tl.indices.size(); i++) {
		indBuffer[2 * i] = tl.indices[i] * pcAttributes.size();
		indBuffer[2 * i + 1] = tl.indices[i] * pcAttributes.size();
	}
	void* d;
	vkMapMemory(g_Device, dl.dlMem, offset, tl.indices.size() * sizeof(uint32_t) * 2, 0, &d);
	memcpy(d, indBuffer, tl.indices.size() * sizeof(uint32_t) * 2);
	vkUnmapMemory(g_Device, dl.dlMem);
	delete[] indBuffer;


	//binding the medianBuffer
	vkBindBufferMemory(g_Device, dl.medianBuffer, dl.dlMem, dl.medianBufferOffset);

	//calculating the medians
#ifdef _DEBUG
	std::cout << "Starting to calulate the medians." << std::endl;
#endif
	float* medianArr = new float[pcAttributes.size() * MEDIANCOUNT];
	//median calulations
	if (calculateMedians) {
		std::vector<int> dataCpy(tl.indices);

		for (int i = 0; i < pcAttributes.size(); i++) {
			std::sort(dataCpy.begin(), dataCpy.end(), [i,ds](int a, int b) {return ds.data[a][i] > ds.data[b][i]; });
			medianArr[MEDIAN * pcAttributes.size() + i] = ds.data[dataCpy[dataCpy.size() >> 1]][i];
		}

		//arithmetic median calculation
		for (int i = 0; i < tl.indices.size(); i++) {
			for (int j = 0; j < pcAttributes.size(); j++) {
				if (i == 0)
					medianArr[ARITHMEDIAN * pcAttributes.size() + j] = 0;
				medianArr[ARITHMEDIAN * pcAttributes.size() + j] += ds.data[tl.indices[i]][j];
			}
		}
		for (int i = 0; i < pcAttributes.size(); i++) {
			medianArr[ARITHMEDIAN * pcAttributes.size() + i] /= tl.indices.size();
		}
		
		//geometric median
		const float epsilon = .001f, phi = .0000001f;
		std::vector<float> last(pcAttributes.size());
		std::vector<float> median(pcAttributes.size());
		std::vector<float> xj;
		float denom,denomsum;
		for (int i = 0; i < median.size(); i++) {
			last[i] = 0;
			median[i] = medianArr[ARITHMEDIAN * pcAttributes.size() + i];
		}

		while (squareDist(last, median) > epsilon) {
			last = std::vector<float>(median);
			for (int i = 0; i < median.size(); i++) {
				median[i] = 0;
			}
			denomsum = 0;
			for (int i = 0; i < tl.indices.size(); i++) {
				xj = std::vector<float>(ds.data[i], ds.data[i] + pcAttributes.size());
				denom = eucDist(xj - last);
				if (denom < phi)
					continue;
				median = median + (xj / denom);
				denomsum += 1/denom;
			}
			median = median / denomsum;
		}

		for (int i = 0; i < pcAttributes.size(); i++) {
			medianArr[GOEMEDIAN * pcAttributes.size() + i] = median[i];
		}
	}
	else {
		for (int i = 0; i < MEDIANCOUNT * pcAttributes.size(); i++) {
			medianArr[i] = 0;
		}
	}
	//copying the median Array
	vkMapMemory(g_Device, dl.dlMem, dl.medianBufferOffset, MEDIANCOUNT* pcAttributes.size() * sizeof(float), 0, &d);
	memcpy(d, medianArr, MEDIANCOUNT* pcAttributes.size() * sizeof(float));
	vkUnmapMemory(g_Device, dl.dlMem);
	delete[] medianArr;

	dl.activeMedian = 0;
	dl.medianColor = { 1,1,1,1 };
#ifdef _DEBUG
	std::cout << "Finished to calulate the medians." << std::endl;
#endif

	//binding the indexBuffer
	vkBindBufferMemory(g_Device, dl.indexBuffer, dl.dlMem, dl.indexBufferOffset);


	//specifying the uniform buffer location
	VkDescriptorBufferInfo desBufferInfo = {};
	desBufferInfo.buffer = dl.ubo;
	desBufferInfo.offset = 0;
	desBufferInfo.range = sizeof(UniformBufferObject);

	VkDescriptorSetAllocateInfo alloc_info = {};
	alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	alloc_info.descriptorPool = g_DescriptorPool;
	alloc_info.descriptorSetCount = 1;
	alloc_info.pSetLayouts = &g_PcPlotDescriptorLayout;
	err = vkAllocateDescriptorSets(g_Device, &alloc_info, &dl.uboDescSet);
	check_vk_result(err);

	VkWriteDescriptorSet descriptorWrite = {};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = dl.uboDescSet;
	descriptorWrite.dstBinding = 0;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pBufferInfo = &desBufferInfo;

	vkUpdateDescriptorSets(g_Device, 1, &descriptorWrite, 0, nullptr);

	int hue = distribution(engine) * 10;
#ifdef _DEBUG
	std::cout << "Hue: " << hue << std::endl;
#endif

	hsl randCol = {  hue,.5f,.6f };
	rgb col = hsl2rgb(randCol);

	dl.name = std::string(listName);
	dl.buffer = tl.buffer;
	dl.color = { (float)col.r,(float)col.g,(float)col.b,alphaDrawLists };
	dl.prefColor = dl.color;
	dl.show = true;
	dl.showHistogramm = true;
	dl.parentDataSet = ds.name;
	dl.indices = std::vector<int>(tl.indices);
	dl.activeInd = std::vector<int>(tl.indices);

	//adding a standard brush for every attribute
	for (Attribute a : pcAttributes) {
		dl.brushes.push_back(std::vector<Brush>());
	}

	g_PcPlotDrawLists.push_back(dl);
}

static void removePcPlotDrawList(DrawList& drawList) {
	for (auto it = g_PcPlotDrawLists.begin(); it != g_PcPlotDrawLists.end(); ++it) {
		if (it->name == drawList.name) {
			it->indices.clear();
			if (it->dlMem) {
				vkFreeMemory(g_Device, it->dlMem, nullptr);
				it->dlMem = VK_NULL_HANDLE;
			}
			if (it->ubo) {
				vkDestroyBuffer(g_Device, it->ubo, nullptr);
				it->ubo = VK_NULL_HANDLE;
			}
			if (it->histogramIndBuffer) {
				vkDestroyBuffer(g_Device, it->histogramIndBuffer, nullptr);
				it->histogramIndBuffer = VK_NULL_HANDLE;
			}
			if (it->medianBuffer) {
				vkDestroyBuffer(g_Device, it->medianBuffer, nullptr);
				it->medianBuffer = VK_NULL_HANDLE;
			}
			if (it->medianUbo) {
				vkDestroyBuffer(g_Device, it->medianUbo, nullptr);
				it->medianUbo = VK_NULL_HANDLE;
			}
			if (it->indexBuffer) {
				vkDestroyBuffer(g_Device, it->indexBuffer, nullptr);
				it->indexBuffer = VK_NULL_HANDLE;
			}
			for (int i = 0; i < it->histogramUbos.size(); i++) {
				vkDestroyBuffer(g_Device, it->histogramUbos[i], nullptr);
			}
			g_PcPlotDrawLists.erase(it);
			break;
		}
	}
	for (GlobalBrush& brush : globalBrushes) {
		brush.lineRatios.erase(drawList.name);
	}
}

static void destroyPcPlotDataSet(DataSet dataSet) {
	auto it = g_PcPlotDataSets.begin();
	for (; it != g_PcPlotDataSets.end(); ++it) {
		if (*it == dataSet) {
			break;
		}
	}

	if (it == g_PcPlotDataSets.end()) {
		std::cout << "DataSet to be destroyed not found" << std::endl;
		return;
	}

	dataSet.drawLists.clear();
	destroyPcPlotVertexBuffer(dataSet.buffer);

	removePcPlotDrawLists(dataSet);

	if (dataSet.oneData) {
		delete[] dataSet.data[0];
	}
	else {
		for (int i = 0; i < dataSet.data.size(); i++) {
			delete[] dataSet.data[i];
		}
	}

	g_PcPlotDataSets.erase(it);

	//if this was the last data set reset the ofther buffer too
	//Attributes also have to be deleted
	if (g_PcPlotDataSets.size() == 0) {
		cleanupPcPlotVertexBuffer();

		pcAttributes.clear();
		pcAttrOrd.clear();
		if (pcAttributeEnabled) {
			delete[] pcAttributeEnabled;
			pcAttributeEnabled = nullptr;
		}
		if (brushTemplateAttrEnabled) {
			delete[] brushTemplateAttrEnabled;
			brushTemplateAttrEnabled = nullptr;
		}
		if (activeBrushAttributes) {
			delete[] activeBrushAttributes;
			activeBrushAttributes = nullptr;
		}
	}
}

//This method automatically also destroys all draw Lists
static void cleanupPcPlotDataSets() {
	for (DataSet ds : g_PcPlotDataSets) {
		ds.drawLists.clear();
		removePcPlotDrawLists(ds);
		
		if (ds.oneData) {
			delete[] ds.data[0];
		}
		else {
			for (int i = 0; i < ds.data.size(); i++) {
				delete[] ds.data[i];
			}
		}
	}

	g_PcPlotDataSets.clear();
	cleanupPcPlotVertexBuffer();
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

	VkClearValue clearColor = { PcPlotBackCol.x,PcPlotBackCol.y,PcPlotBackCol.z,PcPlotBackCol.w };//{ 0.0f,0.0f,0.0f,1.0f };
	
	renderPassInfo.clearValueCount = 1;
	renderPassInfo.pClearValues = &clearColor;

	vkCmdBeginRenderPass(g_PcPlotCommandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	if (g_RenderSplines)
		vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipeline);
	else
		vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotPipeline);
}

static void cleanupPcPlotCommandBuffer() {
	VkResult err;
	vkCmdEndRenderPass(g_PcPlotCommandBuffer);
	
	VkImageMemoryBarrier use_barrier[1] = {};
	use_barrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	use_barrier[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	use_barrier[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	use_barrier[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	use_barrier[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	use_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	use_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	use_barrier[0].image = g_PcPlot;
	use_barrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	use_barrier[0].subresourceRange.levelCount = 1;
	use_barrier[0].subresourceRange.layerCount = 1;
	vkCmdPipelineBarrier(g_PcPlotCommandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, use_barrier);
	
	err = vkEndCommandBuffer(g_PcPlotCommandBuffer);
	check_vk_result(err);

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.signalSemaphoreCount = 0;
	submitInfo.waitSemaphoreCount = 0;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &g_PcPlotCommandBuffer;

	err = vkQueueSubmit(g_Queue, 1, &submitInfo, VK_NULL_HANDLE);
	check_vk_result(err);

	err = vkDeviceWaitIdle(g_Device);
	check_vk_result(err);

	vkFreeCommandBuffers(g_Device, g_PcPlotCommandPool, 1, &g_PcPlotCommandBuffer);
}

static int placeOfInd(int ind) {
	int place = 0;
	for (int i : pcAttrOrd) {
		if (i == ind)
			break;
		if (pcAttributeEnabled[i])
			place++;
	}
	return place;
}

static void drawPcPlot(const std::vector<Attribute>& attributes, const std::vector<int>& attributeOrder, const bool* attributeEnabled, const ImGui_ImplVulkanH_Window* wd) {
#ifdef PRINTRENDERTIME
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	uint32_t amtOfLines = 0;
#endif

	VkResult err;

	err = vkDeviceWaitIdle(g_Device);
	check_vk_result(err);

	//beginning the command buffer
	VkCommandPool command_pool = wd->Frames[wd->FrameIndex].CommandPool;
	VkCommandBuffer command_buffer = wd->Frames[wd->FrameIndex].CommandBuffer;

	err = vkResetCommandPool(g_Device, command_pool, 0);
	check_vk_result(err);
	VkCommandBufferBeginInfo begin_info = {};
	begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	err = vkBeginCommandBuffer(command_buffer, &begin_info);
	check_vk_result(err);

	//now using the memory barrier to transition image state
	VkImageMemoryBarrier use_barrier[1] = {};
	use_barrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	use_barrier[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	use_barrier[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	use_barrier[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	use_barrier[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	use_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	use_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	use_barrier[0].image = g_PcPlot;
	use_barrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	use_barrier[0].subresourceRange.levelCount = 1;
	use_barrier[0].subresourceRange.layerCount = 1;
	vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, use_barrier);

	//ending the command buffer and submitting it
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

	//drawing via copying the indeces into the index buffer
	//the indeces have to have just the right ordering for the vertices
	int amtOfIndeces = 0;
	for (int i = 0; i < attributes.size(); i++) {
		if (attributeEnabled[i])
			amtOfIndeces++;
	}
	

	//filling the uniform buffer and copying it into the end of the uniformbuffer
	UniformBufferObject ubo = {};
	ubo.amtOfVerts = amtOfIndeces;
	ubo.amtOfAttributes = attributes.size();
	ubo.color = { 1,1,1,1 };
	if (drawHistogramm) {
		ubo.padding = histogrammWidth / 2;
	}
	else {
		ubo.padding = 0;
	}
	
	int c = 0;
	
	for (int i : attributeOrder) {
		ubo.VertexTransormations[i].x = c;
		if (attributeEnabled[i])
			c++;
		ubo.VertexTransormations[i].y = attributes[i].min;
		ubo.VertexTransormations[i].z = attributes[i].max;
	}

	std::vector<std::pair<int, int>> order;
	for (int i = 0; i < pcAttributes.size(); i++) {
		if (pcAttributeEnabled[i]) {
			order.push_back(std::pair<int, int>(i, placeOfInd(i)));
		}
	}

	std::sort(order.begin(), order.end(), [](std::pair<int, int>a, std::pair<int, int>b) {return a.second < b.second; });

	//filling the indexbuffer with the used indeces
	uint16_t* ind = new uint16_t[amtOfIndeces];			//contains all indeces to copy
	for (int i = 0; i < order.size(); i++) {
		ind[i] = order[i].first;
	}

#ifdef _DEBUG
	if (order.size() != amtOfIndeces) {
		std::cerr << "There is a severe problem with the indices!" << std::endl;
		exit(-1);
	}
#endif

	void* d;
	//copying the indexbuffer
	if (pcAttributes.size()) {
		vkMapMemory(g_Device, g_PcPlotIndexBufferMemory, 0, sizeof(uint16_t) * attributes.size(), 0, &d);
		memcpy(d, ind, amtOfIndeces * sizeof(uint16_t));
		vkUnmapMemory(g_Device, g_PcPlotIndexBufferMemory);
	}

#ifdef _DEBUG
	if (c != amtOfIndeces) {
		std::cerr << "There is a severe problem with the indices!" << std::endl;
		exit(-1);
	}
#endif

	//copying the uniform buffer
	void* da;
	for (DrawList& ds : g_PcPlotDrawLists) {
		ubo.color = ds.color;
		vkMapMemory(g_Device, ds.dlMem, 0, sizeof(UniformBufferObject), 0, &da);
		memcpy(da, &ubo, sizeof(UniformBufferObject));
		vkUnmapMemory(g_Device, ds.dlMem);
		
		ubo.color = ds.medianColor;
		vkMapMemory(g_Device, ds.dlMem, ds.medianUboOffset, sizeof(UniformBufferObject), 0, &da);
		memcpy(da, &ubo, sizeof(UniformBufferObject));
		vkUnmapMemory(g_Device, ds.dlMem);
		
	}

	//starting the pcPlotCommandBuffer
	createPcPlotCommandBuffer();

	//binding the all needed things
	if(g_RenderSplines)
		vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipelineLayout, 0, 1, &g_PcPlotDescriptorSet, 0, nullptr);
	else
		vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotPipelineLayout, 0, 1, &g_PcPlotDescriptorSet, 0, nullptr);

	if(pcAttributes.size())
		vkCmdBindIndexBuffer(g_PcPlotCommandBuffer, g_PcPlotIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

	//counting the amount of active drawLists for histogramm rendering
	int activeDrawLists = 0;
	
	//now drawing for every draw list in g_pcPlotdrawlists
	for (auto drawList = g_PcPlotDrawLists.rbegin(); g_PcPlotDrawLists.rend() != drawList;++drawList) {
		if (!drawList->show)
			continue;

		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(g_PcPlotCommandBuffer, 0, 1, &drawList->buffer, offsets);
		vkCmdBindIndexBuffer(g_PcPlotCommandBuffer, drawList->indexBuffer, 0, VK_INDEX_TYPE_UINT32);

		//binding the right ubo
		if (g_RenderSplines)
			vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipelineLayout, 0, 1, &drawList->uboDescSet, 0, nullptr);
		else
			vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotPipelineLayout, 0, 1, &drawList->uboDescSet, 0, nullptr);

		vkCmdSetLineWidth(g_PcPlotCommandBuffer, 1.0f);
		
		//ready to draw with draw indexed
		uint32_t amtOfI = drawList->activeInd.size()* (order.size() + 1 + ((g_RenderSplines) ? 2 : 0));
		vkCmdDrawIndexed(g_PcPlotCommandBuffer, amtOfI, 1, 0, 0, 0);

		//draw the Median Line
		if (drawList->activeMedian != 0) {
			vkCmdSetLineWidth(g_PcPlotCommandBuffer, medianLineWidth);
			vkCmdBindVertexBuffers(g_PcPlotCommandBuffer, 0, 1, &drawList->medianBuffer, offsets);

			if (g_RenderSplines)
				vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipelineLayout, 0, 1, &drawList->medianUboDescSet, 0, nullptr);
			else
				vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotPipelineLayout, 0, 1, &drawList->medianUboDescSet, 0, nullptr);

			vkCmdDrawIndexed(g_PcPlotCommandBuffer, amtOfIndeces, 1, 0, (drawList->activeMedian-1)* pcAttributes.size(), 0);

#ifdef PRINTRENDERTIME
			amtOfLines++;
#endif
		}

#ifdef PRINTRENDERTIME
		amtOfLines += drawList->activeInd.size();
#endif
	}

	delete[] ind;

	if (pcPlotDensity && pcAttributes.size() > 0) {
		//ending the pass to blit the image
		vkCmdEndRenderPass(g_PcPlotCommandBuffer);

		//transition image Layouts
		VkUtil::transitionImageLayout(g_PcPlotCommandBuffer, g_PcPlot, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		VkUtil::transitionImageLayout(g_PcPlotCommandBuffer, g_PcPlotDensityImageCopy, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		//blitting the image
		VkUtil::copyImage(g_PcPlotCommandBuffer, g_PcPlot, g_PcPlotWidth, g_PcPlotHeight, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_PcPlotDensityImageCopy, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		//transition image Layouts back
		VkUtil::transitionImageLayout(g_PcPlotCommandBuffer, g_PcPlot, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
		VkUtil::transitionImageLayout(g_PcPlotCommandBuffer, g_PcPlotDensityImageCopy, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		//beginning the density renderpass
		std::vector<VkClearValue> clearColors;
		VkUtil::beginRenderPass(g_PcPlotCommandBuffer, clearColors, g_PcPlotDensityRenderPass, g_PcPlotDensityFrameBuffer, { g_PcPlotWidth,g_PcPlotHeight });
		vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotDensityPipeline);

		VkDeviceSize offsets[1] = { 0 };
		vkCmdBindVertexBuffers(g_PcPlotCommandBuffer, 0, 1, &g_PcPlotDensityRectBuffer, offsets);
		vkCmdBindIndexBuffer(g_PcPlotCommandBuffer, g_PcPlotHistogrammIndex, 0, VK_INDEX_TYPE_UINT16);
		vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotDensityPipelineLayout, 0, 1, &g_PcPlotDensityDescriptorSet, 0, nullptr);

		vkCmdDrawIndexed(g_PcPlotCommandBuffer, 6, 1, 0, 0, 0);
	}

	if (drawHistogramm && pcAttributes.size()>0) {
		//drawing the histogramm background
		RectVertex* rects = new RectVertex[pcAttributes.size() * 4];
		float x = -1;
		for (int i = 0; i < pcAttributes.size(); i++) {
			if (pcAttributeEnabled[i]) {
				RectVertex vert;
				vert.pos = { x,1,0,0 };
				vert.col = histogrammDensity ? densityBackCol : histogrammBackCol;
				rects[i * 4] = vert;
				vert.pos.y = -1;
				rects[i * 4 + 1] = vert;
				vert.pos.x += histogrammWidth;
				rects[i * 4 + 2] = vert;
				vert.pos.y = 1;
				rects[i * 4 + 3] = vert;
				x += (2 - histogrammWidth) / (amtOfIndeces - 1);
			}
			else {
				RectVertex vert;
				vert.pos = { -2,-2,0,0 };
				vert.col = histogrammBackCol;
				rects[i * 4] = vert;
				rects[i * 4 + 1] = vert;
				rects[i * 4 + 2] = vert;
				rects[i * 4 + 3] = vert;
			}
		}
		//uploading the vertexbuffer
		//void* d;
		vkMapMemory(g_Device, g_PcPlotIndexBufferMemory, g_PcPlotHistogrammRectOffset, sizeof(RectVertex) * pcAttributes.size() * 4, 0, &d);
		memcpy(d, rects, sizeof(RectVertex) * pcAttributes.size() * 4);
		vkUnmapMemory(g_Device, g_PcPlotIndexBufferMemory);

		delete[] rects;

		//binding the graphics pipeline
		vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotRectPipeline);

		//binding the buffers and drawing the data
		vkCmdBindIndexBuffer(g_PcPlotCommandBuffer, g_PcPlotHistogrammIndex, 0, VK_INDEX_TYPE_UINT16);
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(g_PcPlotCommandBuffer, 0, 1, &g_PcPlotHistogrammRect, offsets);
		vkCmdDrawIndexed(g_PcPlotCommandBuffer, pcAttributes.size() * 6, 1, 0, 0, 0);

		//starting to draw the histogramm lines
		vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotHistoPipeline);

		//the offset which has to be added to draw the histogramms next to one another
		uint32_t amtOfHisto = 0;
		for (auto dl : g_PcPlotDrawLists) {
			if (dl.showHistogramm)
				amtOfHisto++;
		}
		if (amtOfHisto != 0) {
			HistogramUniformBuffer hubo = {};
			float gap = (2 - histogrammWidth) / (amtOfIndeces - 1);
			float xOffset = .0f;
			float width = histogrammWidth / amtOfHisto;
			for (auto drawList = g_PcPlotDrawLists.begin(); g_PcPlotDrawLists.end() != drawList; ++drawList) {
				//ignore drawLists which are disabled
				if (!drawList->showHistogramm)
					continue;

				//setting the color in the hubo to copy
				hubo.color = drawList->color;
				hubo.width = width;

				//binding the correct vertex and indexbuffer
				VkDeviceSize offsets[] = { 0 };
				vkCmdBindVertexBuffers(g_PcPlotCommandBuffer, 0, 1, &drawList->buffer, offsets);
				vkCmdBindIndexBuffer(g_PcPlotCommandBuffer, drawList->histogramIndBuffer, 0, VK_INDEX_TYPE_UINT32);

				//iterating through the Attributes to render every histogramm
				float x = -1.0f;
				int count = 0;
				for (int i = 0; i < pcAttributes.size(); i++) {
					//setting the missing parameters in the hbuo
					hubo.maxVal = pcAttributes[i].max;
					hubo.minVal = pcAttributes[i].min;
					if (!pcAttributeEnabled[i])
						hubo.x = -2;
					else
						hubo.x = -1 + placeOfInd(i) * gap + xOffset;

					//uploading the ubo
					//void* d;
					vkMapMemory(g_Device, drawList->dlMem, drawList->histogramUbosOffsets[i], sizeof(HistogramUniformBuffer), 0, &d);
					memcpy(d, &hubo, sizeof(HistogramUniformBuffer));
					vkUnmapMemory(g_Device, drawList->dlMem);

					//binding the descriptor set
					vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotHistoPipelineLayout, 0, 1, &drawList->histogrammDescSets[i], 0, nullptr);

					//making the draw call
					vkCmdDrawIndexed(g_PcPlotCommandBuffer, drawList->activeInd.size() * 2, 1, 0, count++, 0);
				}

				//increasing the xOffset for the next drawlist
				xOffset += width;
			}
		}

		if (histogrammDensity) {
			//ending the pass to blit the image
			vkCmdEndRenderPass(g_PcPlotCommandBuffer);

			//transition image Layouts
			VkUtil::transitionImageLayout(g_PcPlotCommandBuffer, g_PcPlot, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
			VkUtil::transitionImageLayout(g_PcPlotCommandBuffer, g_PcPlotDensityImageCopy, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

			//blitting the image
			VkUtil::copyImage(g_PcPlotCommandBuffer, g_PcPlot, g_PcPlotWidth, g_PcPlotHeight, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_PcPlotDensityImageCopy, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

			//transition image Layouts back
			VkUtil::transitionImageLayout(g_PcPlotCommandBuffer, g_PcPlot, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
			VkUtil::transitionImageLayout(g_PcPlotCommandBuffer, g_PcPlotDensityImageCopy, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

			//beginning the density renderpass
			std::vector<VkClearValue> clearColors;
			VkUtil::beginRenderPass(g_PcPlotCommandBuffer, clearColors, g_PcPlotDensityRenderPass, g_PcPlotDensityFrameBuffer, { g_PcPlotWidth,g_PcPlotHeight });
			vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotDensityPipeline);

			Vec4* verts = new Vec4[amtOfIndeces * 4];
			float gap = (2 - histogrammWidth) / (amtOfIndeces - 1);
			for (int i = 0; i < amtOfIndeces; i++) {
				verts[i * 4] = { gap * i - 1,1,0,0 };
				verts[i * 4 + 1] = { gap * i - 1,-1,0,0 };
				verts[i * 4 + 2] = { gap * i + histogrammWidth - 1,-1,0,0 };
				verts[i * 4 + 3] = { gap * i + histogrammWidth - 1,1,0,0 };
			}
			
			vkMapMemory(g_Device, g_PcPlotIndexBufferMemory, g_PcPlotDensityRectBufferOffset + sizeof(Vec4) * 4, sizeof(Vec4)* amtOfIndeces * 4, 0, &d);
			memcpy(d, verts, sizeof(Vec4)* amtOfIndeces * 4);
			vkUnmapMemory(g_Device, g_PcPlotIndexBufferMemory);

			delete[] verts;

			VkDeviceSize offsets[1] = { sizeof(Vec4) * 4 };
			vkCmdBindVertexBuffers(g_PcPlotCommandBuffer, 0, 1, &g_PcPlotDensityRectBuffer, offsets);
			vkCmdBindIndexBuffer(g_PcPlotCommandBuffer, g_PcPlotHistogrammIndex, 0, VK_INDEX_TYPE_UINT16);
			vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotDensityPipelineLayout, 0, 1, &g_PcPlotDensityDescriptorSet, 0, nullptr);

			vkCmdDrawIndexed(g_PcPlotCommandBuffer, amtOfIndeces * 6, 1, 0, 0, 0);
		}
	}

	//when cleaning up the command buffer all data is drawn
	cleanupPcPlotCommandBuffer();

	err = vkDeviceWaitIdle(g_Device);
	check_vk_result(err);

#ifdef PRINTRENDERTIME
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Amount of Lines rendered: " << amtOfLines << std::endl;
	std::cout << "Time for render: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " milliseconds" << std::endl;
#endif
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

#ifdef _DEBUG
		std::cout << "Amount of Gpus: " << gpu_count << std::endl;
#endif

		VkPhysicalDeviceFeatures feat;
		vkGetPhysicalDeviceFeatures(gpus[0], &feat);

#ifdef _DEBUG
		std::cout << "Gometry shader usable:" << feat.geometryShader << std::endl;
		std::cout << "Wide lines usable:" << feat.wideLines << std::endl;
#endif

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
		VkPhysicalDeviceFeatures deviceFeatures = {};
		deviceFeatures.geometryShader = VK_TRUE;
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		deviceFeatures.wideLines = VK_TRUE;
		deviceFeatures.depthClamp = VK_TRUE;
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
		create_info.pEnabledFeatures = &deviceFeatures;
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
	ImDrawData* dd = ImGui::GetDrawData();
	ImGui_ImplVulkan_RenderDrawData(dd, fd->CommandBuffer);

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

static void openCsv(const char* filename) {

	std::ifstream input(filename);

	if (!input.is_open()) {
		std::cout << "The given file was not found" << std::endl;
		return;
	}

	bool firstLine = true;

	//creating the dataset to be drawable
	DataSet ds;
	std::string s(filename);
	int split = (s.find_last_of("\\") > s.find_last_of("/")) ? s.find_last_of("/") : s.find_last_of("\\");
	ds.name = s.substr(split + 1);
	//checking if the filename already exists
	for (auto it = g_PcPlotDataSets.begin(); it != g_PcPlotDataSets.end();++it) {
		if (it->name == ds.name) {
			auto itcop = it;
			int count = 0;
			while (1) {
				if(itcop->name == ds.name + (count==0?"":std::to_string(count)))
					count++;
				if (++itcop == g_PcPlotDataSets.end())
					break;
			}
			ds.name += std::to_string(count);
			break;
		}
	}

	for (std::string line; std::getline(input, line); )
	{
		std::string delimiter = ",";
		size_t pos = 0;
		std::string cur;

		//parsing the attributes in the first line
		if (firstLine) {
			//copying the attributes into a temporary vector to check for correct Attributes
			std::vector<Attribute> tmp;

			while ((pos = line.find(delimiter)) != std::string::npos) {
				cur = line.substr(0, pos);
				line.erase(0, pos + delimiter.length());
				tmp.push_back({ cur,std::numeric_limits<float>::max(),std::numeric_limits<float>::min()});
			}
			//adding the last item which wasn't recognized
			tmp.push_back({ line,std::numeric_limits<float>::max(),std::numeric_limits<float>::min()});

			//checking if the Attributes are correct
			if (pcAttributes.size() != 0) {
				if (tmp.size() != pcAttributes.size()) {
					std::cout << "The Amount of Attributes of the .csv file is not compatible with the currently loaded datasets" << std::endl;
					input.close();
					return;
				}
				
				for (int i = 0; i < tmp.size(); i++) {
					if (tmp[i].name != pcAttributes[i].name) {
						std::cout << "The Attributes of the .csv file have different order or different names." << std::endl;
						input.close();
						return;
					}
				}
			}
			//if this is the first Dataset to be loaded, fill the pcAttributes vector
			else {
				for (Attribute a : tmp) {
					pcAttributes.push_back(a);
				}

				//setting up the boolarray and setting all the attributes to true
				pcAttributeEnabled = new bool[pcAttributes.size()];
				activeBrushAttributes = new bool[pcAttributes.size()];
				for (int i = 0; i < pcAttributes.size(); i++) {
					pcAttributeEnabled[i] = true;
					activeBrushAttributes[i] = false;
					pcAttrOrd.push_back(i);
				}
			}

			firstLine = false;
		}

		//parsing the data which follows the attribute declaration
		else {
			ds.data.push_back(new float[pcAttributes.size()]);
			size_t attr = 0;
			float curF = 0;
			while ((pos = line.find(delimiter)) != std::string::npos) {
				cur = line.substr(0, pos);
				line.erase(0, pos + delimiter.length());

				//checking for an overrunning attribute counter
				if (attr == pcAttributes.size()) {
					std::cerr << "The dataset to open is not consitent!" << std::endl;
					input.close();
					return;
				}

				curF = std::stof(cur);

				//updating the bounds if a new highest value was found in the current data.
				if (curF > pcAttributes[attr].max)
					pcAttributes[attr].max = curF;
				if (curF < pcAttributes[attr].min)
					pcAttributes[attr].min = curF;

				ds.data.back()[attr++] = curF;
			}
			if (attr == pcAttributes.size()) {
				std::cerr << "The dataset to open is not consitent!" << std::endl;
				input.close();
				return;
			}

			//adding the last item which wasn't recognized
			curF = std::stof(line);

			//updating the bounds if a new highest value was found in the current data.
			if (curF > pcAttributes[attr].max)
				pcAttributes[attr].max = curF;
			if (curF < pcAttributes[attr].min)
				pcAttributes[attr].min = curF;
			ds.data.back()[attr] = curF;
		}
	}
	input.close();

	createPcPlotVertexBuffer(pcAttributes, ds.data);

	ds.buffer = g_PcPlotVertexBuffers.back();

	TemplateList tl = {};
	tl.buffer = g_PcPlotVertexBuffers.back().buffer;
	tl.name = "Default Drawlist";
	for (int i = 0; i < ds.data.size(); i++) {
		tl.indices.push_back(i);
	}

	//getting the minimum and maximum values for all attributes. This will later be used for brush creation
	for (int i = 0; i < pcAttributes.size(); i++) {
		tl.minMax.push_back(std::pair<float, float>(std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()));
	}
	for (int i : tl.indices) {
		for (int j = 0; j < pcAttributes.size(); j++) {
			if (ds.data[i][j] < tl.minMax[j].first)
				tl.minMax[j].first = ds.data[i][j];
			if (ds.data[i][j] > tl.minMax[j].second)
				tl.minMax[j].second = ds.data[i][j];
		}
	}

	ds.drawLists.push_back(tl);

	g_PcPlotDataSets.push_back(ds);

#ifdef _DEBUG
	//printing out the loaded attributes for debug reasons
	std::cout << "Attributes: " << std::endl;
	for (auto attribute : pcAttributes) {
		std::cout << attribute.name << ", MinVal: " << attribute.min << ", MaxVal: " << attribute.max << std::endl;
	}

	int dc = 0;
	std::cout << std::endl << "Data:" << std::endl;
	for (auto d : ds.data) {
		for (int i = 0; i < pcAttributes.size(); i++) {
			std::cout << d[i] << " , ";
		}
		std::cout << std::endl;
		if (dc++ > 10)
			break;
	}
#endif
}

//ind1 is the index to which ind2 should be switched
static void switchAttributes(int ind1, int ind2, bool ctrPressed) {
	if (ctrPressed) {
		int tmp = pcAttrOrd[ind2];
		if (ind2 > ind1) {
			for (int i = ind2 ; i != ind1; i--) {
				pcAttrOrd[i] = pcAttrOrd[i - 1];
			}
		}
		else {
			for (int i = ind2; i != ind1; i++) {
				pcAttrOrd[i] = pcAttrOrd[i + 1];
			}
		}
		pcAttrOrd[ind1] = tmp;
	}
	else {
		int tmp = pcAttrOrd[ind1];
		pcAttrOrd[ind1] = pcAttrOrd[ind2];
		pcAttrOrd[ind2] = tmp;
	}
}

static void openDlf(const char* filename) {
	std::ifstream file(filename, std::ifstream::in);
	if (file.is_open()) {
		std::string tmp;
		int amtOfPoints;
		bool newAttr = false;

		while (!file.eof()) {
			file >> tmp;
			if (tmp != std::string("AmtOfPoints:")) {
				std::cout << "AmtOfPoints is missing in the dlf file. Got " << tmp << " instead." << std::endl;
				return;
			}
			else {
				file >> amtOfPoints;
			}
			file >> tmp;
			//checking for the variables section
			if (tmp != std::string("Attributes:")) {
				std::cout << "Attributes section not found. Got " << tmp << " instead" << std::endl;
				return;
			}
			else {
				file >> tmp;
				//checking for the same attributes in the currently loaded Attributes
				if (pcAttributes.size() > 0) {		

					//current max attribute count is 100
					for (int i = 0; tmp != std::string("Data:") && i < 100; file >> tmp, i++) {		
						if (pcAttributes[i].name != tmp) {
							std::cout << "The Attributes are not in the same order are not the same." << std::endl;
							return;
						}
					}
					std::cout << "The Attribute check was successful" << std::endl;
				}

				//reading in new values
				else {								
					for (int i = 0; tmp != std::string("Data:") && i < 100; file >> tmp, i++) {
						pcAttributes.push_back({ tmp,std::numeric_limits<float>::max(),std::numeric_limits<float>::min() -1});
					}

					//check for attributes overflow
					if (pcAttributes.size() == 100) {		
						std::cout << "Too much attributes found, or Datablock not detected." << std::endl;
						pcAttributes.clear();
						return;
					}
					newAttr = true;
				}
			}

			//after Attribute collection reading in the data
			DataSet ds;
			if (tmp != std::string("Data:")) {
				std::cout << "Data Section not found. Got " << tmp << " instead." << std::endl;
				pcAttributes.clear();
				return;
			}
			//reading the data
			else {
				ds.oneData = true;
				ds.name = filename;

				file >> tmp;

				float* d = new float[amtOfPoints * pcAttributes.size()];
				int a = 0;
				for (int i = 0; i < amtOfPoints * pcAttributes.size() && tmp != std::string("Drawlists:"); file >> tmp, i++) {
					d[i] = std::stof(tmp);
					if (pcAttributes[a].min > d[i]) {
						pcAttributes[a].min = d[i];
					}
					if (pcAttributes[a].max < d[i]) {
						pcAttributes[a].max = d[i];
					}
					a = (a + 1) % pcAttributes.size();
				} 

				ds.data = std::vector<float*>(amtOfPoints);
				for (int i = 0; i < amtOfPoints; i++) {
					ds.data[i] = &d[i * pcAttributes.size()];
				}
			}

			createPcPlotVertexBuffer(pcAttributes, ds.data);

			//adding a default drawlist for all attributes
			TemplateList defaultT = {};
			defaultT.buffer = g_PcPlotVertexBuffers.back().buffer;
			defaultT.name = "Default";
			for (int i = 0; i < pcAttributes.size(); i++) {
				defaultT.minMax.push_back(std::pair<float, float>(std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()));
			}
			for (int i = 0; i < ds.data.size(); i++) {
				defaultT.indices.push_back(i);
				for (int j = 0; j < pcAttributes.size(); j++) {
					if (ds.data[i][j] < defaultT.minMax[j].first)
						defaultT.minMax[j].first = ds.data[i][j];
					if (ds.data[i][j] > defaultT.minMax[j].second)
						defaultT.minMax[j].second = ds.data[i][j];
				}
			}
			ds.drawLists.push_back(defaultT);

			//reading the draw lists
			if (tmp != std::string("Drawlists:")) {
				std::cout << "Missing Draw lists section. Got " << tmp << " instead" << std::endl;
				pcAttributes.clear();
				delete[] ds.data[0];
				ds.data.clear();
				return;
			}
			//beginnin to read the drawlists
			else {
				file >> tmp;
				ds.buffer = g_PcPlotVertexBuffers.back();
				while (!file.eof()) {		//Loop for each drawlist
					TemplateList tl;
					tl.buffer = g_PcPlotVertexBuffers.back().buffer;
					tl.name = tmp;
					while (tmp.back() != ':') {
						file >> tmp;
						tl.name += tmp;
					}
					//erasing
					file >> tmp;
					while (std::all_of(tmp.begin(), tmp.end(), ::isdigit) && !file.eof()) {
						tl.indices.push_back(std::stoi(tmp));
						file >> tmp;
					}
					//getting the range of the bounds for each attribute
					for (int i = 0; i < pcAttributes.size(); i++) {
						tl.minMax.push_back(std::pair<float, float>(std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()));
					}
					for (int i : tl.indices) {
						for (int j = 0; j < pcAttributes.size(); j++) {
							if (ds.data[i][j] < tl.minMax[j].first)
								tl.minMax[j].first = ds.data[i][j];
							if (ds.data[i][j] > tl.minMax[j].second)
								tl.minMax[j].second = ds.data[i][j];
						}
					}
					ds.drawLists.push_back(tl);
				}
			}

			if (newAttr) {
				pcAttributeEnabled = new bool[pcAttributes.size()];
				activeBrushAttributes = new bool[pcAttributes.size()];
				for (int i = 0; i < pcAttributes.size(); i++) {
					pcAttributeEnabled[i] = true;
					activeBrushAttributes[i] = false;
					pcAttrOrd.push_back(i);
				}
			}

			//adding the data set finally to the list
			g_PcPlotDataSets.push_back(ds);
		}

		file.close();
	}
	else {
		std::cout << "The dlf File could not be opened." << std::endl;
	}
}

static void openDataset(const char* filename) {
	//checking the datatype and calling the according method
	std::string file = filename;
	if (file.substr(file.find_last_of(".") + 1) == "csv") {
		openCsv(filename);
	}
	else if (file.substr(file.find_last_of(".") + 1) == "dlf") {
		openDlf(filename);
	}
	else {
		std::cout << "The given type of the file is not supported by this programm" << std::endl;
	}
}

static void addIndecesToDs(DataSet& ds,const char* filepath) {
	std::string s(filepath);
	if (s.substr(s.find_last_of(".") + 1) != "idxf") {
		std::cout << "There was an idxf file expected." << std::endl;
		return;
	}
	//opening the file
	std::ifstream file(filepath);
	
	if (file.is_open()) {
		TemplateList tl;
		tl.buffer = ds.buffer.buffer;
		int split = (s.find_last_of("\\") > s.find_last_of("/")) ? s.find_last_of("/") : s.find_last_of("\\");
		tl.name = s.substr(split + 1);

		//reading the values
		for (file >> s ; !file.eof(); file >> s) {
			int index = std::stof(s);
			if (index < ds.data.size()) {
				tl.indices.push_back(index);
			}
		}

		//getting minMax values for each attribute for brush creation
		for (int i = 0; i < pcAttributes.size(); i++) {
			tl.minMax.push_back(std::pair<float, float>(std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()));
		}
		for (int i : tl.indices) {
			for (int j = 0; j < pcAttributes.size(); j++) {
				if (ds.data[i][j] < tl.minMax[j].first)
					tl.minMax[j].first = ds.data[i][j];
				if (ds.data[i][j] > tl.minMax[j].second)
					tl.minMax[j].second = ds.data[i][j];
			}
		}

		//adding the drawlist to ds
		ds.drawLists.push_back(tl);
	}
	else {
		std::cout << "The given indexlist was not found." << std::endl;
		return;
	}
}

static void addMultipleIndicesToDs(DataSet& ds) {
	for (int i = 0; i < droppedPaths.size(); i++) {
		addIndecesToDs(ds, droppedPaths[i].c_str());
		if (createDLForDrop[i]) {
			int split = (droppedPaths[i].find_last_of("\\") > droppedPaths[i].find_last_of("/")) ? droppedPaths[i].find_last_of("/") : droppedPaths[i].find_last_of("\\");
			createPcPlotDrawList(ds.drawLists.back(), ds, droppedPaths[i].substr(split+1).c_str());
			updateActiveIndices(g_PcPlotDrawLists.back());
		}
	}
}

static void updateDrawListIndexBuffer(DrawList& dl) {

	std::vector<std::pair<int, int>> order;
	for (int i = 0; i < pcAttributes.size(); i++) {
		if (pcAttributeEnabled[i]) {
			order.push_back(std::pair<int, int>(i, placeOfInd(i)));
		}
	}

	std::sort(order.begin(), order.end(), [](std::pair<int, int>a, std::pair<int, int>b) {return a.second < b.second; });

	//filling the indexbuffer for drawing
	uint32_t amtOfIndices = dl.activeInd.size() * (order.size() + 1 + ((g_RenderSplines) ? 2 : 0));
	uint32_t* in = new uint32_t[amtOfIndices];
	uint32_t c = 0;
	for (int i : dl.activeInd) {
		if (g_RenderSplines)	//repeating first and last index if spline rendering is active to have adjacent lines to the first and last point of the line
			in[c++] = order[0].first + i * pcAttributes.size();

		for (int j = 0; j < order.size(); j++) {
			in[c++] = order[j].first + i * pcAttributes.size();
		}

		if (g_RenderSplines)
			in[c++] = order[order.size() - 1].first + i * pcAttributes.size();

		in[c++] = 0xFFFFFFFF;
	}

	if (dl.activeInd.size()) {
		void* d;
		vkMapMemory(g_Device, dl.dlMem, dl.indexBufferOffset, sizeof(uint32_t) * amtOfIndices, 0, &d);
		memcpy(d, in, sizeof(uint32_t) * amtOfIndices);
		vkUnmapMemory(g_Device, dl.dlMem);
	}

	delete[] in;
}

static void updateAllDrawListIndexBuffer() {
	for (DrawList& dl : g_PcPlotDrawLists) {
		updateDrawListIndexBuffer(dl);
	}
}

static void updateActiveIndices(DrawList& dl) {
	//getting the parent dataset data
	std::vector<float*>* data;
	for (DataSet& ds : g_PcPlotDataSets) {
		if (ds.name == dl.parentDataSet) {
			data = &ds.data;
			break;
		}
	}

	dl.activeInd.clear();
	for (GlobalBrush& b : globalBrushes) {
		b.lineRatios[dl.name] = 0;
	}
	for (int i : dl.indices) {
		//checking the local brushes
		bool keep = true;
		for (int j = 0; j < pcAttributes.size(); j++) {
			bool good = false;
			for (Brush& b : dl.brushes[j]) {
				if ((*data)[i][j] >= b.minMax.first && (*data)[i][j] <= b.minMax.second) {
					good = true;
					break;
				}
			}
			if (dl.brushes[j].size() == 0) {
				good = true;
			}
			if (!good)
				keep = false;
		}
		if (!keep)
			goto nextInd;

		//checking gloabl brushes
		if (toggleGlobalBrushes) {
			bool or = globalBrushes.size() == 0, and = true;
			for (GlobalBrush& b : globalBrushes) {
				bool lineKeep = true;
				for (auto& brush : b.brushes) {
					if ((*data)[i][brush.first]<brush.second.second.first || (*data)[i][brush.first]>brush.second.second.second) {
						lineKeep = false;
					}
				}
					
				or |= lineKeep;
				and &= lineKeep;

				if( lineKeep)
					b.lineRatios[dl.name] += 1.0f;
			}
			if (brushCombination == 1 && !and) {
				goto nextInd;
			}
			if (brushCombination == 0 && !or ) {
				goto nextInd;
			}
		}

		dl.activeInd.push_back(i);

		nextInd:;
	}
	for (GlobalBrush& b : globalBrushes) {
		b.lineRatios[dl.name] /= dl.indices.size();
	}

	//updating the indexbuffer for histogramm
	if (dl.activeInd.size() == 0)
		return;
	uint32_t* indBuffer = new uint32_t[dl.activeInd.size() * 2];
	for (int i = 0; i < dl.activeInd.size(); i++) {
		indBuffer[2 * i] = dl.activeInd[i] * pcAttributes.size();
		indBuffer[2 * i + 1] = dl.activeInd[i] * pcAttributes.size();
	}
	void* d;
	vkMapMemory(g_Device, dl.dlMem, dl.histIndexBufferOffset, dl.activeInd.size() * sizeof(uint32_t) * 2, 0, &d);
	memcpy(d, indBuffer, dl.activeInd.size() * sizeof(uint32_t) * 2);
	vkUnmapMemory(g_Device, dl.dlMem);
	delete[] indBuffer;

	//updating the standard indexbuffer
	updateDrawListIndexBuffer(dl);
}

//This method does the same as updataActiveIndices, only for ALL drawlists
//whenever possible use updataActiveIndices, not updateAllActiveIndicess
static void updateAllActiveIndices() {
	for (DrawList& dl : g_PcPlotDrawLists) {
		updateActiveIndices(dl);
	}
}

void drop_callback(GLFWwindow* window, int count, const char** paths) {
#ifdef _DEBUG
	std::cout << "Amount of files drag and dropped: " << count << std::endl;
#endif
	createDLForDrop = new bool[count];

	for (int i = 0; i < count; i++) {
		droppedPaths.push_back(std::string(paths[i]));
		createDLForDrop[i] = true;
	}
	pathDropped = true;
}

static void uploadDensityUiformBuffer() {
	DensityUniformBuffer ubo = {};
	ubo.enableMapping = enableDensityMapping;
	ubo.gaussRange = densityRadius;
	ubo.imageHeight = g_PcPlotHeight;
	void* d;
	vkMapMemory(g_Device, g_PcPlotIndexBufferMemory, g_PcPLotDensityUboOffset, sizeof(DensityUniformBuffer), 0, &d);
	memcpy(d, &ubo, sizeof(DensityUniformBuffer));
	vkUnmapMemory(g_Device, g_PcPlotIndexBufferMemory);
}

static void uploadDrawListTo3dView(DrawList& dl, std::string attribute, std::string width, std::string height, std::string depth) {
	int w = SpacialData::rlatSize;
	int d = SpacialData::rlonSize;
	int h = SpacialData::altitudeSize;

	DataSet* parent;
	for (DataSet& ds : g_PcPlotDataSets) {
		if (ds.name == dl.parentDataSet) {
			parent = &ds;
			break;
		}
	}
	
	Attribute a;
	int attributeIndex = 0;
	for (Attribute& at : pcAttributes) {
		if (at.name == attribute) {
			a = at;
			break;
		}
		attributeIndex++;
	}
	float* dat = new float[w * d * h * 4];
	memset(dat, 0, w * d * h * 4 * sizeof(float));
	for (int i : dl.indices) {
		int x = SpacialData::getRlatIndex(parent->data[i][0]);//(parent->data[i][0] - pcAttributes[0].min) / (pcAttributes[0].max +1 - pcAttributes[0].min) * w;
		int y = SpacialData::getAltitudeIndex(parent->data[i][2]);//(parent->data[i][2] - pcAttributes[2].min) / (pcAttributes[2].max +1- pcAttributes[2].min) * h;
		int z = SpacialData::getRlonIndex(parent->data[i][1]);//(parent->data[i][1] - pcAttributes[1].min) / (pcAttributes[1].max +1- pcAttributes[1].min) * d;
		assert(x >= 0);
		assert(y >= 0);
		assert(z >= 0);
		Vec4 col = dl.color;
		//col.w = (parent->data[i][attributeIndex] - a.min) / (a.max - a.min);
#ifdef _DEBUG
		//std::cout << "x: " << x << " y: " << y << " z: " << z << std::endl;
#endif
		
		memcpy(&dat[4 * IDX3D(x, y, z, w, h)], &col.x, sizeof(Vec4));
	}

	for (int i = 0; i < 100; i++) {
		((Vec4*)dat)[IDX3D(i, 100, 100, w, h)] = { 1,0,0,1 };
	}

	view3d->update3dImage(w, h, d, dat);
	delete[] dat;
}

int main(int, char**)
{
#ifdef DETECTMEMLEAK
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

	engine.seed(12);

	//Section for variables
	//float pcLinesAlpha = 1.0f;
	//float pcLinesAlphaCpy = pcLinesAlpha;									//Contains alpha of last fram
	char pcFilePath[200] = {};
	char pcDrawListName[200] = {};
	
	
	//std::vector<float*> pcData = std::vector<float*>();					//Contains all data
	bool pcPlotRender = false;												//If this is true, the pc Plot is rendered in the next frame
	int pcPlotSelectedDrawList = -1;										//Contains the index of the drawlist that is currently selected
	int pcPlotPreviousSlectedDrawList = -1;									//Index of the previously selected drawlist
	bool addIndeces = false;

	// Setup GLFW window
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		return 1;

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "Parallel Coordinates Viewer", NULL, NULL);

	// Setup Drag and drop callback
	glfwSetDropCallback(window, drop_callback);

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
		createPcPlotCommandPool();
		createPcPlotRenderPass();
		createPcPlotHistoPipeline();
		createPcPlotImageView();
		createPcPlotPipeline();
		createPcPlotFramebuffer();

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

		

		//beginning the command buffer
		VkCommandPool command_pool = wd->Frames[wd->FrameIndex].CommandPool;
		VkCommandBuffer command_buffer = wd->Frames[wd->FrameIndex].CommandBuffer;

		err = vkResetCommandPool(g_Device, command_pool, 0);
		check_vk_result(err);
		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		err = vkBeginCommandBuffer(command_buffer, &begin_info);
		check_vk_result(err);

		//now using the memory barrier to transition image state
		VkImageMemoryBarrier use_barrier[1] = {};
		use_barrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		use_barrier[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		use_barrier[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		use_barrier[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		use_barrier[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		use_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		use_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		use_barrier[0].image = g_PcPlot;
		use_barrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		use_barrier[0].subresourceRange.levelCount = 1;
		use_barrier[0].subresourceRange.layerCount = 1;
		vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, use_barrier);

		//transition of the densitiy image
		VkUtil::transitionImageLayout(command_buffer, g_PcPlotDensityImageCopy, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		//ending the command buffer and submitting it
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
	}

#ifdef RENDER3D
	{//creating the 3d viewer and its descriptor set
		view3d = new View3d(800, 800, g_Device, g_PhysicalDevice, g_PcPlotCommandPool, g_Queue, g_DescriptorPool);
		view3d->setImageDescriptorSet((VkDescriptorSet)ImGui_ImplVulkan_AddTexture(view3d->getImageSampler(), view3d->getImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, g_Device, g_DescriptorPool));
	}
#endif

#ifdef NODEVIEW
	{//creating the node viewer and its descriptor set
		nodeViewer = new NodeViewer(800, 800, g_Device, g_PhysicalDevice, g_PcPlotCommandPool, g_Queue, g_DescriptorPool);
		nodeViewer->setImageDescSet((VkDescriptorSet)ImGui_ImplVulkan_AddTexture(nodeViewer->getImageSampler(), nodeViewer->getImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, g_Device, g_DescriptorPool));
		nodeViewer->addSphere(1.0f, glm::vec4( 1,1,1,1 ), glm::vec3( 0,0,0 ));
		nodeViewer->render();
	}
#endif

	{//creating the settngs manager
		settingsManager = new SettingsManager();
	}

	io.ConfigWindowsMoveFromTitleBarOnly = true;
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

		//Check if a drawlist color changed
		for (DrawList& ds : g_PcPlotDrawLists) {
			if (ds.color != ds.prefColor) {
				pcPlotRender = true;
				ds.prefColor = ds.color;
			}
		}

		//check if a path was dropped in the application
		if (pathDropped && !addIndeces) {
			ImGui::OpenPopup("OPENDATASET");
			if (ImGui::BeginPopupModal("OPENDATASET", NULL, ImGuiWindowFlags_AlwaysAutoResize))
			{
				ImGui::Text("Do you really want to open this Dataset:");
				ImGui::Text(droppedPaths.front().c_str());
				ImGui::Separator();

				if (ImGui::Button("Open", ImVec2(120, 0))) {
					ImGui::CloseCurrentPopup();
					openDataset(droppedPaths.front().c_str());
					droppedPaths.clear();
					delete[] createDLForDrop;
					createDLForDrop = NULL;
					pathDropped = false;
				}
				ImGui::SetItemDefaultFocus();
				ImGui::SameLine();
				if (ImGui::Button("Cancel", ImVec2(120, 0))) { 
					ImGui::CloseCurrentPopup(); 
					droppedPaths.clear();
					delete[] createDLForDrop;
					createDLForDrop = NULL;
					pathDropped = false;
				}
				ImGui::EndPopup();
			}
		}

		// Labels for the titels of the attributes
		// Position calculation for each of the Label
		size_t amtOfLabels = 0;
		for (int i = 0; i < pcAttributes.size(); i++)
			if (pcAttributeEnabled[i])
				amtOfLabels++;

		size_t paddingSide = 10;			//padding from left and right screen border
		size_t gap = (io.DisplaySize.x - 2 * paddingSide) / (amtOfLabels - 1);
		ImVec2 buttonSize = ImVec2(70, 20);
		size_t offset = 0;

#ifdef RENDER3D
		//testwindow for the 3d renderer
		if (ImGui::Begin("3dview", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav)) {
			ImGui::Image((ImTextureID)view3d->getImageDescriptorSet(), ImVec2(800, 800), ImVec2(0, 0), ImVec2(1, 1), ImColor(255, 255, 255, 255), ImColor(255, 255, 255, 128));
			
			if ((ImGui::IsMouseDragging()|| io.MouseWheel)&&ImGui::IsItemHovered()) {
				float mousemovement[3];
				mousemovement[0] = -ImGui::GetMouseDragDelta().x;
				mousemovement[1] = ImGui::GetMouseDragDelta().y;
				mousemovement[2] = io.MouseWheel;
				view3d->updateCameraPos(mousemovement);
				view3d->render();
				err = vkDeviceWaitIdle(g_Device);
				check_vk_result(err);
				ImGui::ResetMouseDragDelta();
			}
		}
		ImGui::End();
#endif

#ifdef NODEVIEW
		//testwindow for the node viewer
		if (ImGui::Begin("NodeViewer", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav)) {
			ImGui::Image((ImTextureID)nodeViewer->getImageDescSet(), ImVec2(800, 800), ImVec2(0, 0), ImVec2(1, 1), ImColor(255, 255, 255, 255), ImColor(255, 255, 255, 128));
			if ((ImGui::IsMouseDragging() || io.MouseWheel) && ImGui::IsItemHovered()) {
				float mousemovement[3];
				mousemovement[0] = -ImGui::GetMouseDragDelta().x;
				mousemovement[1] = ImGui::GetMouseDragDelta().y;
				mousemovement[2] = io.MouseWheel;
				nodeViewer->updateCameraPos(mousemovement);
				nodeViewer->render();
				err = vkDeviceWaitIdle(g_Device);
				check_vk_result(err);
				ImGui::ResetMouseDragDelta();
			}
		}
		ImGui::End();
#endif

		//draw the picture of the plotted pc coordinates In the same window the Labels are put as dragable buttons
		ImVec2 window_pos = ImVec2(0, 0);
		ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
		ImGui::SetNextWindowSize({ io.DisplaySize.x,io.DisplaySize.y });
		ImVec2 picPos;
		bool picHovered;
		ImGui::Begin("Plot", NULL, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoScrollbar);

		//Menu for saving settings
		bool openSave = false, openLoad = false, openAttributesManager = false, saveColor = false, openColorManager = false;
		float color[4];
		if (ImGui::BeginMenuBar()) {
			if (ImGui::BeginMenu("Attribute")) {
				if (ImGui::MenuItem("Save Attributes", "Ctrl+S")&&pcAttributes.size()>0) {
					openSave = true;
				}
				if (ImGui::BeginMenu("Load...")) {
					for (SettingsManager::Setting* s : *settingsManager->getSettingsType("AttributeSetting")) {
						if (ImGui::MenuItem(s->id.c_str())) {
							if (((int*)s->data)[0] != pcAttributes.size()) {
								openLoad = true;
								continue;
							}

							std::vector<Attribute> savedAttr;
							char* d = (char*)s->data + sizeof(int);
							for (int i = 0; i < ((int*)s->data)[0]; i++) {
								Attribute a = {};
								a.name = std::string(d);
								d += a.name.size() + 1;
								a.min = *(float*)d;
								d += sizeof(float);
								a.max = *(float*)d;
								d += sizeof(float);
								savedAttr.push_back(a);
								if (pcAttributes[i].name != savedAttr[i].name) {
									openLoad = true;
									continue;
								}
							}
							int* o = (int*)d;
							bool* act = (bool*)(d + pcAttributes.size() * sizeof(int));
							for (int i = 0; i < pcAttributes.size(); i++) {
								pcAttributes[i] = savedAttr[i];
								pcAttrOrd[i] = o[i];
								pcAttributeEnabled[i] = act[i];
							}
							pcPlotRender = true;
						}
					}
					ImGui::EndMenu();
				}
				if (ImGui::MenuItem("Manage")) {
					openAttributesManager = true;
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Colors")) {
				for (auto& s : *settingsManager->getSettingsType("COLOR")) {
					ImGui::ColorEdit4(s->id.c_str(), (float*)s->data, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
				}
				if (ImGui::MenuItem("Manage")) {
					openColorManager = true;
				}

				ImGui::EndMenu();
			}
			//If a color was dropped open saving popup.
			if (ImGui::BeginDragDropTarget()) {
				if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("_COL4F")) {
					saveColor = true;
					memcpy(color, payload->Data, payload->DataSize);
				}
			}

			if (ImGui::BeginMenu("Global Brushes")) {
				if (ImGui::MenuItem("Activate Global Brushing", "", &toggleGlobalBrushes) && !toggleGlobalBrushes) {
					updateAllActiveIndices();
					pcPlotRender = true;
				}

				if (ImGui::BeginMenu("Brush Combination")) {
					static char* const combinations[] = { "OR","AND" };
					if (ImGui::Combo("brushCombination", &brushCombination, combinations, sizeof(combinations) / sizeof(*combinations))) {
						updateAllActiveIndices();
						pcPlotRender = true;
					}

					ImGui::EndMenu();
				}

				ImGui::EndMenu();
			}
			ImGui::EndMenuBar();
		}
		//popup for saving a new Attribute Setting
		if (openSave) {
			ImGui::OpenPopup("Save attribute setting");
		}
		if (openLoad) {
			ImGui::OpenPopup("Load error");
		}
		if (openAttributesManager) {
			ImGui::OpenPopup("Manage attribute settings");
		}
		if (saveColor) {
			ImGui::OpenPopup("Save color");
		}
		if (openColorManager) {
			ImGui::OpenPopup("Color manager");
		}
		if (ImGui::BeginPopupModal("Manage attribute settings", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
			std::string del;
			for (SettingsManager::Setting* s : *settingsManager->getSettingsType("AttributeSetting")) {
				ImGui::Text(s->id.c_str());
				ImGui::SameLine(200);
				if (ImGui::Button(("Delete##" + s->id).c_str())) {
					del = s->id;
				}
			}
			if (del.size() != 0) {
				settingsManager->deleteSetting(del);
			}
			ImGui::Separator();
			if (ImGui::Button("Close")) {
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}
		if (ImGui::BeginPopupModal("Save attribute setting", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
			ImGui::Text("Enter the name for the new setting");
			static char settingName[100] = {};
			ImGui::InputText("setting name", settingName, sizeof(settingName));

			if (ImGui::Button("Save", ImVec2(120, 0))) {
				ImGui::CloseCurrentPopup();
				//creating the new setting
				uint32_t attributesSize = 2 * sizeof(float) * pcAttributes.size();
				for (Attribute& a : pcAttributes) {
					attributesSize += a.name.size() + 1;
				}
				SettingsManager::Setting s = {};
				s.id = std::string(settingName);
				unsigned char* d = new unsigned char[sizeof(int) + attributesSize + pcAttributes.size() * sizeof(int) + pcAttributes.size()];
				s.byteLength = sizeof(int) + attributesSize + pcAttributes.size() * sizeof(int) + pcAttributes.size();
				s.data = d;
				((int*)d)[0] = pcAttributes.size();
				d += 4;
				//Adding the attributes to the dataarray
				for (int i = 0; i < pcAttributes.size(); i++) {
					memcpy(d, pcAttributes[i].name.data(), pcAttributes[i].name.size());
					d += pcAttributes[i].name.size();
					*d = '\0';
					d++;
					((float*)d)[0] = pcAttributes[i].min;
					((float*)d)[1] = pcAttributes[i].max;
					d += 2 * sizeof(float);
				}
				//adding the attributes order
				for (int i : pcAttrOrd) {
					((int*)d)[0] = i;
					d += sizeof(int);
				}
				//adding attribute activation
				for (int i = 0; i < pcAttributes.size(); i++) {
					*d++ = pcAttributeEnabled[i];
				}
				s.type = "AttributeSetting";
				settingsManager->addSetting(s);

				delete[] s.data;
			}
			ImGui::SetItemDefaultFocus();
			ImGui::SameLine();
			if (ImGui::Button("Cancel", ImVec2(120, 0))) {
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}
		//popup for loading error
		if (ImGui::BeginPopupModal("Load error")) {
			ImGui::Text("Error at loading the current setting");
			if (ImGui::Button("Close", ImVec2(120, 0))) {
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}
		//popup for saving a color
		if (ImGui::BeginPopupModal("Save color", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
			static char colorName[200];
			ImGui::ColorEdit4("preview", color, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar | ImGuiColorEditFlags_NoLabel);
			ImGui::SameLine();
			ImGui::InputText("color name", colorName, 200);
			ImGui::Separator();
			if (ImGui::Button("Save", ImVec2(120, 0))) {
				SettingsManager::Setting s = {};
				s.id = colorName;
				s.type = "COLOR";
				s.byteLength = 16;
				s.data = color;
				settingsManager->addSetting(s);

				ImGui::CloseCurrentPopup();
			}
			ImGui::SameLine();
			if (ImGui::Button("Cancel", ImVec2(120, 0))) {
				ImGui::CloseCurrentPopup();
			}

			ImGui::EndPopup();
		}
		//Popup for managing saved colors
		if (ImGui::BeginPopupModal("Color manager", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
			std::string del;
			for (auto color : *settingsManager->getSettingsType("COLOR")) {
				ImGui::ColorEdit4(color->id.c_str(), (float*)color->data, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
				ImGui::SameLine(200);
				if (ImGui::Button(("Delete##" + color->id).c_str())) {
					del = color->id;
				}
			}
			if (!del.empty()) {
				settingsManager->deleteSetting(del);
			}
			ImGui::Separator();
			if (ImGui::Button("Close")) {
				ImGui::CloseCurrentPopup();
			}

			ImGui::EndPopup();
		}

		//drawing the buttons which can be changed via drag and drop
		int c = 0;		//describing the position of the element in the AttrOrd vector
		int c1 = 0;
		for (auto i : pcAttrOrd) {
			//not creating button for unused Attributes
			if (!pcAttributeEnabled[i]) {
				c++;
				continue;
			}

			std::string name = pcAttributes[i].name;
			if(c1!=0)
				ImGui::SameLine(offset-c1*(buttonSize.x/amtOfLabels));
			ImGui::Button(name.c_str(),buttonSize);

			if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
				int p[] = { c,i };		//holding the index in the pcAttriOrd array and the value of it
				ImGui::SetDragDropPayload("ATTRIBUTE", p ,sizeof(p));
				ImGui::Text("Swap %s", name.c_str());
				ImGui::EndDragDropSource();
			}
			if (ImGui::BeginDragDropTarget()) {
				if (const ImGuiPayload * payload = ImGui::AcceptDragDropPayload("ATTRIBUTE")) {
					int* other = (int*)payload->Data;
					
					switchAttributes(c, other[0], io.KeyCtrl);
					updateAllDrawListIndexBuffer();

					pcPlotRender = true;
				}
			}
			
			c++;
			c1++;
			offset += gap;
		}

		//Adding the drag floats for the max values
		c = 0;
		c1 = 0;
		offset = 0;
		for (auto i : pcAttrOrd) {
			if (!pcAttributeEnabled[i]) {
				c++;
				continue;
			}

			std::string name = "max##";
			name += pcAttributes[i].name;
			ImGui::PushItemWidth(buttonSize.x);
			if(c1!=0)
				ImGui::SameLine(offset - c1 * (buttonSize.x / amtOfLabels));

			if (ImGui::DragFloat(name.c_str(), &pcAttributes[i].max, (pcAttributes[i].max - pcAttributes[i].min) * .001f,0.0f,0.0f,"%.05f")) {
				pcPlotRender = true;
				pcPlotPreviousSlectedDrawList = -1;
			}
			ImGui::PopItemWidth();

			c++;
			c1++;
			offset += gap;
		}

		//drawing the Texture
		picPos = ImGui::GetCursorScreenPos();
		picPos.y += 2;
		ImGui::Image((ImTextureID)g_PcPlotImageDescriptorSet, ImVec2(io.DisplaySize.x - 2 * paddingSide, io.DisplaySize.y * 2/5), ImVec2(0, 0), ImVec2(1, 1), ImColor(255, 255, 255, 255), ImColor(255, 255, 255, 255));
		picHovered = ImGui::IsItemHovered();
		if (pcPlotRender) {
			pcPlotRender = false;
			drawPcPlot(pcAttributes, pcAttrOrd,pcAttributeEnabled, wd);
		}

		//Adding the Drag floats for the min values
		c = 0;
		c1 = 0;
		offset = 0;
		for (auto i : pcAttrOrd) {
			if (!pcAttributeEnabled[i]) {
				c++;
				continue;
			}

			std::string name = "min##";
			name += pcAttributes[i].name;
			ImGui::PushItemWidth(buttonSize.x);
			if (c1 != 0)
				ImGui::SameLine(offset - c1 * (buttonSize.x / amtOfLabels));
			if (ImGui::DragFloat(name.c_str(), &pcAttributes[i].min, (pcAttributes[i].max - pcAttributes[i].min) * .001f, .0f, .0f, "%.05f")) {
				pcPlotRender = true;
				pcPlotPreviousSlectedDrawList = -1;
			}
			ImGui::PopItemWidth();

			c++;
			c1++;
			offset += gap;
		}

		ImVec2 picSize(io.DisplaySize.x - 2 * paddingSide + 5, io.DisplaySize.y * 2 / 5);
		if (toggleGlobalBrushes) {
			//drawing checkboxes for activating brush templates
			ImGui::Separator();
			ImGui::Text("Brush templates (Check attributes to create subspace for which brush templates should be shown)");
			c = 0;
			c1 = 0;
			offset = 0;
			for (auto i : pcAttrOrd) {
				if (!pcAttributeEnabled[i]) {
					c++;
					continue;
				}

				std::string name = "##CB";
				name = pcAttributes[i].name + name;
				if (c1 != 0)
					ImGui::SameLine(offset - c1 * (buttonSize.x / amtOfLabels));
				if (ImGui::Checkbox(name.c_str(), &brushTemplateAttrEnabled[i])) {
					if (selectedTemplateBrush != -1) {
						globalBrushes.pop_back();
						selectedTemplateBrush = -1;
						pcPlotRender = true;
					}
					templateBrushes.clear();
					//searching all drawlists for template brushes, and adding the template brushes which fit the subspace selected
					std::set<std::string> subspace;
					for (int i = 0; i < pcAttributes.size(); i++) {
						if (pcAttributeEnabled[i] && brushTemplateAttrEnabled[i]) {
							subspace.insert(pcAttributes[i].name);
						}
					}
					for (const DataSet& ds : g_PcPlotDataSets) {
						if (ds.oneData) {			//oneData indicates a .dlf data -> template brushes are available
							for (const TemplateList& tl : ds.drawLists) {
								//checking if the template list is in the correct subspace and if so adding it to the template Brushes
								std::string s = tl.name.substr(tl.name.find_first_of('[') + 1, tl.name.find_last_of(']') - tl.name.find_first_of('[') - 1);
								std::set<std::string> sub;
								std::size_t current, previous = 0;
								current = s.find(',');
								while (current != std::string::npos) {
									sub.insert(s.substr(previous + 1, current - previous - 2));
									previous = current + 1;
									current = s.find(',', previous);
								}
								sub.insert(s.substr(previous + 1, s.size() - previous - 2));

								if (sub == subspace) {
									TemplateBrush t = {};
									t.name = tl.name;
									for (int i = 0; i < pcAttributes.size(); i++) {
										if (brushTemplateAttrEnabled[i]) {
											t.brushes[i] = tl.minMax[i];
										}
									}
									templateBrushes.push_back(t);
								}
							}
						}
					}
				}

				c++;
				c1++;
				offset += gap;
			}

			//drawing the list for brush templates
			ImGui::BeginChild("brushTemplates", ImVec2(400, 200), true);
			ImGui::Text("Brush Templates");
			ImGui::Separator();
			for (int i = 0; i < templateBrushes.size(); i++) {
				if (ImGui::Selectable(templateBrushes[i].name.c_str(), selectedTemplateBrush == i)) {
					selectedGlobalBrush = -1;
					pcPlotSelectedDrawList = -1;
					if (selectedTemplateBrush != i) {
						if (selectedTemplateBrush != -1)
							globalBrushes.pop_back();
						selectedTemplateBrush = i;
						GlobalBrush preview;
						preview.active = true;
						preview.name = templateBrushes[i].name;
						for (const auto& brush : templateBrushes[i].brushes) {
							preview.brushes[brush.first] = std::pair<unsigned int, std::pair<float, float>>(currentBrushId++, brush.second);
						}
						globalBrushes.push_back(preview);
						updateAllActiveIndices();
						pcPlotRender = true;
					}
					else {
						selectedTemplateBrush = -1;
						globalBrushes.pop_back();
						updateAllActiveIndices();
						pcPlotRender = true;
					}
				}
				if (ImGui::IsItemClicked(2) && selectedTemplateBrush == i) {//creating a permanent Global Brush
					selectedGlobalBrush = globalBrushes.size() - 1;
					selectedTemplateBrush = -1;
				}
			}
			ImGui::EndChild();
			ImGui::SameLine();
			ImGui::BeginChild("GlobalBrushes", ImVec2(400, 200), true);
			ImGui::Text("Global Brushes");
			ImGui::Separator();
			bool popEnd = false;
			for (int i = 0; i < globalBrushes.size(); i++) {
				if (ImGui::Selectable(globalBrushes[i].name.c_str(), selectedGlobalBrush == i)) {
					pcPlotSelectedDrawList = -1;
					if (selectedGlobalBrush != i) {
						selectedGlobalBrush = i;
					}
					else {
						selectedGlobalBrush = -1;
					}
					if (selectedTemplateBrush != -1) {
						selectedTemplateBrush = -1;
						popEnd = true;
					}
				}
				if (ImGui::IsItemClicked(1)) {
					selectedTemplateBrush = -1;
					if (selectedGlobalBrush == i) {
						selectedGlobalBrush = -1;
					}
					if (popEnd) {
						if (globalBrushes.size() == 2) {
							globalBrushes[i] = globalBrushes[globalBrushes.size() - 1];
							globalBrushes.pop_back();
						}
						else {
							globalBrushes[i] = globalBrushes[globalBrushes.size() - 2];
							globalBrushes[globalBrushes.size() - 2] = globalBrushes[globalBrushes.size() - 1];
							globalBrushes.pop_back();
						}
					}
					else {
						globalBrushes[i] = globalBrushes[globalBrushes.size() - 1];
						globalBrushes.pop_back();
						updateAllActiveIndices();
						pcPlotRender = true;
					}
				}
			}
			if (popEnd) {
				globalBrushes.pop_back();
				if (selectedGlobalBrush == globalBrushes.size())
					selectedGlobalBrush = -1;
				updateAllActiveIndices();
				pcPlotRender = true;
			}
			ImGui::EndChild();

			//Statistics for global brushes
			ImGui::SameLine();
			ImGui::BeginChild("Brush statistics", ImVec2(0, 200), true);
			ImGui::Text("Brush statistics: Percentage of lines kept after brushing");
			ImGui::Separator();
			//test for vertical Histogramm
			const float histogrammdata[10] = { 0,.1f,.2f,.3f,.4f,.5f,.6f,.7f,.8f,.9f };
			//int hover = ImGui::PlotHistogramVertical("##testHistogramm", histogrammdata, 10, 0, NULL, 0, 1.0f, ImVec2(50, 200));
			for (auto& brush : globalBrushes) {
				ImGui::BeginChild(("##brushStat" + brush.name).c_str(),ImVec2(200,0),true);
				ImGui::Text(brush.name.c_str());
				float lineHeight = ImGui::GetTextLineHeightWithSpacing();
				static std::vector<float> ratios;
				ImVec2 defaultCursorPos = ImGui::GetCursorPos();
				ImVec2 cursorPos = defaultCursorPos;
				cursorPos.x += 75 + ImGui::GetStyle().ItemInnerSpacing.x;
				ratios.clear();
				for (auto& ratio : brush.lineRatios) {
					ratios.push_back(ratio.second);
					ImGui::SetCursorPos(cursorPos);
					ImGui::Text(ratio.first.c_str());
					cursorPos.y += lineHeight;
				}
				ImGui::SetCursorPos(defaultCursorPos);
				int hover = ImGui::PlotHistogramVertical(("##histo"+brush.name).c_str(), ratios.data(), ratios.size(), 0, NULL, 0, 1.0f, ImVec2(75, lineHeight*ratios.size()));
				if (hover != -1) {
					ImGui::BeginTooltip();
					ImGui::Text("%2.1f%%", ratios[hover] * 100);
					ImGui::EndTooltip();
				}


				ImGui::EndChild();
				ImGui::SameLine();
			}

			ImGui::EndChild();
			
			gap = (picSize.x - ((drawHistogramm) ? histogrammWidth / 2.0 * (double)picSize.x : 0)) / (amtOfLabels - 1);
			//drawing axis lines
			if (enableAxisLines) {
				for (int i = 0; i < amtOfLabels; i++) {
					float x = picPos.x + i * gap + ((drawHistogramm) ? (histogrammWidth / 4.0 * picSize.x) : 0);
					ImVec2 a(x, picPos.y);
					ImVec2 b(x, picPos.y + picSize.y);
					ImGui::GetForegroundDrawList()->AddLine(a, b, IM_COL32((1 - PcPlotBackCol.x) * 255, (1 - PcPlotBackCol.y) * 255, (1 - PcPlotBackCol.z) * 255, 255), 1);
				}
			}

			//drawing the global brush
			//global brushes currently only support change of brush but no adding of new brushes or deletion of brushes
			if (selectedGlobalBrush != -1) {
				for (auto& brush : globalBrushes[selectedGlobalBrush].brushes) {
					if (!pcAttributeEnabled[brush.first])
						continue;

					ImVec2 mousePos = ImGui::GetIO().MousePos;
					float x = gap * placeOfInd(brush.first) + picPos.x - BRUSHWIDTH / 2 + ((drawHistogramm) ? (histogrammWidth / 4.0 * picSize.x) : 0);
					float y = ((brush.second.second.second - pcAttributes[brush.first].max) / (pcAttributes[brush.first].min - pcAttributes[brush.first].max)) * picSize.y + picPos.y;
					float width = BRUSHWIDTH;
					float height = (brush.second.second.second - brush.second.second.first) / (pcAttributes[brush.first].max - pcAttributes[brush.first].min) * picSize.y;
					bool hover = mousePos.x > x&& mousePos.x<x + width && mousePos.y>y&& mousePos.y < y + height;
					//edgeHover = 0 -> No edge is hovered
					//edgeHover = 1 -> Top edge is hovered
					//edgeHover = 2 -> Bot edge is hovered
					int edgeHover = mousePos.x > x&& mousePos.x<x + width && mousePos.y>y - EDGEHOVERDIST && mousePos.y < y + EDGEHOVERDIST ? 1 : 0;
					edgeHover = mousePos.x > x&& mousePos.x<x + width && mousePos.y>y - EDGEHOVERDIST + height && mousePos.y < y + EDGEHOVERDIST + height ? 2 : edgeHover;
					ImGui::GetForegroundDrawList()->AddRect(ImVec2(x, y), ImVec2(x + width, y + height), IM_COL32(30, 0, 200, 255), 1, ImDrawCornerFlags_All, 5);
					//set mouse cursor
					if (edgeHover || brushDragId != -1) {
						ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
					}
					if (hover) {
						ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
					}
					//activate dragging of edge
					if (edgeHover && ImGui::GetIO().MouseClicked[0]) {
						brushDragId = brush.second.first;
						brushDragMode = edgeHover;
					}
					if (hover && ImGui::GetIO().MouseClicked[0]) {
						brushDragId = brush.second.first;
						brushDragMode = 0;
					}
					//drag edge
					if (brushDragId == brush.second.first && ImGui::GetIO().MouseDown[0]) {
						if (brushDragMode == 0) {
							float delta = ImGui::GetIO().MouseDelta.y / picSize.y * (pcAttributes[brush.first].max - pcAttributes[brush.first].min);
							brush.second.second.second -= delta;
							brush.second.second.first -= delta;
						}
						else if (brushDragMode == 1) {
							brush.second.second.second = ((mousePos.y - picPos.y) / picSize.y) * (pcAttributes[brush.first].min - pcAttributes[brush.first].max) + pcAttributes[brush.first].max;
						}
						else {
							brush.second.second.first = ((mousePos.y - picPos.y) / picSize.y) * (pcAttributes[brush.first].min - pcAttributes[brush.first].max) + pcAttributes[brush.first].max;
						}

						//switching edges if max value of brush is smaller than min value
						if (brush.second.second.second < brush.second.second.first) {
							float tmp = brush.second.second.second;
							brush.second.second.second = brush.second.second.first;
							brush.second.second.first = tmp;
							brushDragMode = (brushDragMode == 1) ? 2 : 1;
						}

						if (ImGui::GetIO().MouseDelta.y) {
							updateAllActiveIndices();
							pcPlotRender = true;
						}
					}
					//release edge
					if (brushDragId == brush.second.first && ImGui::GetIO().MouseReleased[0]) {
						brushDragId = -1;
						updateAllActiveIndices();
						pcPlotRender = true;
					}
				}
			}

			//drawing the template brush, these are not changeable
			if (selectedTemplateBrush != -1) {
				for (const auto& brush : globalBrushes.back().brushes) {
					if (!pcAttributeEnabled[brush.first])
						continue;

					float x = gap * placeOfInd(brush.first) + picPos.x - BRUSHWIDTH / 2 + ((drawHistogramm) ? (histogrammWidth / 4.0 * picSize.x) : 0);
					float y = ((brush.second.second.second - pcAttributes[brush.first].max) / (pcAttributes[brush.first].min - pcAttributes[brush.first].max)) * picSize.y + picPos.y;
					float width = BRUSHWIDTH;
					float height = (brush.second.second.second - brush.second.second.first) / (pcAttributes[brush.first].max - pcAttributes[brush.first].min) * picSize.y;
					ImGui::GetForegroundDrawList()->AddRect(ImVec2(x, y), ImVec2(x + width, y + height), IM_COL32(30, 0, 200, 150), 1, ImDrawCornerFlags_All, 5);
				}
			}
		}

		//drawing the brush windows
		if (pcPlotSelectedDrawList != -1) {
			//getting the drawlist;
			DrawList* dl = 0;
			uint32_t c = 0;
			for (DrawList& d : g_PcPlotDrawLists) {
				if (c == pcPlotSelectedDrawList) {
					dl = &d;
					break;
				}
				c++;
			}

			for (int i = 0; i < pcAttributes.size(); i++) {
				if (!pcAttributeEnabled[i])
					continue;
				
				int del = -1;
				int ind = 0;
				bool brushHover = false;
				
				ImVec2 mousePos = ImGui::GetIO().MousePos;
				float x = gap * placeOfInd(i) + picPos.x - BRUSHWIDTH / 2 + ((drawHistogramm) ? (histogrammWidth / 4.0 * picSize.x) : 0);
				//drawing the brushes as foreground objects
				for (Brush& b : dl->brushes[i]) {
					float y = ((b.minMax.second - pcAttributes[i].max) / (pcAttributes[i].min - pcAttributes[i].max)) * picSize.y + picPos.y;
					float width = BRUSHWIDTH;
					float height = (b.minMax.second - b.minMax.first) / (pcAttributes[i].max - pcAttributes[i].min) * picSize.y;
					ImGui::GetForegroundDrawList()->AddRect(ImVec2(x,y), ImVec2(x + width, y + height), IM_COL32(200, 30, 0, 255), 1, ImDrawCornerFlags_All, 5);
					
					bool hover = mousePos.x > x&& mousePos.x<x + width && mousePos.y>y&& mousePos.y < y + height;
					//edgeHover = 0 -> No edge is hovered
					//edgeHover = 1 -> Top edge is hovered
					//edgeHover = 2 -> Bot edge is hovered
					int edgeHover = mousePos.x > x&& mousePos.x<x + width && mousePos.y>y - EDGEHOVERDIST && mousePos.y < y + EDGEHOVERDIST ? 1 : 0;
					edgeHover = mousePos.x > x&& mousePos.x<x + width && mousePos.y>y - EDGEHOVERDIST + height && mousePos.y < y + EDGEHOVERDIST + height ? 2 : edgeHover;
					brushHover |= hover || edgeHover;

					//set mouse cursor
					if (edgeHover || brushDragId != -1) {
						ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
					}
					if (hover) {
						ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
					}
					//activate dragging of edge
					if (edgeHover && ImGui::GetIO().MouseClicked[0]) {
						brushDragId = b.id;
						brushDragMode = edgeHover;
					}
					if (hover && ImGui::GetIO().MouseClicked[0]) {
						brushDragId = b.id;
						brushDragMode = 0;
					}
					//drag edge
					if (brushDragId == b.id && ImGui::GetIO().MouseDown[0]) {
						if (brushDragMode == 0) {
							float delta = ImGui::GetIO().MouseDelta.y / picSize.y * (pcAttributes[i].max - pcAttributes[i].min);
							b.minMax.second -= delta;
							b.minMax.first -= delta;
						}
						else if (brushDragMode == 1) {
							b.minMax.second = ((mousePos.y - picPos.y) / picSize.y) * (pcAttributes[i].min - pcAttributes[i].max) + pcAttributes[i].max;
						}
						else {
							b.minMax.first = ((mousePos.y - picPos.y) / picSize.y) * (pcAttributes[i].min - pcAttributes[i].max) + pcAttributes[i].max;
						}

						//switching edges if max value of brush is smaller than min value
						if (b.minMax.second < b.minMax.first) {
							float tmp = b.minMax.second;
							b.minMax.second = b.minMax.first;
							b.minMax.first = tmp;
							brushDragMode = (brushDragMode == 1) ? 2 : 1;
						}

						if (ImGui::GetIO().MouseDelta.y) {
							updateActiveIndices(*dl);
							pcPlotRender = true;
						}
					}
					//release edge
					if (brushDragId == b.id && ImGui::GetIO().MouseReleased[0]) {
						brushDragId = -1;
						updateActiveIndices(*dl);
						pcPlotRender = true;
					}

					//check for deletion of brush
					if (ImGui::GetIO().MouseClicked[1] && hover)
						del = ind;

					ind++;
				}

				//deleting a brush
				if (del != -1) {
					dl->brushes[i][del] = dl->brushes[i][dl->brushes[i].size() - 1];
					dl->brushes[i].pop_back();
					del = -1;
					updateActiveIndices(*dl);
					pcPlotRender = true;
				}

				//create a new brush
				bool axisHover = mousePos.x > x&& mousePos.x < x + BRUSHWIDTH && mousePos.y > picPos.y && mousePos.y < picPos.y + picSize.y;
				if (!brushHover && axisHover && brushDragId == -1) {
					ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

					if (ImGui::GetIO().MouseClicked[0]) {
						Brush temp = {};
						temp.id = currentBrushId++;
						temp.minMax.first = ((mousePos.y - picPos.y) / picSize.y) * (pcAttributes[i].min - pcAttributes[i].max) + pcAttributes[i].max;
						temp.minMax.second = temp.minMax.first;
						brushDragId = temp.id;
						brushDragMode = 1;
						dl->brushes[i].push_back(temp);
					}
				}
			}
		}

		//Settings section
		ImGui::BeginChild("Settings", ImVec2(500, -1), true);
		ImGui::Text("Settings");
		ImGui::Separator();

		ImGui::Text("Histogramm Settings:");
		if (ImGui::Checkbox("Draw Histogramm", &drawHistogramm)) {
			pcPlotRender = true;
		}
		if (ImGui::SliderFloat("Histogramm Width", &histogrammWidth, 0, .5) && drawHistogramm) {
			pcPlotRender = true;
		}
		if (ImGui::ColorEdit4("Histogramm Background", &histogrammBackCol.x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar) && drawHistogramm) {
			pcPlotRender = true;
		}
		if (ImGui::Checkbox("Show Density", &histogrammDensity) && drawHistogramm) {
			pcPlotRender = true;
		}
		if (ImGui::ColorEdit4("Density Background", &densityBackCol.x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar) && drawHistogramm) {
			pcPlotRender = true;
		}
		ImGui::Separator();
		
		ImGui::Text("Parallel Coordinates Settings:");

		if (ImGui::SliderFloat("Blur radius", &densityRadius, .01f, .5f)) {
			uploadDensityUiformBuffer();
			pcPlotRender = true;
		}

		if (ImGui::Checkbox("Show PcPlot Density", &pcPlotDensity)) {
			pcPlotRender = true;
		}

		if (ImGui::Checkbox("Enable density mapping", &enableDensityMapping)) {
			if (pcAttributes.size()) {
				uploadDensityUiformBuffer();
				pcPlotRender = true;
			}
		}

		if (ImGui::Checkbox("Enable median calc", &calculateMedians)) {
			
		}

		if (ImGui::Checkbox("Enable brushing", &enableBrushing)) {
			for (DrawList& dl : g_PcPlotDrawLists) {
				updateActiveIndices(dl);
			}
			pcPlotRender = true;
		}

		if (ImGui::SliderFloat("Median line width", &medianLineWidth, .5f, 20.0f)) {
			pcPlotRender = true;
		}
		
		if (ImGui::ColorEdit4("Plot Background Color", &PcPlotBackCol.x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar)) {
			pcPlotRender = true;
		}

		if (ImGui::Checkbox("Render Splines", &g_RenderSplines)) {
			updateAllDrawListIndexBuffer();
			pcPlotRender = true;
		}

		if (ImGui::Checkbox("Enable Axis Lines", &enableAxisLines)) {
		}

		ImGui::Separator();

		ImGui::Text("Data Settings:");

		for (int i = 0; i < pcAttributes.size(); i++) {
			if (ImGui::Checkbox(pcAttributes[i].name.c_str(), &pcAttributeEnabled[i])) {
				updateAllDrawListIndexBuffer();
				pcPlotRender = true;
			}
		}

		ImGui::InputText("Directory Path", pcFilePath, 200);

		ImGui::SameLine();

		//Opening a new Dataset into the Viewer
		if (ImGui::Button("Open")) {
			openDataset(pcFilePath);
		}
		ImGui::EndChild();

		//DataSets, from which draw lists can be created
		ImGui::SameLine();
		ImGui::BeginChild("DataSets", ImVec2((io.DisplaySize.x - 500) / 2, -1), true);

		DataSet* destroySet = NULL;
		bool destroy = false;

		ImGui::Text("Datasets");
		ImGui::Separator();
		for (DataSet& ds : g_PcPlotDataSets) {
			if (ImGui::TreeNode(ds.name.c_str())) {
				static const TemplateList* convert = nullptr;
				for (const TemplateList& tl : ds.drawLists) {
					if (ImGui::Button(tl.name.c_str())) {
						ImGui::OpenPopup(tl.name.c_str());
						std::strcpy(pcDrawListName, tl.name.c_str());
					}
					if (ImGui::IsItemClicked(1)) {
						ImGui::OpenPopup("CONVERTTOBRUSH");
						convert = &tl;
					}
					if (ImGui::BeginPopupModal(tl.name.c_str(), NULL, ImGuiWindowFlags_AlwaysAutoResize))
					{
						ImGui::Text((std::string("Creating a drawing list from ")+tl.name+"\n\n").c_str());
						ImGui::Separator();
						ImGui::InputText("Drawlist Name", pcDrawListName, 200);

						if (ImGui::Button("Create", ImVec2(120, 0))) { 
							ImGui::CloseCurrentPopup(); 
							
							createPcPlotDrawList(tl, ds, pcDrawListName);
							updateActiveIndices(g_PcPlotDrawLists.back());
							pcPlotRender = true;
						}
						ImGui::SetItemDefaultFocus();
						ImGui::SameLine();
						if (ImGui::Button("Cancel", ImVec2(120, 0))) { ImGui::CloseCurrentPopup(); }
						ImGui::EndPopup();
					}
				}
				//Popup for converting a template list to brush
				bool convertToGlobalBrush = false;
				bool convertToLokalBrush = false;
				if (ImGui::BeginPopup("CONVERTTOBRUSH")) {
					if (ImGui::MenuItem("Convert to global brush")) {
						convertToGlobalBrush = true;
						ImGui::CloseCurrentPopup();
					}
					if (ImGui::MenuItem("Convert to lokal brush")) {
						convertToLokalBrush = true;
						ImGui::CloseCurrentPopup();
					}
					ImGui::EndPopup();
				}

				//Popup for converting a template list to global brush
				if (convertToGlobalBrush) {
					ImGui::OpenPopup("CONVERTTOGLOBALBRUSH");
				}
				if (ImGui::BeginPopupModal("CONVERTTOGLOBALBRUSH")) {
					static char n[200] = {};
					ImGui::InputText("name of global brush", n, 200);
					ImGui::Text("Please select the axes which should be brushed");
					for (int i = 0; i < pcAttributes.size(); i++) {
						ImGui::Checkbox(pcAttributes[i].name.c_str(), &activeBrushAttributes[i]);
						if (i != pcAttributes.size() - 1) {
							ImGui::SameLine();
						}
					}

					ImGui::Separator();
					if (ImGui::Button("Create")) {
						GlobalBrush brush = {};
						brush.name = std::string(n);
						brush.active = true;
						for (int i = 0; i < pcAttributes.size(); i++) {
							if (activeBrushAttributes[i]) {
								brush.brushes[i] = std::pair<int, std::pair<float, float>>(currentBrushId++, convert->minMax[i]);
							}
						}
						globalBrushes.push_back(brush);

						ImGui::CloseCurrentPopup();
					}
					ImGui::SameLine();
					if (ImGui::Button("Cancel")) {
						ImGui::CloseCurrentPopup();
					}
					ImGui::EndPopup();
				}

				//Popup for converting a template list to a lokal brush
				if (convertToLokalBrush) {
					ImGui::OpenPopup("CONVERTTOLOKALBRUSH");
				}
				if (ImGui::BeginPopupModal("CONVERTTOLOKALBRUSH")) {
					static char n[200] = {};
					ImGui::InputText("name of resulting drawlist", n, 200);
					ImGui::Text("Please select the axis to which the brushes shall be applied");
					for (int i = 0; i < pcAttributes.size(); i++) {
						ImGui::Checkbox(pcAttributes[i].name.c_str(), &activeBrushAttributes[i]);
						if (i != pcAttributes.size() - 1) {
							ImGui::SameLine();
						}
					}

					ImGui::Separator();
					if (ImGui::Button("Create")) {
						createPcPlotDrawList(ds.drawLists.front(), ds, n);
						DrawList& dl = g_PcPlotDrawLists.back();
						for (int i = 0; i < pcAttributes.size(); i++) {
							if (activeBrushAttributes[i]) {
								Brush b = {};
								b.id = currentBrushId++;
								b.minMax = convert->minMax[i];
								dl.brushes[i].push_back(b);
							}
						}
						updateActiveIndices(dl);
						ImGui::CloseCurrentPopup();
					}
					ImGui::SameLine();
					if (ImGui::Button("Cancel")) {
						ImGui::CloseCurrentPopup();
					}

					ImGui::EndPopup();
				}

				//Popup for adding a custom index list
				if (ImGui::Button("ADDINDEXLIST")) {
					ImGui::OpenPopup("ADDINDEXLIST");
					addIndeces = true;
				}
				if (ImGui::BeginPopupModal("ADDINDEXLIST", NULL, ImGuiWindowFlags_AlwaysAutoResize))
				{
					ImGui::Text("Path for the new Indexlist (Alternativley drag and drop here):");
					ImGui::InputText("Path", pcFilePath, 200);
					ImGui::Separator();

					ImGui::BeginChild("ScrollingRegion", ImVec2(0, 400), false, ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysAutoResize);

					if (droppedPaths.size() == 0) {
						ImGui::Text("Drag and drop indexlists here to open them.");
					}
					else {
						ImGui::SliderFloat("Default Alpha Value", &alphaDrawLists, .0f, 1.0f);
					}

					for (int i = 0; i < droppedPaths.size();i++) {
						ImGui::Text(droppedPaths[i].c_str());
						ImGui::SameLine();
						ImGui::Checkbox(("##"+droppedPaths[i]).c_str(), &createDLForDrop[i]);
					}

					ImGui::EndChild();

					if (ImGui::Button("Add Indeces", ImVec2(120, 0))) {
						ImGui::CloseCurrentPopup();
						if (droppedPaths.size() == 0)
							addIndecesToDs(ds, pcFilePath);
						else {
							addMultipleIndicesToDs(ds);
							pcPlotRender = true;
						}
						droppedPaths.clear();
						delete[] createDLForDrop;
						createDLForDrop = NULL;
						pathDropped = false;
						addIndeces = false;
					}
					ImGui::SetItemDefaultFocus();
					ImGui::SameLine();
					if (ImGui::Button("Cancel", ImVec2(120, 0))) { 
						ImGui::CloseCurrentPopup();
						droppedPaths.clear();
						delete[] createDLForDrop;
						createDLForDrop = NULL;
						pathDropped = false;
						addIndeces = false;
					}
					ImGui::EndPopup();
				}

				//Popup for delete menu
				if (ImGui::Button("DELETE"))
					ImGui::OpenPopup("DELETE");
				if (ImGui::BeginPopupModal("DELETE", NULL, ImGuiWindowFlags_AlwaysAutoResize))
				{
					ImGui::Text("Do you really want to delete this data set?");
					ImGui::Separator();

					if (ImGui::Button("Delete", ImVec2(120, 0))) {
						ImGui::CloseCurrentPopup();
						destroySet = &ds;
						destroy = true;
						pcPlotRender = true;
					}
					ImGui::SetItemDefaultFocus();
					ImGui::SameLine();
					if (ImGui::Button("Cancel", ImVec2(120, 0))) { ImGui::CloseCurrentPopup(); }
					ImGui::EndPopup();
				}
				ImGui::TreePop();
			}
		}
		ImGui::EndChild();
		//Destroying a dataset if it was selected
		if(destroy)
			destroyPcPlotDataSet(*destroySet);

		//Showing the Drawlists
		DrawList* changeList;
		destroy = false;
		bool up = false;
		bool down = false;

		ImGui::SameLine();
		ImGui::BeginChild("DrawLists", ImVec2((io.DisplaySize.x - 500) / 2, -1), true);

		ImGui::Text("Draw lists");
		ImGui::Separator();
		int count = 0;

		ImGui::Columns(9, "Columns", false);
		ImGui::SetColumnWidth(0, 250);
		ImGui::SetColumnWidth(1, 25);
		ImGui::SetColumnWidth(2, 25);
		ImGui::SetColumnWidth(3, 25);
		ImGui::SetColumnWidth(4, 25);
		ImGui::SetColumnWidth(5, 25);
		ImGui::SetColumnWidth(6, 25);
		ImGui::SetColumnWidth(7, 100);
		ImGui::SetColumnWidth(8, 25);
		
		//showing texts to describe whats in the corresponding column
		ImGui::Text("Drawlist Name");
		ImGui::NextColumn();
		ImGui::Text("Draw");
		ImGui::NextColumn();
		ImGui::Text("");
		ImGui::NextColumn();
		ImGui::Text("");
		ImGui::NextColumn();
		ImGui::Text("Delete");
		ImGui::NextColumn();
		ImGui::Text("Color");
		ImGui::NextColumn();
		ImGui::Text("Histo");
		ImGui::NextColumn();
		ImGui::Text("Median");
		ImGui::NextColumn();
		ImGui::Text("MColor");
		ImGui::NextColumn();
		for (DrawList& dl : g_PcPlotDrawLists) {
			if (ImGui::Selectable(dl.name.c_str(), count == pcPlotSelectedDrawList)) {
				if (count == pcPlotSelectedDrawList)
					pcPlotSelectedDrawList = -1;
				else
					pcPlotSelectedDrawList = count;
			}
			if (ImGui::IsItemHovered() && io.MouseClicked[1]) {
				ImGui::OpenPopup(("sendTo3d"+dl.name).c_str());
			}
			if (ImGui::BeginPopup(("sendTo3d" + dl.name).c_str())) {
				for (int i = 0; i < pcAttributes.size();i++) {
					if (!pcAttributeEnabled[i])
						continue;
					if (ImGui::MenuItem(("Render "+ pcAttributes[i].name).c_str())) {
						ImGui::CloseCurrentPopup();
						uploadDrawListTo3dView(dl, pcAttributes[i].name, "a", "b", "c");
					}
				}
				
				ImGui::EndPopup();
			}
			ImGui::NextColumn();

			if (ImGui::Checkbox(("##" + dl.name).c_str(), &dl.show)) {
				pcPlotRender = true;
			}
			ImGui::NextColumn();

			float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
			if (ImGui::ArrowButton((std::string("##u")+dl.name).c_str(), ImGuiDir_Up)) {
				changeList = &dl;
				up = true;
				pcPlotRender = true;
			}
			ImGui::NextColumn();

			if (ImGui::ArrowButton((std::string("##d") + dl.name).c_str(), ImGuiDir_Down)) {
				changeList = &dl;
				down = true;
				pcPlotRender = true;
			}
			ImGui::NextColumn();

			if (ImGui::Button((std::string("X##") + dl.name).c_str())) {
				if (count == pcPlotSelectedDrawList) {
					pcPlotSelectedDrawList = -1;
				}
				else if (count < pcPlotSelectedDrawList) {
					pcPlotSelectedDrawList--;
				}
				changeList = &dl;
				destroy = true;
				pcPlotRender = true;
			}
			ImGui::NextColumn();

			int misc_flags = ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar;
			ImGui::ColorEdit4((std::string("Color##") + dl.name).c_str(), (float*)& dl.color, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | misc_flags);
			ImGui::NextColumn();

			if (ImGui::Checkbox((std::string("##dh") + dl.name).c_str(), &dl.showHistogramm) && drawHistogramm) {
				pcPlotRender = true;
			}
			ImGui::NextColumn();

			const char* entrys[] = { "No Median","Synthetic","Arithmetic","Geometric" };
			if (ImGui::Combo((std::string("##c") + dl.name).c_str(), &dl.activeMedian, entrys, sizeof(entrys) / sizeof(*entrys))) {
				pcPlotRender = true;
			}
			ImGui::NextColumn();

			if (ImGui::ColorEdit4((std::string("##CMed") + dl.name).c_str(), (float*)& dl.medianColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | misc_flags)) {
				pcPlotRender = true;
			}
			ImGui::NextColumn();

			count++;
		}
		ImGui::EndChild();
		ImGui::End();
		//main window now closed

		if (destroy) {
			removePcPlotDrawList(*changeList);
		}
		if (up) {
			auto it = g_PcPlotDrawLists.begin();
			while (it!=g_PcPlotDrawLists.end() && it->name != changeList->name)
				++it;
			if (it != g_PcPlotDrawLists.begin()) {
				auto itu = it;
				itu--;
				std::swap(*it, *itu);
			}
		}
		if (down) {
			auto it = g_PcPlotDrawLists.begin();
			while (it != g_PcPlotDrawLists.end() && it->name != changeList->name)
				++it;
			if (it->name != g_PcPlotDrawLists.back().name) {
				auto itu = it;
				itu++;
				std::swap(*it, *itu);
			}
		}

#ifdef _DEBUG
		ImGui::ShowDemoWindow(NULL);
#endif

		// Rendering
		ImGui::Render();
		memcpy(&wd->ClearValue.color.float32[0], &clear_color, 4 * sizeof(float));
		FrameRender(wd);

		FramePresent(wd);
	}

	// Cleanup
	if (pcAttributeEnabled)
		delete[] pcAttributeEnabled;
	if (createDLForDrop)
		delete[] createDLForDrop;
	if (brushTemplateAttrEnabled)
		delete[] brushTemplateAttrEnabled;
	if (activeBrushAttributes)
		delete[] activeBrushAttributes;
	

	err = vkDeviceWaitIdle(g_Device);
	check_vk_result(err);

	{//section to cleanup pcPlot
		vkDestroySampler(g_Device, g_PcPlotSampler, nullptr);
		cleanupPcPlotCommandPool();
		cleanupPcPlotFramebuffer();
		cleanupPcPlotPipeline();
		cleanupPcPlotRenderPass();
		cleanupPcPlotImageView();
		cleanupPcPlotDataSets();
		cleanupPcPlotHistoPipeline();
	}

	{//cleanup 3d view
#ifdef RENDER3D
		delete view3d;
#endif
#ifdef NODEVIEW
		delete nodeViewer;
#endif
		delete settingsManager;
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