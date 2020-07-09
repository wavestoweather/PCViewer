/*
This program is written and maintained by Josef Stumpfegger (josefstumpfegger@outlook.de).
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

//enable this define to print the time needed to render the pc Plot
#define PRINTRENDERTIME
//enable this define to print the time since the last fram
//#define PRINTFRAMETIME
//enable to use gpu sorting (Not implemented yet)
//#define GPUSORT

// build the pcViewer with 3d view
#define RENDER3D
//build hte pcViewer with the node viewer
#define BUBBLEVIEW

#ifdef DETECTMEMLEAK
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#include "PCViewer.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_vulkan.h"
#include "imgui/imgui_internal.h"
#include "Color.h"
#include "VkUtil.h"
#include "PCUtil.h"
#include "BubblePlotter.h"
#include "View3d.h"
#include "SpacialData.h"
#include "SettingsManager.h"
#include "kd_tree.h"
#include "PriorityColorUpdater.h"
#include "GpuBrusher.h"
#include "CameraNav.hpp"
#include "HistogramManager.h"
#include "IsoSurfRenderer.h"
#include "BrushIsoSurfRenderer.h"
#include "MultivariateGauss.h"

#include "ColorPalette.h"

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
#include <math.h>
#include <string.h>
#include <sstream>
#include <memory>
#include <cctype>
#include <Eigen/Dense>
//#include <thrust/sort.h>

#ifdef DETECTMEMLEAK
#define new new( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#endif

//defines for key ids
#define KEYW 87
#define KEYA 65
#define KEYS 83
#define KEYD 68
#define KEYQ 81
#define KEYE 69
#define KEYP 80
#define KEYENTER 257
#define KEYESC 256

//defines for the medians
#define MEDIANCOUNT 3
#define MEDIAN 0
#define ARITHMEDIAN 1
#define GOEMEDIAN 2

#define BRUSHWIDTH 20
#define EDGEHOVERDIST 5
#define DRAGTHRESH .02f


//defines the amount of fractures per axis
#define FRACTUREDEPTH 15

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
static uint32_t					c_QueueFamily = (uint32_t)-1;
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
std::vector<T> operator+(const std::vector<T>& a, const T* b) {
	std::vector<T> result;
	std::transform(a.begin(), a.end(), b, std::back_inserter(result), std::plus<T>());
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
		result += powf(c, 2);
	}
	return result;
}

template <typename T>
T squareDist(const std::vector<T>& a, const float* b) {
	float result = 0;
	float c;
	for (int i = 0; i < a.size(); i++) {
		c = a[i] - b[i];
		result += powf(c, 2);
	}
	return result;
}

template <typename T>
T eucDist(const std::vector<T>& a) {
	float result = 0;
	for (int i = 0; i < a.size(); i++) {
		result += powf(a[i], 2);
	}
	return sqrt(result);
}

std::vector<double> divide(float* arr, float num, int size) {
	std::vector<double> result;
	for (int i = 0; i < size; i++) {
		result.push_back(arr[i] / num);
	}
	return result;
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
	uint32_t attributeInd;
	uint32_t amtOfAttributes;
	uint32_t pad;
	uint32_t padding;
	Vec4 color;
};

struct DensityUniformBuffer {
	uint32_t enableMapping;
	float gaussRange;
	uint32_t imageHeight;
	float gap;
	float compare;
};

struct DrawListComparator {
	std::string parentDataset;
	std::string a;
	std::string b;
	std::vector<uint32_t>aInd;
	std::vector<uint32_t>bInd;
	std::vector<uint32_t>aOrB;
	std::vector<uint32_t>aMinusB;
	std::vector<uint32_t>bMinusA;
	std::vector<uint32_t>aAndb;
};

struct Brush {
	int id;
	std::pair<float, float> minMax;
};

struct TemplateList {
	std::string name;
	VkBuffer buffer;
	std::vector<uint32_t> indices;
	std::vector<std::pair<float, float>> minMax;
	float pointRatio;		//ratio of points in the datasaet(reduced)
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
	int reducedDataSetSize;			//size of the reduced dataset(when clustering was applied). This is set to data.size() on creation.

	bool operator==(const DataSet& other) {
		return this->name == other.name;
	}
};

struct GlobalBrush {
	bool active;										//global brushes can be activated and deactivated
	bool edited;										//indicates if the brush was edited. This is important for kd-tree creation
	TemplateList* parent;
	DataSet* parentDataset;
	std::vector<int> attributes;
	KdTree* kdTree;										//kdTree for the division of cluster
	int fractureDepth;
	bool useMultivariate;								//indicator if brush should use fractions or multivariates
	std::vector<std::vector<std::pair<float, float>>> fractions;
	std::vector<MultivariateGauss::MultivariateBrush> multivariates;
	std::string name;									//the name of a global brush describes the template list it was created from and more...
	std::map<std::string, int> lineRatios;			//contains the ratio of still active lines per drawlist
	std::map<int, std::vector<std::pair<unsigned int, std::pair<float, float>>>> brushes;	//for every brush that exists, one entry in this map exists, where the key is the index of the Attribute in the pcAttributes vector and the pair describes the minMax values
};

struct TemplateBrush {
	std::string name;									//identifier for the template brush
	TemplateList* parent;
	DataSet* parentDataSet;
	std::map<int, std::pair<float, float>> brushes;
};

struct DrawList {
	std::string name;
	std::string parentDataSet;
	TemplateList* parentTemplateList;
	Vec4 color;
	Vec4 prefColor;
	bool show;
	bool showHistogramm;
	std::vector<float> brushedRatioToParent;     // Stores the ratio of points of this data set and points going through the same 1D brushes of the parent.
	bool immuneToGlobalBrushes;
	VkBuffer buffer;
	VkBuffer indexBuffer;
	uint32_t indexBufferOffset;
	VkBuffer ubo;
	//VkBuffer histogramIndBuffer;
	//uint32_t histIndexBufferOffset;
	std::vector<VkBuffer> histogramUbos;
	VkBuffer medianBuffer;
	VkBuffer medianUbo;
	uint32_t priorityColorBufferOffset;
	VkBuffer priorityColorBuffer;
	uint32_t activeIndicesBufferOffset;
	VkBuffer activeIndicesBuffer;					//bool buffer of length n with n being the amount of data in the parent dataset
	uint32_t indicesBufferOffset;
	VkBuffer indicesBuffer;							//graphics buffer with all indices which are in this drawlist
	VkBufferView activeIndicesBufferView;			//buffer view to address the active indices buffer bytewise
	int medianUboOffset;
	VkDescriptorSet medianUboDescSet;
	uint32_t medianBufferOffset;
	Vec4 medianColor;
	int activeMedian;
	std::vector<uint32_t> histogramUbosOffsets;
	std::vector<VkDescriptorSet> histogrammDescSets;
	VkDeviceMemory dlMem;
	VkDescriptorSet uboDescSet;
	std::vector<uint32_t> indices;
	//std::vector<uint32_t> activeInd;
	std::vector<std::vector<Brush>> brushes;		//the pair contains first min and then max for the brush
};

enum ViolinPlacement {
	ViolinLeft,
	ViolinRight,
	ViolinMiddle,
	ViolinMiddleLeft,
	ViolinMiddleRight,
	ViolinLeftHalf,
	ViolinRightHalf
};

enum ViolinScale {
	ViolinScaleSelf,
	ViolinScaleLocal,
	ViolinScaleGlobal,
	ViolinScaleGlobalAttribute
};

enum ViolinYScale {
	ViolinYScaleStandard,
	ViolinYScaleLocalBrush,
	ViolinYScaleGlobalBrush,
	ViolinYScaleBrushes
};

enum ViolinDrawState {
	ViolinDrawStateAll,
	ViolinDrawStateArea,
	ViolinDrawStateLine
};

struct DrawListRef {
	std::string name;
	bool activated;
};

typedef struct ViolinPlot {
	ViolinPlot() {
		colorPaletteManager = new ColorPaletteManager();
	}
	ViolinPlot(const ViolinPlot& obj) {
		colorPaletteManager = new ColorPaletteManager(*(obj.colorPaletteManager));
	}
	~ViolinPlot() { delete colorPaletteManager; }

	std::vector<std::string> attributeNames;		//attribute names are used to identify whether a dataset can be added to this violin plot
	std::vector<DrawListRef> drawLists;				//the name of the drawlists which are shown in the violin plot
	std::vector<ViolinPlacement> violinPlacements;	//the placement of each histogram (defined per histogram)
	std::vector<ViolinScale> violinScalesX;			//the scaling of each histogram in x direction
	std::vector<ImVec4> drawListLineColors;			//line colors for each drawist histogram
	std::vector<ImVec4> drawListFillColors;			//background colors for each drawlist histogram
	bool* activeAttributes;							//bool array containing whether a attribute is active
	std::vector<uint32_t> attributeOrder;			//the first index in this vector corresponds to the first attribute to draw
	float maxGlobalValue;							//max global value accross all histograms
	std::vector<float> maxValues;					//max value of all histogramms

    ColorPaletteManager *colorPaletteManager = new ColorPaletteManager();


} ViolinPlot;

struct ViolinDrawlistPlot {
	ViolinDrawlistPlot() {
		colorPaletteManager = new ColorPaletteManager();
	}
	ViolinDrawlistPlot(const ViolinDrawlistPlot& obj) {
		colorPaletteManager = new ColorPaletteManager(*(obj.colorPaletteManager));
	}
	~ViolinDrawlistPlot() {
		delete colorPaletteManager;
	}

	std::vector<std::string> attributeNames;
	std::vector<float> attributeScalings;
	std::vector<ViolinPlacement> attributePlacements;
	std::vector<std::string> drawLists;
	std::vector<ViolinScale> violinScalesX;
	std::vector<ImVec4> attributeLineColors;
	std::vector<ImVec4> attributeFillColors;
	bool* activeAttributes;
	std::pair<uint32_t, uint32_t> matrixSize;
	std::vector<uint32_t> drawListOrder;
	std::vector<std::vector<uint32_t>> attributeOrder;
	float maxGlobalValue;
	std::vector<float> maxValues;
	std::set<uint32_t> selectedDrawlists;

    std::vector<float> histDistToRepresentative;

	ColorPaletteManager* colorPaletteManager;

//    std::unique_ptr<ColorPaletteManager> colorPaletteManager
//        = ColorPaletteManager();
};

struct Attribute {
	std::string name;
	float min;			//min value of all values
	float max;			//max value of all values
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
static bool						g_RenderSplines = true;
//variables for the histogramm pipeline
static VkPipelineLayout			g_PcPlotHistoPipelineLayout = VK_NULL_HANDLE;
static VkPipelineLayout			g_PcPlotHistoPipelineAdditiveLayout = VK_NULL_HANDLE;
static VkPipeline				g_PcPlotHistoPipeline = VK_NULL_HANDLE;
static VkPipeline				g_PcPlotHistoAdditivePipeline = VK_NULL_HANDLE;
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

//variables for the compute pipeline filling the indexbuffers
static VkPipeline				c_IndexPipeline;
static VkPipelineLayout			c_IndexPipelineLayout;
static VkDescriptorSetLayout	c_IndexPipelineDescSetLayout;
static VkDescriptorSet			c_IndexPipelineDescSet;

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
static char						c_indexShaderPath[] = "shader/indexComp.spv";

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
												,255, 255, 249 ,255 };

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
static bool createDefaultOnLoad = true;
static bool rescaleTableColumns = true;

//variables for the histogramm
static float histogrammWidth = .1f;
static bool drawHistogramm = false;
static bool computeRatioPtsInDLvsIn1axbrushedParent = true;
static bool histogrammDensity = true;
static bool pcPlotDensity = false;
static float densityRadius = .05f;
static bool enableDensityMapping = true;
static bool enableDensityGreyscale = false;
static bool calculateMedians = true;
static bool mapDensity = true;
static int histogrammDrawListComparison = -1;
static Vec4 histogrammBackCol = { .2f,.2f,.2,1 };
static Vec4 densityBackCol = { 0,0,0,1 };
static float medianLineWidth = 1.0f;
static bool enableBrushing = false;

//variables for brush templates
static bool brushTemplatesEnabled = true;
static bool* brushTemplateAttrEnabled = NULL;
static bool showCsvTemplates = false;
static bool updateBrushTemplates = false;
static int selectedTemplateBrush = -1;
static bool drawListForTemplateBrush = false;
static std::vector<TemplateBrush> templateBrushes;
static int liveBrushThreshold = 2e6;

//variables for global brushes
static int selectedGlobalBrush = -1;			//The global brushes are shown in a list where each brush is clickable to then be adaptable.
static std::vector<GlobalBrush> globalBrushes;
static std::map<std::string, float> activeBrushRatios;	//The ratio from lines active after all active brushes have been applied
static std::set<int> brushDragIds;
//information about the dragmode
//0 -> brush dragged
//1 -> top edge dragged
//2 -> bottom edge dragged
static int brushDragMode = 0;
static unsigned int currentBrushId = 0;
static bool* activeBrushAttributes = nullptr;
static bool toggleGlobalBrushes = true;
static int brushCombination = 0;				//How global brushes should be combined. 0->OR, 1->AND
static float brushMuFactor = .001f;				//factor to add a mu to the bounds of a brush

//variables for priority rendering
static int priorityAttribute = -1;
static float priorityAttributeCenterValue = 0;
static bool prioritySelectAttribute = false;
static bool priorityReorder = false;
static int priorityListIndex = 0;

//variables for the 3d views
static View3d* view3d;
static bool view3dAlwaysOnTop = false;
static bool enable3dView = false;
static std::string active3dAttribute;
static bool enableBubbleWindow = false;
static bool enableIsoSurfaceWindow = false;
static bool enableBrushIsoSurfaceWindow = false;
static bool coupleBubbleWindow = true;
static BubblePlotter* bubblePlotter;

static SettingsManager* settingsManager;

static GpuBrusher* gpuBrusher;

static HistogramManager* histogramManager;

static IsoSurfRenderer* isoSurfaceRenderer = nullptr;
static BrushIsoSurfRenderer* brushIsoSurfaceRenderer = nullptr;
static bool coupleIsoSurfaceRenderer = true;
static bool coupleBrushIsoSurfaceRenderer = true;
static bool isoSurfaceRegularGrid = false;
static int isoSurfaceRegularGridDim[3]{ 51,30,81 };
static glm::uvec3 posIndices{ 1,0,2 };

//variables for fractions
static int maxFractionDepth = 24;
static int outlierRank = 11;					//min rank(amount ofdatapoints in a kd tree node) needed to not be an outlier node
static int boundsBehaviour = 2;
static int splitBehaviour = 1;
static int maxRenderDepth = 13;
static float fractionBoxWidth = BRUSHWIDTH;
static int fractionBoxLineWidth = 3;
static float multivariateStdDivThresh = 1.0f;

//variables for animation
static float animationDuration = 2.0f;		//time for every active brush to show in seconds
static std::chrono::steady_clock::time_point animationStart(std::chrono::duration<int>(0));
static bool* animationActiveDatasets = nullptr;
static bool animationItemsDisabled = false;
static int animationCurrentDrawList = -1;

typedef struct {
	bool optimizeSidesNowAttr = false;
	bool optimizeSidesNowDL = false;
	ViolinDrawlistPlot *vdlp = nullptr;
	ViolinPlot *vp = nullptr;
} AdaptViolinSidesAutoStruct;

//variables for violin plots
static int violinPlotHeight = 1000;//550;
static int violinPlotXSpacing = 15;
static bool enableAttributeViolinPlots = false;
static bool enableDrawlistViolinPlots = false;
static float violinPlotThickness = 4;
static float violinPlotBinsSize = 150;
static ImVec4 violinBackgroundColor = { 1,1,1,1 };
static bool coupleViolinPlots = true;
static bool violinPlotDLSendToIso = true;
static bool violinPlotDLInsertCustomColors = true;
static bool violinPlotAttrInsertCustomColors = true;
static bool violinPlotAttrReplaceNonStop = false;
static bool violinPlotAttrConsiderBlendingOrder = true;
static bool violinPlotDLConsiderBlendingOrder = true;
static bool violinPlotDLReplaceNonStop = false;
static bool violinPlotAttrReverseColorPallette = false;
static bool violinPlotDLReverseColorPallette = false;

static std::vector<int> violinPlotDLIdxInListForHistComparison;
static bool violinPlotDLUseRenderedBinsForHistComp = false;
static std::string ttempStr = "abc";

static int violinPlotAttrAutoColorAssignFill = 4;
static int violinPlotDLAutoColorAssignFill = 4;
static int violinPlotAttrAutoColorAssignLine = 4;
static int violinPlotDLAutoColorAssignLine = 4;

static bool yScaleToCurrenMax = false;
static bool violinPlotOverlayLines = true;
static bool renderOrderBasedOnFirstAtt = true;
static bool renderOrderBasedOnFirstDL = true;
static bool renderOrderAttConsider = true;
static bool renderOrderDLConsider = true;
static bool renderOrderAttConsiderNonStop = true;
static bool renderOrderDLConsiderNonStop = true;

static bool renderOrderDLReverse = false;
static bool logScaleDLGlobal = false;
static ViolinYScale violinYScale = ViolinYScaleStandard;
std::vector<ViolinPlot> violinAttributePlots;
std::vector<ViolinDrawlistPlot> violinDrawlistPlots;
AdaptViolinSidesAutoStruct violinAdaptSidesAutoObj;



//method declarations
template <typename T,typename T2>
static bool sortDescPair(std::pair<T, T2> a, std::pair<T, T2> b){ 
	if (isnan(a.second)) { return false; }
	if (isnan(b.second)) { return true; } 
	return a.second > b.second; 
}

template <typename T, typename T2>
static bool sortAscPair(std::pair<T, T2> a, std::pair<T, T2> b) {
	if (isnan(a.second)) { return true; }
	if (isnan(b.second)) { return false; }
	return a.second < b.second;
}

template <typename T, typename T2>
static bool sortAscFloatBasedOnOther(T a, T b, T2 vala, T2 valb) {
    if (isnan(vala)) { return true; }
    if (isnan(valb)) { return false; }

    return vala < valb;


}

template <typename T>
static bool sortMPVPWHistMeasure(T a, T b, std::vector<float> &dists ){
    if (a >= dists.size()) {return false;}
    if (b >= dists.size()) {return true;}

    return sortAscFloatBasedOnOther(a,b, dists[a], dists[b]);

}

// For converting std::vector<std::string>> to char array
char *convertToChar(const std::string & s)
{
	char *pc = new char[s.size() + 1];
	std::strcpy(pc, s.c_str());
	return pc;
}

const char *convertToConstChar(const std::string & s)
{
	return s.c_str();
}


std::vector<const char*> convertStringVecToConstChar(std::vector<std::string> *strV)
{
	std::vector<const char*>  vc;
	std::transform(strV->begin(), strV->end(), std::back_inserter(vc), convertToConstChar);
	return vc;
}


static float determineViolinScaleLocalDiv(std::vector<float>& maxCount, bool **active, std::vector<float> &attributeScalings)
{
	bool bscale = false;
	if (attributeScalings.size() == maxCount.size())
	{
		bscale = true;
	}

	float div = 0;
	for (int i = 0; i < maxCount.size(); ++i)
	{
		if ((*active)[i])
		{
			float currMaxCout = maxCount[i];
			if (bscale) { currMaxCout *= attributeScalings[i]; }

			if (currMaxCout > div)
			{
				div = currMaxCout;
			}
		}
	}
	return div;
}


static void updateDrawListIndexBuffer(DrawList& dl);
static bool updateActiveIndices(DrawList& dl);
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
	shaderModules[0] = VkUtil::createShaderModule(g_Device, vertexBytes);
	//the geometry shader for the pipeline
	std::vector<char> geometryBytes = PCUtil::readByteFile(g_histGeoPath);
	shaderModules[3] = VkUtil::createShaderModule(g_Device, geometryBytes);
	//the fragment shader for the pipeline
	std::vector<char> fragmentBytes = PCUtil::readByteFile(g_histFragPath);
	shaderModules[4] = VkUtil::createShaderModule(g_Device, fragmentBytes);


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

	//VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
	//vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	//vertexInputInfo.vertexBindingDescriptionCount = 1;
	//vertexInputInfo.pVertexBindingDescriptions = &bindingDescripiton;
	//vertexInputInfo.vertexAttributeDescriptionCount = 1;
	//vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

	VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 0;
	vertexInputInfo.pVertexBindingDescriptions = nullptr;
	vertexInputInfo.vertexAttributeDescriptionCount = 0;
	vertexInputInfo.pVertexAttributeDescriptions = nullptr;

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

	uboLayoutBinding.binding = 1;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
	bindings.push_back(uboLayoutBinding);

	uboLayoutBinding.binding = 2;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	bindings.push_back(uboLayoutBinding);

	VkUtil::createDescriptorSetLayout(g_Device, bindings, &g_PcPlotHistoDescriptorSetLayout);
	std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
	descriptorSetLayouts.push_back(g_PcPlotHistoDescriptorSetLayout);

	VkUtil::createPipeline(g_Device, &vertexInputInfo, g_PcPlotWidth, g_PcPlotHeight, dynamicStates, shaderModules, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &g_PcPlotRenderPass, &g_PcPlotHistoPipelineLayout, &g_PcPlotHistoPipeline);

	shaderModules[3] = nullptr;

	//----------------------------------------------------------------------------------------------
	//pipeline for additive histogramm density
	//----------------------------------------------------------------------------------------------
	//the vertex shader for the pipeline
	vertexBytes = PCUtil::readByteFile(g_histVertPath);
	shaderModules[0] = VkUtil::createShaderModule(g_Device, vertexBytes);
	//the geometry shader for the pipeline
	geometryBytes = PCUtil::readByteFile(g_histGeoPath);
	shaderModules[3] = VkUtil::createShaderModule(g_Device, geometryBytes);
	//the fragment shader for the pipeline
	fragmentBytes = PCUtil::readByteFile(g_histFragPath);
	shaderModules[4] = VkUtil::createShaderModule(g_Device, fragmentBytes);

	//blendInfo
	colorBlendAttachment = {};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_TRUE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

	blendInfo.blendAttachment = colorBlendAttachment;
	blendInfo.createInfo = colorBlending;

	VkUtil::createPipeline(g_Device, &vertexInputInfo, g_PcPlotWidth, g_PcPlotHeight, dynamicStates, shaderModules, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &g_PcPlotRenderPass, &g_PcPlotHistoPipelineAdditiveLayout, &g_PcPlotHistoAdditivePipeline);

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

	colorBlendAttachment = {};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_TRUE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

	blendInfo.blendAttachment = colorBlendAttachment;
	blendInfo.createInfo = colorBlending;

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

	VkUtil::createRenderPass(g_Device, VkUtil::PASS_TYPE_COLOR16_OFFLINE_NO_CLEAR, &g_PcPlotDensityRenderPass);

	VkUtil::createPipeline(g_Device, &vertexInputInfo, g_PcPlotWidth, g_PcPlotHeight, dynamicStates, shaderModules, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &g_PcPlotDensityRenderPass, &g_PcPlotDensityPipelineLayout, &g_PcPlotDensityPipeline);
}

static void cleanupPcPlotHistoPipeline() {
	vkDestroyDescriptorSetLayout(g_Device, g_PcPlotHistoDescriptorSetLayout, nullptr);
	vkDestroyPipelineLayout(g_Device, g_PcPlotHistoPipelineLayout, nullptr);
	vkDestroyPipeline(g_Device, g_PcPlotHistoPipeline, nullptr);
	vkDestroyPipelineLayout(g_Device, g_PcPlotRectPipelineLayout, nullptr);
	vkDestroyPipeline(g_Device, g_PcPlotRectPipeline, nullptr);
	vkDestroyPipelineLayout(g_Device, g_PcPlotDensityPipelineLayout, nullptr);
	vkDestroyPipeline(g_Device, g_PcPlotHistoAdditivePipeline, nullptr);
	vkDestroyPipelineLayout(g_Device, g_PcPlotHistoPipelineAdditiveLayout, nullptr);
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
	imageInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
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
	VkUtil::createImage(g_Device, g_PcPlotWidth, g_PcPlotHeight, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &g_PcPlotDensityImageCopy);

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

	VkUtil::createImageView(g_Device, g_PcPlotDensityImageCopy, VK_FORMAT_R16G16B16A16_SFLOAT, 1, VK_IMAGE_ASPECT_COLOR_BIT, &g_PcPlotDensityImageView);
	VkUtil::createImageView(g_Device, g_PcPlotDensityIronMap, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_ASPECT_COLOR_BIT, &g_PcPLotDensityIronMapView);

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
	createInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
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

	VkPipelineViewportStateCreateInfo viewportState = {};
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

	VkDescriptorSetLayoutBinding uboLayoutBindings[3] = {};
	uboLayoutBindings[0].binding = 0;
	uboLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBindings[0].descriptorCount = 1;
	uboLayoutBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	uboLayoutBindings[1].binding = 1;
	uboLayoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	uboLayoutBindings[1].descriptorCount = 1;
	uboLayoutBindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	uboLayoutBindings[2].binding = 2;
	uboLayoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	uboLayoutBindings[2].descriptorCount = 1;
	uboLayoutBindings[2].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 3;
	layoutInfo.pBindings = uboLayoutBindings;

	err = vkCreateDescriptorSetLayout(g_Device, &layoutInfo, nullptr, &g_PcPlotDescriptorLayout);
	check_vk_result(err);

	VkDescriptorPoolSize poolSizes[3] = {};
	poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSizes[0].descriptorCount = 100;
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSizes[1].descriptorCount = 100;
	poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[2].descriptorCount = 1;

	VkDescriptorPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = 3;
	poolInfo.pPoolSizes = poolSizes;
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

	VkDescriptorSetLayoutBinding uboLayoutBinding = {};
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

	//----------------------------------------------------------------------------------------------
	// creating the compute shader for indexbuffer filling
	//----------------------------------------------------------------------------------------------
	std::vector<char> computeBytes = PCUtil::readByteFile(c_indexShaderPath);
	VkShaderModule computeShader = VkUtil::createShaderModule(g_Device, computeBytes);


	//[in]	info buffer
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	std::vector<VkDescriptorSetLayoutBinding> bindings;
	bindings.push_back(uboLayoutBinding);

	//[in]	ordered inex buffer
	uboLayoutBinding.binding = 1;
	bindings.push_back(uboLayoutBinding);

	//[in]	active indices
	uboLayoutBinding.binding = 2;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
	uboLayoutBinding.descriptorCount = 1;
	bindings.push_back(uboLayoutBinding);

	//[out]	index buffer
	uboLayoutBinding.binding = 3;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	bindings.push_back(uboLayoutBinding);

	VkUtil::createDescriptorSetLayout(g_Device, bindings, &c_IndexPipelineDescSetLayout);

	descriptorSetLayouts.clear();
	descriptorSetLayouts.push_back(c_IndexPipelineDescSetLayout);
	VkUtil::createComputePipeline(g_Device, computeShader, descriptorSetLayouts, &c_IndexPipelineLayout, &c_IndexPipeline);
}

static void cleanupPcPlotPipeline() {
	vkDestroyDescriptorPool(g_Device, g_PcPlotDescriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(g_Device, g_PcPlotDescriptorLayout, nullptr);
	vkDestroyPipelineLayout(g_Device, g_PcPlotPipelineLayout, nullptr);
	vkDestroyPipeline(g_Device, g_PcPlotPipeline, nullptr);
	vkDestroyPipelineLayout(g_Device, g_PcPlotSplinePipelineLayout, nullptr);
	vkDestroyPipeline(g_Device, g_PcPlotSplinePipeline, nullptr);
	vkDestroyPipelineLayout(g_Device, c_IndexPipelineLayout, nullptr);
	vkDestroyPipeline(g_Device, c_IndexPipeline, nullptr);
	vkDestroyDescriptorSetLayout(g_Device, c_IndexPipelineDescSetLayout, nullptr);
}

static void createPcPlotRenderPass() {
	VkResult err;

	VkAttachmentDescription colorAttachment = {};
	colorAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
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

static void createPcPlotVertexBuffer(const std::vector<Attribute>& Attributes, const std::vector<float*>& data) {
	VkResult err;

	//creating the command buffer as its needed to do all the operations in here
	//createPcPlotCommandBuffer();

	Buffer vertexBuffer, stagingBuffer;

	uint32_t amtOfVertices = Attributes.size() * data.size();

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = sizeof(Vertex) * Attributes.size() * data.size();
	bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &vertexBuffer.buffer);
	check_vk_result(err);
	VkUtil::createBuffer(g_Device, sizeof(Vertex)*amtOfVertices,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,&stagingBuffer.buffer);

	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(g_Device, vertexBuffer.buffer, &memRequirements);
	vkGetBufferMemoryRequirements(g_Device, stagingBuffer.buffer, &memRequirements);

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	err = vkAllocateMemory(g_Device, &allocInfo, nullptr, &stagingBuffer.memory);
	check_vk_result(err);

	vkBindBufferMemory(g_Device, stagingBuffer.buffer, stagingBuffer.memory, 0);

	allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, 0);
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
	vkMapMemory(g_Device, stagingBuffer.memory, 0, sizeof(Vertex) * amtOfVertices, 0, &mem);
	memcpy(mem, d, amtOfVertices * sizeof(Vertex));
	vkUnmapMemory(g_Device, stagingBuffer.memory);

	delete[] d;

	VkCommandBuffer copyComm;
	VkUtil::createCommandBuffer(g_Device, g_PcPlotCommandPool, &copyComm);
	VkUtil::copyBuffer(copyComm, stagingBuffer.buffer, vertexBuffer.buffer, sizeof(Vertex) * amtOfVertices, 0, 0);
	VkUtil::commitCommandBuffer(g_Queue, copyComm);
	check_vk_result(vkQueueWaitIdle(g_Queue));
	vkFreeCommandBuffers(g_Device, g_PcPlotCommandPool, 1, &copyComm);
	vkDestroyBuffer(g_Device, stagingBuffer.buffer, nullptr);
	vkFreeMemory(g_Device, stagingBuffer.memory, nullptr);

	g_PcPlotVertexBuffers.push_back(vertexBuffer);

	if (g_PcPlotIndexBuffer)
		return;

	//creating the index buffer
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = sizeof(uint16_t) * (Attributes.size() + 2);
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
	VkUtil::updateDescriptorSet(g_Device, g_PcPlotDensityUbo, sizeof(DensityUniformBuffer), 1, g_PcPlotDensityDescriptorSet);
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
	vkMapMemory(g_Device, g_PcPlotIndexBufferMemory, g_PcPlotDensityRectBufferOffset, sizeof(Vec4) * 4, 0, &ind);
	memcpy(ind, rect, sizeof(Vec4) * 4);
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
			activeBrushRatios.erase(it->name);
			if (it->dlMem) {
				vkFreeMemory(g_Device, it->dlMem, nullptr);
				it->dlMem = VK_NULL_HANDLE;
			}
			if (it->ubo) {
				vkDestroyBuffer(g_Device, it->ubo, nullptr);
				it->ubo = VK_NULL_HANDLE;
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
			if (it->priorityColorBuffer) {
				vkDestroyBuffer(g_Device, it->priorityColorBuffer, nullptr);
				it->priorityColorBuffer = VK_NULL_HANDLE;
			}
			if (it->activeIndicesBuffer) {
				vkDestroyBuffer(g_Device, it->activeIndicesBuffer, nullptr);
				it->activeIndicesBuffer = VK_NULL_HANDLE;
			}
			if (it->activeIndicesBufferView) {
				vkDestroyBufferView(g_Device, it->activeIndicesBufferView, nullptr);
				it->activeIndicesBufferView = VK_NULL_HANDLE;
			}
			if (it->indicesBuffer) {
				vkDestroyBuffer(g_Device, it->indicesBuffer, nullptr);
				it->indicesBuffer = VK_NULL_HANDLE;
			}
			for (int i = 0; i < it->histogramUbos.size(); i++) {
				vkDestroyBuffer(g_Device, it->histogramUbos[i], nullptr);
			}
			for (GlobalBrush& brush : globalBrushes) {
				brush.lineRatios.erase(it->name);
			}
			g_PcPlotDrawLists.erase(it++);
		}
		else {
			it++;
		}
	}
}

static void createPcPlotDrawList(TemplateList& tl, const DataSet& ds, const char* listName) {
	VkResult err;

	DrawList dl = {};
	dl.parentTemplateList = &tl;

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
	bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &dl.indexBuffer);
	check_vk_result(err);

	dl.indexBufferOffset = allocInfo.allocationSize;
	vkGetBufferMemoryRequirements(g_Device, dl.indexBuffer, &memRequirements);
	allocInfo.allocationSize += memRequirements.size;

	//priority rendering color buffer
	bufferInfo.size = ds.data.size() * sizeof(float);
	bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &dl.priorityColorBuffer);
	check_vk_result(err);

	dl.priorityColorBufferOffset = allocInfo.allocationSize;
	vkGetBufferMemoryRequirements(g_Device, dl.priorityColorBuffer, &memRequirements);
	allocInfo.allocationSize += memRequirements.size;

	memTypeBits |= memRequirements.memoryTypeBits;

	//active indices buffer
	VkUtil::createBuffer(g_Device, ds.data.size() * sizeof(bool), VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT, &dl.activeIndicesBuffer);

	dl.activeIndicesBufferOffset = allocInfo.allocationSize;
	vkGetBufferMemoryRequirements(g_Device, dl.activeIndicesBuffer, &memRequirements);
	allocInfo.allocationSize += memRequirements.size;
	memTypeBits |= memRequirements.memoryTypeBits;

	//indices buffer
	VkUtil::createBuffer(g_Device, tl.indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &dl.indicesBuffer);

	dl.indicesBufferOffset = allocInfo.allocationSize;
	vkGetBufferMemoryRequirements(g_Device, dl.indicesBuffer, &memRequirements);
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

	//Binding the median uniform Buffer
	vkBindBufferMemory(g_Device, dl.medianUbo, dl.dlMem, dl.medianUboOffset);

	//creating the Descriptor set for the median uniform buffer
	std::vector<VkDescriptorSetLayout> layouts;
	layouts.push_back(g_PcPlotDescriptorLayout);
	VkUtil::createDescriptorSets(g_Device, layouts, g_DescriptorPool, &dl.medianUboDescSet);
	VkUtil::updateDescriptorSet(g_Device, dl.medianUbo, sizeof(UniformBufferObject), 0, dl.medianUboDescSet);

	//creating and uploading the indexbuffer data
	uint32_t* indBuffer = new uint32_t[tl.indices.size() * 2];
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

	//binding the indexBuffer
	vkBindBufferMemory(g_Device, dl.indexBuffer, dl.dlMem, dl.indexBufferOffset);

	//binding the  priority rendering buffer
	vkBindBufferMemory(g_Device, dl.priorityColorBuffer, dl.dlMem, dl.priorityColorBufferOffset);

	//binding the active indices buffer, creating the buffer view and uploading the correct indices to the graphicscard
	vkBindBufferMemory(g_Device, dl.activeIndicesBuffer, dl.dlMem, dl.activeIndicesBufferOffset);
	VkUtil::createBufferView(g_Device, dl.activeIndicesBuffer, VK_FORMAT_R8_SNORM, 0, ds.data.size() * sizeof(bool), &dl.activeIndicesBufferView);
	std::vector<uint8_t> actives(ds.data.size(), 0);			//vector with 0 initialized everywhere
	for (int i : tl.indices) {									//setting all active indices to true
		actives[i] = 1;
	}
	VkUtil::uploadData(g_Device, dl.dlMem, dl.activeIndicesBufferOffset, ds.data.size() * sizeof(bool), actives.data());

	//binding indices buffer and uploading the indices
	vkBindBufferMemory(g_Device, dl.indicesBuffer, dl.dlMem, dl.indicesBufferOffset);
	VkUtil::uploadData(g_Device, dl.dlMem, dl.indicesBufferOffset, tl.indices.size() * sizeof(uint32_t), tl.indices.data());

	//creating the Descriptor sets for the histogramm uniform buffers
	layouts = std::vector<VkDescriptorSetLayout>(dl.histogramUbos.size());
	for (auto& l : layouts) {
		l = g_PcPlotHistoDescriptorSetLayout;
	}

	dl.histogrammDescSets = std::vector<VkDescriptorSet>(layouts.size());
	VkUtil::createDescriptorSets(g_Device, layouts, g_DescriptorPool, dl.histogrammDescSets.data());

	//updating the descriptor sets
	for (int i = 0; i < layouts.size(); i++) {
		VkUtil::updateDescriptorSet(g_Device, dl.histogramUbos[i], sizeof(HistogramUniformBuffer), 0, dl.histogrammDescSets[i]);
		VkUtil::updateTexelBufferDescriptorSet(g_Device, dl.activeIndicesBufferView, 1, dl.histogrammDescSets[i]);
		VkUtil::updateDescriptorSet(g_Device, tl.buffer, ds.data.size() * pcAttributes.size() * sizeof(float), 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, dl.histogrammDescSets[i]);
	}

	//specifying the uniform buffer location
	VkDescriptorBufferInfo desBufferInfos[1] = {};
	desBufferInfos[0].buffer = dl.ubo;
	desBufferInfos[0].offset = 0;
	desBufferInfos[0].range = sizeof(UniformBufferObject);

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
	descriptorWrite.pBufferInfo = desBufferInfos;

	vkUpdateDescriptorSets(g_Device, 1, &descriptorWrite, 0, nullptr);
	VkUtil::updateDescriptorSet(g_Device, dl.priorityColorBuffer, ds.data.size() * sizeof(float), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, dl.uboDescSet);
	VkUtil::updateImageDescriptorSet(g_Device, g_PcPlotDensityIronMapSampler, g_PcPLotDensityIronMapView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2, dl.uboDescSet);

	VkUtil::updateDescriptorSet(g_Device, dl.priorityColorBuffer, ds.data.size() * sizeof(float), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, dl.medianUboDescSet);
	VkUtil::updateImageDescriptorSet(g_Device, g_PcPlotDensityIronMapSampler, g_PcPLotDensityIronMapView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2, dl.medianUboDescSet);

	float hue = distribution(engine) * 10;
#ifdef _DEBUG
	std::cout << "Hue: " << hue << std::endl;
#endif

	hsl randCol = { hue,.5f,.6f };
	rgb col = hsl2rgb(randCol);

	dl.name = std::string(listName);
	dl.buffer = tl.buffer;
	dl.color = { (float)col.r,(float)col.g,(float)col.b,alphaDrawLists };
	dl.prefColor = dl.color;
	dl.show = true;
	dl.showHistogramm = true;
	dl.parentDataSet = ds.name;
	dl.indices = std::vector<uint32_t>(tl.indices);
	dl.brushedRatioToParent = std::vector<float>(pcAttributes.size(), 1);

	//adding a standard brush for every attribute
	for (Attribute a : pcAttributes) {
		dl.brushes.push_back(std::vector<Brush>());
	}

	dl.medianColor = { 1,1,1,1 };

	g_PcPlotDrawLists.push_back(dl);
}

static void removePcPlotDrawList(DrawList& drawList) {
	for (GlobalBrush& brush : globalBrushes) {
		brush.lineRatios.erase(drawList.name);
	}
	activeBrushRatios.erase(drawList.name);
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
			if (it->priorityColorBuffer) {
				vkDestroyBuffer(g_Device, it->priorityColorBuffer, nullptr);
				it->priorityColorBuffer = VK_NULL_HANDLE;
			}
			if (it->activeIndicesBuffer) {
				vkDestroyBuffer(g_Device, it->activeIndicesBuffer, nullptr);
				it->activeIndicesBuffer = VK_NULL_HANDLE;
			}
			if (it->activeIndicesBufferView) {
				vkDestroyBufferView(g_Device, it->activeIndicesBufferView, nullptr);
				it->activeIndicesBufferView = VK_NULL_HANDLE;
			}
			if (it->indicesBuffer) {
				vkDestroyBuffer(g_Device, it->indicesBuffer, nullptr);
				it->indicesBuffer = VK_NULL_HANDLE;
			}
			g_PcPlotDrawLists.erase(it);
			break;
		}
	}
}

static void destroyPcPlotDataSet(DataSet& dataSet) {
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

	updateBrushTemplates = true;

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

	err = vkQueueWaitIdle(g_Queue);
	check_vk_result(err);

	vkFreeCommandBuffers(g_Device, g_PcPlotCommandPool, 1, &g_PcPlotCommandBuffer);
}

// This function assumes that only indices of active attributes are passed. 
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

	err = vkQueueWaitIdle(g_Queue);
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
	ubo.VertexTransormations[0].w = (priorityAttribute != -1) ? 1.f : 0;
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
	uint16_t* ind = new uint16_t[amtOfIndeces + ((g_RenderSplines) ? 2 : 0)];			//contains all indeces to copy
	for (int i = 0; i < order.size(); i++) {
		ind[i + ((g_RenderSplines) ? 1 : 0)] = order[i].first;
	}
	if (g_RenderSplines && pcAttributes.size()) {
		ind[0] = order[0].first;
		ind[order.size()] = order[order.size() - 1].first;
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
		int copyAmount = sizeof(uint16_t) * (attributes.size() + ((g_RenderSplines) ? 2 : 0));
		vkMapMemory(g_Device, g_PcPlotIndexBufferMemory, 0, copyAmount, 0, &d);
		memcpy(d, ind, copyAmount);
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
	c = 0;
	for (DrawList& ds : g_PcPlotDrawLists) {
		ubo.VertexTransormations[0].w = (priorityAttribute != -1 && c == priorityListIndex) ? 1.f : 0;
		ubo.color = ds.color;
		vkMapMemory(g_Device, ds.dlMem, 0, sizeof(UniformBufferObject), 0, &da);
		memcpy(da, &ubo, sizeof(UniformBufferObject));
		vkUnmapMemory(g_Device, ds.dlMem);

		ubo.VertexTransormations[0].w = 0;
		ubo.color = ds.medianColor;
		vkMapMemory(g_Device, ds.dlMem, ds.medianUboOffset, sizeof(UniformBufferObject), 0, &da);
		memcpy(da, &ubo, sizeof(UniformBufferObject));
		vkUnmapMemory(g_Device, ds.dlMem);

		c++;
	}

	//starting the pcPlotCommandBuffer
	createPcPlotCommandBuffer();

	//binding the all needed things
	if (g_RenderSplines)
		vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipelineLayout, 0, 1, &g_PcPlotDescriptorSet, 0, nullptr);
	else
		vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotPipelineLayout, 0, 1, &g_PcPlotDescriptorSet, 0, nullptr);

	if (pcAttributes.size())
		vkCmdBindIndexBuffer(g_PcPlotCommandBuffer, g_PcPlotIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

	//counting the amount of active drawLists for histogramm rendering
	int activeDrawLists = 0;

	//now drawing for every draw list in g_pcPlotdrawlists
	for (auto drawList = g_PcPlotDrawLists.rbegin(); g_PcPlotDrawLists.rend() != drawList; ++drawList) {
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
		uint32_t amtOfI = drawList->indices.size() * (order.size() + 1 + ((g_RenderSplines) ? 2 : 0));
		vkCmdDrawIndexed(g_PcPlotCommandBuffer, amtOfI, 1, 0, 0, 0);

		//draw the Median Line
		if (drawList->activeMedian != 0) {
			vkCmdSetLineWidth(g_PcPlotCommandBuffer, medianLineWidth);
			vkCmdBindVertexBuffers(g_PcPlotCommandBuffer, 0, 1, &drawList->medianBuffer, offsets);
			vkCmdBindIndexBuffer(g_PcPlotCommandBuffer, g_PcPlotIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

			if (g_RenderSplines)
				vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipelineLayout, 0, 1, &drawList->medianUboDescSet, 0, nullptr);
			else
				vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotPipelineLayout, 0, 1, &drawList->medianUboDescSet, 0, nullptr);

			vkCmdDrawIndexed(g_PcPlotCommandBuffer, amtOfIndeces + ((g_RenderSplines) ? 2 : 0), 1, 0, (drawList->activeMedian - 1) * pcAttributes.size(), 0);

#ifdef PRINTRENDERTIME
			amtOfLines++;
#endif
		}

#ifdef PRINTRENDERTIME
		uint32_t boolSize;
		for (DataSet& ds : g_PcPlotDataSets) {
			if (ds.name == drawList->parentDataSet) {
				boolSize = ds.data.size();
				break;
			}
		}
		bool* active = new bool[boolSize];
		VkUtil::downloadData(g_Device, drawList->dlMem, drawList->activeIndicesBufferOffset, boolSize * sizeof(bool), active);
		for (int i = 0; i < boolSize; ++i) {
			if (active[i]) ++amtOfLines;
		}
		delete[] active;
		//amtOfLines += drawList->activeInd.size();
#endif
	}

	delete[] ind;

	if (pcPlotDensity && pcAttributes.size() > 0) {
		//ending the pass to blit the image
		vkCmdEndRenderPass(g_PcPlotCommandBuffer);

		//transition image Layouts
		VkUtil::transitionImageLayout(g_PcPlotCommandBuffer, g_PcPlot, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		VkUtil::transitionImageLayout(g_PcPlotCommandBuffer, g_PcPlotDensityImageCopy, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		//blitting the image
		VkUtil::copyImage(g_PcPlotCommandBuffer, g_PcPlot, g_PcPlotWidth, g_PcPlotHeight, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_PcPlotDensityImageCopy, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		//transition image Layouts back
		VkUtil::transitionImageLayout(g_PcPlotCommandBuffer, g_PcPlot, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
		VkUtil::transitionImageLayout(g_PcPlotCommandBuffer, g_PcPlotDensityImageCopy, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

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

	if (drawHistogramm && pcAttributes.size() > 0) {
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
		if (histogrammDensity && enableDensityMapping) {
			vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotHistoAdditivePipeline);
		}
		else {
			vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotHistoPipeline);
		}

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
				//vkCmdBindVertexBuffers(g_PcPlotCommandBuffer, 0, 1, &drawList->buffer, offsets);
				vkCmdBindIndexBuffer(g_PcPlotCommandBuffer, drawList->indicesBuffer, 0, VK_INDEX_TYPE_UINT32);

				//iterating through the Attributes to render every histogramm
				float x = -1.0f;
				int count = 0;
				for (int i = 0; i < pcAttributes.size(); i++) {
					//setting the missing parameters in the hubo
					hubo.maxVal = pcAttributes[i].max;
					hubo.minVal = pcAttributes[i].min;
					hubo.attributeInd = i;
					hubo.amtOfAttributes = pcAttributes.size();
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
					if (histogrammDensity && enableDensityMapping) {
						vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotHistoPipelineAdditiveLayout, 0, 1, &drawList->histogrammDescSets[i], 0, nullptr);
					}
					else {
						vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotHistoPipelineLayout, 0, 1, &drawList->histogrammDescSets[i], 0, nullptr);
					}

					//making the draw call
					vkCmdDrawIndexed(g_PcPlotCommandBuffer, drawList->indices.size(), 1, 0, 0, 0);
				}

				//increasing the xOffset for the next drawlist
				xOffset += width;
			}
		}

		if (histogrammDensity) {
			//ending the pass to blit the image
			vkCmdEndRenderPass(g_PcPlotCommandBuffer);

			//transition image Layouts
			VkUtil::transitionImageLayout(g_PcPlotCommandBuffer, g_PcPlot, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
			VkUtil::transitionImageLayout(g_PcPlotCommandBuffer, g_PcPlotDensityImageCopy, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

			//blitting the image
			VkUtil::copyImage(g_PcPlotCommandBuffer, g_PcPlot, g_PcPlotWidth, g_PcPlotHeight, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_PcPlotDensityImageCopy, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

			//transition image Layouts back
			VkUtil::transitionImageLayout(g_PcPlotCommandBuffer, g_PcPlot, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
			VkUtil::transitionImageLayout(g_PcPlotCommandBuffer, g_PcPlotDensityImageCopy, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

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

			vkMapMemory(g_Device, g_PcPlotIndexBufferMemory, g_PcPlotDensityRectBufferOffset + sizeof(Vec4) * 4, sizeof(Vec4) * amtOfIndeces * 4, 0, &d);
			memcpy(d, verts, sizeof(Vec4) * amtOfIndeces * 4);
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

	err = vkQueueWaitIdle(g_Queue);
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

	// Select graphics queue families for graphic pipelines
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
		for (uint32_t i = 0; i < count; ++i) {
			if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
				c_QueueFamily = i;
				break;
			}
		}
		free(queues);
		IM_ASSERT(g_QueueFamily != (uint32_t)-1);
		IM_ASSERT(c_QueueFamily != (uint32_t)-1);
#ifdef _DEBUG
		std::cout << "graphics queue: " << g_QueueFamily << std::endl << "cpompute queue: " << c_QueueFamily << std::endl;
#endif
	}

	// Create Logical Device (with 2 queues)
	{
		int device_extension_count = 1;
		const char* device_extensions[] = { "VK_KHR_swapchain" };

		VkPhysicalDeviceDescriptorIndexingFeaturesEXT indexingFeatures{};
		indexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
		indexingFeatures.pNext = nullptr;
		indexingFeatures.descriptorBindingPartiallyBound = VK_TRUE;
		indexingFeatures.runtimeDescriptorArray = VK_TRUE;

		VkPhysicalDeviceFeatures deviceFeatures = {};
		deviceFeatures.geometryShader = VK_TRUE;
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		deviceFeatures.wideLines = VK_TRUE;
		deviceFeatures.depthClamp = VK_TRUE;
		deviceFeatures.vertexPipelineStoresAndAtomics = VK_TRUE;
		deviceFeatures.shaderStorageImageExtendedFormats = VK_TRUE;
		deviceFeatures.shaderTessellationAndGeometryPointSize = VK_TRUE;
		deviceFeatures.fragmentStoresAndAtomics = VK_TRUE;
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
		create_info.pNext = &indexingFeatures;
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
static void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd, VkSurfaceKHR surface, int width, int height)
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

static void FrameRender(ImGui_ImplVulkanH_Window* wd)
{
	VkResult err;

	VkSemaphore image_acquired_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
	VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
	err = vkAcquireNextImageKHR(g_Device, wd->Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &wd->FrameIndex);
	if (err == VK_ERROR_OUT_OF_DATE_KHR) {
		return;
	}
	else if (err != VK_SUCCESS && err != VK_SUBOPTIMAL_KHR)
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

	render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
	VkPresentInfoKHR info = {};
	info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	info.waitSemaphoreCount = 1;
	info.pWaitSemaphores = &render_complete_semaphore;
	info.swapchainCount = 1;
	info.pSwapchains = &wd->Swapchain;
	info.pImageIndices = &wd->FrameIndex;
	err = vkQueuePresentKHR(g_Queue, &info);
	if (err == VK_ERROR_OUT_OF_DATE_KHR) {
		return;
	}
	else if (err != VK_SUCCESS && err != VK_SUBOPTIMAL_KHR)
		check_vk_result(err);
	wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->ImageCount; // Now we can use the next set of semaphores
}

static void FramePresent(ImGui_ImplVulkanH_Window* wd)
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
	if (err == VK_ERROR_OUT_OF_DATE_KHR) {
		return;
	}
	else if (err != VK_SUCCESS && err != VK_SUBOPTIMAL_KHR)
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
	rescaleTableColumns = true;
}

//checks if the attributes a are the same as the ones in pcAttributes and are giving back a permutation to order the new data correctly
//if the attributes arent equal the retuned vector is empty
//If a value is read at index i in the datum, it should be placed at index permutation[i] in the data array.
static std::vector<int> checkAttriubtes(std::vector<std::string>& a) {
	if (pcAttributes.size() == 0) {
		std::vector<int> permutation;
		for (int i = 0; i < a.size(); i++) {
			permutation.push_back(i);
		}
		return permutation;
	}

	if (a.size() != pcAttributes.size())
		return std::vector<int>();

	//creating sets to compare the attributes
	std::set<std::string> pcAttr, attr;
	for (Attribute& a : pcAttributes) {
		std::string s = a.name;
		std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
		if (s == "rlon") s = "longitude";
		if (s == "rlat") s = "latitude";
		pcAttr.insert(s);
	}
	for (std::string s : a) {
		std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
		if (s == "rlon") s = "longitude";
		if (s == "rlat") s = "latitude";
		attr.insert(s);
	}

	if (pcAttr == attr) {
		//getting the right permutation
		std::vector<std::string> lowerCaseA(a), lowerCaseAttr;
		for (int i = 0; i < lowerCaseA.size(); i++) {
			std::transform(lowerCaseA[i].begin(), lowerCaseA[i].end(), lowerCaseA[i].begin(), [](unsigned char c) { return std::tolower(c); });
			if (lowerCaseA[i] == "rlon") lowerCaseA[i] = "longitude";
			if (lowerCaseA[i] == "rlat") lowerCaseA[i] = "latitude";
			std::string attribute = pcAttributes[i].name;
			std::transform(attribute.begin(), attribute.end(), attribute.begin(), [](unsigned char c) { return std::tolower(c); });
			if (attribute == "rlon") attribute = "longitude";
			if (attribute == "rlat") attribute = "latitude";
			lowerCaseAttr.push_back(attribute);
		}

		std::vector<int> permutation;
		for (std::string& s : lowerCaseA) {
			int i = 0;
			for (; i < lowerCaseAttr.size(); i++) {
				if (lowerCaseAttr[i] == s) break;
			}
			permutation.push_back(i);
		}

		return permutation;
	}

	return std::vector<int>();
}

static void openCsv(const char* filename) {

	std::ifstream f(filename, std::ios::in | std::ios::binary);
	std::stringstream input;
	input << f.rdbuf();

	if (!f.is_open()) {
		std::cout << "The given file was not found" << std::endl;
		return;
	}

	bool firstLine = true;

	//creating the dataset to be drawable
	DataSet ds;
	std::string s(filename);
    float minRangeEps = 0.00001;//10e-16;
	int split = (s.find_last_of("\\") > s.find_last_of("/")) ? s.find_last_of("/") : s.find_last_of("\\");
	ds.name = s.substr(split + 1);
	//checking if the filename already exists
	for (auto it = g_PcPlotDataSets.begin(); it != g_PcPlotDataSets.end(); ++it) {
		if (it->name == ds.name) {
			auto itcop = it;
			int count = 0;
			while (1) {
				if (itcop->name == ds.name + (count == 0 ? "" : std::to_string(count)))
					count++;
				if (++itcop == g_PcPlotDataSets.end())
					break;
			}
			ds.name += std::to_string(count);
			break;
		}
	}

	std::vector<int> permutation;
	for (std::string line; std::getline(input, line); )
	{
		std::string delimiter = ",";
		size_t pos = 0;
		std::string cur;

		//parsing the attributes in the first line
		if (firstLine) {
			//copying the attributes into a temporary vector to check for correct Attributes
			std::vector<Attribute> tmp;
			std::vector<std::string> attributes;

			while ((pos = line.find(delimiter)) != std::string::npos) {
				cur = line.substr(0, pos);
				line.erase(0, pos + delimiter.length());
				tmp.push_back({ cur,std::numeric_limits<float>::max(),std::numeric_limits<float>::min() });
				attributes.push_back(tmp.back().name);
			}
			//adding the last item which wasn't recognized
			line = line.substr(0, line.find("\r"));
			tmp.push_back({ line,std::numeric_limits<float>::max(),std::numeric_limits<float>::min() });
			attributes.push_back(tmp.back().name);

			//checking if the Attributes are correct
			permutation = checkAttriubtes(attributes);
			if (pcAttributes.size() != 0) {
				if (tmp.size() != pcAttributes.size()) {
					std::cout << "The Amount of Attributes of the .csv file is not compatible with the currently loaded datasets" << std::endl;
					f.close();
					return;
				}

				if (!permutation.size()) {
					std::cout << "The attributes of the .csv data are not the same as the ones already loaded in the program." << std::endl;
					return;
				}
			}
			//if this is the first Dataset to be loaded, fill the pcAttributes vector
			else {
				for (Attribute& a : tmp) {
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
					f.close();
					return;
				}

				curF = std::stof(cur);

				//updating the bounds if a new highest value was found in the current data.
				if (curF > pcAttributes[permutation[attr]].max)
					pcAttributes[permutation[attr]].max = curF;
				if (curF < pcAttributes[permutation[attr]].min)
					pcAttributes[permutation[attr]].min = curF;

				ds.data.back()[permutation[attr++]] = curF;
			}
			if (attr == pcAttributes.size()) {
				std::cerr << "The dataset to open is not consitent!" << std::endl;
				f.close();
				return;
			}

			//adding the last item which wasn't recognized
			curF = std::stof(line);

			//updating the bounds if a new highest value was found in the current data.
			if (curF > pcAttributes[permutation[attr]].max)
				pcAttributes[permutation[attr]].max = curF;
			if (curF < pcAttributes[permutation[attr]].min)
				pcAttributes[permutation[attr]].min = curF;
			ds.data.back()[permutation[attr]] = curF;
		}
	}

    for (unsigned int k = 0; k < pcAttributes.size(); ++k){
        if (pcAttributes[k].max == pcAttributes[k].min)   {
            pcAttributes[k].max += minRangeEps;
        }
    }


	f.close();

	ds.reducedDataSetSize = ds.data.size();

	createPcPlotVertexBuffer(pcAttributes, ds.data);

	ds.buffer = g_PcPlotVertexBuffers.back();

	TemplateList tl = {};
	tl.buffer = g_PcPlotVertexBuffers.back().buffer;
	tl.name = "Default Drawlist";
	for (int i = 0; i < ds.data.size(); i++) {
		tl.indices.push_back(i);
	}
	tl.pointRatio = tl.indices.size() / (float)ds.data.size();

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
			for (int i = ind2; i != ind1; i--) {
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

static void switchViolinAttributes(int ind1, int ind2, bool ctrPressed, std::vector<uint32_t>& order) {
	if (ctrPressed) {
		int tmp = order[ind2];
		if (ind2 > ind1) {
			for (int i = ind2; i != ind1; i--) {
				order[i] = order[i - 1];
			}
		}
		else {
			for (int i = ind2; i != ind1; i++) {
				order[i] = order[i + 1];
			}
		}
		order[ind1] = tmp;
	}
	else {
		int tmp = order[ind1];
		order[ind1] = order[ind2];
		order[ind2] = tmp;
	}
}

static void openDlf(const char* filename) {
	std::ifstream f(filename, std::ios::in | std::ios::binary);
	std::stringstream file;
	file << f.rdbuf();
	if (f.is_open()) {
		std::string tmp;
		int amtOfPoints;
		bool newAttr = false;
		std::vector<int> permutation;

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
				std::vector<std::string> attributes;
				for (int i = 0; tmp != std::string("Data:") && i < 100; file >> tmp, i++) {
					attributes.push_back(tmp);
				}
				permutation = checkAttriubtes(attributes);
				if (pcAttributes.size() > 0) {
					if (!permutation.size()) {
						std::cout << "The attributes of the dataset to be loaded are not the same as the attributes already used by other datasets" << std::endl;
						return;
					}

#ifdef _DEBUG
					std::cout << "The Attribute check was successful" << std::endl;
#endif
				}

				//reading in new values
				else {
					for (int i = 0; i < attributes.size(); i++) {
						pcAttributes.push_back({ attributes[i],std::numeric_limits<float>::max(),std::numeric_limits<float>::min() - 1 });
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

			if (newAttr) {
				pcAttributeEnabled = new bool[pcAttributes.size()];
				activeBrushAttributes = new bool[pcAttributes.size()];
				for (int i = 0; i < pcAttributes.size(); i++) {
					pcAttributeEnabled[i] = true;
					activeBrushAttributes[i] = false;
					pcAttrOrd.push_back(i);
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
				std::string fname(filename);
				int offset = (fname.find_last_of("/") < fname.find_last_of("\\")) ? fname.find_last_of("/") : fname.find_last_of("\\");
				ds.name = fname.substr(offset + 1);

				file >> tmp;

				float* d = new float[amtOfPoints * pcAttributes.size()];
				int a = 0;
				for (int i = 0; i < amtOfPoints * pcAttributes.size() && tmp != std::string("Drawlists:"); file >> tmp, i++) {
					int datum = i / pcAttributes.size();
					int index = i % pcAttributes.size();
					index = datum * pcAttributes.size() + permutation[index];
					d[index] = std::stof(tmp);
					if (pcAttributes[a].min > d[index]) {
						pcAttributes[a].min = d[index];
					}
					if (pcAttributes[a].max < d[index]) {
						pcAttributes[a].max = d[index];
					}
					a = (a + 1) % pcAttributes.size();
				}

				ds.data = std::vector<float*>(amtOfPoints);
				for (int i = 0; i < amtOfPoints; i++) {
					ds.data[i] = &d[i * pcAttributes.size()];
				}
			}

			createPcPlotVertexBuffer(pcAttributes, ds.data);

			ds.reducedDataSetSize = ds.data.size();

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
					tl.pointRatio = tl.indices.size() / ((float)ds.data.size());
					ds.drawLists.push_back(tl);
				}
			}

			//adding the data set finally to the list
			g_PcPlotDataSets.push_back(ds);
		}

		f.close();
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
		return;
	}
	//printing Amount of data loaded
	std::cout << "Amount of data loaded: " << g_PcPlotDataSets.back().data.size() << std::endl;

	//standard things which should be done on loading of a dataset
	//adding a standard attributes saving
	histogrammWidth = 1.0f / (pcAttributes.size() * 5);
	uint32_t attributesSize = 2 * sizeof(float) * pcAttributes.size();
	for (Attribute& a : pcAttributes) {
		attributesSize += a.name.size() + 1;
	}
	SettingsManager::Setting s = {};
	int split = (file.find_last_of("\\") > file.find_last_of("/")) ? file.find_last_of("/") : file.find_last_of("\\");
	s.id = file.substr(split + 1);
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

static void addIndecesToDs(DataSet& ds, const char* filepath) {
	std::string s(filepath);
	if (s.substr(s.find_last_of(".") + 1) != "idxf") {
		std::cout << "There was an idxf file expected." << std::endl;
		return;
	}
	//opening the file
	std::ifstream f(filepath, std::ios::in | std::ios::binary);
	std::stringstream file;
	file << f.rdbuf();

	if (f.is_open()) {
		TemplateList tl;
		tl.buffer = ds.buffer.buffer;
		int split = (s.find_last_of("\\") > s.find_last_of("/")) ? s.find_last_of("/") : s.find_last_of("\\");
		tl.name = s.substr(split + 1);

		//reading the values
		for (file >> s; !file.eof(); file >> s) {
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
		tl.pointRatio = ((float)tl.indices.size()) / ds.data.size();

		//adding the drawlist to ds
		ds.drawLists.push_back(tl);
		std::cout << "Amount of indices loaded: " << tl.indices.size() << std::endl;
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
			createPcPlotDrawList(ds.drawLists.back(), ds, droppedPaths[i].substr(split + 1).c_str());
			updateActiveIndices(g_PcPlotDrawLists.back());
		}
	}
}


//ToDo: To speed up the programm, this function could be called only once per renderloop.
static void getLocalBrushLimits(DrawList* dl, std::vector<std::pair<float, float>>& localMinMax) {
	//std::vector<std::pair<float, float>> localMinMax(pcAttributes.size(), { std::numeric_limits<float>().max(),std::numeric_limits<float>().min() });

	if (violinYScale == ViolinYScaleLocalBrush || violinYScale == ViolinYScaleBrushes) {
		for (int k = 0; k < pcAttributes.size(); ++k) {
			for (int mi = 0; mi < dl->brushes[k].size(); ++mi) {
				if (dl->brushes[k][mi].minMax.first < localMinMax[k].first) localMinMax[k].first = dl->brushes[k][mi].minMax.first;
				if (dl->brushes[k][mi].minMax.second > localMinMax[k].second) localMinMax[k].second = dl->brushes[k][mi].minMax.second;
			}
		}
		for (int k = 0; k < pcAttributes.size(); ++k) {
			if (localMinMax[k].first == std::numeric_limits<float>().max()) {
				localMinMax[k].first = pcAttributes[k].min;
				localMinMax[k].second = pcAttributes[k].max;
			}
		}
	}
	//return localMinMax;
}


//ToDo: To speed up the programm, this function could be called only once per renderloop.
static void getGlobalBrushLimits(std::vector<std::pair<float, float>>& globalMinMax) {
	//std::vector<std::pair<float, float>> globalMinMax(pcAttributes.size(), { std::numeric_limits<float>().max(),std::numeric_limits<float>().min() });

	if (violinYScale == ViolinYScaleGlobalBrush || violinYScale == ViolinYScaleBrushes) {
		for (auto& brush : globalBrushes) {
			if (!brush.active) { continue; }
			for (auto& br : brush.brushes) {
				for (auto& minMax : br.second) {
					if (minMax.second.first < globalMinMax[br.first].first) globalMinMax[br.first].first = minMax.second.first;
					if (minMax.second.second > globalMinMax[br.first].second) globalMinMax[br.first].second = minMax.second.second;
				}
			}
		}
		for (int in = 0; in < globalMinMax.size(); ++in) {
			if (globalMinMax[in].first == std::numeric_limits<float>().max()) {
				globalMinMax[in].first = pcAttributes[in].min;
				globalMinMax[in].second = pcAttributes[in].max;
			}
		}
	}
	//return globalMinMax;
}

// Call this function for every attribute independently. Since the order is determined using the first drawlist in the violine plots,
// this function only needs one drawlist.
// Returns a pair of histogram Min and Max.
static void getyScaleDL(unsigned int& dlNr,
	ViolinDrawlistPlot& violinDrawlistPlot,
	std::vector<std::pair<float, float>>& violinMinMax

)
{
	//std::vector<std::pair<float, float>> violinMinMax(pcAttributes.size(), { std::numeric_limits<float>().max(),std::numeric_limits<float>().min() });

	if (violinYScale == ViolinYScaleStandard)
	{
		for (int k = 0; k < pcAttributes.size(); ++k) {
			// Find the attribute in the PC plot to determine min and max values.
			std::string currAttributeName = violinDrawlistPlot.attributeNames[k];
			auto it = std::find_if(pcAttributes.begin(), pcAttributes.end(),
				[&violinDrawlistPlot, k](const Attribute& currObj) {return currObj.name == violinDrawlistPlot.attributeNames[k];  });

			violinMinMax[k] = std::pair(it->min, it->max);

		}
		return;
		//return violinMinMax;
	}


	DrawList* dl = nullptr;
	if (violinYScale == ViolinYScaleLocalBrush || violinYScale == ViolinYScaleBrushes) {
		for (DrawList& draw : g_PcPlotDrawLists) {
			if (draw.name == violinDrawlistPlot.drawLists[dlNr]) {
				dl = &draw;
			}
		}
	}

	switch (violinYScale) {
	case ViolinYScaleLocalBrush:
		getLocalBrushLimits(dl, violinMinMax);
		return;
		break;
	case ViolinYScaleGlobalBrush:
		getGlobalBrushLimits(violinMinMax);
		return;
		break;
	case ViolinYScaleBrushes:
		getGlobalBrushLimits(violinMinMax);
		getLocalBrushLimits(dl, violinMinMax);
		return;
		break;
	}
}

// Call this function for every attribute independently. Since the order is determined using the first drawlist in the violine plots,
// this function only needs one drawlist.
// Returns a pair of histogram Min and Max.
static void getyScaleDLForAttributeViolins(unsigned int& dlNr,
    ViolinPlot& violinAttrPlot,
    std::vector<std::pair<float, float>>& violinMinMax

)
{
    //std::vector<std::pair<float, float>> violinMinMax(pcAttributes.size(), { std::numeric_limits<float>().max(),std::numeric_limits<float>().min() });

    if (violinYScale == ViolinYScaleStandard)
    {
        for (int k = 0; k < pcAttributes.size(); ++k) {
            // Find the attribute in the PC plot to determine min and max values.
            std::string currAttributeName = violinAttrPlot.attributeNames[k];
            auto it = std::find_if(pcAttributes.begin(), pcAttributes.end(),
                [&violinAttrPlot, k](const Attribute& currObj) {return currObj.name == violinAttrPlot.attributeNames[k];  });

            violinMinMax[k] = std::pair(it->min, it->max);

        }
        return;
        //return violinMinMax;
    }


    DrawList* dl = nullptr;
    if (violinYScale == ViolinYScaleLocalBrush || violinYScale == ViolinYScaleBrushes) {
        for (DrawList& draw : g_PcPlotDrawLists) {
            if (draw.name == violinAttrPlot.drawLists[dlNr].name) {
                dl = &draw;
            }
        }
    }

    switch (violinYScale) {
    case ViolinYScaleLocalBrush:
        getLocalBrushLimits(dl, violinMinMax);
        return;
        break;
    case ViolinYScaleGlobalBrush:
        getGlobalBrushLimits(violinMinMax);
        return;
        break;
    case ViolinYScaleBrushes:
        getGlobalBrushLimits(violinMinMax);
        getLocalBrushLimits(dl, violinMinMax);
        return;
        break;
    }
}

/** Computes histogram values for one single Drawlist. */
static void exeComputeHistogram(std::string& name, std::vector<std::pair<float, float>>& minMax, VkBuffer data, uint32_t amtOfData, VkBuffer indices, uint32_t amtOfIndices, VkBufferView indicesActivations, bool callForviolinAttributePlots = false) {
	if (histogramManager->adaptMinMaxToBrush) {
		std::vector<std::pair<float, float>> violinMinMax(minMax.size(), { std::numeric_limits<float>().max(),std::numeric_limits<float>().min() });

		unsigned int currDLNr = 0;
        if (callForviolinAttributePlots || (violinDrawlistPlots.size() == 0))
        {
            getyScaleDLForAttributeViolins(currDLNr, violinAttributePlots[0], violinMinMax);
        }
        else
        {
            getyScaleDL(currDLNr, violinDrawlistPlots[0], violinMinMax);
        }


		float minimalRelativeRange = 0.001;
		// Only use the new range if it is large enough to be computationally stable.
		for (int i = 0; i < minMax.size(); ++i) {
			if ((violinMinMax[i].second - violinMinMax[i].first) > minimalRelativeRange * (minMax[i].second - minMax[i].first)) {
				minMax[i] = violinMinMax[i];
			}
			else {
				minMax[i].first = violinMinMax[i].first;
				minMax[i].second = minMax[i].first + minimalRelativeRange * (minMax[i].second - minMax[i].first);
			}
		}
	}

	histogramManager->computeHistogramm(name, minMax, data, amtOfData, indices, amtOfIndices, indicesActivations);

}


static void updateHistogramComparisonDL(unsigned int& idVioDLPlts)
{
    if (violinPlotDLIdxInListForHistComparison[idVioDLPlts] == -1) {return;}
    std::string nameRep = violinDrawlistPlots[idVioDLPlts].drawLists[violinPlotDLIdxInListForHistComparison[idVioDLPlts]];
    for (unsigned int i = 0; i < violinDrawlistPlots[idVioDLPlts].drawLists.size(); ++i)
    {
        if (violinDrawlistPlots[idVioDLPlts].histDistToRepresentative.size() <= i){violinDrawlistPlots[idVioDLPlts].histDistToRepresentative.push_back(0);}
        std::string name =  violinDrawlistPlots[idVioDLPlts].drawLists[i];
        float dist = histogramManager->computeHistogramDistance(nameRep, name, &(violinDrawlistPlots[idVioDLPlts].activeAttributes) ,(int) violinPlotDLUseRenderedBinsForHistComp);


        violinDrawlistPlots[idVioDLPlts].histDistToRepresentative[i] = dist;//(float)((int)(dist * 100 + .5));

        std::cout << std::to_string(violinDrawlistPlots[idVioDLPlts].histDistToRepresentative[i]) << std::endl;
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

	//ordering active indices if priority rendering is enabled
	if (priorityReorder) {
		priorityReorder = false;
		std::vector<float*>* data;
		for (DataSet& ds : g_PcPlotDataSets) {
			if (ds.name == dl.parentDataSet) {
				data = &ds.data;
				break;
			}
		}
		int p = priorityAttribute;
#ifdef GPUSORT
		//sorting with thrus
		std::vector<float> keys(dl.activeInd.size());
		int c = 0;
		for (int i : dl.activeInd) {
			keys[c++] = (*data)[i][p] - priorityAttributeCenterValue;
		}
		thrust::sort_by_key(keys.begin(), keys.end(), dl.activeInd.begin(), thrust::greater<float>());

#else
		std::sort(dl.indices.begin(), dl.indices.end(), [data, p](int a, int b) {return fabs((*data)[a][p] - priorityAttributeCenterValue) > fabs((*data)[b][p] - priorityAttributeCenterValue); });
		VkUtil::uploadData(g_Device, dl.dlMem, dl.indicesBufferOffset, dl.indices.size() * sizeof(uint32_t), dl.indices.data());
		//std::sort(dl.activeInd.begin(), dl.activeInd.end(), [data, p](int a, int b) {return fabs((*data)[a][p] - priorityAttributeCenterValue) > fabs((*data)[b][p] - priorityAttributeCenterValue); });
#endif
	}

	/*
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
	*/

	//using the compute pipeline to update the indexbuffer for rendering
	VkBuffer infos;
	VkDeviceMemory memory;
	uint32_t infosSize = (4 + order.size()) * sizeof(uint32_t);
	VkUtil::createBuffer(g_Device, infosSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &infos);
	VkMemoryRequirements memReq;
	vkGetBufferMemoryRequirements(g_Device, infos, &memReq);
	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memReq.size;
	allocInfo.memoryTypeIndex = VkUtil::findMemoryType(g_PhysicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	VkResult err = vkAllocateMemory(g_Device, &allocInfo, nullptr, &memory);
	check_vk_result(err);
	err = vkBindBufferMemory(g_Device, infos, memory, 0);
	check_vk_result(err);

	uint32_t* inf = new uint32_t[4 + order.size()];
	inf[0] = order.size();
	inf[1] = pcAttributes.size();
	inf[2] = dl.indices.size();
	//inf[3] is padding
	for (int i = 0; i < order.size(); ++i) {
		inf[4 + i] = order[i].first;
	}
	VkUtil::uploadData(g_Device, memory, 0, infosSize, inf);

	std::vector<VkDescriptorSetLayout> layouts;
	layouts.push_back(c_IndexPipelineDescSetLayout);
	VkUtil::createDescriptorSets(g_Device, layouts, g_DescriptorPool, &c_IndexPipelineDescSet);
	VkUtil::updateDescriptorSet(g_Device, infos, infosSize, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, c_IndexPipelineDescSet);
	VkUtil::updateDescriptorSet(g_Device, dl.indicesBuffer, dl.indices.size() * sizeof(uint32_t), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, c_IndexPipelineDescSet);
	VkUtil::updateTexelBufferDescriptorSet(g_Device, dl.activeIndicesBufferView, 2, c_IndexPipelineDescSet);
	VkUtil::updateDescriptorSet(g_Device, dl.indexBuffer, dl.indices.size() * (pcAttributes.size() + 3) * sizeof(uint32_t), 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, c_IndexPipelineDescSet);

	VkCommandBuffer indexCommands;
	VkUtil::createCommandBuffer(g_Device, g_PcPlotCommandPool, &indexCommands);
	vkCmdBindPipeline(indexCommands, VK_PIPELINE_BIND_POINT_COMPUTE, c_IndexPipeline);
	vkCmdBindDescriptorSets(indexCommands, VK_PIPELINE_BIND_POINT_COMPUTE, c_IndexPipelineLayout, 0, 1, &c_IndexPipelineDescSet, 0, {});
	int patchAmount = dl.indices.size() / LOCALSIZE;
	patchAmount += (dl.indices.size() % LOCALSIZE) ? 1 : 0;
	vkCmdDispatch(indexCommands, patchAmount, 1, 1);
	VkUtil::commitCommandBuffer(g_Queue, indexCommands);
	err = vkQueueWaitIdle(g_Queue);
	check_vk_result(err);

	/*
	//Debugging code for the creation of the index buffer
	uint32_t* downloadedInd = new uint32_t[dl.indices.size()];
	VkUtil::downloadData(g_Device, dl.dlMem, dl.indicesBufferOffset, dl.indices.size() * sizeof(uint32_t), downloadedInd);
	for (int i = 0; i < 100; ++i) {
		std::cout << downloadedInd[i] << std::endl;
	}

	uint32_t* updatedInd = new uint32_t[dl.indices.size() * (pcAttributes.size() + 3)];
	VkUtil::downloadData(g_Device, dl.dlMem, dl.indexBufferOffset, dl.indices.size() * (pcAttributes.size() + 3) * sizeof(uint32_t), updatedInd);
	for (int i = 0; i < 1000; ++i) {
		std::cout << updatedInd[i] << ",";
		if ((i+1) % ((pcAttributes.size() + 3)) == 0)
			std::cout << std::endl;
	}
	delete[] updatedInd;
	*/

	vkFreeCommandBuffers(g_Device, g_PcPlotCommandPool, 1, &indexCommands);
	vkFreeDescriptorSets(g_Device, g_DescriptorPool, 1, &c_IndexPipelineDescSet);
	vkDestroyBuffer(g_Device, infos, nullptr);
	vkFreeMemory(g_Device, memory, nullptr);
	delete[] inf;
}

static void updateAllDrawListIndexBuffer() {
	for (DrawList& dl : g_PcPlotDrawLists) {
		updateDrawListIndexBuffer(dl);
	}
}

static void upatePriorityColorBuffer() {
	priorityListIndex = (priorityListIndex < g_PcPlotDrawLists.size()) ? priorityListIndex : g_PcPlotDrawLists.size() - 1;
	auto it = g_PcPlotDrawLists.begin();
	std::advance(it, priorityListIndex);
	DrawList* dl = &(*it);

	std::vector<float*>* data;
	for (DataSet& ds : g_PcPlotDataSets) {
		if (ds.name == dl->parentDataSet) {
			data = &ds.data;
			break;
		}
	}

	//float denom = (pcAttributes[priorityAttribute].max - pcAttributes[priorityAttribute].min);
	float denom = (fabs(pcAttributes[priorityAttribute].max - priorityAttributeCenterValue) > fabs(pcAttributes[priorityAttribute].min - priorityAttributeCenterValue)) ? fabs(pcAttributes[priorityAttribute].max - priorityAttributeCenterValue) : fabs(pcAttributes[priorityAttribute].min - priorityAttributeCenterValue);

	//uint8_t* color = new uint8_t[dl.indices.size() * 3];
	//for (int i : dl.indices) {
	//	int index = (((*data)[i][priorityAttribute] - pcAttributes[priorityAttribute].min) / denom) * (sizeof(colorPalette)/(4*sizeof(*colorPalette)));
	//	color[i * 3] = colorPalette[index * 4];
	//	color[i * 3 + 1] = colorPalette[index * 4 + 1];
	//	color[i * 3 + 2] = colorPalette[index * 4 + 2];
	//}
	float* color = new float[data->size()];
	for (int i : dl->indices) {
		color[i] = 1 - .9f * (fabs((*data)[i][priorityAttribute] - priorityAttributeCenterValue) / denom);
	}

	void* d;
	vkMapMemory(g_Device, dl->dlMem, dl->priorityColorBufferOffset, data->size() * sizeof(float), 0, &d);
	memcpy(d, color, data->size() * sizeof(float));
	vkUnmapMemory(g_Device, dl->dlMem);

	delete[] color;

	priorityReorder = true;
	updateDrawListIndexBuffer(*dl);
}

static void updateMaxHistogramValues(ViolinDrawlistPlot& plot) {
	plot.maxGlobalValue = 0;
	plot.maxValues = std::vector<float>(plot.maxValues.size(), 0);
	
	for (int index: plot.drawListOrder) {
		if (index == 0xffffffff)
			continue;

		HistogramManager::Histogram& hist = histogramManager->getHistogram(plot.drawLists[index]);
		for (int j = 0; j < hist.maxCount.size(); ++j) {
			if (hist.maxCount[j] > plot.maxValues[j]) {
				plot.maxValues[j] = hist.maxCount[j];
			}
			if (hist.maxCount[j] > plot.maxGlobalValue) {
				plot.maxGlobalValue = hist.maxCount[j];
			}
		}
	}
}

static void updateIsoSurface(GlobalBrush& gb) {
	int amtOfLines = 0;
	for (auto& dl : g_PcPlotDrawLists) amtOfLines += dl.indices.size();
	if ((ImGui::IsMouseDown(0) && liveBrushThreshold < amtOfLines) || !coupleIsoSurfaceRenderer) return;
	if (coupleBrushIsoSurfaceRenderer && enableBrushIsoSurfaceWindow) {
		if (brushIsoSurfaceRenderer->brushColors.find(gb.name) != brushIsoSurfaceRenderer->brushColors.end()) {
			std::vector<std::vector<std::pair<float, float>>> minMax(pcAttributes.size());
			for (auto& axis : gb.brushes) {
				for (auto& m : axis.second) {
					minMax[axis.first].push_back(m.second);
				}
			}
			brushIsoSurfaceRenderer->updateBrush(gb.name, minMax);
		}
	}
	int index = -1;
	for (auto& db : isoSurfaceRenderer->drawlistBrushes) {
		++index;
		if (gb.name != db.brush) continue;
		DrawList* dl = nullptr;
		for (DrawList& draw : g_PcPlotDrawLists) {
			if (draw.name == db.drawlist) {
				dl = &draw;
				break;
			}
		}
		if (!dl) continue;
		std::vector<float*>* data = nullptr;
		for (DataSet& ds : g_PcPlotDataSets) {
			if (dl->parentDataSet == ds.name) {
				data = &ds.data;
				break;
			}
		}

		std::vector<unsigned int> attr;
		std::vector<std::pair<float, float>> minMax;
		for (int i = 0; i < pcAttributes.size(); ++i) {
			attr.push_back(i);
			minMax.push_back({ pcAttributes[i].min, pcAttributes[i].max });
		}
		std::vector<std::vector<std::pair<float, float>>> miMa(pcAttributes.size());
		std::vector<uint32_t> brushIndices;
		for (auto axis : globalBrushes[selectedGlobalBrush].brushes) {
			if (axis.second.size()) brushIndices.push_back(axis.first);
			for (auto& brush : axis.second) {
				miMa[axis.first].push_back(brush.second);
			}
		}

		glm::uvec3 posIndices{ 0,2,1 };

		if (isoSurfaceRegularGrid) {
			isoSurfaceRenderer->update3dBinaryVolume(isoSurfaceRegularGridDim[0], isoSurfaceRegularGridDim[1], isoSurfaceRegularGridDim[2], pcAttributes.size(), brushIndices, minMax, posIndices, dl->buffer, data->size() * pcAttributes.size() * sizeof(float), dl->indicesBuffer, dl->indices.size(), miMa, index);
		}
		else {
			if (!ImGui::IsMouseDown(0))
				isoSurfaceRenderer->update3dBinaryVolume(SpacialData::rlatSize, SpacialData::altitudeSize, SpacialData::rlonSize, pcAttributes.size(), attr, minMax, posIndices, *data, dl->indices, miMa, index);
		}
	}
}

static void updateIsoSurface(DrawList& dl) {
	int amtOfLines = 0;
	for (auto& dl : g_PcPlotDrawLists) amtOfLines += dl.indices.size();
	if ((ImGui::IsMouseDown(0) && liveBrushThreshold < amtOfLines) || !coupleIsoSurfaceRenderer) return;

	int index = -1;
	for (auto& db : isoSurfaceRenderer->drawlistBrushes) {
		++index;
		if (dl.name != db.drawlist || db.brush.size()) continue;
		uint32_t w = db.gridDimensions[0];
		uint32_t h = db.gridDimensions[1];
		uint32_t d = db.gridDimensions[2];
		uint32_t posIndices[3];
		isoSurfaceRenderer->getPosIndices(index, posIndices);
		std::vector<std::pair<float, float>> posBounds(3);
		for (int i = 0; i < 3; ++i) {
			posBounds[i].first = pcAttributes[posIndices[i]].min;
			posBounds[i].second = pcAttributes[posIndices[i]].max;
		}
		if (!isoSurfaceRegularGrid) {
			posBounds[0].first = SpacialData::rlat[0];
			posBounds[0].second = SpacialData::rlat[SpacialData::rlatSize - 1];
			posBounds[1].first = SpacialData::altitude[0];
			posBounds[1].second = SpacialData::altitude[SpacialData::altitudeSize - 1];
			posBounds[2].first = SpacialData::rlon[0];
			posBounds[2].second = SpacialData::rlon[SpacialData::rlonSize - 1];
		}
		std::vector<float*>* data;
		for (DataSet& ds : g_PcPlotDataSets) {
			if (ds.name == dl.parentDataSet) {
				data = &ds.data;
				break;
			}
		}
		isoSurfaceRenderer->update3dBinaryVolume(w, h, d, posIndices, posBounds, pcAttributes.size(), data->size(), dl.buffer, dl.activeIndicesBufferView, dl.indices.size(), dl.indicesBuffer, isoSurfaceRegularGrid, index);
	}
}

static bool updateActiveIndices(DrawList& dl) {
	//safety check to avoid updates of large drawlists. Update only occurs when mouse was released
	if (dl.indices.size() > liveBrushThreshold) {
		if (ImGui::GetIO().MouseDown[0]) return false;
	}

	//getting the parent dataset data
	std::vector<float*>* data;
	for (DataSet& ds : g_PcPlotDataSets) {
		if (ds.name == dl.parentDataSet) {
			data = &ds.data;
			break;
		}
	}
	activeBrushRatios[dl.name] = 0;
	for (GlobalBrush& b : globalBrushes) {
		b.lineRatios[dl.name] = 0;
	}
	/*
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

		//checking gloabl brushes
		if (toggleGlobalBrushes) {
			bool orr = globalBrushes.size() == 0, andd = true, anyActive = false;
			for (GlobalBrush& b : globalBrushes) {
				bool lineKeep = true;
				if (b.fractureDepth > 0) { //fractured brush
					std::vector<std::vector<std::pair<float, float>>>& fractures = b.fractions;
					lineKeep = false;
					for (auto& frac : fractures) {
						bool good = true;
						for (int j = 0; j < frac.size(); j++) {
							if ((*data)[i][b.attributes[j]] < frac[j].first || (*data)[i][b.attributes[j]] > frac[j].second) {
								good = false;
								break;
							}
						}
						if (good)
							lineKeep = true;
					}
				}
				else { //standard global brush
					//bool lineKeep = true;
					continue;
					for (auto& br : b.brushes) {
						bool good = false;
						for (auto& brush : br.second) {
							if ((*data)[i][br.first] >= brush.second.first && (*data)[i][br.first] <= brush.second.second) {
								good = true;
								break;
							}
						}
						if (!good)
							lineKeep = false;
					}
				}

				if (b.active) {
					orr |= lineKeep;
					andd &= lineKeep;
					anyActive = true;
				}

				if (lineKeep)
					b.lineRatios[dl.name] ++;
			}
			if (brushCombination == 1 && !andd) {
				//goto nextInd;
				keep = false;
			}
			if (brushCombination == 0 && !orr && anyActive) {
				//goto nextInd;
				keep = false;
			}
		}

		if (keep) {
			dl.activeInd.push_back(i);
			activeBrushRatios[dl.name] += 1;
		}
	}*/
	std::map<int, std::vector<std::pair<float, float>>> brush;

	bool firstBrush = true;
	int globalRemainingLines = dl.indices.size();

	//apply local brush
	brush.clear();
	for (int i = 0; i < pcAttributes.size(); i++) {
		for (Brush& b : dl.brushes[i]) {
			brush[i].push_back(b.minMax);
		}
	}
	if (brush.size()) {
		std::pair<uint32_t, int> res = gpuBrusher->brushIndices(brush, data->size(), dl.buffer, dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView, pcAttributes.size(), true, brushCombination == 1, globalBrushes.size() == 0);
		globalRemainingLines = res.second;
		firstBrush = false;
	}
	
	//apply global brushes
	std::vector<int> globalIndices;
	bool globalBrushesActive = false;
	if (toggleGlobalBrushes && !dl.immuneToGlobalBrushes) {
		int c = 1;
		for (GlobalBrush& gb : globalBrushes) {
			if (gb.fractureDepth > 0) { //fractured brush
				if (!gb.active) continue;
				for (auto& a : gb.brushes) {
					if (a.second.size()) {
						globalBrushesActive = true;
						break;
					}
				}
				std::pair<uint32_t, int> res;// = gpuBrusher->brushIndices(gb.fractions, gb.attributes, data->size(), dl.buffer, dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView, pcAttributes.size(), firstBrush, brushCombination == 1, c == globalBrushes.size());
				if (gb.useMultivariate) {
					std::vector<std::pair<float, float>> origBounds = gb.kdTree->getOriginalBounds();
					float x[30]{};
					std::vector<int> activeInd;
					//for (uint32_t lineIndex : dl.indices) {
					//	//fill x
					//	int amtOfMultvarAxes = 0;
					//	for (int j = 0; j < gb.attributes.size(); ++j) {
					//		x[j] = (data[lineIndex][gb.attributes[j]] - origBounds[j].first) / (origBounds[j].second - origBounds[j].first);
					//	}
					//
					//	bool lineKeep = false;
					//	for (auto& multvar:gb.multivariates) {
					//		//if (multvar.detCov < TINY) continue;
					//		//
					//		////doing calculation of: (x - mu)' * COV^(-1) * (x - mu)
					//		//Eigen::MatrixXd invCo = multvar.cov;
					//		//double s = 0;
					//		//for (int c = 0; c < multvar.invCov.size(); ++c) {
					//		//	double m = 0;
					//		//	for (int c1 = 0; c1 < multvar.invCov[c].size(); ++c1) {
					//		//		m += (x[c1] - multvar.mean[c1]) * invCo(c, c1);//multvar.invCov[c][c1];
					//		//	}
					//		//
					//		//	s += (x[c] - multvar.mean[c]) * m;
					//		//}
					//		////s = multvar.m[preFactorBase] * exp(-.5f * s);
					//		//float gaussMin = 1 * gb.attributes.size();	//vector of 3's squared (amtOfMultvarAxes 3's are in the vector)
					//		////checking if the gauss value is in range of 3 sigma(over 99% of the points are then accounted for)
					//		//if (s <= gaussMin) {			//we are only comparing the exponents, as the prefactors of the mulivariate normal distributions are the same
					//		//	lineKeep = true;
					//		//	break;
					//		//}
					//
					//		//doing the check via pca
					//		Eigen::MatrixXd& pc = multvar.pc;
					//		int multvarBoundsInd = -1;
					//		double s = 0;
					//		bool nope = false;
					//		for (int c = 0; c < multvar.invCov.size(); ++c) {
					//			double m = 0;
					//			for (int c1 = 0; c1 < multvar.invCov[c].size(); ++c1) {
					//				m += (x[c1] - multvar.mean[c1]) * pc(c1, c);				//project point onto the pca axis
					//			}
					//			if (multvar.sv(c) > 1e-20) {									//standard gaussian check
					//				s += std::pow(m, 2) / std::pow(multvar.sv(c), 2);			//x^2 / sigma^2
					//			}
					//			else {
					//				++multvarBoundsInd;
					//				if (m<multvar.pcBounds[multvarBoundsInd].first || m > multvar.pcBounds[multvarBoundsInd].second) {
					//					nope = true;
					//					break;
					//				}
					//			}
					//		}
					//		assert(multvarBoundsInd == multvar.pcBounds.size() - 1);
					//		//s = multvar.m[preFactorBase] * exp(-.5f * s);
					//		float gaussMin = std::pow(multivariateStdDivThresh,2) * multvar.pcInd.size();	//vector of 3's squared (amtOfMultvarAxes 3's are in the vector)
					//		//checking if the gauss value is in range of 3 sigma(over 99% of the points are then accounted for)
					//		if (s <= gaussMin && !nope) {			//we are only comparing the exponents, as the prefactors of the mulivariate normal distributions are the same
					//			lineKeep = true;
					//			break;
					//		}
					//	}
					//	if (lineKeep)
					//		activeInd.push_back(lineIndex);
					//}
					//bool* actives = new bool[data.size()]{};
					//for (int i : activeInd) {
					//	actives[i] = true;
					//}
					//VkUtil::uploadData(g_Device, dl.dlMem, dl.activeIndicesBufferOffset, data.size(), actives);
					//delete[] actives;
					//res = { activeInd.size(),activeInd.size() };
					res = gpuBrusher->brushIndices(gb.multivariates, gb.kdTree->getOriginalBounds(), gb.attributes, data->size(), dl.buffer, dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView, pcAttributes.size(), firstBrush, brushCombination == 1, c == globalBrushes.size(), multivariateStdDivThresh);
				}
				else {
					res = gpuBrusher->brushIndices(gb.fractions, gb.attributes, data->size(), dl.buffer, dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView, pcAttributes.size(), firstBrush, brushCombination == 1, c == globalBrushes.size());
				}
				gb.lineRatios[dl.name] = res.first;
				globalRemainingLines = res.second;
				firstBrush = false;
				++c;
			}
			else {
				if (!gb.active) continue;
				for (auto& a : gb.brushes) {
					if (a.second.size()) {
						globalBrushesActive = true;
						break;
					}
				}
				brush.clear();
				for (auto b : gb.brushes) {
					for (auto br : b.second) {
						brush[b.first].push_back(br.second);
					}
				}
				if (!brush.size()) continue;
				std::pair<uint32_t, int> res = gpuBrusher->brushIndices(brush, data->size(), dl.buffer, dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView, pcAttributes.size(), firstBrush, brushCombination == 1, c == globalBrushes.size());
				gb.lineRatios[dl.name] = res.first;
				globalRemainingLines = res.second;
				firstBrush = false;
				++c;
			}
		}
	}

	//if no brush is active, reset the active indices
	if (!brush.size() && !globalBrushesActive) {
		std::vector<float*>* data;
		for (DataSet& ds : g_PcPlotDataSets) {
			if (ds.name == dl.parentDataSet) {
				data = &ds.data;
				break;
			}
		}
		std::vector<uint8_t> actives(data->size(), 0);			//vector with 0 initialized everywhere
		for (int i : dl.indices) {								//setting all active indices to true
			actives[i] = 1;
		}
		VkUtil::uploadData(g_Device, dl.dlMem, dl.activeIndicesBufferOffset, data->size() * sizeof(bool), actives.data());
	}

	activeBrushRatios[dl.name] = globalRemainingLines;

	// Computing ratios for the pie charts
	if (computeRatioPtsInDLvsIn1axbrushedParent && drawHistogramm) {
		//Todo: dl.indices,size() of parent!
		//dl.brushedRatioToParent = std::vector<float>(pcAttributes.size(), (float)globalRemainingLines/dl.indices.size());			//instantiate with the standard active lines

		DataSet* parentDS = nullptr;
		// DataSet* currParentDataSet = nullptr;
		// Determine parent drawlist
		for (auto& ds : g_PcPlotDataSets)
		{
			for (auto& currdl : ds.drawLists)
			{
				// Checking the buffer Reference should be enough, nevertheless, we check all 3 conditions.
				if ((currdl.name == dl.parentTemplateList->name) && (currdl.indices.size() == dl.parentTemplateList->indices.size())  && (&currdl.buffer == &(dl.parentTemplateList->buffer) ) )
				{
					parentDS = &ds;
					break;

				}
			}
			if (parentDS != nullptr) { break; }
		}

		dl.brushedRatioToParent = std::vector<float>(pcAttributes.size(), (float)globalRemainingLines / parentDS->data.size());			//instantiate with the standard active lines
		


		if (false)
		for (GlobalBrush& gb : globalBrushes) {
			if (!gb.active) continue;
			for (auto b : gb.brushes) {
				for (auto br : b.second) {
					brush.clear();
					brush[b.first].push_back(br.second);
					std::pair<uint32_t, int> res = gpuBrusher->brushIndices(brush, data->size(), dl.buffer, dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView, pcAttributes.size());
					if(res.first)
						dl.brushedRatioToParent[b.first] = float(globalRemainingLines) / res.first;
					else
						dl.brushedRatioToParent[b.first] = 0;
				}
			}
			break;
		}

		if (true) {
			int parentCount = 0;
			std::vector<bool> parentActive;
			parentActive.clear();

			for (auto& gb : globalBrushes) {
				//for (int iax = 0; iax < dl.parentTemplateList->minMax.size(); iax++) {
				if (gb.fractureDepth > 0) {
					if (parentActive.size() == 0) {
						parentActive.resize(parentDS->data.size());
						for (int i = 0; i < parentActive.size(); ++i) { parentActive[i] = false; }
					}

					int iax = 0;
					for (auto& ax : gb.attributes) {
						// if (gb.fractureDepth > 0) {}if (gb.useMultivariate) {}gb.fractions

							// Multivariate works the same as normal brushes for those pie-charts.
							//if (gb.useMultivariate) {
							//
							//}
							//else {
						for (std::vector<std::pair<float, float>>& currFr : gb.fractions) {
							
							float currBrMin = currFr[iax].first;
							float currBrMax = currFr[iax].second;

							int iVal = -1;
							for (auto& val : parentDS->data)
							{
								iVal++;
								if ((val[ax] >= currBrMin) && (val[ax] <= currBrMax))
								{
									parentActive[iVal] = true;
									++parentCount;
								}
							}
						}
						if (parentCount != 0)
						{	
							int nrActiveInParent = 0;
							for (int i = 0; i < parentActive.size(); ++i) {
								nrActiveInParent += int(parentActive[i]);
							}

							dl.brushedRatioToParent[ax] = float(globalRemainingLines) / nrActiveInParent;
							for (int i = 0; i < parentActive.size(); ++i) {
								parentActive[i] = false;
							}
						}
						parentCount = 0;
						//}
						++iax;
					}
				}
				else {
					for (int iax = 0; iax < dl.parentTemplateList->minMax.size(); iax++) {
						//for (auto br : b.second) {
						for (auto& currBr : gb.brushes.at(iax)){
						
							brush.clear();
							brush[currBr.first].push_back(currBr.second);
							std::pair<uint32_t, int> res = gpuBrusher->brushIndices(brush, data->size(), dl.buffer, dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView, pcAttributes.size());

							std::cout << res.first << "\n";

							float currBrMin = currBr.second.first;
							float currBrMax = currBr.second.second;
							for (auto& val : parentDS->data)
							{
								if ((val[iax] >= currBrMin) && (val[iax] <= currBrMax))
								{
									++parentCount;
								}
							}
						}
						if (parentCount != 0)
						{
							dl.brushedRatioToParent[iax] = float(globalRemainingLines) / parentCount;
						}

						parentCount = 0;
					}
				}
				break;
			}
		}


	}
	//for (GlobalBrush& b : globalBrushes) {
		//b.lineRatios[dl.name] /= dl.indices.size();
	//}
	for (auto& it : activeBrushRatios) {
		if (it.first == dl.name)
			it.second /= dl.indices.size();
	}

	//todo: delete this
	// Compute the ratio of points in the DL vs the one in the parents' 1 axis brush
	//if (computeRatioPtsInDLvsIn1axbrushedParent && drawHistogramm)
	//{
	//	dl.brushedRatioToParent.clear();
	//	// Assume that there are no global brushes atm. So active indices / parentds indices.
	//	// ax, min,max
	//	/*
	//	std::pair<float, float> pp (std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
	//	std::vector<std::pair<float, float>> vpp{ pp };
	//	std::map<int, std::vector<std::pair<float, float>>> extremeMinMaxBrush{ {0,vpp} };
	//
	//	std::pair<uint32_t, int> res = gpuBrusher->brushIndices(
	//		extremeMinMaxBrush, data->size(),
	//		dl.buffer, 
	//		dl.indicesBuffer, 
	//		dl.indices.size(), 
	//		dl.activeIndicesBufferView, 
	//		pcAttributes.size(), 
	//		firstBrush, 
	//		brushCombination == 1, 
	//		true);
	//		*/
	//
	//	for (int i = 0; i < dl.parentTemplateList->minMax.size(); ++i)
	//	{
	//		// Divide number of all points in dl and parent dl by each other. This is the default, if no brushes are active.
	//		float currRatio = dl.indices.size() / dl.parentTemplateList->indices.size();
	//		dl.brushedRatioToParent.push_back(currRatio);
	//	}
	//
	//	for (auto& gb : globalBrushes)
	//	{
	//		
	//		if (!gb.active) { continue; }
	//		std::map<int, std::vector<std::pair<float, float>>> brush;
	//		brush.clear();
	//		for (auto b : gb.brushes) {
	//			for (auto br : b.second) {
	//				brush[b.first].push_back(br.second);
	//			}
	//		}
	//
	//
	//		/*res = gpuBrusher->brushIndices(
	//			brush, data->size(),
	//			dl.buffer,
	//			dl.indicesBuffer,
	//			dl.indices.size(),
	//			dl.activeIndicesBufferView,
	//			pcAttributes.size(),
	//			firstBrush,
	//			brushCombination == 1,
	//			true);*/
	//		
	//		for (int i = 0; i < dl.parentTemplateList->minMax.size(); ++i)
	//		{
	//			// Divide number of all points in dl and parent dl by each other. This is the default, if no brushes are active.
//To//DO : Hier die anzahl aktiver Indices von Dl  / Anzahl indices in ParentDataset gesamt
	//			float currRatio = dl.indices.size() / dl.parentTemplateList->indices.size();
	//		}
	//
	//
	//		// Loop through the attributes which are brushed
	//		for (auto& ax : gb.attributes)
	//		{
//To//DO Hier die anzahl aktiver Indices von Dl, kann auch vor den for loop, bzw. wurde da ja dann schon berechnet 
	//			int activeIndicesInDL = dl.indices.size();
	//			//int indicesInDL = globalRemainingLines;
	//
	//			// Compute indices brushed in parent by 1D brush
	//
	//			DrawList* currParentDrawlist = nullptr;
	//			DataSet* currParentDataSet = nullptr;
	//			// Determine parent drawlist
	//			for (auto& currdl : g_PcPlotDrawLists)
	//			{
	//				if (currdl.name == dl.parentDataSet)
	//				{
	//					currParentDrawlist = &currdl;
	//
	//				}
	//			}
	//			
	//			for (auto& currds : g_PcPlotDataSets)
	//			{
	//				if (currds.name == dl.parentDataSet)
	//				{
	//					currParentDataSet = &currds;
	//				}
	//			}
	//			
	//			int parentCount = 0;
	//			// Access histogramm data, loop through all brushes for the one axis, and add up the indices in the range of the brushes which are in the histogram
// T//oDo : Geht das besser? Wir haben einen nDBrush. Zersplitten in 1D Brushes, und jeden 1D Brush auf Parent anwenden, Linien zählen
	//			if (true)//(!gb.fractureDepth)
	//			{
	//				// auto it = gb.brushes.find(ax);
	//				for (auto& currBr : gb.brushes.at(ax))
	//				{
	//					{
	//						float currBrMin = currBr.second.first;
	//						float currBrMax = currBr.second.second;
	//						for (auto& val : currParentDataSet->data)
	//						{
	//							if ((val[ax] > currBrMin) && (val[ax] < currBrMax))
	//							{
	//								++parentCount;
	//							}
	//						}
	//					}
	//
	//				}
	//
	//			}
	//
	//			// Ratio das gespeichert werden muss ist jetzt  indicesInDL / parentCount
	//			dl.brushedRatioToParent[ax] = activeIndicesInDL / parentCount;
	//			
	//			
	//
	//		}
	//
	//		for (int i = 0; i < dl.parentTemplateList->minMax.size(); ++i)
	//		{
	//			std::cout << "Pie-chart percent for brushed axes: " << dl.brushedRatioToParent[i] << std::endl;
	//		}
	//		// Only do this for the first global brush which is active...
	//		break;
	//	}
	//}




	//updating the standard indexbuffer
	updateDrawListIndexBuffer(dl);

	//rendering the updated active points in the bubble plotter
	if (coupleBubbleWindow) {
		bubblePlotter->render();
	}

	if (coupleViolinPlots && histogramManager->containsHistogram(dl.name)) {
		std::vector<std::pair<float, float>> minMax;
		for (Attribute& a : pcAttributes) {
			minMax.push_back({ a.min,a.max });
		}
		DataSet* ds;
		for (DataSet& d : g_PcPlotDataSets) {
			if (d.name == dl.parentDataSet) {
				ds = &d;
				break;
			}
		}
		exeComputeHistogram(dl.name, minMax, dl.buffer, ds->data.size(), dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView);

		//histogramManager->computeHistogramm(dl.name, minMax, dl.buffer, ds->data.size(), dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView);
		HistogramManager::Histogram& hist = histogramManager->getHistogram(dl.name);
        for (unsigned int i = 0; i < violinDrawlistPlots.size(); ++i) {
			bool contains = false;
			for (auto& s : violinDrawlistPlots[i].drawLists) {
				if (s == dl.name) {
					contains = true;
					break;
				}
			}
			if (!contains) continue;

			updateMaxHistogramValues(violinDrawlistPlots[i]);
            updateHistogramComparisonDL(i);
		}
	}

	if (coupleIsoSurfaceRenderer && enableIsoSurfaceWindow) {
		updateIsoSurface(dl);
	}

	//setting the median to no median to enforce median recalculation
	dl.activeMedian = 0;
	return true;
}

//This method does the same as updataActiveIndices, only for ALL drawlists
//whenever possible use updataActiveIndices, not updateAllActiveIndicess
static bool updateAllActiveIndices() {
	bool ret = false;
	for (DrawList& dl : g_PcPlotDrawLists) {
		ret = updateActiveIndices(dl);
	}
	return ret;
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
	ubo.enableMapping = enableDensityMapping | ((uint8_t)(histogrammDensity && enableDensityMapping)) * 2 | uint32_t(enableDensityGreyscale)<<2;
	ubo.gaussRange = densityRadius;
	ubo.imageHeight = g_PcPlotHeight;
	int amtOfIndices = 0;
	for (int i = 0; i < pcAttributes.size(); i++) {
		if (pcAttributeEnabled[i]) amtOfIndices++;
	}
	ubo.gap = (1 - histogrammWidth / 2) / (amtOfIndices - 1);
	if (histogrammDrawListComparison != -1) {
		float offset = 0;
		int activeHists = 0;
		int c = 0;
		for (auto it = g_PcPlotDrawLists.begin(); it != g_PcPlotDrawLists.end(); ++it, c++) {
			if (it->showHistogramm) {
				activeHists++;
				if (c == histogrammDrawListComparison) {
					offset = activeHists;
				}
			}
			else if (c == histogrammDrawListComparison) {
				std::cout << "Histogramm to compare to is not active." << std::endl;
			}
		}
		ubo.compare = (offset / activeHists - (1 / (2.0f * activeHists))) * histogrammWidth / 2;
	}
	else {
		ubo.compare = -1;
	}
	void* d;
	vkMapMemory(g_Device, g_PcPlotIndexBufferMemory, g_PcPLotDensityUboOffset, sizeof(DensityUniformBuffer), 0, &d);
	memcpy(d, &ubo, sizeof(DensityUniformBuffer));
	vkUnmapMemory(g_Device, g_PcPlotIndexBufferMemory);
}

static void uploadDrawListTo3dView(DrawList& dl, std::string attribute, std::string width, std::string depth, std::string height) {
	int w = SpacialData::rlatSize;
	int d = SpacialData::rlonSize;
	int h = SpacialData::altitudeSize + 22;	//the top 22 layer of the dataset are twice the size of the rest

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
		int x = SpacialData::getRlatIndex(parent->data[i][0]);
		int y = SpacialData::getAltitudeIndex(parent->data[i][2]);
		if (y > h - 44)
			y = (y - h + 44) * 2 + (h - 44);
		int z = SpacialData::getRlonIndex(parent->data[i][1]);
		assert(x >= 0);
		assert(y >= 0);
		assert(z >= 0);
		Vec4 col = dl.color;
		col.w = (parent->data[i][attributeIndex] - a.min) / (a.max - a.min);
#ifdef _DEBUG
		//std::cout << "x: " << x << " y: " << y << " z: " << z << std::endl;
#endif

		memcpy(&dat[4 * IDX3D(x, y, z, w, h)], &col.x, sizeof(Vec4));
		if (y >= h - 44)
			memcpy(&dat[4 * IDX3D(x, y + 1, z, w, h)], &col.x, sizeof(Vec4));
	}

	view3d->update3dImage(w, h, d, dat);
	delete[] dat;
}

static void exportBrushAsCsv(DrawList& dl, const  char* filepath) {
	std::string path(filepath);
	if (path.substr(path.find_last_of('.')) != ".csv") {
		return;
#ifdef _DEBUG
		std::cout << "The filepath with filename given was not a .csv file. Instead " << path.substr(path.find_last_of('.')) << " was found." << std::endl;
#endif
	}

	//getting the parent dataset for the data
	DataSet* ds = nullptr;
	for (DataSet& d : g_PcPlotDataSets) {
		if (d.name == dl.parentDataSet) {
			ds = &d;
			break;
		}
	}

	std::ofstream file(filepath);
	//adding the attributes
	for (int i = 0; i < pcAttributes.size(); i++) {
		file << pcAttributes[i].name;
		if (i != pcAttributes.size() - 1)
			file << ",";
	}
	file << "\n";
	//adding the data;
	bool* act = new bool[dl.indices.size()];
	VkUtil::downloadData(g_Device, dl.dlMem, dl.activeIndicesBufferOffset, dl.indices.size() * sizeof(bool), act);
	for (int i : dl.indices) {
		if (!act[i]) continue;
		for (int j = 0; j < pcAttributes.size(); j++) {
			file << ds->data[i][j];
			if (j != pcAttributes.size() - 1)
				file << ",";
		}
		file << "\n";
	}
	delete[] act;
}

static void exportBrushAsIdxf(DrawList& dl, const char* filepath) {
	std::string path(filepath);
	if (path.substr(path.find_last_of('.')) != ".idxf") {
		return;
#ifdef _DEBUG
		std::cout << "The filepath with filename given was not a .idxf file. Instead " << path.substr(path.find_last_of('.')) << " was found." << std::endl;
#endif
	}
	bool* act = new bool[dl.indices.size()];
	VkUtil::downloadData(g_Device, dl.dlMem, dl.activeIndicesBufferOffset, dl.indices.size() * sizeof(bool), act);
	std::ofstream file(filepath);
	for (int i : dl.indices) {
		if (!act[i]) continue;
		file << i << "\n";
	}
	file.close();
	delete[] act;
}

//calculates the length of a vector of size pcAttributes.size()
static double length(double* vec) {
	double result = 0;
	for (int i = 0; i < pcAttributes.size(); i++) {
		result += std::pow(vec[i], 2);
	}
	return std::sqrt(result);
}

//adds up the distance from point a to the data points indices
static double summedDistance(double* a, std::vector<int>& indices, float** data) {
	double dist = 0;
	for (int i : indices) {
		double d = 0;
		for (int j = 0; j < pcAttributes.size(); j++) {
			d += std::pow(data[i][j] - a[j], 2);
		}
		dist += std::sqrt(d);
	}
	return dist;
}

static void invertGlobalBrush(GlobalBrush& b) {
	std::map<int, std::vector<std::pair<float, float>>> reducedAttributeBrush;
	for (auto& axis : b.brushes) {
		for (auto& brush : axis.second) {
			bool addNew = true;
			for (auto& reducedBrush : reducedAttributeBrush[axis.first]) {
				if (brush.second.first > reducedBrush.first&& brush.second.first < reducedBrush.second) {
					reducedBrush.second = (reducedBrush.second > brush.second.second) ? reducedBrush.second : brush.second.second;
					addNew = false;
					break;
				}
				if (brush.second.second > reducedBrush.first&& brush.second.second < reducedBrush.second)
				{
					reducedBrush.first = (reducedBrush.first < brush.second.first) ? reducedBrush.first : brush.second.first;
					addNew = false;
					break;
				}
			}
			if (addNew) {
				reducedAttributeBrush[axis.first].push_back(brush.second);
			}
		}
	}
	//now inverting the reduced Attribute brushes
	b.brushes.clear();
	for (auto& axis : reducedAttributeBrush) {
		std::sort(axis.second.begin(), axis.second.end(), [](std::pair<float, float>lhs, std::pair<float, float> rhs) {return lhs.first < rhs.first; });
		float tmp = pcAttributes[axis.first].min;
		int c = 0;
		while (tmp < pcAttributes[axis.first].max) {
			if (c < axis.second.size()) {
				if (tmp != axis.second[c].first)
					b.brushes[axis.first].push_back(std::pair<int, std::pair<float, float>>(currentBrushId++, std::pair<float, float>(tmp, axis.second[c].first)));
			}
			else {
				b.brushes[axis.first].push_back(std::pair<int, std::pair<float, float>>(currentBrushId++, std::pair<float, float>(tmp, pcAttributes[axis.first].max)));
				break;
			}
			tmp = axis.second[c++].second;
		}
	}
}

static void calculateDrawListMedians(DrawList& dl) {
	if (!calculateMedians)
		return;

	float* medianArr = new float[pcAttributes.size() * MEDIANCOUNT];

	DataSet* ds = nullptr;
	for (DataSet& d : g_PcPlotDataSets) {
		if (d.name == dl.parentDataSet) {
			ds = &d;
			break;
		}
	}

	bool* act = new bool[dl.indices.size()];
	VkUtil::downloadData(g_Device, dl.dlMem, dl.activeIndicesBufferOffset, dl.indices.size() * sizeof(bool), act);
	std::vector<uint32_t> actIndices;
	for (int i : dl.indices) {
		if (act[i]) actIndices.push_back(i);
	}
	delete[] act;

	std::vector<uint32_t> dataCpy(actIndices);

	for (int i = 0; i < pcAttributes.size(); i++) {
		std::sort(dataCpy.begin(), dataCpy.end(), [i, ds](int a, int b) {return ds->data[a][i] > ds->data[b][i]; });
		medianArr[MEDIAN * pcAttributes.size() + i] = ds->data[dataCpy[dataCpy.size() >> 1]][i];
	}

	//arithmetic median calculation
	for (int i = 0; i < actIndices.size(); i++) {
		for (int j = 0; j < pcAttributes.size(); j++) {
			if (i == 0)
				medianArr[ARITHMEDIAN * pcAttributes.size() + j] = 0;
			medianArr[ARITHMEDIAN * pcAttributes.size() + j] += ds->data[actIndices[i]][j];
		}
	}
	for (int i = 0; i < pcAttributes.size(); i++) {
		medianArr[ARITHMEDIAN * pcAttributes.size() + i] /= actIndices.size();
	}

	//geometric median. Computed via gradient descent
	//geometric median
	//const float epsilon = .05f;
	//
	//std::vector<double> last(pcAttributes.size());
	//std::vector<double> median(pcAttributes.size());
	//for (int i = 0; i < median.size(); i++) {
	//	last[i] = 0;
	//	median[i] = medianArr[ARITHMEDIAN * pcAttributes.size() + i];
	//}
	//
	//while (squareDist(last, median) > epsilon) {
	//	std::vector<double> numerator(median.size());
	//	double denominator = 0;
	//	for (int i = 0; i < median.size(); i++) {
	//		numerator[i] = 0;
	//}
	//	for (int i = 0; i < dl.activeInd.size(); i++) {
	//		double dist = std::sqrt(squareDist(median, ds.data[dl.activeInd[i]]));
	//		if (dist == 0)
	//			continue;
	//		numerator = numerator + divide(ds.data[dl.activeInd[i]], dist, median.size());
	//		denominator += 1 / dist;
	//	}
	//	last = median;
	//	median = numerator / denominator;
	//}
	//
	//for (int i = 0; i < pcAttributes.size(); i++) {
	//	medianArr[GOEMEDIAN * pcAttributes.size() + i] = median[i];
	//}

	void* d;
	vkMapMemory(g_Device, dl.dlMem, dl.medianBufferOffset, MEDIANCOUNT * pcAttributes.size() * sizeof(float), 0, &d);
	memcpy(d, medianArr, MEDIANCOUNT * pcAttributes.size() * sizeof(float));
	vkUnmapMemory(g_Device, dl.dlMem);

	delete[] medianArr;
}

//x is in range [0,1] to indicate the position
static inline float getBinVal(float x, std::vector<float>& bins) {
	x *= bins.size() - 1;
	int ind = x;
	float mul = 1 - x + ind;
	if (ind < -1)
		return 0;
	if (ind == -1)
		return bins[0];
	if (ind >= bins.size())
		return 0;
	if (ind == bins.size() - 1)
		return bins[bins.size() - 1];
	return mul * bins[ind] + (1 - mul) * bins[ind + 1];
}



static void changeColorsToCustomAlternatingColors(ColorPaletteManager *cpm, 
	unsigned int nrAttributes, 
	std::vector<ImVec4> *violinLineColors, 
	std::vector<ImVec4> *violinFillColors, 
	HistogramManager::Histogram &hist, 
	bool **activeAttributes,
	bool custColors = false)
{
	// Get complete colorpalette.
	//const std::string colorStr = std::string("Dark2ReorderSplitYellowExtended");
	const std::string colorStrFill = cpm->chosenAutoColorPaletteFill;
	const std::string colorStrLine = cpm->chosenAutoColorPaletteLine;

	std::vector<ImVec4> retrievedColorsFill;
	std::vector<ImVec4> retrievedColorsLine;
	if (custColors)
	{
		retrievedColorsFill = cpm->colorPalette->getPallettAsImVec4(0, 0, 20, cpm->alphaFill, colorStrFill);
		retrievedColorsLine = cpm->colorPalette->getPallettAsImVec4(0, 0, 20, cpm->alphaFill, colorStrLine);
	}
	else
	{
		retrievedColorsFill = cpm->colorPalette->getPallettAsImVec4(cpm->chosenCategoryNr, cpm->chosenPaletteNr, cpm->chosenNrColorNr, cpm->alphaFill);
		retrievedColorsLine = cpm->colorPalette->getPallettAsImVec4(cpm->chosenCategoryNr, cpm->chosenPaletteNr, cpm->chosenNrColorNr, cpm->alphaFill);
	}

	unsigned int colorCount = 0;
	// So far, only 12 colors are available.
	for (unsigned int i = 0; (i < nrAttributes) && (i < 12) ; ++i)
	{
		unsigned int times0 = 0;
		unsigned int times1 = 0;
		// Colors are sorted alternatingly for right and left side. So, all plots on the right have to get 'right-colors'.
		for (unsigned int item :hist.attributeColorOrderIdx)
		{
			ImVec4 currColorFill;
			ImVec4 currColorLine;
			unsigned int currColorIdx = 0;
			(!(hist.side[item])) ? currColorIdx = (2 * times0++) : currColorIdx = (2 * times1++ + 1);
			if ((retrievedColorsFill.size() > currColorIdx) && (retrievedColorsLine.size() > currColorIdx))
			 {
				currColorFill = retrievedColorsFill[currColorIdx];
				currColorLine = retrievedColorsLine[currColorIdx];
			}
			else
			{
				break;
			}
			//(!(hist.side[item])) ? currColor = retrievedColors[(2 * times0++)] : currColor= retrievedColors[(2 * times1++ + 1)];
			if (cpm->applyToLineColor) {
				(*violinLineColors)[item] = currColorLine;
				(*violinLineColors)[item].w = cpm->alphaLines / 255.;
			}
			if (cpm->applyToFillColor) {
				(*violinFillColors)[item] = currColorFill;
				(*violinFillColors)[item].w = cpm->alphaFill / 255.;
			}



		}
	}
}



static void includeColorbrewerToViolinPlot(ColorPaletteManager *cpm, std::vector<ImVec4> *violinLineColors, std::vector<ImVec4> *violinFillColors)
{

//    std::vector<ViolinPlot> violinAttributePlots;
//    std::vector<ViolinDrawlistPlot> violinDrawlistPlots;
    ImGui::Separator();
    int previousNrOfColumns = ImGui::GetColumnsCount();
    ImGui::Columns(5);
    ImGui::Checkbox("Apply Palette", &cpm->useColorPalette);
    ImGui::NextColumn();

    const char*  categoryDefault[] = { "div", "qual", "seq", "cust" };
    unsigned int currCategory = cpm->chosenCategoryNr;
    if(ImGui::BeginCombo("Category",  categoryDefault[currCategory]))
    {
        for (unsigned int ik = 0; ik < 4; ++ik) {
            if (ImGui::MenuItem(categoryDefault[ik], nullptr)) {
                cpm->setChosenCategoryNr(ik);
                currCategory = ik;
            }
        }
        ImGui::EndCombo();
    }


    const std::vector<std::string> pNames = cpm->colorPalette->paletteNamesVec.at(cpm->chosenCategoryNr);
    unsigned int currPaletteNr = cpm->chosenPaletteNr;
    char const * pName;
    if (currPaletteNr > pNames.size())
    {
        currPaletteNr = 0;
        cpm->setChosenPaletteNr(currPaletteNr);
        pName = pNames[currPaletteNr].c_str();
    }
    else
    {
        pName = pNames[currPaletteNr].c_str();
    }


    ImGui::NextColumn();


    if(ImGui::BeginCombo("Palette",  &pName[0])){
        for (unsigned int ij = 0; ij < pNames.size(); ++ij){
            if (ImGui::MenuItem(pNames[ij].c_str(), nullptr)){
                cpm->setChosenPaletteNr(ij);
            }
        }
        ImGui::EndCombo();
    }
    ImGui::NextColumn();

    const char*  numbers[] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x" };
    unsigned int currColorNr = cpm->chosenNrColorNr;
    CPalette* currPalette =  cpm->colorPalette->getPalletteWithName(pNames[currPaletteNr]);

    if(ImGui::BeginCombo("Nr colors",  numbers[currColorNr])){

        for (unsigned int il =1; il < currPalette->maxcolors+1 ;++il)
        {
            if (ImGui::MenuItem(numbers[il], nullptr)){
                cpm->setChosenNrColorNr(il);

            }
        }
        ImGui::EndCombo();
    }
    ImGui::NextColumn();
    unsigned int skipFirstAttributes = cpm->skipFirstAttributes;
    if(ImGui::BeginCombo("Skip first x attrbts",  numbers[skipFirstAttributes])){

        for (unsigned int il =0; il < violinLineColors->size();++il)
        {
            if (ImGui::MenuItem(numbers[il], nullptr)){
                cpm->setChosenSkipFirstAttributes(il);

            }
        }
        ImGui::EndCombo();
    }




    ImGui::Separator();
    ImGui::Columns(6);
    ImGui::Checkbox("Adjust Line Color", &cpm->applyToLineColor);
    ImGui::NextColumn();
    ImGui::Checkbox("Adjust Fill Color", &cpm->applyToFillColor);
    ImGui::NextColumn();
    ImGui::Checkbox("Backup Line Color", &cpm->backupLineColor);
    ImGui::NextColumn();
    ImGui::Checkbox("Backup Fill Color", &cpm->backupFillColor);
    ImGui::NextColumn();
    // TODO: Change to int boxes
    if(ImGui::SliderInt("Alpha value lines", &cpm->alphaLines, 0, 255))
    {
        cpm->bvaluesChanged = true;
    }
    ImGui::NextColumn();
    if (ImGui::SliderInt("Alpha value fill", &cpm->alphaFill, 0, 255))
    {
        cpm->bvaluesChanged = true;
    }

    ImGui::Separator();


    // Now exchange as many colors as selected if something was changed.
    if ((cpm->useColorPalette) &&
            (cpm->getBValuesChanged())){
        // Backup existing colors. Only backsup the ones after skipRange...
        cpm->backupColors(*violinLineColors, *violinFillColors);

        std::vector<ImVec4> retrievedColors = cpm->colorPalette->getPallettAsImVec4(
                    cpm->chosenCategoryNr,
                    cpm->chosenPaletteNr,
                    cpm->chosenNrColorNr);
        for (unsigned int iColor = 0; iColor < cpm->chosenNrColorNr; ++iColor)
        {
            if ((iColor + cpm->skipFirstAttributes < (*violinFillColors).size())
                && (iColor < retrievedColors.size()))
            {
                if (cpm->applyToLineColor){
                    (*violinLineColors)[iColor + cpm->skipFirstAttributes] = retrievedColors[iColor];
                    (*violinLineColors)[iColor + cpm->skipFirstAttributes].w = cpm->alphaLines / 255.;
                }
                if (cpm->applyToFillColor){
                    (*violinFillColors)[iColor + cpm->skipFirstAttributes] = retrievedColors[iColor];
                    (*violinFillColors)[iColor + cpm->skipFirstAttributes].w = cpm->alphaFill / 255.;
                }

            }
        }

    }
    ImGui::Columns(previousNrOfColumns);

}


static std::vector<uint32_t> sortHistogram(HistogramManager::Histogram& hist, 
											ViolinDrawlistPlot& violinDrawlistPlot, 
											bool changeRenderOrder = true, 
											bool reverseRenderOrder = false) {
	std::vector<std::pair<uint32_t, float>> area;
	float div = determineViolinScaleLocalDiv(hist.maxCount, &(violinDrawlistPlot.activeAttributes), violinDrawlistPlot.attributeScalings);
	
	// Determine how to use the y-scaling option
	std::vector<std::pair<float, float>> violinMinMax(pcAttributes.size(), { std::numeric_limits<float>().max(),std::numeric_limits<float>().min() });
	unsigned int dlNr = 0;
	getyScaleDL(dlNr, violinDrawlistPlot, violinMinMax);
	
	for (unsigned int k = 0; k < hist.area.size(); ++k) {

		float deltaBin = (hist.ranges[k].second - hist.ranges[k].first) / hist.bins[k].size();
        if (deltaBin == 0.0){deltaBin = 1;}

		// casting to int works as floor, since startBin is always positive. 
		int startBin = std::max(0,(int)((violinMinMax[k].first - hist.ranges[k].first) / deltaBin));
		int endBin = std::min((int)hist.bins[k].size()-1, (int)(std::ceil((violinMinMax[k].second - hist.ranges[k].first) / deltaBin) +0.0000001) );
		hist.areaShown[k] = 0;
		for (unsigned int i = startBin + int(histogramManager->ignoreZeroBins); i <= endBin; ++i) {
			// If the minRange of the violin is within the current bin, it is used.
			hist.areaShown[k] += hist.bins[k][i];
		}
		hist.areaShown[k] /= (endBin - startBin + 1* int(histogramManager->ignoreZeroBins));


		float a = 0;
		switch (violinDrawlistPlot.violinScalesX[k]) {
		case ViolinScaleSelf:
			a = hist.areaShown[k] / hist.maxCount[k];
			break;
		case ViolinScaleLocal:
			//a = hist.area[k] / hist.maxGlobalCount;
			a = hist.areaShown[k] / div;
			break;
		case ViolinScaleGlobal:
			a = hist.areaShown[k] / violinDrawlistPlot.maxGlobalValue;
			break;
		case ViolinScaleGlobalAttribute:
			a = hist.areaShown[k] / violinDrawlistPlot.maxValues[k];
			break;
		}

		a *= violinDrawlistPlot.attributeScalings[k];
		if (violinDrawlistPlot.attributePlacements[k] == ViolinLeftHalf ||
			violinDrawlistPlot.attributePlacements[k] == ViolinRightHalf ||
			violinDrawlistPlot.attributePlacements[k] == ViolinMiddleLeft ||
			violinDrawlistPlot.attributePlacements[k] == ViolinMiddleRight)
			a *= .5f;

		area.push_back({ k,a });
	}

	if (changeRenderOrder) {
		if (!reverseRenderOrder) {
			std::sort(area.begin(), area.end(), [](std::pair<uint32_t, float>& a, std::pair<uint32_t, float>& b) { return sortDescPair(a, b); });
		}
		else
		{
			std::sort(area.begin(), area.end(), [](std::pair<uint32_t, float>& a, std::pair<uint32_t, float>& b) {return sortAscPair(a, b); });
		}
	}
	std::vector<uint32_t> ret(area.size());
	for (int i = 0; i < area.size(); ++i) ret[i] = area[i].first;
	return ret;
}


static void sortAllHistograms(std::string option)
{
	if (option == "dl") {
		if (!renderOrderDLConsider) {
			return;
		}

		for (auto& drawListPlot : violinDrawlistPlots) {
			int drawL = 0;
			for (auto& drawList : drawListPlot.drawLists) {
				DrawList* dl = &(*std::find_if(g_PcPlotDrawLists.begin(), g_PcPlotDrawLists.end(), [drawList](DrawList& draw) {return draw.name == drawList; }));

				DataSet* ds;
				for (DataSet& d : g_PcPlotDataSets) {
					if (d.name == dl->parentDataSet) {
						ds = &d;
						break;
					}
				}
				HistogramManager::Histogram& hist = histogramManager->getHistogram(dl->name);
				if (renderOrderDLConsider && ((drawL == 0) || (!renderOrderBasedOnFirstDL))) {
					drawListPlot.attributeOrder[drawL] = sortHistogram(hist, drawListPlot, renderOrderDLConsider, renderOrderDLReverse);
				}
				else if ((renderOrderBasedOnFirstDL && drawL > 0)) {
					break;
				}
			}
			if ((renderOrderBasedOnFirstDL && drawL > 0)) {
				break;
			}
		}
	}
	else if (option == "attr") {
		std::cout << "Automatic non-stop reordering of attribute violins is not implemented yet. \n";
		return;
	}


}

inline void updateAllViolinPlotMaxValues(bool renderOrderBasedOnFirst = false) {
	for (auto& drawListPlot : violinDrawlistPlots) {
		drawListPlot.maxGlobalValue = 0;
		for (int j = 0; j < drawListPlot.maxValues.size(); ++j) {
			drawListPlot.maxValues[j] = 0;
		}
		int drawL = 0;
		for (auto& drawList : drawListPlot.drawLists) {
			HistogramManager::Histogram& hist = histogramManager->getHistogram(drawList);
			std::vector<std::pair<uint32_t, float>> area;
			for (int j = 0; j < hist.maxCount.size(); ++j) {
				if (hist.maxCount[j] > drawListPlot.maxValues[j]) {
					drawListPlot.maxValues[j] = hist.maxCount[j];
				}
				if (hist.maxCount[j] > drawListPlot.maxGlobalValue) {
					drawListPlot.maxGlobalValue = hist.maxCount[j];
				}
				area.push_back({ j, drawListPlot.attributeScalings[j] / hist.maxCount[j] });
			}

			if (renderOrderDLConsider && ((drawL == 0) || (!renderOrderBasedOnFirst)))
			{
				drawListPlot.attributeOrder[drawL] = sortHistogram(hist, drawListPlot, renderOrderDLConsider, renderOrderDLReverse);
				//std::sort(area.begin(), area.end(), [](std::pair<uint32_t, float>& a, std::pair<uint32_t, float>& b) {return sortDescPair(a, b); });
				//for (int j = 0; j < pcAttributes.size(); ++j)drawListPlot.attributeOrder[drawL][j] = area[j].first;
			}
			else
			{
				drawListPlot.attributeOrder[drawL] = drawListPlot.attributeOrder[0];
			}
			++drawL;
		}
	}
}


static void optimizeViolinSidesAndAssignCustColors() {
	if (violinAdaptSidesAutoObj.optimizeSidesNowAttr)
	{
		auto& hist = histogramManager->getHistogram(violinAdaptSidesAutoObj.vp->drawLists[0].name);
		histogramManager->determineSideHist(hist, &(violinAdaptSidesAutoObj.vp->activeAttributes), violinPlotAttrConsiderBlendingOrder);

		violinAdaptSidesAutoObj.vp->violinPlacements.clear();
		for (int j = 0; j < violinAdaptSidesAutoObj.vp->attributeNames.size(); ++j) {
			violinAdaptSidesAutoObj.vp->violinPlacements.push_back((hist.side[j] % 2) ? ViolinLeft : ViolinRight);


		}
		if (violinPlotAttrInsertCustomColors || violinPlotAttrConsiderBlendingOrder) {
			changeColorsToCustomAlternatingColors((violinAdaptSidesAutoObj.vp->colorPaletteManager), violinAdaptSidesAutoObj.vp->attributeNames.size(), &(violinAdaptSidesAutoObj.vp->drawListLineColors), &(violinAdaptSidesAutoObj.vp->drawListFillColors),
				hist, &(violinAdaptSidesAutoObj.vp->activeAttributes), violinPlotAttrInsertCustomColors);
		}
		violinAdaptSidesAutoObj.optimizeSidesNowAttr = false;
	}
	///


	if (violinAdaptSidesAutoObj.optimizeSidesNowDL)
	{
		auto& hist = histogramManager->getHistogram(violinAdaptSidesAutoObj.vdlp->drawLists[0]);
		histogramManager->determineSideHist(hist, &(violinAdaptSidesAutoObj.vdlp->activeAttributes), violinPlotDLConsiderBlendingOrder);

		violinAdaptSidesAutoObj.vdlp->attributePlacements.clear();
		for (int j = 0; j < violinAdaptSidesAutoObj.vdlp->attributeNames.size(); ++j) {
			violinAdaptSidesAutoObj.vdlp->attributePlacements.push_back((hist.side[j] % 2) ? ViolinMiddleLeft : ViolinMiddleRight);
		}

		if (violinPlotDLInsertCustomColors || violinPlotDLConsiderBlendingOrder) {
			changeColorsToCustomAlternatingColors((violinAdaptSidesAutoObj.vdlp->colorPaletteManager), violinAdaptSidesAutoObj.vdlp->attributeNames.size(), &(violinAdaptSidesAutoObj.vdlp->attributeLineColors), &(violinAdaptSidesAutoObj.vdlp->attributeFillColors),
				hist, &(violinAdaptSidesAutoObj.vdlp->activeAttributes), violinPlotDLInsertCustomColors);
		}
		violinAdaptSidesAutoObj.optimizeSidesNowDL = false;
	}
}

int main(int, char**)
{
#ifdef DETECTMEMLEAK
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	engine.seed(15);

	//test of multivariate gauss calculations
	//float determinant;
	std::vector<std::vector<double>> X{ {10,0,-3,10}, {-2,-4,1,.5},{3,0,2,7},{-3,5,9,0} };
	//std::vector<std::vector<double>> S(X[0].size(),std::vector<double>(X[0].size())), I(X[0].size(), std::vector<double>(X[0].size())), D(X[0].size(), std::vector<double>(X[0].size()));
	//std::vector<double> mean(X[0].size());
	//MultivariateGauss::compute_average_vector(X, mean);
	//MultivariateGauss::compute_covariance_matrix(X, S);
	//MultivariateGauss::compute_matrix_inverse(X, I);
	//MultivariateGauss::compute_matrix_times_matrix(X, I, I);
	//MultivariateGauss::compute_matrix_determinant(X, determinant);
	//PCUtil::matrixdump(X);
	//PCUtil::matrixdump(S);
	//PCUtil::matrixdump(I);
	//PCUtil::matrixdump(D);
	//std::cout << determinant << std::endl;

	//test of eigen
	//Eigen::MatrixXd m(4,4);
	//for (int i = 0; i < X.size(); ++i) {
	//	for (int j = 0; j < X[i].size(); ++j) {
	//		m(i, j) = X[i][j];
	//	}
	//}
	//std::cout << m.inverse() * m << std::endl;

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
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;	// Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;		// Enable docking
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
	io.ConfigViewportsNoDecoration = false;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;	// Enable Gamepad Controls
	ImFontConfig fontConf{};
	fontConf.OversampleH = 2;
	fontConf.OversampleV = 2;
	io.Fonts->AddFontFromFileTTF("fonts/Roboto-Regular.ttf", 15.0f, &fontConf, io.Fonts->GetGlyphRangesDefault());
	io.Fonts->AddFontFromFileTTF("fonts/Roboto-Regular.ttf", 10.0f, &fontConf, io.Fonts->GetGlyphRangesDefault());
	io.Fonts->AddFontFromFileTTF("fonts/Roboto-Regular.ttf", 25.0f, &fontConf, io.Fonts->GetGlyphRangesDefault());
	io.Fonts->AddFontFromFileTTF("fonts/Roboto-Medium.ttf", 10.0f, &fontConf, io.Fonts->GetGlyphRangesDefault());
	io.Fonts->AddFontFromFileTTF("fonts/Roboto-Medium.ttf", 15.0f, &fontConf, io.Fonts->GetGlyphRangesDefault());
	io.Fonts->AddFontFromFileTTF("fonts/Roboto-Medium.ttf", 25.0f, &fontConf, io.Fonts->GetGlyphRangesDefault());

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

		ImGui_ImplVulkan_CreateFontsTexture(command_buffer);// , g_Device, g_DescriptorPool);

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

		g_PcPlotImageDescriptorSet = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(g_PcPlotSampler, g_PcPlotView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, g_Device, g_DescriptorPool);



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
		VkUtil::transitionImageLayout(command_buffer, g_PcPlotDensityImageCopy, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

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

#ifdef BUBBLEVIEW
	{//creating the node viewer and its descriptor set
		bubblePlotter = new BubblePlotter(800, 800, g_Device, g_PhysicalDevice, g_PcPlotCommandPool, g_Queue, g_DescriptorPool);
		bubblePlotter->setImageDescSet((VkDescriptorSet)ImGui_ImplVulkan_AddTexture(bubblePlotter->getImageSampler(), bubblePlotter->getImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, g_Device, g_DescriptorPool));
		bubblePlotter->addSphere(1.0f, glm::vec4(1, 1, 1, 1), glm::vec3(0, 0, 0));
		bubblePlotter->render();
	}
#endif

	{//iso surface renderer
		isoSurfaceRenderer = new IsoSurfRenderer(800, 800, g_Device, g_PhysicalDevice, g_PcPlotCommandPool, g_Queue, g_DescriptorPool);
		isoSurfaceRenderer->setImageDescriptorSet((VkDescriptorSet)ImGui_ImplVulkan_AddTexture(isoSurfaceRenderer->getImageSampler(), isoSurfaceRenderer->getImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, g_Device, g_DescriptorPool));
		brushIsoSurfaceRenderer = new BrushIsoSurfRenderer(800, 800, g_Device, g_PhysicalDevice, g_PcPlotCommandPool, g_Queue, g_DescriptorPool);
		brushIsoSurfaceRenderer->setImageDescriptorSet((VkDescriptorSet)ImGui_ImplVulkan_AddTexture(brushIsoSurfaceRenderer->getImageSampler(), brushIsoSurfaceRenderer->getImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, g_Device, g_DescriptorPool));
	}

	{//creating the settngs manager
		settingsManager = new SettingsManager();
	}

	{//brushing gpu
		gpuBrusher = new GpuBrusher(g_Device, g_PhysicalDevice, g_PcPlotCommandPool, g_Queue, g_DescriptorPool);
	}

	{//histogram manager
		histogramManager = new HistogramManager(g_Device, g_PhysicalDevice, g_PcPlotCommandPool, g_Queue, g_DescriptorPool, violinPlotBinsSize);
	}

	io.ConfigWindowsMoveFromTitleBarOnly = true;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	{//Set imgui style
		ImGuiStyle& style = ImGui::GetStyle();
		style.ChildRounding = 5;
		style.FrameRounding = 3;
		style.GrabRounding = 3;
		style.WindowRounding = 0;
		style.PopupRounding = 3;
	}

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

		if (animationStart != std::chrono::steady_clock::time_point(std::chrono::duration<int>(0))) {
			//disabling inputs when animating
			ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
			animationItemsDisabled = true;
			//remembering the original show flags for every drawlist
			if (!animationActiveDatasets) {
				animationActiveDatasets = new bool[g_PcPlotDrawLists.size()];
				int i = 0;
				for (DrawList& dl : g_PcPlotDrawLists) {
					animationActiveDatasets[i++] = dl.show;
					dl.show = false;
				}
			}
			//rendering a new drawlist if current drawlist to show changed
			if (animationCurrentDrawList != (int)(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - animationStart).count() / animationDuration)) {
				//disabling current drawlist
				auto it = g_PcPlotDrawLists.begin();
				int c = -1, i = 0;
				for (; c < animationCurrentDrawList && it != g_PcPlotDrawLists.end(); i++) {
					if (animationActiveDatasets[i]) c++;
					if (c != 0) ++it;
				}

				if (it == g_PcPlotDrawLists.end()) {
					animationStart = std::chrono::steady_clock::time_point(std::chrono::duration<int>(0));
					animationCurrentDrawList = -1;
				}
				else {
					if (c != -1) {
						it->show = false;
						it++;
						i++;
					}
					while (!animationActiveDatasets[i] && it != g_PcPlotDrawLists.end()) {
						i++;
						++it;
					}
					if (it != g_PcPlotDrawLists.end()) {
						it->show = true;
						animationCurrentDrawList = c + 1;
						pcPlotRender = true;
					}
					else {
						animationStart = std::chrono::steady_clock::time_point(std::chrono::duration<int>(0));
						animationCurrentDrawList = -1;
					}
				}
			}
		}
		else {
			if (animationActiveDatasets) {
				int i = 0;
				for (DrawList& dl : g_PcPlotDrawLists) {
					dl.show = animationActiveDatasets[i++];
				}
				delete[] animationActiveDatasets;
				animationActiveDatasets = nullptr;
				pcPlotRender = true;
			}
		}

		//Check if a drawlist color changed
		//for (DrawList& ds : g_PcPlotDrawLists) {
		//	if (ds.color != ds.prefColor) {
		//		pcPlotRender = true;
		//		ds.prefColor = ds.color;
		//	}
		//}

		//if animation is active disable imgui widgets

		//check if mouse was clicked for frametime
#ifdef PRINTFRAMETIME
		static std::chrono::steady_clock::time_point begin(std::chrono::duration<int>(0));
		if (begin != std::chrono::steady_clock::time_point(std::chrono::duration<int>(0))) {
			std::cout << "Time for frame after mouseclick: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() << " milliseconds." << std::endl;
			begin = std::chrono::steady_clock::time_point(std::chrono::duration<int>(0));
		}
		if (ImGui::GetIO().MouseDown[0]) {
			begin = std::chrono::steady_clock::now();
		}
#endif

		//Check if button f5 was pressed for rendering
		if (ImGui::GetIO().KeysDown[294]) {
			pcPlotRender = true;
		}

		//check if a path was dropped in the application
		if (pathDropped && !addIndeces) {
			ImGui::OpenPopup("OPENDATASET");
			if (ImGui::BeginPopupModal("OPENDATASET", NULL, ImGuiWindowFlags_AlwaysAutoResize))
			{
				ImGui::Text("Do you really want to open these Datasets?");
				for (std::string& s : droppedPaths) {
					ImGui::Text(s.c_str());
				}
				ImGui::Separator();

				if (ImGui::Button("Open", ImVec2(120, 0)) || ImGui::IsKeyPressed(KEYENTER)) {
					ImGui::CloseCurrentPopup();
					for (std::string& s : droppedPaths) {
						openDataset(s.c_str());
						if (createDefaultOnLoad) {
							createPcPlotDrawList(g_PcPlotDataSets.back().drawLists.front(), g_PcPlotDataSets.back(), g_PcPlotDataSets.back().name.c_str());
							updateActiveIndices(g_PcPlotDrawLists.back());
							pcPlotRender = true;
						}
					}
					droppedPaths.clear();
					delete[] createDLForDrop;
					createDLForDrop = NULL;
					pathDropped = false;
				}
				ImGui::SetItemDefaultFocus();
				ImGui::SameLine();
				if (ImGui::Button("Cancel", ImVec2(120, 0)) || ImGui::IsKeyPressed(KEYESC)) {
					ImGui::CloseCurrentPopup();
					droppedPaths.clear();
					delete[] createDLForDrop;
					createDLForDrop = NULL;
					pathDropped = false;
				}
				ImGui::EndPopup();
			}
		}

		//Main docking window including the main menu
		ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(viewport->GetWorkPos());
		ImGui::SetNextWindowSize(viewport->GetWorkSize());
		ImGui::SetNextWindowViewport(viewport->ID);
		ImGuiWindowFlags dockingWindow_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBringToFrontOnFocus;
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

		ImGui::Begin("DockSpace", NULL, dockingWindow_flags);

		ImGui::PopStyleVar(3);
		ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
		if (ImGui::DockBuilderGetNode(dockspace_id) == NULL) {
			ImGui::DockBuilderRemoveNode(dockspace_id);
			ImGuiDockNodeFlags dockSpaceFlags = 0;
			dockSpaceFlags |= ImGuiDockNodeFlags_DockSpace;
			ImGui::DockBuilderAddNode(dockspace_id, dockSpaceFlags);

#ifdef _DEBUG
			ImGui::DockBuilderDockWindow("Dear ImGui Demo", dockspace_id);
#endif
			ImGui::DockBuilderDockWindow("Parallel coordinates", dockspace_id);
		}
		auto id = ImGui::DockBuilderGetNode(dockspace_id)->SelectedTabId;
		ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

		bool openSave = ImGui::GetIO().KeyCtrl && ImGui::IsKeyDown(83), openLoad = false, openAttributesManager = false, saveColor = false, openColorManager = false;
		float color[4];
		if (ImGui::BeginMenuBar()) {
			if (ImGui::BeginMenu("Gui")) {
				//ImGui::ShowFontSelector("Select font");
				//ImGui::ShowStyleSelector("Select gui style");
				ImGui::ShowStyleEditor();
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Maximize")) {
				ImGui::DragInt("Max window Width", (int*)&windowWidth, 10, 200, 10000);
				ImGui::DragInt("Max window Height", (int*)&windowHeight, 10, 200, 10000);
				if (ImGui::MenuItem("Maximize!")) {
					glfwSetWindowSize(window, windowWidth, windowHeight);
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Attribute")) {
				if (ImGui::MenuItem("Save Attributes", "Ctrl+S") && pcAttributes.size() > 0) {
					openSave = true;
				}
				if (ImGui::BeginMenu("Load...")) {
					for (SettingsManager::Setting* s : *settingsManager->getSettingsType("AttributeSetting")) {
						if (ImGui::MenuItem(s->id.c_str())) {
							if (((int*)(s->data))[0] != pcAttributes.size()) {
								openLoad = true;
								continue;
							}

							std::vector<Attribute> savedAttr;
							char* d = (char*)s->data + sizeof(int);
							bool cont = false;
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
									//goto end;
									cont = true;
								}
							}
							if (cont)
								continue;

							int* o = (int*)d;
							bool* act = (bool*)(d + pcAttributes.size() * sizeof(int));
							for (int i = 0; i < pcAttributes.size(); i++) {
								pcAttributes[i] = savedAttr[i];
								pcAttrOrd[i] = o[i];
								pcAttributeEnabled[i] = act[i];
							}
							pcPlotRender = true;

						end:;
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

			if (ImGui::BeginMenu("Global brush")) {
				if (ImGui::MenuItem("Activate Global Brushing", "", &toggleGlobalBrushes) && !toggleGlobalBrushes) {
					pcPlotRender = updateAllActiveIndices();
				}

				if (ImGui::BeginMenu("Brush Combination")) {
					static char* const combinations[] = { "OR","AND" };
					if (ImGui::Combo("brushCombination", &brushCombination, combinations, sizeof(combinations) / sizeof(*combinations))) {
						pcPlotRender = updateAllActiveIndices();
					}

					ImGui::EndMenu();
				}
				ImGui::InputFloat("Mu add factor", &brushMuFactor, 0.000001, 0.001,10);

				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Fractioning")) {
				if (ImGui::InputInt("Max fraction depth", &maxFractionDepth, 1, 1)) {
					if (maxFractionDepth < 1) maxFractionDepth = 1;
					if (maxFractionDepth > 30) maxFractionDepth = 30;
				}

				if (ImGui::InputInt("Outlier rank", &outlierRank, 1, 1)) {
					if (outlierRank < 1) outlierRank = 1;
				}

				static char* boundsTypes[] = { "No adjustment","Pull in outside", "Pull in both sides" };
				if (ImGui::BeginCombo("Bounds behaviour", boundsTypes[boundsBehaviour])) {
					for (int i = 0; i < 3; i++) {
						if (ImGui::MenuItem(boundsTypes[i])) boundsBehaviour = i;
					}
					ImGui::EndCombo();
				}

				static char* splitTypes[] = { "Split half","SAH" };
				if (ImGui::BeginCombo("Split behaviour", splitTypes[splitBehaviour])) {
					for (int i = 0; i < 2; ++i) {
						if (ImGui::MenuItem(splitTypes[i])) splitBehaviour = i;
					}
					ImGui::EndCombo();
				}

				ImGui::DragFloat("Fractionbox width", &fractionBoxWidth, 1, 0, 100);

				if (ImGui::InputInt("Fractionbox linewidth", &fractionBoxLineWidth, 1, 1)) {
					if (fractionBoxLineWidth < 1) maxFractionDepth = 1;
					if (fractionBoxLineWidth > 30) maxFractionDepth = 30;
				}
				
				ImGui::SliderFloat("Multivariate std dev thresh", &multivariateStdDivThresh, .01f, 5);
				
				if (ImGui::IsItemDeactivatedAfterEdit()) {
					pcPlotRender = updateAllActiveIndices();
				}

				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Animation")) {
				ImGui::SliderFloat("Animation duration per drawlist", &animationDuration, .1f, 10);
				if (ImGui::MenuItem("Start drawlist animation")) {
					animationStart = std::chrono::steady_clock::now();
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Workbenches")) {
				ImGui::MenuItem("Bubbleplot workbench", "", &enableBubbleWindow);
				ImGui::MenuItem("3d View", "", &enable3dView);
				if(ImGui::BeginMenu("Iso surface workbenches")) {
					ImGui::MenuItem("Iso surface workbench", "", &enableIsoSurfaceWindow);
					ImGui::MenuItem("Direct iso surface workbench", "", &enableBrushIsoSurfaceWindow);
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Violinplot workbenches")) {
					ImGui::MenuItem("Violin attribute major", "", &enableAttributeViolinPlots);
					ImGui::MenuItem("Violin drawlist major", "", &enableDrawlistViolinPlots);
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
			if ((ImGui::Button("Close")) || ImGui::IsKeyPressed(KEYESC)) {
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
			if ((ImGui::Button("Close", ImVec2(120, 0))) || ImGui::IsKeyPressed(KEYESC)) {
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
			if ((ImGui::Button("Cancel", ImVec2(120, 0))) || ImGui::IsKeyPressed(KEYESC)) {
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
			if ((ImGui::Button("Close")) || ImGui::IsKeyPressed(KEYESC)) {
				ImGui::CloseCurrentPopup();
			}

			ImGui::EndPopup();
		}

		ImGui::End();

#ifdef RENDER3D
		//testwindow for the 3d renderer
		ImGui::SetNextWindowSize(ImVec2(800, 800));
		if (enable3dView) {
			ImGui::Begin("3dview", &enable3dView, ImGuiWindowFlags_NoSavedSettings);
			ImGui::Image((ImTextureID)view3d->getImageDescriptorSet(), ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionMax().y), ImVec2(0, 0), ImVec2(1, 1), ImColor(255, 255, 255, 255), ImColor(255, 255, 255, 128));

			//if (ImGui::IsWindowHovered() && ImGui::GetIO().MouseReleased[0]);
			//	view3d->resize(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowHeight());

			if ((ImGui::IsMouseDragging(ImGuiMouseButton_Left) || io.MouseWheel) && ImGui::IsItemHovered()) {
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
			ImGui::End();
		}
#endif

#ifdef _DEBUG
		ImGui::ShowDemoWindow(NULL);
#endif
		
		//Parallel coordinates plot ----------------------------------------------------------------------------------------
		ImVec2 picPos;
		bool picHovered;
		if (ImGui::Begin("Parallel coordinates", NULL)) {
			float windowW = ImGui::GetWindowWidth();
			// Labels for the titels of the attributes
			// Position calculation for each of the Label
			size_t amtOfLabels = 0;
			for (int i = 0; i < pcAttributes.size(); i++)
				if (pcAttributeEnabled[i])
					amtOfLabels++;

			size_t paddingSide = 10;			//padding from left and right screen border
			size_t gap = (windowW - 2 * paddingSide) / (amtOfLabels - 1);
			ImVec2 buttonSize = ImVec2(70, 20);
			size_t offset = 0;

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
				if (c1 != 0)
					ImGui::SameLine(offset - c1 * (buttonSize.x / amtOfLabels));
				ImGui::Button(name.c_str(), buttonSize);

				if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
					int p[] = { c,i };		//holding the index in the pcAttriOrd array and the value of it
					ImGui::SetDragDropPayload("ATTRIBUTE", p, sizeof(p));
					ImGui::Text("Swap %s", name.c_str());
					ImGui::EndDragDropSource();
				}
				if (ImGui::BeginDragDropTarget()) {
					if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ATTRIBUTE")) {
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
				if (c1 != 0)
					ImGui::SameLine(offset - c1 * (buttonSize.x / amtOfLabels));

				if (ImGui::DragFloat(name.c_str(), &pcAttributes[i].max, (pcAttributes[i].max - pcAttributes[i].min) * .001f, 0.0f, 0.0f, "%.05f")) {
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
			ImGui::Image((ImTextureID)g_PcPlotImageDescriptorSet, ImVec2(ImGui::GetWindowWidth() - 2 * paddingSide, io.DisplaySize.y * 2 / 5), ImVec2(0, 0), ImVec2(1, 1), ImColor(255, 255, 255, 255), ImColor(255, 255, 255, 255));
			picHovered = ImGui::IsItemHovered();
			if (pcPlotRender) {
				pcPlotRender = false;
				drawPcPlot(pcAttributes, pcAttrOrd, pcAttributeEnabled, wd);
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

			ImVec2 picSize(ImGui::GetWindowWidth() - 2 * paddingSide + 5, io.DisplaySize.y * 2 / 5);
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
					if (ImGui::Checkbox(name.c_str(), &brushTemplateAttrEnabled[i]) || updateBrushTemplates) {
						updateBrushTemplates = false;
						if (selectedTemplateBrush != -1) {
							if (drawListForTemplateBrush) {
								removePcPlotDrawList(g_PcPlotDrawLists.back());
								drawListForTemplateBrush = false;
							}
							if (globalBrushes.back().kdTree) delete globalBrushes.back().kdTree;
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
						for (DataSet& ds : g_PcPlotDataSets) {
							if (ds.oneData) {			//oneData indicates a .dlf data -> template brushes are available
								for (TemplateList& tl : ds.drawLists) {
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
										t.name = ds.name + " | " + tl.name;
										for (int i = 0; i < pcAttributes.size(); i++) {
											if (brushTemplateAttrEnabled[i]) {
												t.brushes[i] = tl.minMax[i];
											}
										}
										t.parent = &tl;
										t.parentDataSet = &ds;
										templateBrushes.push_back(t);
									}
								}
							}
							else if (showCsvTemplates && !ds.oneData) {
								auto tl = ++ds.drawLists.begin();
								for (; tl != ds.drawLists.end(); ++tl) {
									TemplateBrush t = {};
									t.name = ds.name + " | " + tl->name;
									for (int i = 0; i < pcAttributes.size(); i++) {
										if (brushTemplateAttrEnabled[i]) {
											t.brushes[i] = tl->minMax[i];
										}
									}
									t.parent = &(*tl);
									t.parentDataSet = &ds;
									templateBrushes.push_back(t);
								}
							}
						}
					}

					c++;
					c1++;
					offset += gap;
				}
				if (ImGui::Checkbox("Show .idxf brush templates", &showCsvTemplates)) {
					updateBrushTemplates = true;
				}

				ImGui::SameLine(250);
				if (ImGui::Button("Combine active global brushes")) {
					GlobalBrush combo;
					combo.name = "Combined(";
					bool any = false;
					for (auto& brush : globalBrushes) {
						if (!brush.active)
							continue;
						
						any = true;
						for (auto& br : brush.brushes) {
							combo.brushes[br.first].insert(combo.brushes[br.first].end(), br.second.begin(), br.second.end());
						}
						brush.active = false;
						combo.name += brush.name.substr(std::min(brush.name.length() ,(size_t)5)) + "|";
					}
					combo.active = true;
					combo.edited = true;
					combo.name += ")";
					combo.parent = nullptr;
					combo.kdTree = nullptr;

					if (any) {
						globalBrushes.push_back(combo);
						updateAllActiveIndices();
					}
				}
				ImGui::SameLine(500);
				if (ImGui::Button("Activate/deactivate parameter checkboxes")) {
					int activate = -1;
					for (auto i : pcAttrOrd) {
						if (!pcAttributeEnabled[i]) {
							c++;
							continue;
						}
						if (activate == -1) {
							activate = int((!brushTemplateAttrEnabled[i]));
						}
						(activate) ? brushTemplateAttrEnabled[i] = true :brushTemplateAttrEnabled[i] = false;
					}
					updateBrushTemplates = true;
				}

				//drawing the list for brush templates
				ImGui::BeginChild("brushTemplates", ImVec2(400, 200), true, ImGuiWindowFlags_HorizontalScrollbar);
				ImGui::Text("Brush Templates");
				ImGui::Separator();
				for (int i = 0; i < templateBrushes.size(); i++) {
					if (ImGui::Selectable(templateBrushes[i].name.c_str(), selectedTemplateBrush == i)) {
						selectedGlobalBrush = -1;
						pcPlotSelectedDrawList = -1;
						if (selectedTemplateBrush != i) {
							if (selectedTemplateBrush != -1) {
								if (drawListForTemplateBrush) {
									removePcPlotDrawList(g_PcPlotDrawLists.back());
									drawListForTemplateBrush = false;
								}
								if (globalBrushes.back().kdTree) delete globalBrushes.back().kdTree;
								globalBrushes.pop_back();
							}
							selectedTemplateBrush = i;
							GlobalBrush preview{};
							preview.active = true;
							preview.edited = false;
							preview.useMultivariate = false;
							preview.name = templateBrushes[i].name;
							for (int i = 0; i < pcAttributes.size(); ++i) {
								preview.brushes[i] = {};
							}
							for (const auto& brush : templateBrushes[i].brushes) {
								preview.brushes[brush.first].push_back(std::pair<unsigned int, std::pair<float, float>>(currentBrushId++, brush.second));
							}
							preview.parent = templateBrushes[i].parent;
							preview.parentDataset = templateBrushes[i].parentDataSet;
							preview.kdTree = nullptr;
							preview.fractureDepth = 0;
							globalBrushes.push_back(preview);
							if (std::find_if(g_PcPlotDrawLists.begin(), g_PcPlotDrawLists.end(), [preview](DrawList& dl) {return dl.name == preview.parent->name; }) == g_PcPlotDrawLists.end()) {
								drawListForTemplateBrush = true;
								createPcPlotDrawList(preview.parentDataset->drawLists.front(), *templateBrushes[i].parentDataSet, preview.parent->name.c_str());
							}
							pcPlotRender = updateAllActiveIndices();
							if (active3dAttribute.size())
								uploadDrawListTo3dView(g_PcPlotDrawLists.front(), active3dAttribute, "a", "b", "c");
						}
						else {
							selectedTemplateBrush = -1;
							if (drawListForTemplateBrush) {
								removePcPlotDrawList(g_PcPlotDrawLists.back());
								drawListForTemplateBrush = false;
							}
							if (globalBrushes.back().kdTree) delete globalBrushes.back().kdTree;
							globalBrushes.pop_back();
							pcPlotRender = updateAllActiveIndices();
							if (active3dAttribute.size())
								uploadDrawListTo3dView(g_PcPlotDrawLists.front(), active3dAttribute, "a", "b", "c");
						}
					}
					if (ImGui::IsItemClicked(2) && selectedTemplateBrush == i) {//creating a permanent Global Brush
						selectedGlobalBrush = globalBrushes.size() - 1;
						selectedTemplateBrush = -1;
					}
				}
				ImGui::EndChild();
				ImGui::SameLine();
				//Drawing the list of global brushes
				ImGui::BeginChild("GlobalBrushes", ImVec2(400, 200), true, ImGuiWindowFlags_HorizontalScrollbar);
				ImGui::Text("Global Brushes");
				ImGui::Separator();
				//child for names and selection
				bool popEnd = false;
				static int openConvertToLokal = -1, setParent = -1;
				for (int i = 0; i < globalBrushes.size(); i++) {
					if (ImGui::Selectable(globalBrushes[i].name.c_str(), selectedGlobalBrush == i, ImGuiSelectableFlags_None, ImVec2(350, 0))) {
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
					if (ImGui::BeginDragDropSource()) {
						GlobalBrush* brush = &globalBrushes[i];
						ImGui::SetDragDropPayload("GlobalBrush", &brush, sizeof(GlobalBrush*));
						ImGui::Text("%s", brush->name.c_str());
						ImGui::EndDragDropSource();
					}
					if (ImGui::IsItemClicked(1)) {
						ImGui::OpenPopup(("GlobalBrushPopup##" + globalBrushes[i].name).c_str());
					}
					if (ImGui::BeginPopup(("GlobalBrushPopup##" + globalBrushes[i].name).c_str(), ImGuiWindowFlags_AlwaysAutoResize)) {
						if (globalBrushes[i].kdTree) {
							if (ImGui::BeginCombo("Fracture depth", std::to_string(globalBrushes[i].fractureDepth).c_str())) {
								for (int j = 0; j < maxFractionDepth; j++) {
									if (ImGui::Selectable(std::to_string(j).c_str())) {
										globalBrushes[i].fractureDepth = j;
										globalBrushes[i].fractions = globalBrushes[i].kdTree->getBounds(j, outlierRank);
										globalBrushes[i].multivariates = globalBrushes[i].kdTree->getMultivariates(j);
										pcPlotRender = updateAllActiveIndices();
									}
								}
								ImGui::EndCombo();
							}
							if (ImGui::MenuItem("Recreate brush fractures")) {
								delete globalBrushes[i].kdTree;
								std::vector<std::vector<std::pair<float, float>>> bounds;
								globalBrushes[i].attributes.clear();
								//NOTE: only the first brush for each axis is taken
								int index = 0;
								for (auto brush : globalBrushes[i].brushes) {
									if (!brush.second.size()) continue;
									globalBrushes[i].attributes.push_back(brush.first);
									bounds.push_back({});
									for (auto& minMax : brush.second) {
										bounds[index].push_back(minMax.second);
									}
									index++;
								}
#ifdef _DEBUG
								std::cout << "Starting to build the kd tree for fracturing." << std::endl;
#endif
								if (globalBrushes[i].edited)
									globalBrushes[i].kdTree = new KdTree(globalBrushes[i].parentDataset->drawLists.front().indices, globalBrushes[i].parentDataset->data, globalBrushes[i].attributes, bounds, maxFractionDepth, (KdTree::BoundsBehaviour) boundsBehaviour, (KdTree::SplitBehaviour) splitBehaviour);
								else
									globalBrushes[i].kdTree = new KdTree(globalBrushes[i].parent->indices, globalBrushes[i].parentDataset->data, globalBrushes[i].attributes, bounds, maxFractionDepth, (KdTree::BoundsBehaviour) boundsBehaviour, (KdTree::SplitBehaviour) splitBehaviour);
#ifdef _DEBUG
								std::cout << "Kd tree done." << std::endl;
#endif
							}
							ImGui::MenuItem("Use normal dist", "", &globalBrushes[i].useMultivariate);
							ImGui::Separator();
						}
						else {
							if (ImGui::MenuItem("Create brush fractures")) {
								std::vector<std::vector<std::pair<float, float>>> bounds;
								globalBrushes[i].attributes.clear();
								//NOTE: only the first brush for each axis is taken
								int index = 0;
								for (auto brush : globalBrushes[i].brushes) {
									if (!brush.second.size()) continue;
									globalBrushes[i].attributes.push_back(brush.first);
									bounds.push_back({});
									for (auto& minMax : brush.second) {
										bounds[index].push_back(minMax.second);
									}
									index++;
								}
#ifdef _DEBUG
								std::cout << "Starting to build the kd tree for fracturing." << std::endl;
#endif
								if(globalBrushes[i].edited)
									globalBrushes[i].kdTree = new KdTree(globalBrushes[i].parentDataset->drawLists.front().indices, globalBrushes[i].parentDataset->data, globalBrushes[i].attributes, bounds, maxFractionDepth, (KdTree::BoundsBehaviour) boundsBehaviour, (KdTree::SplitBehaviour) splitBehaviour);
								else
									globalBrushes[i].kdTree = new KdTree(globalBrushes[i].parent->indices, globalBrushes[i].parentDataset->data, globalBrushes[i].attributes, bounds, maxFractionDepth, (KdTree::BoundsBehaviour) boundsBehaviour, (KdTree::SplitBehaviour) splitBehaviour);
#ifdef _DEBUG
								std::cout << "Kd tree done." << std::endl;
#endif
							}
						}
						ImGui::MenuItem("Brush edited", "", &globalBrushes[i].edited);
						if (ImGui::MenuItem("Focus axis on brush")) {
							std::pair<float, float> mm;
							for (auto& axis : globalBrushes[i].brushes) {
								mm.first = std::numeric_limits<float>::infinity();
								mm.second = -std::numeric_limits<float>::infinity();
								for (auto b : axis.second) {
									if (b.second.first < mm.first)
										mm.first = b.second.first;
									if (b.second.second > mm.second)
										mm.second = b.second.second;
								}
								pcAttributes[axis.first].min = mm.first;
								pcAttributes[axis.first].max = mm.second;
							}
							pcPlotRender = true;
							ImGui::CloseCurrentPopup();
						}
						if (ImGui::MenuItem("Convert to lokal brush")) {
							openConvertToLokal = i;
							ImGui::CloseCurrentPopup();
						}
						if (ImGui::MenuItem("Set parent Dataset")) {
							setParent = i;
							ImGui::CloseCurrentPopup();
						}
						if (ImGui::MenuItem("Delete")) {
							selectedTemplateBrush = -1;
							if (selectedGlobalBrush == i) {
								selectedGlobalBrush = -1;
							}
							if (popEnd) {
								if (globalBrushes.size() == 2) {
									globalBrushes[i] = globalBrushes[globalBrushes.size() - 1];
									if (drawListForTemplateBrush) {
										removePcPlotDrawList(g_PcPlotDrawLists.back());
										drawListForTemplateBrush = false;
									}
									if (globalBrushes.back().kdTree) delete globalBrushes.back().kdTree;
									globalBrushes.pop_back();
								}
								else {
									globalBrushes[i] = globalBrushes[globalBrushes.size() - 2];
									globalBrushes[globalBrushes.size() - 2] = globalBrushes[globalBrushes.size() - 1];
									if (drawListForTemplateBrush) {
										removePcPlotDrawList(g_PcPlotDrawLists.back());
										drawListForTemplateBrush = false;
									}
									if (globalBrushes.back().kdTree) delete globalBrushes.back().kdTree;
									globalBrushes.pop_back();
								}
							}
							else {
								globalBrushes[i] = globalBrushes[globalBrushes.size() - 1];
								if (drawListForTemplateBrush) {
									removePcPlotDrawList(g_PcPlotDrawLists.back());
									drawListForTemplateBrush = false;
								}
								if (globalBrushes.back().kdTree) delete globalBrushes.back().kdTree;
								globalBrushes.pop_back();
								pcPlotRender = updateAllActiveIndices();
							}
							ImGui::CloseCurrentPopup();
						}
						if (ImGui::MenuItem("Invert Brush")) {
							invertGlobalBrush(globalBrushes[i]);
							pcPlotRender = updateAllActiveIndices();
						}

						ImGui::EndPopup();
					}

					ImGui::SameLine();
					if (i < globalBrushes.size() && ImGui::Checkbox(("##cbgb_" + globalBrushes[i].name).c_str(), &globalBrushes[i].active)) {
						pcPlotRender = updateAllActiveIndices();
					}
				}
				if (popEnd) {
					if (drawListForTemplateBrush) {
						removePcPlotDrawList(g_PcPlotDrawLists.back());
						drawListForTemplateBrush = false;
					}
					if (globalBrushes.back().kdTree) delete globalBrushes.back().kdTree;
					globalBrushes.pop_back();
					if (selectedGlobalBrush == globalBrushes.size())
						selectedGlobalBrush = -1;
					pcPlotRender = updateAllActiveIndices();
				}
				if (openConvertToLokal != -1 && !ImGui::IsPopupOpen("Global to lokal brush")) {
					ImGui::OpenPopup("Global to lokal brush");
				}
				if (ImGui::BeginPopupModal("Global to lokal brush", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
					static char name[200] = {};
					ImGui::InputText("Name for the new DrawList", name, 200);
					ImGui::Text("Choose the Dataset with which a lokaly brushed drawlist will be created");
					ImGui::BeginChild("Datasets", ImVec2(300, 150));
					static int selected = -1;
					int c = 0;
					for (DataSet& ds : g_PcPlotDataSets) {
						if (ImGui::Selectable(ds.name.c_str(), c == selected)) {
							if (c == selected)
								selected = -1;
							else
								selected = c;
						}
						c++;
					}
					ImGui::EndChild();

					if ((ImGui::Button("Cancel")) || ImGui::IsKeyPressed(KEYESC)) {
						openConvertToLokal = -1;
						ImGui::CloseCurrentPopup();
					}
					ImGui::SameLine();
					if ((ImGui::Button("Create")) || ImGui::IsKeyPressed(KEYENTER)) {
						if (selected != -1) {
							auto ds = g_PcPlotDataSets.begin();
							for (int i = 0; i < selected; i++) {
								++ds;
							}
							createPcPlotDrawList(ds->drawLists.front(), *ds, name);
							DrawList& dl = g_PcPlotDrawLists.back();
							//copying all brushes to the new drawlsit
							for (int i = 0; i < pcAttributes.size(); i++) {
								dl.brushes.push_back(std::vector<Brush>());
							}
							for (auto axis : globalBrushes[openConvertToLokal].brushes) {
								for (auto brush : axis.second) {
									Brush b = {};
									b.id = currentBrushId++;
									b.minMax = brush.second;
									dl.brushes[axis.first].push_back(b);
								}
							}
							updateActiveIndices(dl);
							openConvertToLokal = -1;
							ImGui::CloseCurrentPopup();
						}
					}

					ImGui::EndPopup();
				}
				if (setParent != -1 && !ImGui::IsPopupOpen("Set brush parent")) {
					ImGui::OpenPopup("Set brush parent");
				}
				if (ImGui::BeginPopupModal("Set brush parent", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
					static int selds = 0;
					auto ds = g_PcPlotDataSets.begin();
					std::advance(ds, selds);
					if (ImGui::BeginCombo("Select new parent dataset", ds->name.c_str())) {
						ds = g_PcPlotDataSets.begin();
						for (int i = 0; i < g_PcPlotDataSets.size(); i++) {
							if (ImGui::MenuItem(ds->name.c_str())) {
								selds = i;
							}
							++ds;
						}
						ImGui::EndCombo();
					}
					if ((ImGui::Button("Cancel")) || ImGui::IsKeyPressed(KEYESC)) {
						setParent = -1;
						ImGui::CloseCurrentPopup();
					}
					ImGui::SameLine();
					if ((ImGui::Button("Confirm")) || ImGui::IsKeyPressed(KEYENTER)) {
						auto gb = globalBrushes.begin();
						std::advance(gb, setParent);
						auto ds = g_PcPlotDataSets.begin();
						std::advance(ds, selds);
						gb->parentDataset = &(*ds);
						setParent = -1;
						ImGui::CloseCurrentPopup();
					}

					ImGui::EndPopup();
				}

				ImGui::EndChild();

				//converting a lokal brush to a global one
				if (ImGui::BeginDragDropTarget()) {
					if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Drawlist")) {
						DrawList* dl = *((DrawList**)payload->Data);

						GlobalBrush gb = {};
						gb.active = true;
						gb.edited = true;
						gb.useMultivariate = false;
						gb.name = dl->name;
						gb.parent = dl->parentTemplateList;
						gb.kdTree = nullptr;
						DataSet* ds = &(*std::find_if(g_PcPlotDataSets.begin(), g_PcPlotDataSets.end(), [dl](auto d) {return d.name == dl->parentDataSet; }));
						gb.parentDataset = ds;
						for (int i = 0; i < dl->brushes.size(); ++i) {	//Attribute Index
							bool first = true;
							gb.brushes[i] = {};
							for (Brush& b : dl->brushes[i]) {
								gb.brushes[i].push_back(std::pair<unsigned int, std::pair<float, float>>(currentBrushId++, b.minMax));
								if (first) {
									gb.attributes.push_back(i);
									first = false;
								}
							}
						}
						globalBrushes.push_back(gb);
						//pcPlotRender = true;
						pcPlotRender = updateAllActiveIndices();
					}
					ImGui::EndDragDropTarget();
				}

				//Statistics for global brushes
				ImGui::SameLine();
				ImGui::BeginChild("Brush statistics", ImVec2(0, 200), true, ImGuiWindowFlags_HorizontalScrollbar);
				ImGui::Text("Brush statistics: Percentage of lines kept after brushing");
				if (ImGui::IsItemHovered())
					ImGui::SetTooltip("The bar shows the percentage of points active in comparison to all points in the drawlist. Exception: For the index list which is the parent of the brush, the ratio of points active in the index list vs. its parent data set is shown.");
				ImGui::Separator();
				//int hover = ImGui::PlotHistogramVertical("##testHistogramm", histogrammdata, 10, 0, NULL, 0, 1.0f, ImVec2(50, 200));
				for (auto& brush : globalBrushes) {
					ImGui::BeginChild(("##brushStat" + brush.name).c_str(), ImVec2(400, 0), true);
					ImGui::Text(brush.name.c_str());
					float lineHeight = ImGui::GetTextLineHeightWithSpacing();
					static std::vector<float> ratios;
					ImVec2 defaultCursorPos = ImGui::GetCursorPos();
					ImVec2 cursorPos = defaultCursorPos;
					ImVec2 screenCursorPos = ImGui::GetCursorScreenPos();
					cursorPos.x += 75 + ImGui::GetStyle().ItemInnerSpacing.x;
					ratios.clear();
					for (auto& ratio : brush.lineRatios) {
						static const float xOffset = 200; //offset for ratio comp
						ratios.push_back(ratio.second);
						ImGui::SetCursorPos(cursorPos);
						ImVec2 textSize = ImGui::CalcTextSize(ratio.first.c_str());
						if (textSize.x > xOffset / 2) {
							int c = 0;
							std::find_if(g_PcPlotDrawLists.begin(), g_PcPlotDrawLists.end(), [&c, ratio](DrawList& d) { c++; return d.name == ratio.first; });
							ImGui::Text("Drawlist %d", c);
							if (ImGui::IsItemHovered()) {
								ImGui::BeginTooltip();
								ImGui::Text(ratio.first.c_str());
								ImGui::EndTooltip();
							}
						}
						else {
							ImGui::Text(ratio.first.c_str());
						}
						DrawList* dl;
						for (auto it = g_PcPlotDrawLists.begin(); it != g_PcPlotDrawLists.end(); ++it) {
							if (it->name == ratio.first) {
								dl = &(*it);
								break;
							}
						}
						DataSet* ds;
						for (auto it = g_PcPlotDataSets.begin(); it != g_PcPlotDataSets.end(); ++it) {
							if (it->name == dl->parentDataSet) {
								ds = &(*it);
							}
						}

						// Todo: If this is the ratio for the first bar, then it should be divided by dl.indices.size() instead, since we want to know,
						// how many lines of the DL are still active
						// ratios.back() /= ds->data.size();
						if (brush.parent != nullptr && brush.parent->name == dl->name){
							// This would be the ratio of points in the index list / points in the data set (not dependent on brush)
							//ratios.back() = float(dl->indices.size()) / ds->data.size(); 

							// This is the ratio  (active points in idx list(only this global brush considered)) / (points in data set)
							ratios.back() /= ds->data.size();
						}
						else {
							// This is the ratio of active points in the dl (only this global brush considered) vs. total points in the drawlist
							ratios.back() /= dl->indices.size(); 
						}
						
						//drawing the line ratios
						ImGui::SetCursorPos(cursorPos);
						if (brush.parent != nullptr && brush.lineRatios.find(brush.parent->name) != brush.lineRatios.end()) {
							static const float width = 180;
							ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(screenCursorPos.x + xOffset, screenCursorPos.y), ImVec2(screenCursorPos.x + xOffset + width, screenCursorPos.y + lineHeight - 1), ImGui::ColorConvertFloat4ToU32(ImGui::GetStyle().Colors[ImGuiCol_FrameBg]), ImGui::GetStyle().FrameRounding);
							float linepos = width / 2;
							if (brush.parent->name == dl->name) {	//identity dataset
								// It cannot move past the middle line, since once all points of the idx-list are contained, there are no more to add.
								linepos += (brush.lineRatios[brush.parent->name] / (float)ds->data.size() > brush.parent->pointRatio) ? 
									(1 - (brush.parent->pointRatio / (brush.lineRatios[brush.parent->name] / (float)ds->data.size()))) * linepos : 
									-(1 - ((brush.lineRatios[brush.parent->name] / (float)ds->data.size()) / brush.parent->pointRatio)) * linepos;
								//linepos += (dl->activeInd.size()/(float)ds->data.size() > brush.parent->pointRatio) ? (1 - (brush.par ent->pointRatio / (dl->activeInd.size() / (float)ds->data.size()))) * linepos : -(1 - ((dl->activeInd.size() / (float)ds->data.size()) / brush.parent->pointRatio)) * linepos;
								//linepos += (brush.lineRatios[brush.parent->name] > brush.parent->pointRatio) ? (1 - (brush.parent->pointRatio / (brush.lineRatios[brush.parent->name]))) * linepos : -(1 - ((brush.lineRatios[brush.parent->name]) / brush.parent->pointRatio)) * linepos;
							}
							else {
								//linepos += (dl->activeInd.size()/(float)ds->data.size() > brush.parent->pointRatio) ? (1 - (brush.parent->pointRatio / (dl->activeInd.size() / (float)ds->data.size()))) * linepos : -(1 - ((dl->activeInd.size() / (float)ds->data.size()) / brush.parent->pointRatio)) * linepos;
								//linepos += (ratio.second > brush.lineRatios[brush.parent->name]) ? (1 - (brush.lineRatios[brush.parent->name] / ratio.second)) * linepos : -(1 - (ratio.second / brush.lineRatios[brush.parent->name])) * linepos;
								//linepos += (dl->activeInd.size()/(float)ds->data.size() > brush.lineRatios[brush.parent->name]) ? (1 - (brush.lineRatios[brush.parent->name] / (dl->activeInd.size() / (float)ds->data.size()))) * linepos : -(1 - ((dl->activeInd.size() / (float)ds->data.size()) / brush.lineRatios[brush.parent->name])) * linepos;
								linepos += (ratio.second / (float)ds->data.size() > (brush.lineRatios[brush.parent->name] / (float)brush.parentDataset->data.size())) ? (1 - ((brush.lineRatios[brush.parent->name] / (float)brush.parentDataset->data.size()) / (ratio.second / (float)ds->data.size()))) * linepos : -(1 - ((ratio.second / (float)ds->data.size()) / (brush.lineRatios[brush.parent->name] / (float)brush.parentDataset->data.size()))) * linepos;
							}
							ImGui::GetWindowDrawList()->AddLine(ImVec2(screenCursorPos.x + xOffset + linepos, screenCursorPos.y), ImVec2(screenCursorPos.x + xOffset + linepos, screenCursorPos.y + lineHeight - 1), IM_COL32(255, 0, 0, 255), 5);
							ImGui::GetWindowDrawList()->AddLine(ImVec2(screenCursorPos.x + xOffset + width / 2, screenCursorPos.y), ImVec2(screenCursorPos.x + xOffset + width / 2, screenCursorPos.y + lineHeight - 1), IM_COL32(255, 255, 255, 255));

							if (ImGui::IsMouseHoveringRect(ImVec2(screenCursorPos.x + xOffset, screenCursorPos.y), ImVec2(screenCursorPos.x + xOffset + width, screenCursorPos.y + lineHeight - 1))) {
								ImGui::BeginTooltip();
								if (linepos < width / 2) {
									ImGui::Text("Ratio is  %2.1f%%", ((linepos / (width / 2))) * 100);
								}
								else {
									ImGui::Text("Ratio is  %2.1f%%", (1 - ((linepos - width / 2) / width)) * 100);
								}
								ImGui::EndTooltip();
							}
						}

						screenCursorPos.y += lineHeight;
						cursorPos.y += lineHeight;
					}
					ImGui::SetCursorPos(defaultCursorPos);
					int hover = ImGui::PlotHistogramVertical(("##histo" + brush.name).c_str(), ratios.data(), ratios.size(), 0, NULL, 0, 1.0f, ImVec2(75, lineHeight * ratios.size()));
					if (hover != -1) {
						ImGui::BeginTooltip();
						ImGui::Text("%2.1f%%", ratios[hover] * 100);
						ImGui::EndTooltip();
					}


					ImGui::EndChild();
					ImGui::SameLine();
				}
				if (activeBrushRatios.size()) {
					ImGui::BeginChild("activeBrushRatios", ImVec2(200, 0), true);
					ImGui::Text("Active brushes combined");
					float lineHeight = ImGui::GetTextLineHeightWithSpacing();
					static std::vector<float> ratios;
					ImVec2 defaultCursorPos = ImGui::GetCursorPos();
					ImVec2 cursorPos = defaultCursorPos;
					cursorPos.x += 75 + ImGui::GetStyle().ItemInnerSpacing.x;
					ratios.clear();
					for (auto& ratio : activeBrushRatios) {
						ratios.push_back(ratio.second);
						ImGui::SetCursorPos(cursorPos);
						ImGui::Text(ratio.first.c_str());
						cursorPos.y += lineHeight;
					}
					ImGui::SetCursorPos(defaultCursorPos);
					int hover = ImGui::PlotHistogramVertical("##activeBrushesRatioHist", ratios.data(), ratios.size(), 0, NULL, 0, 1.0f, ImVec2(75, lineHeight * ratios.size()));
					if (hover != -1) {
						ImGui::BeginTooltip();
						ImGui::Text("%2.1f%%", ratios[hover] * 100);
						ImGui::EndTooltip();
					}


					ImGui::EndChild();
				}

				ImGui::EndChild();

				gap = (picSize.x - ((drawHistogramm) ? histogrammWidth / 2.0f * picSize.x : 0)) / (amtOfLabels - 1);
				//drawing axis lines
				if (enableAxisLines) {
					for (int i = 0; i < amtOfLabels; i++) {
						float x = picPos.x + i * gap + ((drawHistogramm) ? (histogrammWidth / 4.0 * picSize.x) : 0);
						ImVec2 a(x, picPos.y);
						ImVec2 b(x, picPos.y + picSize.y - 1);
						ImGui::GetWindowDrawList()->AddLine(a, b, IM_COL32((1 - PcPlotBackCol.x) * 255, (1 - PcPlotBackCol.y) * 255, (1 - PcPlotBackCol.z) * 255, 255), 1);
					}
				}

				//drawing pie chart for the first drawlist
				if (computeRatioPtsInDLvsIn1axbrushedParent && drawHistogramm) {
					// Count, how many histograms are drawn
					int nrActiveHists = 0;
					for (auto &currdl : g_PcPlotDrawLists)
					{
						nrActiveHists += int(currdl.showHistogramm);
					}
					float xStartOffset = -histogrammWidth / 4.0 * picSize.x;
					float xOffsetPerAttr = (histogrammWidth * picSize.x) / (2 * nrActiveHists);
					float xoffset = 0;
					xStartOffset +=  0.5 * xOffsetPerAttr;
					float pieBorder = 3;
					float radius = (xOffsetPerAttr / 2.0 - 3) * 0.9;

//					for (int i = 0; i < amtOfLabels; i++) { 
					// Loop through all attributes, since placeOfInd expects ids of active attributes only.
					int iActAttr = -1;
					for (int i = 0; i < pcAttrOrd.size(); i++) {
						if (!pcAttributeEnabled[i]) { continue; }

						++iActAttr;
						for (auto &currdl : g_PcPlotDrawLists){
							
							if (!currdl.showHistogramm) { continue; }



							float x = picPos.x + iActAttr * gap + ((drawHistogramm) ? (histogrammWidth / 4.0 * picSize.x) : 0);
							x += xStartOffset + xoffset;

							// x is the center of the axis. Now, the hist goes to the left and right, no matter how many are drawn. So, calculate the min_x, max_x, h*2 +1 axes, every second is the middle of a histogrm

							//std::cout << placeOfInd(i) << "\n";
							//std::cout << pcAttrOrd[i] << "\n";
							
							ImVec2 a(x, picPos.y + std::max(14.f, radius + 4.f));
							ImGui::GetWindowDrawList()->AddPie(a, radius, IM_COL32(255, 255, 255, 255), currdl.brushedRatioToParent[pcAttrOrd[i]], -1,  pieBorder);
							// ImGui::GetWindowDrawList()->AddPie(a, radius, IM_COL32(255, 255, 255, 255), currdl.brushedRatioToParent[placeOfInd(i)], -1, pieBorder);

							xoffset += xOffsetPerAttr;
						}
						xoffset = 0;
					}
				}

				//clearing the dragged brushes if ctrl key is released
				if (!ImGui::GetIO().MouseDown && !ImGui::GetIO().KeyCtrl) {
					brushDragIds.clear();
				}

				//drawing the global brush
				//global brushes currently only support change of brush but no adding of new brushes or deletion of brushes
				if (selectedGlobalBrush != -1) {
					if (globalBrushes[selectedGlobalBrush].fractureDepth) {
						GlobalBrush& globalBrush = globalBrushes[selectedGlobalBrush];
						for (int i = 0; i < globalBrush.fractions.size(); i++) {
							for (int j = 0; j < globalBrush.fractions[i].size(); j++) {
								int axis = globalBrush.attributes[j];
								float x = gap * placeOfInd(axis) + picPos.x - fractionBoxWidth / 2 + ((drawHistogramm) ? (histogrammWidth / 4.0 * picSize.x) : 0);
								float width = fractionBoxWidth;
								float y = ((globalBrush.fractions[i][j].second - pcAttributes[axis].max) / (pcAttributes[axis].min - pcAttributes[axis].max)) * picSize.y + picPos.y;
								float height = (globalBrush.fractions[i][j].second - globalBrush.fractions[i][j].first) / (pcAttributes[axis].max - pcAttributes[axis].min) * picSize.y;
								if (i < pow(2, maxRenderDepth))
									ImGui::GetWindowDrawList()->AddRect(ImVec2(x, y), ImVec2(x + width, y + height), IM_COL32(0, 230, 100, 255), 2, ImDrawCornerFlags_All, fractionBoxLineWidth);
							}
						}
					}
					else {
						for (auto& brush : globalBrushes[selectedGlobalBrush].brushes) {
							if (!pcAttributeEnabled[brush.first])
								continue;

							ImVec2 mousePos = ImGui::GetIO().MousePos;
							float x = gap * placeOfInd(brush.first) + picPos.x - BRUSHWIDTH / 2 + ((drawHistogramm) ? (histogrammWidth / 4.0 * picSize.x) : 0);
							float width = BRUSHWIDTH;

							int del = -1;
							int ind = 0;
							bool brushHover = false;
							for (auto& br : brush.second) {
								float y = ((br.second.second - pcAttributes[brush.first].max) / (pcAttributes[brush.first].min - pcAttributes[brush.first].max)) * picSize.y + picPos.y;
								float height = (br.second.second - br.second.first) / (pcAttributes[brush.first].max - pcAttributes[brush.first].min) * picSize.y;
								bool hover = mousePos.x > x&& mousePos.x<x + width && mousePos.y>y&& mousePos.y < y + height;
								//edgeHover = 0 -> No edge is hovered
								//edgeHover = 1 -> Top edge is hovered
								//edgeHover = 2 -> Bot edge is hovered
								int edgeHover = mousePos.x > x&& mousePos.x<x + width && mousePos.y>y - EDGEHOVERDIST && mousePos.y < y + EDGEHOVERDIST ? 1 : 0;
								edgeHover = mousePos.x > x&& mousePos.x<x + width && mousePos.y>y - EDGEHOVERDIST + height && mousePos.y < y + EDGEHOVERDIST + height ? 2 : edgeHover;
								int r = (brushDragIds.find(br.first) != brushDragIds.end()) ? 200 : 30;
								ImGui::GetWindowDrawList()->AddRect(ImVec2(x, y), ImVec2(x + width, y + height), IM_COL32(r, 0, 200, 255), 1, ImDrawCornerFlags_All, 5);
								brushHover |= hover || edgeHover;
								//set mouse cursor
								if (edgeHover || brushDragIds.size()) {
									ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
								}
								if (hover) {
									ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
								}
								//activate dragging of edge
								if (edgeHover && ImGui::GetIO().MouseClicked[0]) {
									brushDragIds.insert(br.first);
									brushDragMode = edgeHover;
								}
								if (hover && ImGui::GetIO().MouseClicked[0]) {
									brushDragIds.insert(br.first);
									brushDragMode = 0;
								}
								//drag edge
								if (brushDragIds.find(br.first) != brushDragIds.end() && ImGui::GetIO().MouseDown[0]) {
									globalBrushes[selectedGlobalBrush].edited = true;
									if (brushDragMode == 0) {
										float delta = ImGui::GetIO().MouseDelta.y / picSize.y * (pcAttributes[brush.first].max - pcAttributes[brush.first].min);
										br.second.second -= delta;
										br.second.first -= delta;
									}
									else if (brushDragMode == 1) {
										br.second.second -= ImGui::GetIO().MouseDelta.y / picSize.y * (pcAttributes[brush.first].max - pcAttributes[brush.first].min); //((mousePos.y - picPos.y) / picSize.y) * (pcAttributes[brush.first].min - pcAttributes[brush.first].max) + pcAttributes[brush.first].max;
									}
									else {
										br.second.first -= ImGui::GetIO().MouseDelta.y / picSize.y * (pcAttributes[brush.first].max - pcAttributes[brush.first].min); //((mousePos.y - picPos.y) / picSize.y) * (pcAttributes[brush.first].min - pcAttributes[brush.first].max) + pcAttributes[brush.first].max;
									}

									//switching edges if max value of brush is smaller than min value
									if (br.second.second < br.second.first) {
										float tmp = br.second.second;
										br.second.second = br.second.first;
										br.second.first = tmp;
										brushDragMode = (brushDragMode == 1) ? 2 : 1;
									}

									if (ImGui::GetIO().MouseDelta.y) {
										pcPlotRender = updateAllActiveIndices();
										updateIsoSurface(globalBrushes[selectedGlobalBrush]);
									}
								}
								//release edge
								if (brushDragIds.find(br.first) != brushDragIds.end() && ImGui::GetIO().MouseReleased[0] && !ImGui::GetIO().KeyCtrl) {
									brushDragIds.clear();
									pcPlotRender = updateAllActiveIndices();
									updateIsoSurface(globalBrushes[selectedGlobalBrush]);
								}

								//check for deletion of brush
								if (ImGui::GetIO().MouseClicked[1] && hover) {
									del = ind;
									brushDragIds.clear();
								}

								//adjusting the bounds of the brush by a mu
								if (brushHover && ImGui::GetIO().MouseWheel) {
									if (ImGui::GetIO().MouseWheel > 0) {
										br.second.first += ImGui::GetIO().MouseWheel * (pcAttributes[brush.first].max - pcAttributes[brush.first].min) * brushMuFactor;
									}
									else {
										br.second.second += ImGui::GetIO().MouseWheel * (pcAttributes[brush.first].max - pcAttributes[brush.first].min) * brushMuFactor;
									}
									pcPlotRender = updateAllActiveIndices();
									updateIsoSurface(globalBrushes[selectedGlobalBrush]);
								}

								//draw tooltip on hover for min and max value
								if (hover || edgeHover || brushDragIds.find(br.first) != brushDragIds.end()) {
									float xAnchor = .5f;
									if (pcAttrOrd[brush.first] == 0) xAnchor = 0;
									if (pcAttrOrd[brush.first] == pcAttributes.size() - 1) xAnchor = 1;

									ImGui::SetNextWindowPos({ x + width / 2,y }, 0, { xAnchor,1 });
									ImGui::SetNextWindowBgAlpha(ImGui::GetStyle().Colors[ImGuiCol_PopupBg].w * 0.60f);
									ImGuiWindowFlags flags = ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking;
									ImGui::Begin("Tooltip brush max", NULL, flags);
									ImGui::Text("%f", br.second.second);
									ImGui::End();

									ImGui::SetNextWindowPos({ x + width / 2, y + height }, 0, { xAnchor,0 });
									ImGui::SetNextWindowBgAlpha(ImGui::GetStyle().Colors[ImGuiCol_PopupBg].w * 0.60f);
									ImGui::Begin("Tooltip brush min", NULL, flags);
									ImGui::Text("%f", br.second.first);
									ImGui::End();
								}

								ind++;
							}
							//deleting a brush
							if (del != -1) {
								globalBrushes[selectedGlobalBrush].edited = true;
								brush.second[del] = brush.second[brush.second.size() - 1];
								brush.second.pop_back();
								del = -1;
								pcPlotRender = updateAllActiveIndices();
								updateIsoSurface(globalBrushes[selectedGlobalBrush]);
							}

							//create a new brush
							bool axisHover = mousePos.x > x&& mousePos.x < x + BRUSHWIDTH && mousePos.y > picPos.y&& mousePos.y < picPos.y + picSize.y;
							if (!brushHover && axisHover && brushDragIds.size() == 0) {
								ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

								if (ImGui::GetIO().MouseClicked[0]) {
									globalBrushes[selectedGlobalBrush].edited = true;
									std::pair<unsigned int, std::pair<float, float>> temp = {};
									temp.first = currentBrushId++;
									temp.second.first = ((mousePos.y - picPos.y) / picSize.y) * (pcAttributes[brush.first].min - pcAttributes[brush.first].max) + pcAttributes[brush.first].max;
									temp.second.second = temp.second.first;
									brushDragIds.insert(temp.first);
									brushDragMode = 1;
									brush.second.push_back(temp);
								}
							}
						}
					}
				}

				//drawing the template brush, these are not changeable
				if (selectedTemplateBrush != -1) {
					for (const auto& brush : globalBrushes.back().brushes) {
						if (!pcAttributeEnabled[brush.first] || !brush.second.size())
							continue;

						float x = gap * placeOfInd(brush.first) + picPos.x - BRUSHWIDTH / 2 + ((drawHistogramm) ? (histogrammWidth / 4.0 * picSize.x) : 0);
						float y = ((brush.second[0].second.second - pcAttributes[brush.first].max) / (pcAttributes[brush.first].min - pcAttributes[brush.first].max)) * picSize.y + picPos.y;
						float width = BRUSHWIDTH;
						float height = (brush.second[0].second.second - brush.second[0].second.first) / (pcAttributes[brush.first].max - pcAttributes[brush.first].min) * picSize.y;
						ImGui::GetWindowDrawList()->AddRect(ImVec2(x, y), ImVec2(x + width, y + height), IM_COL32(30, 0, 200, 150), 1, ImDrawCornerFlags_All, 5);
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
					//drawing the brushes
					for (Brush& b : dl->brushes[i]) {
						float y = ((b.minMax.second - pcAttributes[i].max) / (pcAttributes[i].min - pcAttributes[i].max)) * picSize.y + picPos.y;
						float width = BRUSHWIDTH;
						float height = (b.minMax.second - b.minMax.first) / (pcAttributes[i].max - pcAttributes[i].min) * picSize.y;
						int g = (brushDragIds.find(b.id) != brushDragIds.end()) ? 200 : 30;
						ImGui::GetWindowDrawList()->AddRect(ImVec2(x, y), ImVec2(x + width, y + height), IM_COL32(200, g, 0, 255), 1, ImDrawCornerFlags_All, 5);

						bool hover = mousePos.x > x&& mousePos.x<x + width && mousePos.y>y&& mousePos.y < y + height;
						//edgeHover = 0 -> No edge is hovered
						//edgeHover = 1 -> Top edge is hovered
						//edgeHover = 2 -> Bot edge is hovered
						int edgeHover = mousePos.x > x&& mousePos.x<x + width && mousePos.y>y - EDGEHOVERDIST && mousePos.y < y + EDGEHOVERDIST ? 1 : 0;
						edgeHover = mousePos.x > x&& mousePos.x<x + width && mousePos.y>y - EDGEHOVERDIST + height && mousePos.y < y + EDGEHOVERDIST + height ? 2 : edgeHover;
						brushHover |= hover || edgeHover;

						//set mouse cursor
						if (edgeHover || brushDragIds.size()) {
							ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
						}
						if (hover) {
							ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
						}
						//activate dragging of edge
						if (edgeHover && ImGui::GetIO().MouseClicked[0]) {
							brushDragIds.insert(b.id);
							brushDragMode = edgeHover;
						}
						if (hover && ImGui::GetIO().MouseClicked[0]) {
							brushDragIds.insert(b.id);
							brushDragMode = 0;
						}
						//drag edge
						if (brushDragIds.find(b.id) != brushDragIds.end() && ImGui::GetIO().MouseDown[0]) {
							if (brushDragMode == 0) {
								float delta = ImGui::GetIO().MouseDelta.y / picSize.y * (pcAttributes[i].max - pcAttributes[i].min);
								b.minMax.second -= delta;
								b.minMax.first -= delta;
							}
							else if (brushDragMode == 1) {
								b.minMax.second -= ImGui::GetIO().MouseDelta.y / picSize.y * (pcAttributes[i].max - pcAttributes[i].min); //((mousePos.y - picPos.y) / picSize.y) * (pcAttributes[i].min - pcAttributes[i].max) + pcAttributes[i].max;
							}
							else {
								b.minMax.first -= ImGui::GetIO().MouseDelta.y / picSize.y * (pcAttributes[i].max - pcAttributes[i].min); //((mousePos.y - picPos.y) / picSize.y) * (pcAttributes[i].min - pcAttributes[i].max) + pcAttributes[i].max;
							}

							//switching edges if max value of brush is smaller than min value
							if (b.minMax.second < b.minMax.first) {
								float tmp = b.minMax.second;
								b.minMax.second = b.minMax.first;
								b.minMax.first = tmp;
								brushDragMode = (brushDragMode == 1) ? 2 : 1;
							}

							if (ImGui::GetIO().MouseDelta.y) {
								pcPlotRender = updateActiveIndices(*dl);
							}
						}
						//release edge
						if (brushDragIds.find(b.id) != brushDragIds.end() && ImGui::GetIO().MouseReleased[0] && !ImGui::GetIO().KeyCtrl) {
							brushDragIds.clear();
							pcPlotRender = updateActiveIndices(*dl);
						}

						//check for deletion of brush
						if (ImGui::GetIO().MouseClicked[1] && hover) {
							del = ind;
							brushDragIds.clear();
						}

						//draw tooltip on hover for min and max value
						if (hover || edgeHover || brushDragIds.find(b.id) != brushDragIds.end()) {
							float xAnchor = .5f;
							if (pcAttrOrd[i] == 0) xAnchor = 0;
							if (pcAttrOrd[i] == pcAttributes.size() - 1) xAnchor = 1;

							ImGui::SetNextWindowPos({ x + width / 2,y }, 0, { xAnchor,1 });
							ImGui::SetNextWindowBgAlpha(ImGui::GetStyle().Colors[ImGuiCol_PopupBg].w * 0.60f);
							ImGuiWindowFlags flags = ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking;
							ImGui::Begin("Tooltip brush max", NULL, flags);
							ImGui::Text("%f", b.minMax.second);
							ImGui::End();

							ImGui::SetNextWindowPos({ x + width / 2, y + height }, 0, { xAnchor,0 });
							ImGui::SetNextWindowBgAlpha(ImGui::GetStyle().Colors[ImGuiCol_PopupBg].w * 0.60f);
							ImGui::Begin("Tooltip brush min", NULL, flags);
							ImGui::Text("%f", b.minMax.first);
							ImGui::End();
						}

						ind++;
					}

					//deleting a brush
					if (del != -1) {
						dl->brushes[i][del] = dl->brushes[i][dl->brushes[i].size() - 1];
						dl->brushes[i].pop_back();
						del = -1;
						pcPlotRender = updateActiveIndices(*dl);
					}

					//create a new brush
					bool axisHover = mousePos.x > x&& mousePos.x < x + BRUSHWIDTH && mousePos.y > picPos.y&& mousePos.y < picPos.y + picSize.y;
					if (!brushHover && axisHover && brushDragIds.size() == 0) {
						ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

						if (ImGui::GetIO().MouseClicked[0]) {
							Brush temp = {};
							temp.id = currentBrushId++;
							temp.minMax.first = ((mousePos.y - picPos.y) / picSize.y) * (pcAttributes[i].min - pcAttributes[i].max) + pcAttributes[i].max;
							temp.minMax.second = temp.minMax.first;
							brushDragIds.insert(temp.id);
							brushDragMode = 1;
							dl->brushes[i].push_back(temp);
						}
					}
				}
			}

			//handling priority selection
			if (prioritySelectAttribute) {
				pcPlotSelectedDrawList = -1;
				selectedGlobalBrush = -1;
				ImGui::GetWindowDrawList()->AddRect(picPos, ImVec2(picPos.x + picSize.x, picPos.y + picSize.y), IM_COL32(255, 255, 0, 255), 0, 15, 5);
				if ((ImGui::GetIO().MousePos.x<picPos.x || ImGui::GetIO().MousePos.x>picPos.x + picSize.x || ImGui::GetIO().MousePos.y<picPos.y || ImGui::GetIO().MousePos.y>picPos.y + picSize.y) && ImGui::GetIO().MouseClicked[0]) {
					prioritySelectAttribute = false;
				}

				for (int i = 0; i < pcAttributes.size(); i++) {
					if (!pcAttributeEnabled[i])
						continue;

					int del = -1;
					int ind = 0;
					bool brushHover = false;

					ImVec2 mousePos = ImGui::GetIO().MousePos;
					float x = gap * placeOfInd(i) + picPos.x - BRUSHWIDTH / 2 + ((drawHistogramm) ? (histogrammWidth / 4.0 * picSize.x) : 0);
					bool axisHover = mousePos.x > x&& mousePos.x < x + BRUSHWIDTH && mousePos.y > picPos.y&& mousePos.y < picPos.y + picSize.y;

					if (axisHover) {
						ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

						if (ImGui::GetIO().MouseClicked[0]) {
							prioritySelectAttribute = false;
							priorityAttribute = i;
							priorityAttributeCenterValue = ((mousePos.y - picPos.y) / picSize.y) * (pcAttributes[i].min - pcAttributes[i].max) + pcAttributes[i].max;
							upatePriorityColorBuffer();
							pcPlotRender = true;
						}
					}
				}
			}

			// reorder Histograms in ViolinPlots if pcPlotRender==true and if requested.
			if (pcPlotRender && renderOrderDLConsider && renderOrderDLConsiderNonStop && enableDrawlistViolinPlots) {
				sortAllHistograms(std::string("dl"));
			}
			if (pcPlotRender && renderOrderAttConsider && renderOrderAttConsiderNonStop && enableAttributeViolinPlots) {
				sortAllHistograms(std::string("attr"));
			}



			//Settings section
			ImGui::BeginChild("Settings", ImVec2(500, -1), true);
			ImGui::Text("Settings");
			ImGui::Separator();

			ImGui::Text("Histogram Settings:");
			ImGui::Columns(2);
			if (ImGui::Checkbox("Draw Histogram", &drawHistogramm)) {
				pcPlotRender = true;
				if (computeRatioPtsInDLvsIn1axbrushedParent)
				{
					pcPlotRender = updateAllActiveIndices();
				}
			}
			ImGui::NextColumn();
			if (ImGui::Checkbox("Draw Pie-Ratio", &computeRatioPtsInDLvsIn1axbrushedParent)) {
				if (drawHistogramm) {
					pcPlotRender = updateAllActiveIndices();
				}
			}


			ImGui::Columns(1);
			if (ImGui::SliderFloat("Histogram Width", &histogrammWidth, 0, .5) && drawHistogramm) {
				if (histogrammDrawListComparison != -1) {
					uploadDensityUiformBuffer();
				}
				pcPlotRender = true;
			}
			if (ImGui::ColorEdit4("Histogram Background", &histogrammBackCol.x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar) && drawHistogramm) {
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

			if (ImGui::Checkbox("Enable grayscale density", &enableDensityGreyscale)) {
				if (pcAttributes.size()) {
					uploadDensityUiformBuffer();
					pcPlotRender = true;
				}
			}

			//if (ImGui::Checkbox("Enable additive density", &pcPlotLinDensity)) {
			//	if (pcAttributes.size()) {
			//		uploadDensityUiformBuffer();
			//		pcPlotRender = true;
			//	}
			//}

			if (ImGui::Checkbox("Enable median calc", &calculateMedians)) {
				for (DrawList& dl : g_PcPlotDrawLists) {
					dl.activeMedian = 0;
				}
			}

			if (ImGui::Checkbox("Enable brushing", &enableBrushing)) {
				pcPlotRender = updateAllActiveIndices();
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

			if (ImGui::InputInt("Priority draw list index", &priorityListIndex, 1, 1) && priorityAttribute != -1) {
				if (priorityListIndex < 0)priorityListIndex = 0;
				if (priorityListIndex >= g_PcPlotDrawLists.size())priorityListIndex = g_PcPlotDrawLists.size() - 1;
				upatePriorityColorBuffer();
				pcPlotRender = true;
			}

			if (ImGui::BeginCombo("Priority rendering", (priorityAttribute == -1) ? "Off" : pcAttributes[priorityAttribute].name.c_str())) {
				if (ImGui::MenuItem("Off")) {
					priorityAttribute = -1;
					pcPlotRender = true;
				}
				for (int i = 0; i < pcAttributes.size(); i++) {
					if (pcAttributeEnabled[i]) {
						if (ImGui::MenuItem(pcAttributes[i].name.c_str()) && g_PcPlotDrawLists.size()) {
							priorityAttribute = i;
							priorityAttributeCenterValue = pcAttributes[i].max;
							upatePriorityColorBuffer();
							pcPlotRender = true;
						}
					}
				}

				ImGui::EndCombo();
			}

			if (ImGui::IsKeyPressed(KEYP)) {
				if (prioritySelectAttribute) {
					prioritySelectAttribute = false;
				}
				else {
					prioritySelectAttribute = true;
				}
			}


			if (ImGui::Button("Set Priority center")) {
				if (ImGui::IsItemHovered()) {
					ImGui::SetTooltip("or press 'P' to set a priority rendering center");
				}

				prioritySelectAttribute = true;
			}

			if (ImGui::Checkbox("Put 3d view always in focus", &view3dAlwaysOnTop)) {

			}

			auto histComp = g_PcPlotDrawLists.begin();
			if (histogrammDrawListComparison != -1) std::advance(histComp, histogrammDrawListComparison);
			if (ImGui::BeginCombo("Histogramm Comparison", (histogrammDrawListComparison == -1) ? "Off" : histComp->name.c_str())) {
				if (ImGui::MenuItem("Off")) {
					histogrammDrawListComparison = -1;
					uploadDensityUiformBuffer();
					if (drawHistogramm) {
						pcPlotRender = true;
					}
				}
				auto it = g_PcPlotDrawLists.begin();
				for (int i = 0; i < g_PcPlotDrawLists.size(); i++, ++it) {
					if (ImGui::MenuItem(it->name.c_str())) {
						histogrammDrawListComparison = i;
						uploadDensityUiformBuffer();
						if (drawHistogramm) {
							pcPlotRender = true;
						}
					}
				}

				ImGui::EndCombo();
			}

			ImGui::Checkbox("Create default drawlist on load", &createDefaultOnLoad);

			ImGui::DragInt("Live brush threshold", &liveBrushThreshold, 1000);

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
				if (createDefaultOnLoad) {
					//pcPlotRender = true;
					createPcPlotDrawList(g_PcPlotDataSets.back().drawLists.front(), g_PcPlotDataSets.back(), g_PcPlotDataSets.back().name.c_str());
					pcPlotRender = updateActiveIndices(g_PcPlotDrawLists.back());
				}
			}
			ImGui::EndChild();

			//DataSets, from which draw lists can be created
			ImGui::SameLine();

			ImGui::BeginChild("DataSets", ImVec2((ImGui::GetWindowWidth() - 500) / 2, -1), true, ImGuiWindowFlags_HorizontalScrollbar);

			DataSet* destroySet = NULL;
			bool destroy = false;

			ImGui::Text("Datasets");
			ImGui::Separator();
			for (DataSet& ds : g_PcPlotDataSets) {
				if (ImGui::TreeNode(ds.name.c_str())) {
					static TemplateList* convert = nullptr;
					int c = 0;		//counter to reduce the amount of template lists being drawn
					for (TemplateList& tl : ds.drawLists) {
						if (c++ > 10000)break;
						if (ImGui::Button(tl.name.c_str())) {
							ImGui::OpenPopup(tl.name.c_str());
							strcpy(pcDrawListName, tl.name.c_str());
						}
						if (ImGui::IsItemClicked(1)) {
							ImGui::OpenPopup("CONVERTTOBRUSH");
							convert = &tl;
						}
						if (ImGui::BeginPopupModal(tl.name.c_str(), NULL, ImGuiWindowFlags_AlwaysAutoResize))
						{
							ImGui::Text((std::string("Creating a drawing list from ") + tl.name + "\n\n").c_str());
							ImGui::Separator();
							ImGui::InputText("Drawlist Name", pcDrawListName, 200);

							if ((ImGui::Button("Create", ImVec2(120, 0))) || ImGui::IsKeyPressed(KEYENTER))
							{
								ImGui::CloseCurrentPopup();

								createPcPlotDrawList(tl, ds, pcDrawListName);
								pcPlotRender = updateActiveIndices(g_PcPlotDrawLists.back());
							}
							ImGui::SetItemDefaultFocus();
							ImGui::SameLine();
							if ((ImGui::Button("Cancel", ImVec2(120, 0))) || ImGui::IsKeyPressed(KEYESC)) 
							{ ImGui::CloseCurrentPopup(); }
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
						if ((ImGui::Button("Create")) || ImGui::IsKeyPressed(KEYENTER)) {
							GlobalBrush brush = {};
							brush.name = std::string(n);
							brush.active = true;
							brush.useMultivariate = false;
							brush.edited = false;
							brush.parent = convert;
							brush.kdTree = nullptr;
							brush.parentDataset = &ds;
							for (int i = 0; i < pcAttributes.size(); i++) {
								if (activeBrushAttributes[i]) {
									brush.brushes[i].push_back(std::pair<int, std::pair<float, float>>(currentBrushId++, convert->minMax[i]));
								}
								else {
									brush.brushes[i] = {};
								}
							}
							globalBrushes.push_back(brush);
							pcPlotRender = updateAllActiveIndices();

							ImGui::CloseCurrentPopup();
						}
						ImGui::SameLine();
						if ((ImGui::Button("Cancel")) || ImGui::IsKeyPressed(KEYESC)) {
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
						if ((ImGui::Button("Create")) || ImGui::IsKeyPressed(KEYENTER)) {
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
						if ((ImGui::Button("Cancel")) || ImGui::IsKeyPressed(KEYESC)) {
							ImGui::CloseCurrentPopup();
						}

						ImGui::EndPopup();
					}

					//Popup for adding a custom index list
					ImGui::PushStyleColor(ImGuiCol_Button, (ImGuiCol)IM_COL32(20, 220, 0, 255));
					if (ImGui::Button("ADDINDEXLIST")) {
						ImGui::OpenPopup("ADDINDEXLIST");
						addIndeces = true;
					}
					ImGui::PopStyleColor();

					if (ImGui::BeginPopupModal("ADDINDEXLIST", NULL, ImGuiWindowFlags_AlwaysAutoResize))
					{
						ImGui::Text("Path for the new Indexlist (Alternativley drag and drop here):");
						ImGui::InputText("Path", pcFilePath, 200);
						ImGui::Separator();
						if (ImGui::Button("Select all")) {
							for (int i = 0; i < droppedPaths.size(); i++) {
								createDLForDrop[i] = true;
							}
						}
						ImGui::SameLine();
						if (ImGui::Button("Deselect all")) {
							for (int i = 0; i < droppedPaths.size(); i++) {
								createDLForDrop[i] = false;
							}
						}

						ImGui::BeginChild("ScrollingRegion", ImVec2(0, 400), false, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_HorizontalScrollbar);

						if (droppedPaths.size() == 0) {
							ImGui::Text("Drag and drop indexlists here to open them.");
						}
						else {
							ImGui::SliderFloat("Default Alpha Value", &alphaDrawLists, .0f, 1.0f);
						}

						for (int i = 0; i < droppedPaths.size(); i++) {
							ImGui::Text(droppedPaths[i].c_str());
							ImGui::SameLine();
							ImGui::Checkbox(("##" + droppedPaths[i]).c_str(), &createDLForDrop[i]);
						}

						ImGui::EndChild();

						if ((ImGui::Button("Add Indeces", ImVec2(120, 0))) || ImGui::IsKeyPressed(KEYENTER)) {
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
						if ((ImGui::Button("Cancel", ImVec2(120, 0))) || ImGui::IsKeyPressed(KEYESC)) {
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
					ImGui::PushStyleColor(ImGuiCol_Button, (ImGuiCol)IM_COL32(220, 20, 0, 255));
					if (ImGui::Button("DELETE"))
						ImGui::OpenPopup("DELETE");
					ImGui::PopStyleColor();
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
						if ((ImGui::Button("Cancel", ImVec2(120, 0))) || ImGui::IsKeyPressed(KEYESC)) 
						{ ImGui::CloseCurrentPopup(); }
						ImGui::EndPopup();
					}
					ImGui::TreePop();
				}
				ImGui::OpenPopupOnItemClick("ReducedAttributesSize", 1);
				if (ImGui::BeginPopupModal("ReducedAttributesSize", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
					static float reducedSize = 1.0f;
					if (ImGui::InputFloat("Input the ratio of the reduced dataset size", &reducedSize, 0.0f, 0.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue) ||
						ImGui::Button("Save")) {
						for (TemplateList& tl : ds.drawLists) {
							tl.pointRatio = (float)tl.indices.size() / (ds.data.size() * reducedSize);

						}

						ImGui::CloseCurrentPopup();
					}
					ImGui::SameLine();
					if ((ImGui::Button("Cancel")) || ImGui::IsKeyPressed(KEYESC)) {
						ImGui::CloseCurrentPopup();
					}

					ImGui::EndPopup();
				}

			}
			ImGui::EndChild();
			//Destroying a dataset if it was selected
			if (destroy)
				destroyPcPlotDataSet(*destroySet);

			//Showing the Drawlists
			DrawList* changeList;
			destroy = false;
			bool up = false;
			bool down = false;

			ImGui::SameLine();
			ImGui::BeginChild("DrawLists", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);

			ImGui::Text("Draw lists");
			ImGui::Separator();
			int count = 0;

			ImGui::Columns(9, "Columns", true);
			if (rescaleTableColumns) {
				ImGui::SetColumnWidth(0, ImGui::GetWindowContentRegionWidth() - 275);
				ImGui::SetColumnWidth(1, 25);
				ImGui::SetColumnWidth(2, 25);
				ImGui::SetColumnWidth(3, 25);
				ImGui::SetColumnWidth(4, 25);
				ImGui::SetColumnWidth(5, 25);
				ImGui::SetColumnWidth(6, 25);
				ImGui::SetColumnWidth(7, 100);
				ImGui::SetColumnWidth(8, 25);
			}

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
			ImGui::Separator();
			bool compareDrawLists = false;
			static DrawListComparator drawListComparator;
			bool exportIdxf = false;
			bool exportCsv = false;
			static DrawList* exportDl;
			for (DrawList& dl : g_PcPlotDrawLists) {
				if (ImGui::Selectable(dl.name.c_str(), count == pcPlotSelectedDrawList)) {
					selectedGlobalBrush = -1;
					if (count == pcPlotSelectedDrawList)
						pcPlotSelectedDrawList = -1;
					else
						pcPlotSelectedDrawList = count;
				}
				if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
					DrawList* point = &dl;
					ImGui::SetDragDropPayload("Drawlist", &point, sizeof(DrawList*));
					ImGui::Text("%s", dl.name.c_str());
					ImGui::EndDragDropSource();
				}
				if (ImGui::IsItemHovered() && io.MouseClicked[1]) {
					ImGui::OpenPopup(("drawListMenu" + dl.name).c_str());
				}
				if (ImGui::BeginPopup(("drawListMenu" + dl.name).c_str())) {
					if (ImGui::MenuItem("Immune to global brushes", "", &dl.immuneToGlobalBrushes)) {
						pcPlotRender = updateActiveIndices(dl);
					}
					if (ImGui::BeginCombo("##combo", "Compare to")) // The second parameter is the label previewed before opening the combo.
					{
						auto draw = g_PcPlotDrawLists.begin();
						for (int n = 0; n < g_PcPlotDrawLists.size(); n++)
						{
							if (draw->name == dl.name || draw->parentDataSet != dl.parentDataSet) {
								++draw;
								continue;
							}

							if (ImGui::Selectable(draw->name.c_str(), false)) {
								std::vector<uint32_t> activeDraw, activeDl;
								uint32_t boolSize;
								for (DataSet& ds : g_PcPlotDataSets) {
									if (ds.name == draw->parentDataSet) {
										boolSize = ds.data.size();
										break;
									}
								}
								bool* actDraw = new bool[boolSize];
								bool* actDl = new bool[boolSize];
								VkUtil::downloadData(g_Device, draw->dlMem, draw->activeIndicesBufferOffset, boolSize * sizeof(bool), actDraw);
								VkUtil::downloadData(g_Device, dl.dlMem, dl.activeIndicesBufferOffset, boolSize * sizeof(bool), actDl);
								for (int i : draw->indices) {
									if (actDraw[i]) activeDraw.push_back(i);
								}
								for (int i : dl.indices) {
									if (actDl[i]) activeDl.push_back(i);
								}
								delete[] actDraw;
								delete[] actDl;
								std::set<int> a(activeDl.begin(), activeDl.end());
								std::set<int> b(activeDraw.begin(), activeDraw.end());
								std::vector<uint32_t> aOrB;
								std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(aOrB));
								std::vector<uint32_t> aMinusB;
								std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(aMinusB));
								std::vector<uint32_t> bMinusA;
								std::set_difference(b.begin(), b.end(), a.begin(), a.end(), std::back_inserter(bMinusA));
								std::vector<uint32_t> aAndb;
								std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(aAndb));

								drawListComparator.parentDataset = dl.parentDataSet;
								drawListComparator.a = dl.name;
								drawListComparator.b = draw->name;
								drawListComparator.aInd = activeDl;
								drawListComparator.bInd = activeDraw;
								drawListComparator.aOrB = aOrB;
								drawListComparator.aMinusB = aMinusB;
								drawListComparator.bMinusA = bMinusA;
								drawListComparator.aAndb = aAndb;
								compareDrawLists = true;
							}
							++draw;
						}
						ImGui::EndCombo();
					}
					if (ImGui::MenuItem("Export as .idxf")) {
						exportIdxf = true;
						exportDl = &dl;
						ImGui::CloseCurrentPopup();
					}
					if (ImGui::MenuItem("Export as .csv")) {
						exportCsv = true;
						exportDl = &dl;
						ImGui::CloseCurrentPopup();
					}
					ImGui::Separator();
					if (ImGui::MenuItem("Send to Bubble plotter")) {
						DataSet* parent;
						for (auto it = g_PcPlotDataSets.begin(); it != g_PcPlotDataSets.end(); ++it) {
							if (it->name == dl.parentDataSet) {
								parent = &(*it);
							}
						}
						std::vector<uint32_t> ids;
						std::vector<std::string> attributeNames;
						std::vector<std::pair<float, float>> attributeMinMax;
						for (int i = 0; i < pcAttributes.size(); ++i) {
							attributeNames.push_back(pcAttributes[i].name);
							attributeMinMax.push_back({ pcAttributes[i].min,pcAttributes[i].max });
						}
						glm::uvec3 posIndices(0, 2, 1);
						bubblePlotter->setBubbleData(posIndices, dl.indices, attributeNames, attributeMinMax, parent->data, dl.buffer, dl.activeIndicesBufferView, attributeNames.size(), parent->data.size());

						//Debugging of histogramms
						//histogramManager->setNumberOfBins(100);
						//histogramManager->computeHistogramm(dl.name, dl.activeInd, attributeMinMax, parent->buffer.buffer, parent->data.size());
						//for (auto& i : histogramManager->getHistogram(dl.name).bins)
						//{
						//	std::for_each(i.begin(), i.end(), [](uint32_t a) {std::cout << a << ","; });
						//	std::cout << std::endl;
						//}
					}

					ImGui::Separator();
					for (int i = 0; i < pcAttributes.size(); i++) {
						if (!pcAttributeEnabled[i])
							continue;
						if (ImGui::MenuItem(("Render " + pcAttributes[i].name).c_str())) {
							ImGui::CloseCurrentPopup();
							uploadDrawListTo3dView(dl, pcAttributes[i].name, "a", "b", "c");
							active3dAttribute = pcAttributes[i].name;
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
				if (ImGui::ArrowButton((std::string("##u") + dl.name).c_str(), ImGuiDir_Up)) {
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
				if (ImGui::ColorEdit4((std::string("Color##") + dl.name).c_str(), (float*)&dl.color, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | misc_flags)) {
					pcPlotRender = true;
				}
				ImGui::NextColumn();

				if (ImGui::Checkbox((std::string("##dh") + dl.name).c_str(), &dl.showHistogramm) && drawHistogramm) {
					pcPlotRender = true;
				}
				ImGui::NextColumn();

				const char* entrys[] = { "No Median","Synthetic","Arithmetic","Geometric" };
				int prevActive = dl.activeMedian;
				if (ImGui::Combo((std::string("##c") + dl.name).c_str(), &dl.activeMedian, entrys, sizeof(entrys) / sizeof(*entrys) - 1)) {
					if (prevActive == 0) {
						calculateDrawListMedians(dl);
					}
					pcPlotRender = true;
				}
				ImGui::NextColumn();

				if (ImGui::ColorEdit4((std::string("##CMed") + dl.name).c_str(), (float*)&dl.medianColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | misc_flags)) {
					pcPlotRender = true;
				}
				ImGui::NextColumn();

				count++;
			}
			ImGui::Columns(1);
			ImGui::Separator();
			//open compare popup
			if (compareDrawLists)
				ImGui::OpenPopup("Compare Drawlists");
			if (ImGui::BeginPopupModal("Compare Drawlists", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
				static char name[250];
				ImGui::InputText("Name for new Drawlist", name, 250);
				ImGui::BeginChild("A", ImVec2(400, 200));
				ImGui::Text("Statistics for %s", drawListComparator.a.c_str());
				ImGui::Text("Contains %8d points.", drawListComparator.aInd.size());
				ImGui::Text("The intersection has %d points.", drawListComparator.aAndb.size());
				if (ImGui::Button("Create intersection##a")) {
					TemplateList tl = {};
					DataSet& parent = g_PcPlotDataSets.front();
					for (DataSet& ds : g_PcPlotDataSets) {
						if (ds.name == drawListComparator.parentDataset) {
							parent = ds;
							break;
						}
					}
					tl.buffer = parent.buffer.buffer;
					tl.name = name;
					tl.indices = drawListComparator.aAndb;
					createPcPlotDrawList(tl, parent, name);
					pcPlotRender = updateActiveIndices(g_PcPlotDrawLists.back());
				}
				ImGui::Text("Union has %8d points", drawListComparator.aOrB.size());
				if (ImGui::Button("Create union##a")) {
					TemplateList tl = {};
					DataSet& parent = g_PcPlotDataSets.front();
					for (DataSet& ds : g_PcPlotDataSets) {
						if (ds.name == drawListComparator.parentDataset) {
							parent = ds;
							break;
						}
					}
					tl.buffer = parent.buffer.buffer;
					tl.name = name;
					tl.indices = drawListComparator.aOrB;
					createPcPlotDrawList(tl, parent, name);
					pcPlotRender = updateActiveIndices(g_PcPlotDrawLists.back());
				}
				ImGui::Text("Difference has %8d points", drawListComparator.aMinusB.size());
				if (ImGui::Button("Create difference##a")) {
					TemplateList tl = {};
					DataSet& parent = g_PcPlotDataSets.front();
					for (DataSet& ds : g_PcPlotDataSets) {
						if (ds.name == drawListComparator.parentDataset) {
							parent = ds;
							break;
						}
					}
					tl.buffer = parent.buffer.buffer;
					tl.name = name;
					tl.indices = drawListComparator.aMinusB;
					createPcPlotDrawList(tl, parent, name);
					pcPlotRender = updateActiveIndices(g_PcPlotDrawLists.back());
				}
				ImGui::EndChild();
				ImGui::SameLine();

				ImGui::BeginChild("B", ImVec2(400, 200));
				ImGui::Text("Statistics for %s", drawListComparator.b.c_str());
				ImGui::Text("Contains %8d points.", drawListComparator.bInd.size());
				ImGui::Text("The intersection has %d points.", drawListComparator.aAndb.size());
				if (ImGui::Button("Create intersection##b")) {
					TemplateList tl = {};
					DataSet& parent = g_PcPlotDataSets.front();
					for (DataSet& ds : g_PcPlotDataSets) {
						if (ds.name == drawListComparator.parentDataset) {
							parent = ds;
							break;
						}
					}
					tl.buffer = parent.buffer.buffer;
					tl.name = name;
					tl.indices = drawListComparator.aAndb;
					createPcPlotDrawList(tl, parent, name);
					pcPlotRender = updateActiveIndices(g_PcPlotDrawLists.back());
				}
				ImGui::Text("Union has %8d points", drawListComparator.aOrB.size());
				if (ImGui::Button("Create union##b")) {
					TemplateList tl = {};
					DataSet& parent = g_PcPlotDataSets.front();
					for (DataSet& ds : g_PcPlotDataSets) {
						if (ds.name == drawListComparator.parentDataset) {
							parent = ds;
							break;
						}
					}
					tl.buffer = parent.buffer.buffer;
					tl.name = name;
					tl.indices = drawListComparator.aOrB;
					createPcPlotDrawList(tl, parent, name);
					pcPlotRender = updateActiveIndices(g_PcPlotDrawLists.back());
				}
				ImGui::Text("Difference has %8d points", drawListComparator.bMinusA.size());
				if (ImGui::Button("Create difference##b")) {
					TemplateList tl = {};
					DataSet& parent = g_PcPlotDataSets.front();
					for (DataSet& ds : g_PcPlotDataSets) {
						if (ds.name == drawListComparator.parentDataset) {
							parent = ds;
							break;
						}
					}
					tl.buffer = parent.buffer.buffer;
					tl.name = name;
					tl.indices = drawListComparator.bMinusA;
					createPcPlotDrawList(tl, parent, name);
					pcPlotRender = updateActiveIndices(g_PcPlotDrawLists.back());
				}
				ImGui::EndChild();

				if ((ImGui::Button("Close")) || ImGui::IsKeyPressed(KEYESC)) {
					ImGui::CloseCurrentPopup();
				}

				ImGui::EndPopup();
			}
			//open export popup
			if (exportIdxf) {
				ImGui::OpenPopup("Export Drawlist");
			}
			if (ImGui::BeginPopupModal("Export Drawlist", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
				static char filepath[250];
				ImGui::InputText("filepath", filepath, 250);
				if ((ImGui::Button("Cancel")) || ImGui::IsKeyPressed(KEYESC)) {
					ImGui::CloseCurrentPopup();
				}
				ImGui::SameLine();
				if (ImGui::Button("Save")) {
					exportBrushAsIdxf(*exportDl, filepath);
					ImGui::CloseCurrentPopup();
				}

				ImGui::EndPopup();
			}
			if (exportCsv) {
				ImGui::OpenPopup("Export Drawlist to .csv");
			}
			if (ImGui::BeginPopupModal("Export Drawlist to .csv", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
				static char filepath[250];
				ImGui::InputText("filepath (has to include the filename with .csv ending)", filepath, 250);
				if ((ImGui::Button("Cancel")) || ImGui::IsKeyPressed(KEYESC)) {
					ImGui::CloseCurrentPopup();
				}
				ImGui::SameLine();
				if (ImGui::Button("Save")) {
					exportBrushAsCsv(*exportDl, filepath);
					ImGui::CloseCurrentPopup();
				}

				ImGui::EndPopup();
			}

			ImGui::EndChild();
			
			//main window now closed -----------------------------------------------------------------------
			if (destroy) {
				removePcPlotDrawList(*changeList);
				updateBrushTemplates = true;
				if (!pcAttributes.size()) {
					globalBrushes.clear();
				}
			}
			if (up) {
				auto it = g_PcPlotDrawLists.begin();
				while (it != g_PcPlotDrawLists.end() && it->name != changeList->name)
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
		}
		ImGui::End();

		//bubble window ----------------------------------------------------------------------------------
		int bubbleWindowSize = 0;
		if (enableBubbleWindow) {
			ImGui::Begin("Bubble window", &enableBubbleWindow, ImGuiWindowFlags_MenuBar);

			bubbleWindowSize = ImGui::GetWindowSize().y;

			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("Coupling")) {
					ImGui::MenuItem("Couple to Parallel Coordinates", "", &coupleBubbleWindow);
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Navigation")) {
					ImGui::SliderFloat("fly speed", &bubblePlotter->flySpeed, 0.01, 10);
					ImGui::SliderFloat("fast fly multiplier", &bubblePlotter->fastFlyMultiplier, 1, 10);
					ImGui::SliderFloat("rotation speed", &bubblePlotter->rotationSpeed, 0.01, 5);
					ImGui::SliderFloat("fov speed", &bubblePlotter->fovSpeed, 1, 100);
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Visualization")) {
					if (ImGui::DragFloat3("Min position Values", &bubblePlotter->boundingRectMin.x))bubblePlotter->render();
					if (ImGui::DragFloat3("Max position Values", &bubblePlotter->boundingRectMax.x))bubblePlotter->render();
					if (ImGui::SliderFloat("max point size", &bubblePlotter->maxPointSize, .1f, 200))bubblePlotter->render();
					if (ImGui::ColorEdit4("Clip Color", bubblePlotter->grey, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar)) bubblePlotter->render();
					if (ImGui::MenuItem("Enable clipping", "", &bubblePlotter->clipping))bubblePlotter->render();
					if (ImGui::MenuItem("Enable normalization", "", &bubblePlotter->normalization))bubblePlotter->render();
					if (ImGui::DragFloat("Spacing", &bubblePlotter->layerSpacing, bubblePlotter->layerSpacing / 100.0f, 0.0001, 100)) {
						bubblePlotter->render();
					}
					if (ImGui::DragInt3("Position indices", (int*)&bubblePlotter->posIndices.x, .05f, 0, pcAttributes.size())) {
						bubblePlotter->render();
					}
					ImGui::EndMenu();
				}
				ImGui::EndMenuBar();
			}

			ImGui::Image((ImTextureID)bubblePlotter->getImageDescSet(), ImVec2(800, 800), ImVec2(0, 0), ImVec2(1, 1), ImColor(255, 255, 255, 255), ImColor(255, 255, 255, 128));
			if (ImGui::IsItemHovered() && (ImGui::IsMouseDragging(ImGuiMouseButton_Left) || io.MouseWheel || io.KeysDown[KEYW] || io.KeysDown[KEYA] || io.KeysDown[KEYS] || io.KeysDown[KEYD])) {
				CamNav::NavigationInput nav = {};
				nav.mouseDeltaX = ImGui::GetMouseDragDelta().x;
				nav.mouseDeltaY = ImGui::GetMouseDragDelta().y;
				nav.mouseScrollDelta = io.MouseWheel;
				nav.w = io.KeysDown[KEYW];
				nav.a = io.KeysDown[KEYA];
				nav.s = io.KeysDown[KEYS];
				nav.d = io.KeysDown[KEYD];
				nav.q = io.KeysDown[KEYQ];
				nav.e = io.KeysDown[KEYE];
				nav.shift = io.KeyShift;
				bubblePlotter->updateCameraPos(nav, io.DeltaTime);
				bubblePlotter->render();
				err = vkDeviceWaitIdle(g_Device);
				check_vk_result(err);
				ImGui::ResetMouseDragDelta();
			}

			ImGui::SameLine();
			ImGui::BeginChild("Attribute Settings", ImVec2(-1, 800));
			ImGui::Columns(4);
			ImGui::Separator();
			ImGui::Text("Variable"); ImGui::NextColumn();
			ImGui::Text("Scale"); ImGui::NextColumn();
			ImGui::Text("Min/Max"); ImGui::NextColumn();
			ImGui::Text("Color"); ImGui::NextColumn();
			ImGui::Separator();
			for (int i = 0; i < bubblePlotter->attributeNames.size(); ++i) {
				if (i == bubblePlotter->posIndices.x || i == bubblePlotter->posIndices.y || i == bubblePlotter->posIndices.z)
					continue;
				if (ImGui::Checkbox((bubblePlotter->attributeNames[i] + "##cb").c_str(), &bubblePlotter->attributeActivations[i])) {		//redistribute the remaining variales over the free layer space
					float count = 0;
					for (int j = 0; j < bubblePlotter->attributeNames.size(); ++j) {
						if (bubblePlotter->attributeActivations[j] && j != bubblePlotter->posIndices.x && j != bubblePlotter->posIndices.y && j != bubblePlotter->posIndices.z)
							count += 1;
					}
					count = 1 / (count - 1); //converting count to the percentage step
					float curP = 0;
					for (int j = 0; j < bubblePlotter->attributeNames.size(); ++j) {
						if (!bubblePlotter->attributeActivations[j] || j == bubblePlotter->posIndices.x || j == bubblePlotter->posIndices.y || j == bubblePlotter->posIndices.z) {
							continue;
						}
						bubblePlotter->attributeTopOffsets[j] = curP;
						curP += count;
					}
					bubblePlotter->render();
				}
				ImGui::NextColumn();
				static char* scales[] = { "Normal","Squareroot","Logarithmic" };
				static int selectedScale = 0;
				if (ImGui::BeginCombo(("Scale##" + std::to_string(i)).c_str(), scales[bubblePlotter->attributeScales[i]])) {
					for (int j = 0; j < 3; ++j) {
						if (ImGui::Selectable(scales[j])) {
							selectedScale = j;
							bubblePlotter->attributeScales[i] = (BubblePlotter::Scale)selectedScale;
							bubblePlotter->render();
						}
					}
					ImGui::EndCombo();
				}
				ImGui::NextColumn();
				if (ImGui::DragFloat2(("##minmax" + bubblePlotter->attributeNames[i]).c_str(), &bubblePlotter->attributeMinMaxValues[i].first)) {
					bubblePlotter->render();
				}
				ImGui::NextColumn();
				if (ImGui::ColorEdit4((std::string("##col") + bubblePlotter->attributeNames[i]).c_str(), (float*)&bubblePlotter->attributeColors[i].x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar)) {
					bubblePlotter->render();
				}
				ImGui::NextColumn();
			}
			ImGui::Columns(1);
			ImGui::Separator();
			ImGui::EndChild();
			
			//set bubble plot data via drag and drop
			ImGui::SetCursorPos(ImGui::GetWindowContentRegionMin());
			ImGui::Dummy(ImGui::GetWindowContentRegionMax() - ImGui::GetWindowContentRegionMin());
			if (ImGui::BeginDragDropTarget()) {
				if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Drawlist")) {
					DrawList* dl = *((DrawList**)payload->Data);
					DataSet* parent;
					for (auto it = g_PcPlotDataSets.begin(); it != g_PcPlotDataSets.end(); ++it) {
						if (it->name == dl->parentDataSet) {
							parent = &(*it);
						}
					}
					std::vector<uint32_t> ids;
					std::vector<std::string> attributeNames;
					std::vector<std::pair<float, float>> attributeMinMax;
					for (int i = 0; i < pcAttributes.size(); ++i) {
						attributeNames.push_back(pcAttributes[i].name);
						attributeMinMax.push_back({ pcAttributes[i].min,pcAttributes[i].max });
					}
					glm::uvec3 posIndices(0, 2, 1);
					bubblePlotter->setBubbleData(posIndices, dl->indices, attributeNames, attributeMinMax, parent->data, dl->buffer, dl->activeIndicesBufferView, attributeNames.size(), parent->data.size());
				}
				ImGui::EndDragDropTarget();
			}

			ImGui::End();
		}
			
		//end of bubble window ---------------------------------------------------------------------------

		//begin of iso surface window --------------------------------------------------------------------
		if (enableIsoSurfaceWindow) {
			ImGui::Begin("Isosurface Renderer",&enableIsoSurfaceWindow,ImGuiWindowFlags_MenuBar);
			int dlbExport = -1;
			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("Settings")) {
					ImGui::Checkbox("Couple to brush", &coupleIsoSurfaceRenderer);
					ImGui::Checkbox("Regular grid", &isoSurfaceRegularGrid);
					ImGui::InputInt3("Regular grid dimensions", isoSurfaceRegularGridDim);
					
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Rendering")) {
					static float boxSize[3]{ 1.5f,1.f,1.5f };
					if (ImGui::DragFloat3("Box dimensions", boxSize, .001f)) {
						isoSurfaceRenderer->resizeBox(boxSize[0], boxSize[1], boxSize[2]);
						isoSurfaceRenderer->render();
					}
					if (ImGui::Checkbox("Activate shading", &isoSurfaceRenderer->shade)) {
						isoSurfaceRenderer->render();
					}
					if (ImGui::SliderFloat("Iso value", &isoSurfaceRenderer->isoValue, .01f, .99f)) {
						isoSurfaceRenderer->render();
					}
					if (ImGui::SliderFloat("Ray march step size", &isoSurfaceRenderer->stepSize, 0.0005f, .05f, "%.5f")) {
						isoSurfaceRenderer->render();
					}
					if (ImGui::SliderFloat("Step size for normal calc", &isoSurfaceRenderer->shadingStep, .1f, 10)) {
						isoSurfaceRenderer->render();
					}
					if (ImGui::SliderFloat("Wireframe width", &isoSurfaceRenderer->gridLineWidth, 0, .1f)) {
						isoSurfaceRenderer->render();
					}
					static float stdDiv = 1;
					static bool copyOnes = true;
					if (ImGui::SliderFloat("Smoothing kernel size", &stdDiv, 0, 10)) {
						isoSurfaceRenderer->setBinarySmoothing(stdDiv,copyOnes);
						isoSurfaceRenderer->render();
					}
					if (ImGui::Checkbox("Copy 1 entries after smoothing", &copyOnes)) {
						isoSurfaceRenderer->setBinarySmoothing(stdDiv, copyOnes);
						isoSurfaceRenderer->render();
					}
					if (ImGui::DragFloat3("Ligt direction", &isoSurfaceRenderer->lightDir.x)) {
						isoSurfaceRenderer->render();
					}
					if (ImGui::ColorEdit4("Image background", isoSurfaceRenderer->imageBackground.color.float32, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar)) {
						isoSurfaceRenderer->imageBackGroundUpdated();
						isoSurfaceRenderer->render();
					}
					ImGui::EndMenu();
				}


				if (ImGui::BeginMenu("Camera position")) {
					ImGui::InputFloat3("Position", isoSurfaceRenderer->cameraPositionGUI, 3);
					ImGui::InputFloat2("Rotation", isoSurfaceRenderer->cameraRotationGUI, 3);
					if (ImGui::Button("get current camera position"))
					{
						float *p = isoSurfaceRenderer->cameraRotationGUI;
						isoSurfaceRenderer->getCameraPos(isoSurfaceRenderer->cameraPositionGLMGUI, &p);
						isoSurfaceRenderer->cameraPositionGUI[0] = isoSurfaceRenderer->cameraPositionGLMGUI.x;
						isoSurfaceRenderer->cameraPositionGUI[1] = isoSurfaceRenderer->cameraPositionGLMGUI.y;
						isoSurfaceRenderer->cameraPositionGUI[2] = isoSurfaceRenderer->cameraPositionGLMGUI.z;


					}
					if (ImGui::Button("set camera position"))
					{
						float* p = isoSurfaceRenderer->cameraRotationGUI;
						isoSurfaceRenderer->cameraPositionGLMGUI.x = isoSurfaceRenderer->cameraPositionGUI[0];
						isoSurfaceRenderer->cameraPositionGLMGUI.y = isoSurfaceRenderer->cameraPositionGUI[1];
						isoSurfaceRenderer->cameraPositionGLMGUI.z = isoSurfaceRenderer->cameraPositionGUI[2];
						isoSurfaceRenderer->setCameraPos(isoSurfaceRenderer->cameraPositionGLMGUI, &p);
						isoSurfaceRenderer->render();
						err = vkQueueWaitIdle(g_Queue);
						check_vk_result(err);
						ImGui::ResetMouseDragDelta();
					}
					if (ImGui::Button("sync direct iso renderer's camera")) {
						
						if (brushIsoSurfaceRenderer)
						{
							float* p = isoSurfaceRenderer->cameraRotationGUI;
							brushIsoSurfaceRenderer->directIsoRendererCameraPositionGLM.x = isoSurfaceRenderer->cameraPositionGUI[0];
							brushIsoSurfaceRenderer->directIsoRendererCameraPositionGLM.y = isoSurfaceRenderer->cameraPositionGUI[1];
							brushIsoSurfaceRenderer->directIsoRendererCameraPositionGLM.z = isoSurfaceRenderer->cameraPositionGUI[2];
							brushIsoSurfaceRenderer->setCameraPos(brushIsoSurfaceRenderer->directIsoRendererCameraPositionGLM, &p);
							brushIsoSurfaceRenderer->render();
							err = vkQueueWaitIdle(g_Queue);
							check_vk_result(err);
							ImGui::ResetMouseDragDelta();
						}
					}

					ImGui::EndMenu();
				}


				if (ImGui::BeginMenu("Export")) {
					int ind = 0;
					for (auto& dlb : isoSurfaceRenderer->drawlistBrushes) {
						if (ImGui::MenuItem((dlb.drawlist + dlb.brush).c_str())) {
							dlbExport = ind;
						}
						++ind;
					}
					ImGui::EndMenu();
				}

				ImGui::EndMenuBar();
			}

			static int dlbEx;
			if (dlbExport != -1) { 
				ImGui::OpenPopup("Export binary volume"); 
				dlbEx = dlbExport;
			}
			if (ImGui::BeginPopupModal("Export binary volume", 0, ImGuiWindowFlags_AlwaysAutoResize)) {
				
				static char path[200]{};
				ImGui::InputText("filepath (including filename and file ending)", path, 200);

				if ((ImGui::Button("Save", ImVec2(120, 0)) || ImGui::IsKeyPressed(KEYENTER)) && std::string(path).size()) {
					isoSurfaceRenderer->exportBinaryCsv(std::string(path), dlbEx);
					ImGui::CloseCurrentPopup();
				}

				ImGui::SameLine();
				if (ImGui::Button("Cancel", ImVec2(120, 0)) || ImGui::IsKeyPressed(KEYESC)) {
					ImGui::CloseCurrentPopup();
				}

				ImGui::EndPopup();
			}
			
			ImGui::Image((ImTextureID)isoSurfaceRenderer->getImageDescriptorSet(), ImVec2{ 800,800 }, { 0,0 }, { 1,1 }, { 1,1,1,1 }, { 0,0,0,1 });
			if (ImGui::IsItemHovered() && (ImGui::IsMouseDragging(ImGuiMouseButton_Left) || io.MouseWheel || 
				ImGui::IsKeyDown(KEYA) || ImGui::IsKeyDown(KEYS) || ImGui::IsKeyDown(KEYD) || ImGui::IsKeyDown(KEYQ) || ImGui::IsKeyDown(KEYW) || ImGui::IsKeyDown(KEYE))) {
				CamNav::NavigationInput nav = {};
				nav.mouseDeltaX = ImGui::GetMouseDragDelta().x;
				nav.mouseDeltaY = ImGui::GetMouseDragDelta().y;
				nav.mouseScrollDelta = io.MouseWheel;
				nav.w = io.KeysDown[KEYW];
				nav.a = io.KeysDown[KEYA];
				nav.s = io.KeysDown[KEYS];
				nav.d = io.KeysDown[KEYD];
				nav.q = io.KeysDown[KEYQ];
				nav.e = io.KeysDown[KEYE];
				nav.shift = io.KeyShift;
				isoSurfaceRenderer->updateCameraPos(nav, io.DeltaTime);
				isoSurfaceRenderer->render();
				err = vkDeviceWaitIdle(g_Device);
				check_vk_result(err);
				ImGui::ResetMouseDragDelta();
			}

			ImGui::Text("Add iso surface");
			static char choose[]{ "choose" };
			static int selectedDrawlist = -1;
			static int selectedGlobalBrush = -1;
			ImGui::PushItemWidth(300);
			if(ImGui::BeginCombo("Drawlist", (selectedDrawlist == -1) ? choose : std::next(g_PcPlotDrawLists.begin(), selectedDrawlist)->name.c_str())) {
				if (ImGui::Selectable(choose, selectedDrawlist == -1)) selectedDrawlist = -1;
				auto dl = g_PcPlotDrawLists.begin();
				for (int i = 0; i < g_PcPlotDrawLists.size(); ++i) {
					if (ImGui::Selectable(dl->name.c_str(), selectedDrawlist == i)) selectedDrawlist = i;
					++dl;
				}
				ImGui::EndCombo();
			}
			ImGui::SameLine();
			if (selectedGlobalBrush != -1 && !globalBrushes.size()) selectedGlobalBrush = -1;
			if (ImGui::BeginCombo("Brush(if now brush is selected, active indices are used for iso Surface)", (selectedGlobalBrush == -1) ? choose : globalBrushes[selectedGlobalBrush].name.c_str())) {
				if (ImGui::Selectable(choose, selectedGlobalBrush == -1)) selectedGlobalBrush = -1;
				for (int i = 0; i < globalBrushes.size(); ++i) {
					if (ImGui::Selectable(globalBrushes[i].name.c_str(), selectedGlobalBrush == i)) selectedGlobalBrush = i;
				}
				ImGui::EndCombo();
			}
			ImGui::PopItemWidth();

			ImGui::PushItemWidth(300);
			
			ImGui::DragInt3("Position indices (Order: lat, alt, lon)", (int*)&posIndices.x, 0.00000001f, 0, pcAttributes.size());
			ImGui::PopItemWidth();

			static bool showError = false;
			static bool positionError = false;
			if (ImGui::Button("Add new iso surface")) {
				if (selectedDrawlist == -1 || posIndices.x==posIndices.y || posIndices.y==posIndices.z || posIndices.x==posIndices.z) {
					showError = true;
				}
				else {
					DrawList* dl = &*std::next(g_PcPlotDrawLists.begin(), selectedDrawlist);
					std::vector<float*>* data = nullptr;
					for (DataSet& ds : g_PcPlotDataSets) {
						if (dl->parentDataSet == ds.name) {
							data = &ds.data;
							break;
						}
					}

					std::vector<unsigned int> attr;
					std::vector<std::pair<float, float>> minMax;
					for (int i = 0; i < pcAttributes.size(); ++i) {
						attr.push_back(i);
						minMax.push_back({ pcAttributes[i].min, pcAttributes[i].max });
					}
					std::vector<std::vector<std::pair<float, float>>> miMa(pcAttributes.size());
					std::vector<uint32_t> brushIndices;
					if (selectedGlobalBrush != -1) {
						for (auto axis : globalBrushes[selectedGlobalBrush].brushes) {
							if (axis.second.size()) brushIndices.push_back(axis.first);
							for (auto& brush : axis.second) {
								miMa[axis.first].push_back(brush.second);
							}
						}
					}

					int index = -1;
					for (int i = 0; i < isoSurfaceRenderer->drawlistBrushes.size(); ++i) {
						if (dl->name == isoSurfaceRenderer->drawlistBrushes[i].drawlist){
							if (isoSurfaceRenderer->drawlistBrushes[i].brush.size() && globalBrushes[selectedGlobalBrush].name == isoSurfaceRenderer->drawlistBrushes[i].brush) {
								index = i;
								break;
							}
							else if (!isoSurfaceRenderer->drawlistBrushes[i].brush.size()) {
								index = i;
								break;
							}
						}
					}
					if (index == -1) {
						uint32_t wi = (isoSurfaceRegularGrid) ? isoSurfaceRegularGridDim[0] : SpacialData::rlatSize;
						uint32_t he = (isoSurfaceRegularGrid) ? isoSurfaceRegularGridDim[1] : SpacialData::altitudeSize + 22;
						uint32_t de = (isoSurfaceRegularGrid) ? isoSurfaceRegularGridDim[2] : SpacialData::rlonSize;
						isoSurfaceRenderer->drawlistBrushes.push_back({ dl->name,(selectedGlobalBrush == -1) ? "" : globalBrushes[selectedGlobalBrush].name,{ 1,0,0,1 }, {wi, he, de} });
					}
					if (selectedGlobalBrush == -1) {
						uint32_t w = (isoSurfaceRegularGrid) ? isoSurfaceRegularGridDim[0] : SpacialData::rlatSize;
						uint32_t h = (isoSurfaceRegularGrid) ? isoSurfaceRegularGridDim[1] : SpacialData::altitudeSize + 22;
						uint32_t d = (isoSurfaceRegularGrid) ? isoSurfaceRegularGridDim[2] : SpacialData::rlonSize;
						std::vector<std::pair<float, float>> posBounds(3);
						for (int i = 0; i < 3; ++i) {
							posBounds[i].first = pcAttributes[posIndices[i]].min;
							posBounds[i].second = pcAttributes[posIndices[i]].max;
						}
						if (!isoSurfaceRegularGrid) {
							posBounds[0].first = SpacialData::rlat[0];
							posBounds[0].second = SpacialData::rlat[SpacialData::rlatSize - 1];
							posBounds[1].first = SpacialData::altitude[0];
							posBounds[1].second = SpacialData::altitude[SpacialData::altitudeSize - 1];
							posBounds[2].first = SpacialData::rlon[0];
							posBounds[2].second = SpacialData::rlon[SpacialData::rlonSize - 1];
						}
						isoSurfaceRenderer->update3dBinaryVolume(w, h, d, &posIndices.x, posBounds, pcAttributes.size(), data->size(), dl->buffer, dl->activeIndicesBufferView, dl->indices.size(), dl->indicesBuffer, isoSurfaceRegularGrid, index);
					}
					else {
						if (isoSurfaceRegularGrid) {
							isoSurfaceRenderer->update3dBinaryVolume(isoSurfaceRegularGridDim[0], isoSurfaceRegularGridDim[1], isoSurfaceRegularGridDim[2], pcAttributes.size(), brushIndices, minMax, posIndices, dl->buffer, data->size() * pcAttributes.size() * sizeof(float), dl->indicesBuffer, dl->indices.size(), miMa, index);
						}
						else {
							if (!isoSurfaceRenderer->update3dBinaryVolume(SpacialData::rlatSize, SpacialData::altitudeSize, SpacialData::rlonSize, pcAttributes.size(), attr, minMax, posIndices, *data, dl->indices, miMa, index) && index == -1) {
								isoSurfaceRenderer->drawlistBrushes.push_back({ dl->name,globalBrushes[selectedGlobalBrush].name,{ 1,0,0,1 }, {uint32_t(SpacialData::rlatSize), uint32_t(SpacialData::altitudeSize), uint32_t(SpacialData::rlonSize)} });
								positionError = true;
							}
							else {
								positionError = false;
							}
						}
					}
				}
			}
			if (showError) {
				ImGui::TextColored({ 1,0,0,1 }, "You have to select a drawlist and a global brush!");
				ImGui::TextColored({ 1,0,0,1 }, "Or you made a mistake when setting the positon indices!");
			}
			if (positionError) {
				ImGui::TextColored({ 1,0,0,1 }, "The binary volume coulndt be created! Perhaps there is a error with the position indices.");
			}
			ImGui::Separator();
			ImGui::Text("Active iso sufaces:");
			ImGui::Columns(4);
			int index = 0;
			int del = -1;
			for (IsoSurfRenderer::DrawlistBrush& db : isoSurfaceRenderer->drawlistBrushes) {
				ImGui::Text(db.drawlist.c_str());
				ImGui::NextColumn();
				ImGui::Text(db.brush.c_str());
				ImGui::NextColumn();
				if (ImGui::ColorEdit4((std::string("##col") + db.drawlist + db.brush).c_str(), (float*)&db.brushSurfaceColor.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar)) {
					isoSurfaceRenderer->render();
				}
				ImGui::NextColumn();
				if (ImGui::Button(("X##" + std::to_string(index)).c_str())) {
					del = index;
				}
				ImGui::NextColumn();
				++index;
			}
			if (del != -1) {
				isoSurfaceRenderer->deleteBinaryVolume(del);
				isoSurfaceRenderer->render();
			}
			ImGui::Columns(1);

			//ImGui::SetCursorPos(ImGui::GetWindowContentRegionMin() + ImVec2(ImGui::GetScrollX(), 2 * ImGui::GetScrollY()));
			//ImGui::Dummy(ImGui::GetWindowContentRegionMax() - ImGui::GetWindowContentRegionMin());
			////set drawlist data via drag and drop
			//if (ImGui::BeginDragDropTarget()) {
			//	if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Drawlist")) {
			//		DrawList* dl = *((DrawList**)payload->Data);
			//		std::vector<float*>* data = nullptr;
			//		for (DataSet& ds : g_PcPlotDataSets) {
			//			if (dl->parentDataSet == ds.name) {
			//				data = &ds.data;
			//			}
			//		}
			//		
			//		std::vector<unsigned int> attr;
			//		std::vector<std::pair<float, float>> minMax;
			//		for (int i = 0; i < pcAttributes.size(); ++i) {
			//			attr.push_back(i);
			//			minMax.push_back({ pcAttributes[i].min, pcAttributes[i].max });
			//		}
			//		minMax[0] = { SpacialData::rlat[0],SpacialData::altitude[SpacialData::rlatSize - 1] };
			//		minMax[1] = { SpacialData::rlon[0],SpacialData::altitude[SpacialData::rlonSize - 1] };
			//		minMax[2] = { SpacialData::altitude[0],SpacialData::altitude[SpacialData::altitudeSize - 1] };
			//		//isoSurfaceRenderer->update3dBinaryVolume(SpacialData::rlatSize, SpacialData::altitudeSize, SpacialData::rlonSize, pcAttributes.size(), attr, minMax, glm::uvec3{ 0,2,1 }, *data, dl->indices);
			//	}
			//	ImGui::EndDragDropTarget();
			//}
			//
			//if (ImGui::BeginDragDropTarget()) {
			//	if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("GlobalBrush")) {
			//		GlobalBrush* brush = *((GlobalBrush**)payload->Data);
			//
			//		std::vector<std::vector<std::pair<float, float>>> minMax(pcAttributes.size());
			//		for (auto& axis : brush->brushes) {
			//			for (auto& m : axis.second) {
			//				minMax[axis.first].push_back(m.second);
			//			}
			//		}
			//		isoSurfaceRenderer->addBrush(brush->name, minMax);
			//	}
			//
			//	ImGui::EndDragDropTarget();
			//}
			ImGui::End();
		}
		//end of iso surface window -----------------------------------------------------------------------

		//brush iso surface window -----------------------------------------------------------------------
		if (enableBrushIsoSurfaceWindow) {
			ImGui::Begin("Brush Isosurface Renderer", &enableBrushIsoSurfaceWindow, ImGuiWindowFlags_MenuBar);
			int dlbExport = -1;
			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("Settings")) {
					ImGui::Checkbox("Couple to brush", &coupleBrushIsoSurfaceRenderer);
					ImGui::Checkbox("Regular grid", &isoSurfaceRegularGrid);
					ImGui::InputInt3("Regular grid dimensions", isoSurfaceRegularGridDim);

					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Rendering")) {
					static float boxSize[3]{ 1.5f,1.f,1.5f };
					if (ImGui::DragFloat3("Box dimensions", boxSize, .001f)) {
						brushIsoSurfaceRenderer->resizeBox(boxSize[0], boxSize[1], boxSize[2]);
						brushIsoSurfaceRenderer->render();
					}
					if (ImGui::Checkbox("Activate shading", &brushIsoSurfaceRenderer->shade)) {
						brushIsoSurfaceRenderer->render();
					}
					if (ImGui::SliderFloat("Ray march step size", &brushIsoSurfaceRenderer->stepSize, 0.0005f, .05f, "%.5f")) {
						brushIsoSurfaceRenderer->render();
					}
					if (ImGui::SliderFloat("Step size for normal calc", &brushIsoSurfaceRenderer->shadingStep, .1f, 10)) {
						brushIsoSurfaceRenderer->render();
					}
					if (ImGui::DragFloat3("Ligt direction", &brushIsoSurfaceRenderer->lightDir.x)) {
						brushIsoSurfaceRenderer->render();
					}
					if (ImGui::ColorEdit4("Image background", brushIsoSurfaceRenderer->imageBackground.color.float32, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar)) {
						brushIsoSurfaceRenderer->imageBackGroundUpdated();
						brushIsoSurfaceRenderer->render();
					}
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Camera position")) {
					ImGui::InputFloat3("Position", brushIsoSurfaceRenderer->directIsoRendererCameraPosition, 3);
					ImGui::InputFloat2("Rotation", brushIsoSurfaceRenderer->cameraRotationGUI, 3);
					if (ImGui::Button("get current camera position"))
					{
						float* p = brushIsoSurfaceRenderer->cameraRotationGUI;
						brushIsoSurfaceRenderer->getCameraPos(brushIsoSurfaceRenderer->directIsoRendererCameraPositionGLM, &p);
						brushIsoSurfaceRenderer->directIsoRendererCameraPosition[0] = brushIsoSurfaceRenderer->directIsoRendererCameraPositionGLM.x;
						brushIsoSurfaceRenderer->directIsoRendererCameraPosition[1] = brushIsoSurfaceRenderer->directIsoRendererCameraPositionGLM.y;
						brushIsoSurfaceRenderer->directIsoRendererCameraPosition[2] = brushIsoSurfaceRenderer->directIsoRendererCameraPositionGLM.z;
							
					}
					if (ImGui::Button("set camera position"))
					{
						float* p = brushIsoSurfaceRenderer->cameraRotationGUI;
						brushIsoSurfaceRenderer->directIsoRendererCameraPositionGLM.x = brushIsoSurfaceRenderer->directIsoRendererCameraPosition[0];
						brushIsoSurfaceRenderer->directIsoRendererCameraPositionGLM.y = brushIsoSurfaceRenderer->directIsoRendererCameraPosition[1];
						brushIsoSurfaceRenderer->directIsoRendererCameraPositionGLM.z = brushIsoSurfaceRenderer->directIsoRendererCameraPosition[2];
						brushIsoSurfaceRenderer->setCameraPos(brushIsoSurfaceRenderer->directIsoRendererCameraPositionGLM, &p);
						brushIsoSurfaceRenderer->render();
						err = vkDeviceWaitIdle(g_Device);
						check_vk_result(err);
						ImGui::ResetMouseDragDelta();
					}

					if (ImGui::Button("sync iso renderer's camera")) {

						if (isoSurfaceRenderer)
						{

							float* p = brushIsoSurfaceRenderer->cameraRotationGUI;
							isoSurfaceRenderer->cameraPositionGLMGUI.x = brushIsoSurfaceRenderer->directIsoRendererCameraPosition[0];
							isoSurfaceRenderer->cameraPositionGLMGUI.y = brushIsoSurfaceRenderer->directIsoRendererCameraPosition[1];
							isoSurfaceRenderer->cameraPositionGLMGUI.z = brushIsoSurfaceRenderer->directIsoRendererCameraPosition[2];
							isoSurfaceRenderer->setCameraPos(isoSurfaceRenderer->cameraPositionGLMGUI, &p);
							isoSurfaceRenderer->render();
							err = vkDeviceWaitIdle(g_Device);
							check_vk_result(err);
							ImGui::ResetMouseDragDelta();
						}
					}

					ImGui::EndMenu();
				}

				ImGui::EndMenuBar();
			}

			ImGui::Image((ImTextureID)brushIsoSurfaceRenderer->getImageDescriptorSet(), ImVec2{ 800,800 }, { 0,0 }, { 1,1 }, { 1,1,1,1 }, { 0,0,0,1 });
			if (ImGui::IsItemHovered() && (ImGui::IsMouseDragging(ImGuiMouseButton_Left) || io.MouseWheel ||
				ImGui::IsKeyDown(KEYA) || ImGui::IsKeyDown(KEYS) || ImGui::IsKeyDown(KEYD) || ImGui::IsKeyDown(KEYQ) || ImGui::IsKeyDown(KEYW) || ImGui::IsKeyDown(KEYE))) {
				CamNav::NavigationInput nav = {};
				nav.mouseDeltaX = ImGui::GetMouseDragDelta().x;
				nav.mouseDeltaY = ImGui::GetMouseDragDelta().y;
				nav.mouseScrollDelta = io.MouseWheel;
				nav.w = io.KeysDown[KEYW];
				nav.a = io.KeysDown[KEYA];
				nav.s = io.KeysDown[KEYS];
				nav.d = io.KeysDown[KEYD];
				nav.q = io.KeysDown[KEYQ];
				nav.e = io.KeysDown[KEYE];
				nav.shift = io.KeyShift;
				brushIsoSurfaceRenderer->updateCameraPos(nav, io.DeltaTime);
				brushIsoSurfaceRenderer->render();
				err = vkDeviceWaitIdle(g_Device);
				check_vk_result(err);
				ImGui::ResetMouseDragDelta();
			}

			ImGui::Text("To set the data for iso surface rendering, drag and drop a drawlist onto this window.\nTo Add a brush iso surface, darg and drop a global brush onto this window");
			static uint32_t posIndices[3]{ 1,0,2 };
			ImGui::DragInt3("Position indices(order: lat, alt, lon)", (int*)posIndices, 1, 0, pcAttributes.size());

			ImGui::Separator();
			ImGui::Text("Selected drawlist: ");
			ImGui::SameLine();
			ImGui::Button(brushIsoSurfaceRenderer->activeDrawlist.c_str());
			ImGui::Text("Brushes:");
			if (brushIsoSurfaceRenderer->brushColors.size() > 1) {
				for (auto& col : brushIsoSurfaceRenderer->brushColors) {
					if (ImGui::ColorEdit4(col.first.c_str(), &col.second[0], ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar)) {
						brushIsoSurfaceRenderer->render();
					}
				}
			}
			else if(brushIsoSurfaceRenderer->brushColors.size()){
				for (int i = 0; i < pcAttributes.size(); ++i) {
					if (ImGui::ColorEdit4(pcAttributes[i].name.c_str(), &brushIsoSurfaceRenderer->firstBrushColors[i][0], ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar)) {
						brushIsoSurfaceRenderer->render();
					}
				}
			}

			static float pad = 10;
			ImGui::SetCursorScreenPos(ImGui::GetWindowPos() + ImVec2{ pad,pad });
			ImGui::Dummy(ImGui::GetWindowSize() - ImVec2{ 2 * pad,2 * pad });
			if (ImGui::BeginDragDropTarget()) {
				if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("GlobalBrush")) {
					GlobalBrush* brush = *((GlobalBrush**)payload->Data);

					std::vector<std::vector<std::pair<float, float>>> minMax(pcAttributes.size());
					for (auto& axis : brush->brushes) {
						for (auto& m : axis.second) {
							minMax[axis.first].push_back(m.second);
						}
					}
					brushIsoSurfaceRenderer->updateBrush(brush->name, minMax);
				}

				if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Drawlist")) {
					DrawList* dl = *((DrawList**)payload->Data);
					DataSet* ds = nullptr;
					for (DataSet& d : g_PcPlotDataSets) {
						if (d.name == dl->parentDataSet) {
							ds = &d;
							break;
						}
					}
					uint32_t w = (isoSurfaceRegularGrid) ? isoSurfaceRegularGridDim[0] : SpacialData::rlatSize;
					uint32_t h = (isoSurfaceRegularGrid) ? isoSurfaceRegularGridDim[1] : SpacialData::altitudeSize + 22;
					uint32_t d = (isoSurfaceRegularGrid) ? isoSurfaceRegularGridDim[2] : SpacialData::rlonSize;
					std::vector<uint32_t> densityInds(pcAttributes.size());
					for (int i = 0; i < pcAttributes.size(); ++i) densityInds[i] = i;
					std::vector<std::pair<float, float>> bounds;// { {pcAttributes[posIndices[0]].min, pcAttributes[posIndices[0]].max}, { pcAttributes[posIndices[1]].min,pcAttributes[posIndices[1]].max }, { pcAttributes[posIndices[2]].min,pcAttributes[posIndices[2]].max } };
					for (int i = 0; i < pcAttributes.size(); ++i) {
						bounds.emplace_back(pcAttributes[i].min, pcAttributes[i].max);
					}
					if (!isoSurfaceRegularGrid) {
						bounds[posIndices[0]].first = SpacialData::rlat[0];
						bounds[posIndices[0]].second = SpacialData::rlat[SpacialData::rlatSize - 1];
						bounds[posIndices[1]].first = SpacialData::altitude[0];
						bounds[posIndices[1]].second = SpacialData::altitude[SpacialData::altitudeSize - 1];
						bounds[posIndices[2]].first = SpacialData::rlon[0];
						bounds[posIndices[2]].second = SpacialData::rlon[SpacialData::rlonSize - 1];
					}
					brushIsoSurfaceRenderer->update3dBinaryVolume(w, h, d, pcAttributes.size(), densityInds, posIndices, bounds, dl->buffer, ds->data.size(), dl->indicesBuffer, dl->indices.size(),isoSurfaceRegularGrid);
					brushIsoSurfaceRenderer->activeDrawlist = dl->name;
				}
				ImGui::EndDragDropTarget();
			}
			ImGui::End();
		}
		//end of brush iso surface window -----------------------------------------------------------------------
		
		//begin of violin plots attribute major ----------------------------------------------------------
		std::vector<std::pair<float, float>> globalMinMax(pcAttributes.size(), { std::numeric_limits<float>().max(),std::numeric_limits<float>().min() });
		std::vector<std::pair<float, float>> localMinMax(pcAttributes.size(), { std::numeric_limits<float>().max(),std::numeric_limits<float>().min() });
		if (violinYScale == ViolinYScaleGlobalBrush || violinYScale == ViolinYScaleBrushes) {
			for (auto& brush : globalBrushes) {
				for (auto& br : brush.brushes) {
					for (auto& minMax : br.second) {
						if (minMax.second.first < globalMinMax[br.first].first) globalMinMax[br.first].first = minMax.second.first;
						if (minMax.second.second > globalMinMax[br.first].second) globalMinMax[br.first].second = minMax.second.second;
					}
				}
			}
			for (int in = 0; in < globalMinMax.size(); ++in) {
				if (globalMinMax[in].first == std::numeric_limits<float>().max()) {
					globalMinMax[in].first = pcAttributes[in].min;
					globalMinMax[in].second = pcAttributes[in].max;
				}
			}
		}
		if (enableAttributeViolinPlots) {
			ImGui::Begin("Violin attribute window", &enableAttributeViolinPlots, ImGuiWindowFlags_MenuBar);
			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("Settings")) {
					ImGui::Checkbox("Couple to Brushing", &coupleViolinPlots);
					ImGui::SliderInt("Violin plots height", &violinPlotHeight, 1, 4000);
					ImGui::SliderInt("Violin plots x spacing", &violinPlotXSpacing, 0, 40);
					ImGui::SliderFloat("Violin plots line thickness", &violinPlotThickness, 0, 10);
					ImGui::ColorEdit4("Violin plots background", &violinBackgroundColor.x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
					if (ImGui::Checkbox("Ignore zero values",&histogramManager->ignoreZeroValues)) {
// ToDo: Check whether violinPlots vs violinDrawlistPlots. This here is for the attributes, not the drawlists...
						for (auto& drawListPlot : violinDrawlistPlots) {
							int drawL = 0;
							drawListPlot.maxGlobalValue = 0;
							for (int j = 0; j < drawListPlot.maxValues.size(); ++j) {
								drawListPlot.maxValues[j] = 0;
							}
							for (auto& drawList : drawListPlot.drawLists) {
								DrawList* dl = &(*std::find_if(g_PcPlotDrawLists.begin(), g_PcPlotDrawLists.end(), [drawList](DrawList& draw) {return draw.name == drawList; }));

								std::vector<std::pair<float, float>> minMax;
								for (Attribute& a : pcAttributes) {
									minMax.push_back({ a.min,a.max });
								}
								DataSet* ds;
								for (DataSet& d : g_PcPlotDataSets) {
									if (d.name == dl->parentDataSet) {
										ds = &d;
										break;
									}
								}
                                exeComputeHistogram(dl->name, minMax, dl->buffer, ds->data.size(), dl->indicesBuffer, dl->indices.size(), dl->activeIndicesBufferView, true);
								//histogramManager->computeHistogramm(dl->name, minMax, dl->buffer, ds->data.size(), dl->indicesBuffer, dl->indices.size(), dl->activeIndicesBufferView);
								HistogramManager::Histogram& hist = histogramManager->getHistogram(dl->name);
								std::vector<std::pair<uint32_t, float>> area;
								for (int j = 0; j < hist.maxCount.size(); ++j) {
									if (hist.maxCount[j] > drawListPlot.maxValues[j]) {
										drawListPlot.maxValues[j] = hist.maxCount[j];
									}
									if (hist.maxCount[j] > drawListPlot.maxGlobalValue) {
										drawListPlot.maxGlobalValue = hist.maxCount[j];
									}
									area.push_back({ j, drawListPlot.attributeScalings[j] / hist.maxCount[j] });
								}

								if (renderOrderAttConsider && ((drawL == 0) || (!renderOrderBasedOnFirstAtt)))
								{
									std::sort(area.begin(), area.end(), [](std::pair<uint32_t, float>& a, std::pair<uint32_t, float>& b) { return sortDescPair(a, b); });
									for (int j = 0; j < pcAttributes.size(); ++j)drawListPlot.attributeOrder[drawL][j] = area[j].first;
								}
								else
								{
									drawListPlot.attributeOrder[drawL] = drawListPlot.attributeOrder[0];
								}
								++drawL;
							}
						}
					}
					if (ImGui::Checkbox("Ignore zero bins", &histogramManager->ignoreZeroBins)) {
						histogramManager->updateSmoothedValues();
						for (auto& drawListPlot : violinDrawlistPlots) {
							drawListPlot.maxGlobalValue = 0;
							for (int j = 0; j < drawListPlot.maxValues.size(); ++j) {
								drawListPlot.maxValues[j] = 0;
							}
							int drawL = 0;
							for (auto& drawList : drawListPlot.drawLists) {
								HistogramManager::Histogram& hist = histogramManager->getHistogram(drawList);
								std::vector<std::pair<uint32_t, float>> area;
								for (int j = 0; j < hist.maxCount.size(); ++j) {
									if (hist.maxCount[j] > drawListPlot.maxValues[j]) {
										drawListPlot.maxValues[j] = hist.maxCount[j];
									}
									if (hist.maxCount[j] > drawListPlot.maxGlobalValue) {
										drawListPlot.maxGlobalValue = hist.maxCount[j];
									}
									area.push_back({ j, drawListPlot.attributeScalings[j] / hist.maxCount[j] });
								}

								if (renderOrderAttConsider && ((drawL == 0) || (!renderOrderBasedOnFirstAtt)))
								{
									std::sort(area.begin(), area.end(), [](std::pair<uint32_t, float>& a, std::pair<uint32_t, float>& b) {return sortDescPair(a, b); });
									for (int j = 0; j < pcAttributes.size(); ++j)drawListPlot.attributeOrder[drawL][j] = area[j].first;
								}
								else
								{
								drawListPlot.attributeOrder[drawL] = drawListPlot.attributeOrder[0];
								}
								++drawL;
							}
						}
					}
					static float stdDev = 1.5;
					if (ImGui::SliderFloat("Smoothing kernel stdDev", &stdDev, -1, 25)) {
						histogramManager->setSmoothingKernelSize(stdDev);
						updateAllViolinPlotMaxValues(renderOrderBasedOnFirstAtt);
					}
					static char* violinYs[] = { "Standard","Local brush","Global brush","All brushes" };
					if (ImGui::BeginCombo("Y Scale", violinYs[violinYScale])) {
                        ImGui::SetTooltip("This only affects to which range the bins are fitted. The y min and max are the PCPlot axis borders.");
						for (int v = 0; v < 4; ++v) {
							if (ImGui::MenuItem(violinYs[v])) {
								violinYScale =(ViolinYScale) v;
							}
						}
						ImGui::EndCombo();
					}
					ImGui::Columns(3);
					ImGui::Checkbox("Overlay lines", &violinPlotOverlayLines);
					ImGui::NextColumn();
					ImGui::Checkbox("Base render order on first attribute", &renderOrderBasedOnFirstAtt);
					ImGui::NextColumn();
					ImGui::Checkbox("Optimize render order", &renderOrderAttConsider);
					ImGui::Separator();

					ImGui::Checkbox("Optimize non-stop", &renderOrderAttConsiderNonStop);
					
					//ImGui::EndMenu();

					ImGui::EndMenu();
				}
				ImGui::EndMenuBar();
			}

			const static int plusWidth = 100;
            for (unsigned int i = 0; i < violinAttributePlots.size(); ++i) {
				ImGui::BeginChild(std::to_string(i).c_str(), ImVec2(-1, violinPlotHeight), true);
				ImGui::PushItemWidth(150);
				//listing all histograms available
				for (int j = 0; j < violinAttributePlots[i].drawLists.size(); ++j) {
					ImGui::Checkbox(violinAttributePlots[i].drawLists[j].name.c_str(), &violinAttributePlots[i].drawLists[j].activated);
					static char* plotPositions[] = { "Left","Right","Middle" };
					ImGui::SameLine(200);
					if (ImGui::BeginCombo(("Position##" + std::to_string(j)).c_str(), plotPositions[violinAttributePlots[i].violinPlacements[j]])) {
						for (int k = 0; k < 3; ++k) {
							if (ImGui::MenuItem(plotPositions[k], nullptr)) {
								violinAttributePlots[i].violinPlacements[j] = (ViolinPlacement)k;
							}
						}
						ImGui::EndCombo();
					}
					static char* violinScales[] = { "Self","Local","Global","Global Attribute" };
					ImGui::SameLine(480);
					if (ImGui::BeginCombo(("Scale##" + std::to_string(j)).c_str(), violinScales[violinAttributePlots[i].violinScalesX[j]])) {
						for (int k = 0; k < 4; ++k) {
							if (ImGui::MenuItem(violinScales[k], nullptr)) {
								violinAttributePlots[i].violinScalesX[j] = (ViolinScale)k;
							}
						}
						ImGui::EndCombo();
					}
					ImGui::SameLine(730);
					ImGui::ColorEdit4(("Line Col" + std::to_string(j)).c_str(), &violinAttributePlots[i].drawListLineColors[j].x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
					ImGui::SameLine(900);
					ImGui::ColorEdit4(("Fill Col" + std::to_string(j)).c_str(), &violinAttributePlots[i].drawListFillColors[j].x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
					ImGui::SameLine(1200);
					if (ImGui::Checkbox(("##log" + std::to_string(j)).c_str(), &histogramManager->logScale[j])) {
						histogramManager->updateSmoothedValues();
						updateAllViolinPlotMaxValues(renderOrderBasedOnFirstAtt);
					};
				}
				static char choose[] = "Choose drawlist";
				if (ImGui::BeginCombo("Add drawlistdata", choose)) {
					for (auto k = g_PcPlotDrawLists.begin(); k != g_PcPlotDrawLists.end(); ++k) {
						if (ImGui::MenuItem(k->name.c_str(), "", false)) {
							std::vector<std::string> attrNames;
							std::vector<std::pair<float, float>> minMax;
							for (Attribute& a : pcAttributes) {
								minMax.push_back({ a.min,a.max });
								attrNames.push_back(a.name);
							}

							if (violinAttributePlots[i].attributeNames.size()) {		//checking if the attributes of the dataset to be added are the same as the already existing attributes in this violin plot
								bool attributeCheckFail = false;
								if (violinAttributePlots[i].attributeNames.size() != pcAttributes.size()) continue;
								for (int l = 0; l < pcAttributes.size(); ++l) {
									if (pcAttributes[l].name != violinAttributePlots[i].attributeNames[l]) {
										attributeCheckFail = true;
										break;
									}
								}
								if (attributeCheckFail) {
									continue;
#ifdef _DEBUG
									std::cout << "The attribute check for the drawlist to add failed." << std::endl;
#endif
								}
							}
							else {											//instantiating the values of the violin plot, as this is the first drawlist to be added to this plot
								violinAttributePlots[i].activeAttributes = new bool[pcAttributes.size()];
								for (int l = 0; l < pcAttributes.size(); ++l) {
									violinAttributePlots[i].activeAttributes[l] = true;
									violinAttributePlots[i].maxValues.push_back(std::numeric_limits<float>::min());
									violinAttributePlots[i].attributeOrder.push_back(l);
								}
								violinAttributePlots[i].attributeNames = attrNames;
							}

							DataSet* parent;
							for (DataSet& ds : g_PcPlotDataSets) {
								if (ds.name == k->parentDataSet)
									parent = &ds;
							}
                            exeComputeHistogram(k->name, minMax, k->buffer, parent->data.size(), k->indicesBuffer, k->indices.size(), k->activeIndicesBufferView, true);
							//histogramManager->computeHistogramm(k->name, minMax, k->buffer, parent->data.size(), k->indicesBuffer, k->indices.size(), k->activeIndicesBufferView);
							bool datasetIncluded = false;
							for (int j = 0; j < violinAttributePlots[i].drawLists.size(); ++j) {
								if (k->name == violinAttributePlots[i].drawLists[j].name) {
									datasetIncluded = true;
									break;
								}
							}
							if (!datasetIncluded) {
								violinAttributePlots[i].drawLists.push_back({ k->name, true });
								violinAttributePlots[i].violinPlacements.push_back(ViolinLeft);
								violinAttributePlots[i].violinScalesX.push_back(ViolinScaleSelf);
								violinAttributePlots[i].drawListLineColors.push_back({ 0,0,0,1 });
								violinAttributePlots[i].drawListFillColors.push_back({ 0,0,0,.1f });
							}
							HistogramManager::Histogram& h = histogramManager->getHistogram(k->name);
							for (int l = 0; l < h.maxCount.size(); ++l) {
								if (violinAttributePlots[i].maxValues[l] < h.maxCount[l]) {
									violinAttributePlots[i].maxValues[l] = h.maxCount[l];
								}
							}
							if (h.maxGlobalCount > violinAttributePlots[i].maxGlobalValue) {
								violinAttributePlots[i].maxGlobalValue = h.maxGlobalCount;
							}
						}
					}
					ImGui::EndCombo();


				}

                // Draw everything to load Colorbrewer Colorpalettes
                if (violinAttributePlots[i].attributeNames.size() > 0){
                    includeColorbrewerToViolinPlot((violinAttributePlots[i].colorPaletteManager),
                                                   &(violinAttributePlots[i].drawListLineColors),
                                                   &(violinAttributePlots[i].drawListFillColors));
                }

				int amtOfAttributes = 0;
				for (int j = 0; j < violinAttributePlots[i].maxValues.size(); ++j) {
					if (j != 0)ImGui::SameLine();
                    ImGui::Checkbox(pcAttributes[j].name.c_str(), violinAttributePlots[i].activeAttributes + j);

					if (violinAttributePlots[i].activeAttributes[j]) ++amtOfAttributes;
				}



				int previousNrOfColumns = ImGui::GetColumnsCount();
				ImGui::Separator();
				ImGui::Columns(5);


				if ((ImGui::Button("Optimize sides <right/left>")) || (violinPlotAttrReplaceNonStop)) {
					if (violinAttributePlots[i].drawLists.size() != 0) {
						violinAdaptSidesAutoObj.vp = &(violinAttributePlots[i]);
						violinAdaptSidesAutoObj.optimizeSidesNowAttr = true;

						// Only compute the order for the first histogram in the list (the first one in the matrix. Is that the same?)
						//auto& hist = histogramManager->getHistogram(violinAttributePlots[i].drawLists[0].name);
						//histogramManager->determineSideHist(hist, &violinAttributePlots[i].activeAttributes);

						//violinAttributePlots[i].violinPlacements.clear();
						//for (int j = 0; j < violinAttributePlots[i].attributeNames.size(); ++j) {
						//	violinAttributePlots[i].violinPlacements.push_back((hist.side[j] % 2) ? ViolinLeft : ViolinRight);


						//}
						//if (violinPlotAttrInsertCustomColors) {
						//	changeColorsToCustomAlternatingColors(&(violinAttributePlots[i].colorPaletteManager), violinAttributePlots[i].attributeNames.size(), &(violinAttributePlots[i].drawListLineColors), &(violinAttributePlots[i].drawListFillColors),
						//		hist, &(violinAttributePlots[i].activeAttributes) );
						//}


						// Create a render-order object, which is used just before drawing the violins to compute the order. This way, it will be computed when all the scaling factors are known. 
						// ViolinDrawlistPlot *dlp = &(violinAttributePlots[i]);



						// Here


						// ViolinPlot *dlp = &(violinAttributePlots[i]);
						// bool optimizeSidesNow = true;
						
						// for drawing
						


					}
				}
				ImGui::NextColumn();
				ImGui::Checkbox("", &violinPlotAttrInsertCustomColors);
				ImGui::SameLine(50);
				if (ImGui::BeginMenu("Apply colors of Dark2YellowSplit")) {
					std::vector<std::string> *availablePalettes = 
						violinAttributePlots[i].colorPaletteManager->colorPalette->getQualPaletteNames();

					std::vector<const char*>  vc = convertStringVecToConstChar(availablePalettes);
				
					if (vc.size() > 0) {
						//static char* violinYs[] = { "Standard","Local brush","Global brush","All brushes" };
						if (ImGui::BeginCombo("Line Palette", vc[violinPlotAttrAutoColorAssignLine])) {
							for (int v = 0; v < vc.size(); ++v) {
								if (ImGui::MenuItem(vc[v])) {
									violinAttributePlots[i].colorPaletteManager->chosenAutoColorPaletteLine =
										(*availablePalettes)[v];
									violinPlotAttrAutoColorAssignFill = v;
								}
							}
							ImGui::EndCombo();
						}
						if (ImGui::BeginCombo("Fill Palette", vc[violinPlotAttrAutoColorAssignFill])) {
							for (int v = 0; v < vc.size(); ++v) {
								if (ImGui::MenuItem(vc[v])) {
									violinAttributePlots[i].colorPaletteManager->chosenAutoColorPaletteFill =
										(*availablePalettes)[v];
									violinPlotAttrAutoColorAssignFill = v;
								}
							}
							ImGui::EndCombo();
                        }
                    }
					ImGui::EndMenu();
				}
				

				ImGui::NextColumn();
				ImGui::Checkbox("Re-place constantly", &violinPlotAttrReplaceNonStop);

				ImGui::NextColumn();
				ImGui::Checkbox("Consider blending order", &violinPlotAttrConsiderBlendingOrder);
				ImGui::NextColumn();
				if(ImGui::Checkbox("Reverse color pallette", &violinPlotAttrReverseColorPallette))
				{
					violinAttributePlots[i].colorPaletteManager->setReverseColorOrder(violinPlotDLReverseColorPallette);
				}

				if (ImGui::Button("Fix order and colors"))
				{
					violinPlotAttrReplaceNonStop = false;
					violinAttributePlots[i].colorPaletteManager->useColorPalette = false;
					renderOrderAttConsiderNonStop = false;
					renderOrderAttConsider = false;
					//violinDrawlistPlots[i].colorPaletteManager->useColorPalette = false;				
				}


				ImGui::Columns(previousNrOfColumns);


				//labels for the plots
				ImGui::Separator();
				int c = 0;
				int c1 = 0;
				float xGap = (ImGui::GetWindowContentRegionWidth() - (amtOfAttributes - 1) * violinPlotXSpacing) / amtOfAttributes + violinPlotXSpacing;
				for (uint32_t j : violinAttributePlots[i].attributeOrder) {
					if (!violinAttributePlots[i].activeAttributes[j]) {
						c++;
						continue;
					}

					if (c1 != 0) {
						ImGui::SameLine(c1 * xGap + 10);
					}
					ImGui::Button((violinAttributePlots[i].attributeNames[j] + "##violinattr").c_str());
					if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
						int p[] = { c,i };		//holding the index in the pcAttriOrd array and the value of it
						ImGui::SetDragDropPayload("ViolinATTRIBUTE", p, sizeof(p));
						ImGui::Text("Swap %s", violinAttributePlots[i].attributeNames[j].c_str());
						ImGui::EndDragDropSource();
					}
					if (ImGui::BeginDragDropTarget()) {
						if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ViolinATTRIBUTE")) {
							int* other = (int*)payload->Data;

							switchViolinAttributes(c, other[0], io.KeyCtrl, violinAttributePlots[i].attributeOrder);
						}
						ImGui::EndDragDropTarget();
					}

					c++;
					c1++;
				}

				// Drawing the violin plots
				ImVec2 leftUpperCorner = ImGui::GetCursorScreenPos();
				ImVec2 origLeftUpper = leftUpperCorner;
				ImVec2 size((ImGui::GetWindowContentRegionWidth() - (amtOfAttributes - 1) * violinPlotXSpacing) / amtOfAttributes, ImGui::GetWindowContentRegionMax().y - leftUpperCorner.y + ImGui::GetWindowPos().y);
				ViolinDrawState drawState = (violinPlotOverlayLines) ? ViolinDrawStateArea : ViolinDrawStateAll;
				bool done = false;
				while (!done) {
					leftUpperCorner = origLeftUpper;
					for (int j : violinAttributePlots[i].attributeOrder) {		//Drawing the plots per Attribute
						if (!violinAttributePlots[i].activeAttributes[j]) continue;
						if (drawState == ViolinDrawStateAll || drawState == ViolinDrawStateArea) ImGui::RenderFrame(leftUpperCorner, leftUpperCorner + size, ImGui::GetColorU32(violinBackgroundColor), true, ImGui::GetStyle().FrameRounding);
						ImGui::PushClipRect(leftUpperCorner, leftUpperCorner + size + ImVec2{ 1,1 }, false);
						for (int k = 0; k < violinAttributePlots[i].drawLists.size(); ++k) {
							if (!violinAttributePlots[i].drawLists[k].activated) continue;
							HistogramManager::Histogram& hist = histogramManager->getHistogram(violinAttributePlots[i].drawLists[k].name);
							DrawList* dl = nullptr;
                            if (true || yScaleToCurrenMax) {
								for (DrawList& draw : g_PcPlotDrawLists) {
									if (draw.name == violinAttributePlots[i].drawLists[k].name) {
										dl = &draw;
									}
								}
							}
							//std::vector<std::pair<float, float>> localMinMax(pcAttributes.size(), { std::numeric_limits<float>().max(),std::numeric_limits<float>().min() });
							if (violinYScale == ViolinYScaleLocalBrush || violinYScale == ViolinYScaleBrushes) {
								for (int j = 0; j < pcAttributes.size(); ++j) {
									for (int mi = 0; mi < dl->brushes[k].size(); ++mi) {
										if (dl->brushes[j][mi].minMax.first < localMinMax[j].first) localMinMax[j].first = dl->brushes[j][mi].minMax.first;
										if (dl->brushes[j][mi].minMax.second > localMinMax[j].second) localMinMax[j].second = dl->brushes[j][mi].minMax.second;
									}
								}
								for (int j = 0; j < pcAttributes.size(); ++j) {
									if (localMinMax[j].first == std::numeric_limits<float>().max()) {
										localMinMax[j].first = pcAttributes[j].min;
										localMinMax[j].second = pcAttributes[j].max;
									}
								}
							}

							float histYStart;
							float histYEnd;
							switch (violinYScale) {
							case ViolinYScaleStandard:
								histYStart = 0;
								histYEnd = size.y;
								break;
							case ViolinYScaleLocalBrush:
								histYStart = ((hist.ranges[j].second - localMinMax[j].second) / (localMinMax[j].first - localMinMax[j].second) * size.y);
								histYEnd = ((hist.ranges[j].first - localMinMax[j].second) / (localMinMax[j].first - localMinMax[j].second) * size.y);
								break;
							case ViolinYScaleGlobalBrush:
								histYStart = ((hist.ranges[j].second - globalMinMax[j].second) / (globalMinMax[j].first - globalMinMax[j].second) * size.y);
								histYEnd = ((hist.ranges[j].first - globalMinMax[j].second) / (globalMinMax[j].first - globalMinMax[j].second) * size.y);
								break;
							case ViolinYScaleBrushes:
								float min = (localMinMax[j].first < globalMinMax[j].first) ? localMinMax[j].first : globalMinMax[j].first;
								float max = (localMinMax[j].second < globalMinMax[j].second) ? localMinMax[j].second : globalMinMax[j].second;
								histYStart = ((hist.ranges[j].second - max) / (min - max) * size.y);
								histYEnd = ((hist.ranges[j].first - max) / (min - max) * size.y);
								break;
							}
							//if (dl && dl->brushes[j].size()) {
							//	float max = dl->brushes[j][0].minMax.second;
							//	float min = dl->brushes[j][0].minMax.first;
							//	for (int mi = 1; mi < dl->brushes[j].size(); ++mi) {
							//		if (dl->brushes[j][mi].minMax.first < min) min = dl->brushes[j][mi].minMax.first;
							//		if (dl->brushes[j][mi].minMax.second > max) max = dl->brushes[j][mi].minMax.second;
							//	}
							//	histYStart = ((hist.ranges[k].second - max) / (min - max) * size.y);
							//	histYEnd = ((hist.ranges[k].first - max) / (min - max) * size.y);
							//}
							//else {
							//	histYStart = ((hist.ranges[j].second - pcAttributes[j].max) / (pcAttributes[j].min - pcAttributes[j].max) * size.y);
							//	histYEnd = ((hist.ranges[j].first - pcAttributes[j].max) / (pcAttributes[j].min - pcAttributes[j].max) * size.y);
							//}
							//if (!yScaleToCurrenMax) {
							//	histYStart = 0;
							//	histYEnd = size.y;
							//}
							float histYFillStart = (histYStart < 0) ? 0 : histYStart;
							float histYFillEnd = (histYEnd > size.y) ? size.y : histYEnd;
							float histYLineStart = histYStart + leftUpperCorner.y;
							float histYLineEnd = histYEnd + leftUpperCorner.y;
							float histYLineDiff = histYLineEnd - histYLineStart;

							float div = 0;
							std::vector<float> scals({});
							switch (violinAttributePlots[i].violinScalesX[k]) {
							case ViolinScaleSelf:
								div = hist.maxCount[j];
								break;
							case ViolinScaleLocal:
								div = determineViolinScaleLocalDiv(hist.maxCount, &(violinAttributePlots[i].activeAttributes), scals);
								//div = hist.maxGlobalCount;
								break;
							case ViolinScaleGlobal:
								div = violinAttributePlots[i].maxGlobalValue;
								break;
							case ViolinScaleGlobalAttribute:
								div = violinDrawlistPlots[i].maxValues[k];
								break;
							}


							switch (violinAttributePlots[i].violinPlacements[k]) {
							case ViolinLeft:
								//filling
								if (drawState == ViolinDrawStateArea || drawState == ViolinDrawStateAll) {
									hist.binsRendered[j].clear();
									hist.areaRendered[j] = 0;
									for (int p = histYFillStart; p < histYFillEnd; ++p) {
										ImVec2 a(leftUpperCorner.x, leftUpperCorner.y + p);
										float v = getBinVal(((1 - (p + .5f) + histYEnd) / (histYEnd - histYStart)), hist.bins[j]);
										ImVec2 b(leftUpperCorner.x + v / div * size.x, leftUpperCorner.y + p + 1);
										hist.binsRendered[j].push_back(std::abs(b.x - a.x));
										hist.areaRendered[j] += std::abs(b.x - a.x);
										if (b.x - a.x >= 1)
											ImGui::GetWindowDrawList()->AddRectFilled(a, b, ImColor(violinAttributePlots[i].drawListFillColors[k]));
									}
								}
								//outline
								if (drawState == ViolinDrawStateLine || drawState == ViolinDrawStateAll) {
									for (int l = 1; l < hist.bins[j].size(); ++l) {
										ImGui::GetWindowDrawList()->AddLine(ImVec2(leftUpperCorner.x + hist.bins[j][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[k].size() - 1) * histYLineDiff),
											ImVec2(leftUpperCorner.x + hist.bins[j][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[k].size() - 1) * histYLineDiff), ImColor(violinAttributePlots[i].drawListLineColors[k]), violinPlotThickness);
									}
								}
								break;
							case ViolinRight:
								//filling
								if (drawState == ViolinDrawStateArea || drawState == ViolinDrawStateAll) {
									hist.binsRendered[j].clear();
									hist.areaRendered[j] = 0;
									for (int p = histYFillStart; p < histYFillEnd; ++p) {
										ImVec2 a(leftUpperCorner.x + size.x, leftUpperCorner.y + p);
										float v = getBinVal(((1 - (p + .5f) + histYEnd) / (histYEnd - histYStart)), hist.bins[j]);
										ImVec2 b(leftUpperCorner.x + size.x - v / div * size.x, leftUpperCorner.y + p + 1);
										hist.binsRendered[j].push_back(std::abs(b.x - a.x));
										hist.areaRendered[j] += std::abs(b.x - a.x);
										if (a.x - b.x >= 1)
											ImGui::GetWindowDrawList()->AddRectFilled(a, b, ImColor(violinAttributePlots[i].drawListFillColors[k]));
									}
								}
								//outline
								if (drawState == ViolinDrawStateLine || drawState == ViolinDrawStateAll) {
									for (int l = 1; l < hist.bins[j].size(); ++l) {
										ImGui::GetWindowDrawList()->AddLine(ImVec2(leftUpperCorner.x + size.x - hist.bins[j][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[k].size() - 1) * histYLineDiff),
											ImVec2(leftUpperCorner.x + size.x - hist.bins[j][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[k].size() - 1) * histYLineDiff), ImColor(violinAttributePlots[i].drawListLineColors[k]), violinPlotThickness);
									}
								}
								break;
							case ViolinMiddle:
								float xBase = leftUpperCorner.x + .5f * size.x;
								//filling
								if (drawState == ViolinDrawStateArea || drawState == ViolinDrawStateAll) {
									hist.binsRendered[j].clear();
									hist.areaRendered[j] = 0;
									for (int p = histYFillStart; p < histYFillEnd; ++p) {
										float v = getBinVal(((1 - (p + .5f) + histYEnd) / (histYEnd - histYStart)), hist.bins[j]);
										ImVec2 a(xBase - .5f * v / div * size.x, leftUpperCorner.y + p);
										ImVec2 b(xBase + .5f * v / div * size.x, leftUpperCorner.y + p + 1);
										hist.binsRendered[j].push_back(std::abs(b.x - a.x));
										hist.areaRendered[j] += std::abs(b.x - a.x);
										if (b.x - a.x >= 1)
											ImGui::GetWindowDrawList()->AddRectFilled(a, b, ImColor(violinAttributePlots[i].drawListFillColors[k]));
									}
								}
								if (drawState == ViolinDrawStateLine || drawState == ViolinDrawStateAll) {
									for (int l = 1; l < hist.bins[j].size(); ++l) {
										//left Line
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase - .5f * hist.bins[j][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[k].size() - 1) * histYLineDiff),
											ImVec2(xBase - .5f * hist.bins[j][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[k].size() - 1) * histYLineDiff), ImColor(violinAttributePlots[i].drawListLineColors[k]), violinPlotThickness);
										//right Line
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase + .5f * hist.bins[j][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[k].size() - 1) * histYLineDiff),
											ImVec2(xBase + .5f * hist.bins[j][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[k].size() - 1) * histYLineDiff), ImColor(violinAttributePlots[i].drawListLineColors[k]), violinPlotThickness);
									}
								}

								break;
							}
							
						}
						optimizeViolinSidesAndAssignCustColors();

						ImGui::PopClipRect();
						leftUpperCorner.x += size.x + violinPlotXSpacing;
					}

					if (drawState == ViolinDrawStateAll || drawState == ViolinDrawStateLine) done = true;
					if (drawState == ViolinDrawStateArea) drawState = ViolinDrawStateLine;
				}
				ImGui::PopItemWidth();
				ImGui::EndChild();
			}

			//adding new Plots
			ImGui::SetCursorPosX(ImGui::GetWindowWidth() / 2 - plusWidth / 2);
			if (ImGui::Button("+", ImVec2(plusWidth, 0))) {
				//ViolinPlot *currVP = new ViolinPlot();
				violinAttributePlots.emplace_back();// *currVP);
				//operator delete(currVP);
				//currVP = nullptr;
			}

			ImGui::End(); 
		}

		//begin of violin plots drawlist major --------------------------------------------------------------------------
		if (enableDrawlistViolinPlots) {
			ImGui::Begin("Violin drawlist window", &enableDrawlistViolinPlots, ImGuiWindowFlags_MenuBar);
			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("Settings")) {
					ImGui::Checkbox("Couple to Brushing", &coupleViolinPlots);
					ImGui::Checkbox("Send to iso renderer on select", &violinPlotDLSendToIso);
					ImGui::SliderInt("Violin plots height", &violinPlotHeight, 1, 4000);
					ImGui::SliderInt("Violin plots x spacing", &violinPlotXSpacing, 0, 40);
					ImGui::SliderFloat("Violin plots line thickness", &violinPlotThickness, 0, 10);
					ImGui::ColorEdit4("Violin plots background", &violinBackgroundColor.x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
					if (ImGui::Checkbox("Ignore zero values", &histogramManager->ignoreZeroValues)) {		//updating all histogramms if 0 values should be ignored
                        unsigned int currViolinDrawlistPlotIdx = 0;
                        for (auto& drawListPlot : violinDrawlistPlots) {

							int drawL = 0;
							drawListPlot.maxGlobalValue = 0;
							for (int j = 0; j < drawListPlot.maxValues.size(); ++j) {
								drawListPlot.maxValues[j] = 0;
							}
							for (auto& drawList : drawListPlot.drawLists) {
								DrawList* dl = &(*std::find_if(g_PcPlotDrawLists.begin(), g_PcPlotDrawLists.end(), [drawList](DrawList& draw) {return draw.name == drawList; }));

								std::vector<std::pair<float, float>> minMax;
								for (Attribute& a : pcAttributes) {
									minMax.push_back({ a.min,a.max });
								}
								DataSet* ds;
								for (DataSet& d : g_PcPlotDataSets) {
									if (d.name == dl->parentDataSet) {
										ds = &d;
										break;
									}
								}
								exeComputeHistogram(dl->name, minMax, dl->buffer, ds->data.size(), dl->indicesBuffer, dl->indices.size(), dl->activeIndicesBufferView);

								//histogramManager->computeHistogramm(dl->name, minMax, dl->buffer, ds->data.size(), dl->indicesBuffer, dl->indices.size(), dl->activeIndicesBufferView);
								HistogramManager::Histogram& hist = histogramManager->getHistogram(dl->name);
								std::vector<std::pair<uint32_t, float>> area;
								for (int j = 0; j < hist.maxCount.size(); ++j) {
									if (hist.maxCount[j] > drawListPlot.maxValues[j]) {
										drawListPlot.maxValues[j] = hist.maxCount[j];
									}
									if (hist.maxCount[j] > drawListPlot.maxGlobalValue) {
										drawListPlot.maxGlobalValue = hist.maxCount[j];
									}
									area.push_back({ j, drawListPlot.attributeScalings[j] / hist.maxCount[j] });
								}

								// Only sort the first drawlist. All others should have the same sorting (if option is taken)
								if (renderOrderDLConsider && ((drawL == 0) || (!renderOrderBasedOnFirstDL)))
								{
									//if (!renderOrderDLReverse) {
										drawListPlot.attributeOrder[drawL] = sortHistogram(hist, drawListPlot, renderOrderDLConsider, renderOrderDLReverse);

										//std::sort(area.begin(), area.end(), [](std::pair<uint32_t, float>& a, std::pair<uint32_t, float>& b) {return sortDescPair(a, b); });
										//for (int j = 0; j < pcAttributes.size(); ++j)drawListPlot.attributeOrder[drawL][j] = area[j].first;
									//}
									//else {
										//std::sort(area.begin(), area.end(), [](std::pair<uint32_t, float>& a, std::pair<uint32_t, float>& b) {return sortAscPair(a, b); });
										//for (int j = 0; j < pcAttributes.size(); ++j)drawListPlot.attributeOrder[drawL][j] = area[j].first;
									//}
								}
								else
								{
									drawListPlot.attributeOrder[drawL] = drawListPlot.attributeOrder[0];
								}
								++drawL;

                                updateHistogramComparisonDL(currViolinDrawlistPlotIdx);
							}
                            ++currViolinDrawlistPlotIdx;
						}
					}
					if (ImGui::Checkbox("Ignore zero bins", &histogramManager->ignoreZeroBins)) {
						histogramManager->updateSmoothedValues();
						updateAllViolinPlotMaxValues(renderOrderBasedOnFirstDL);
                        for (unsigned int cpdlI; cpdlI < violinDrawlistPlots.size(); ++cpdlI){updateHistogramComparisonDL(cpdlI);}
					}
					static float stdDev = 1.5;
					if (ImGui::SliderFloat("Smoothing kernel stdDev", &stdDev, -1, 25)) {
						histogramManager->setSmoothingKernelSize(stdDev);
						updateAllViolinPlotMaxValues(renderOrderBasedOnFirstDL);
                        for (unsigned int cpdlI; cpdlI < violinDrawlistPlots.size(); ++cpdlI){updateHistogramComparisonDL(cpdlI);}
					}
					static char* violinYs[] = { "Standard","Local brush","Global brush","All brushes" };

					ImGui::Columns(2);
					if (ImGui::BeginCombo("Y Scale", violinYs[violinYScale])) {
						for (int v = 0; v < 4; ++v) {
							if (ImGui::MenuItem(violinYs[v])) {
								violinYScale = (ViolinYScale)v;
							}
						}
						ImGui::EndCombo();
					}
					ImGui::NextColumn();
					if (ImGui::Checkbox("Fit bins for selected range", &histogramManager->adaptMinMaxToBrush)) {
						//std::vector<std::pair<float, float>> violinMinMax(pcAttributes.size(), { std::numeric_limits<float>().max(),std::numeric_limits<float>().min() });
						//unsigned int currDLNr = 0;
						//getyScaleDL(currDLNr, violinDrawlistPlots[0], violinMinMax);
                        for (unsigned int cpdlI; cpdlI < violinDrawlistPlots.size(); ++cpdlI){updateHistogramComparisonDL(cpdlI);}
					}

					ImGui::Columns(3);
					ImGui::Checkbox("Overlay lines", &violinPlotOverlayLines);
					ImGui::NextColumn();
					ImGui::Checkbox("Base render order on first DL", &renderOrderBasedOnFirstDL);
					ImGui::NextColumn();
					ImGui::Checkbox("Optimize render order", &renderOrderDLConsider);
					ImGui::Columns(2);
					if (ImGui::Checkbox("Reverse render order", &renderOrderDLReverse)) {
						for (unsigned int i = 0; i < violinDrawlistPlots.size(); ++i) {
							for (int jj = 0; jj < violinDrawlistPlots[i].drawLists.size(); ++jj) {
								HistogramManager::Histogram& hist = histogramManager->getHistogram(violinDrawlistPlots[i].drawLists[jj]);
								(renderOrderDLConsider && ((jj == 0) || (!renderOrderBasedOnFirstDL))) ? violinDrawlistPlots[i].attributeOrder[jj] = sortHistogram(hist, violinDrawlistPlots[i], renderOrderDLConsider, renderOrderDLReverse) : violinDrawlistPlots[i].attributeOrder[jj] = violinDrawlistPlots[i].attributeOrder[0];
							}
						}
					}
					ImGui::NextColumn();
					ImGui::Checkbox("Optimize non-stop", &renderOrderDLConsiderNonStop);

                    ImGui::Separator();
                    ImGui::Columns(1);
                    if(ImGui::Checkbox("Use rendered bins for Hist Comparison", &violinPlotDLUseRenderedBinsForHistComp)){
                        for (unsigned int i = 0; i < violinDrawlistPlots.size(); ++i){
                            updateHistogramComparisonDL(i);
                        }
                    }

					// Option to change all attributes at once.
					ImGui::Separator();
					ImGui::Columns(3);

					static char* plotPositions[] = { "Left","Right","Middle","Middle|Left","Middle|Right","Left|Half","Right|Half" };
					if (ImGui::BeginCombo("ChangePosition", plotPositions[0])) {
						for (int k = 0; k < 7; ++k) {
							if (ImGui::MenuItem(plotPositions[k], nullptr)) {
								for (unsigned int i = 0; i < violinDrawlistPlots.size(); ++i) {
									for (int j = 0; j < violinDrawlistPlots[i].attributeFillColors.size(); ++j)
									{
										violinDrawlistPlots[i].attributePlacements[j] = (ViolinPlacement)k;
									}
								}
							}
						}
						ImGui::EndCombo();
					}
					ImGui::NextColumn();
					static char* violinScales[] = { "Self","Local","Global","Global Attribute" };
					if (ImGui::BeginCombo("ChangeScale", violinScales[0])) {
						for (int k = 0; k < 4; ++k) {
							if (ImGui::MenuItem(violinScales[k], nullptr)) {
								for (unsigned int i = 0; i < violinDrawlistPlots.size(); ++i) {
									for (int j = 0; j < violinDrawlistPlots[i].attributeFillColors.size(); ++j) {
										violinDrawlistPlots[i].violinScalesX[j] = (ViolinScale)k;
									}
								}
							}
						}
						ImGui::EndCombo();
                        for (unsigned int cpdlI; cpdlI < violinDrawlistPlots.size(); ++cpdlI){updateHistogramComparisonDL(cpdlI);}
					}
					ImGui::NextColumn();
					if (ImGui::Checkbox("ChangeLogScale", &logScaleDLGlobal)) {
						for (unsigned int i = 0; i < violinDrawlistPlots.size(); ++i) {
							for (int j = 0; j < violinDrawlistPlots[i].attributeFillColors.size(); ++j) {
								//if (histogramManager->logScale[j]) {
								(histogramManager->logScale[j]) = logScaleDLGlobal;
								//}
							}
							histogramManager->updateSmoothedValues();
							updateAllViolinPlotMaxValues(renderOrderBasedOnFirstDL);
							for (int jj = 0; jj < violinDrawlistPlots[i].drawLists.size(); ++jj) {
								HistogramManager::Histogram& hist = histogramManager->getHistogram(violinDrawlistPlots[i].drawLists[jj]);
								(renderOrderDLConsider && ((jj == 0) || (!renderOrderBasedOnFirstDL))) ? violinDrawlistPlots[i].attributeOrder[jj] = sortHistogram(hist, violinDrawlistPlots[i], renderOrderDLConsider, renderOrderDLReverse) : violinDrawlistPlots[i].attributeOrder[jj] = violinDrawlistPlots[i].attributeOrder[0];
							}
							
						}
                        for (unsigned int cpdlI; cpdlI < violinDrawlistPlots.size(); ++cpdlI){updateHistogramComparisonDL(cpdlI);}
					}

					ImGui::EndMenu();

					
				}
				ImGui::EndMenuBar();
			}

			const static int plusWidth = 100;
			for (unsigned int i = 0; i < violinDrawlistPlots.size(); ++i) {
				ImGui::BeginChild(std::to_string(i).c_str(), ImVec2(-1, violinPlotHeight), true);
				ImGui::PushItemWidth(150);
				ImGui::Columns(7);
				ImGui::Separator();
				ImGui::Text("Attributes"); ImGui::NextColumn();
				ImGui::Text("Position"); ImGui::NextColumn();
				ImGui::Text("Scale"); ImGui::NextColumn();
				ImGui::Text("Scale Multiplier"); ImGui::NextColumn();
				ImGui::Text("Log Scale"); ImGui::NextColumn();
				ImGui::Text("Line Color"); ImGui::NextColumn();
				ImGui::Text("Fill Color"); ImGui::NextColumn();
				ImGui::Separator();
				//settings for the attributes
				for (unsigned int j = 0; j < violinDrawlistPlots[i].attributeNames.size(); ++j) {
                    if (ImGui::Checkbox(violinDrawlistPlots[i].attributeNames[j].c_str(), &violinDrawlistPlots[i].activeAttributes[j]))
                    {
                        updateHistogramComparisonDL(i);
                    }
					static char* plotPositions[] = { "Left","Right","Middle","Middle|Left","Middle|Right","Left|Half","Right|Half" };
					ImGui::NextColumn();
					if (ImGui::BeginCombo(("##Position" + std::to_string(j)).c_str(), plotPositions[violinDrawlistPlots[i].attributePlacements[j]])) {
						for (int k = 0; k < 7; ++k) {
							if (ImGui::MenuItem(plotPositions[k], nullptr)) {
								violinDrawlistPlots[i].attributePlacements[j] = (ViolinPlacement)k;
							}
						}
						ImGui::EndCombo();
					}
					static char* violinScales[] = { "Self","Local","Global","Global Attribute" };
					ImGui::NextColumn();
					if (ImGui::BeginCombo(("##Scale" + std::to_string(j)).c_str(), violinScales[violinDrawlistPlots[i].violinScalesX[j]])) {
						for (int k = 0; k < 4; ++k) {
							if (ImGui::MenuItem(violinScales[k], nullptr)) {
								violinDrawlistPlots[i].violinScalesX[j] = (ViolinScale)k;
							}
						}
						ImGui::EndCombo();
                        updateHistogramComparisonDL(i);
					}
					ImGui::NextColumn();
					if (ImGui::SliderFloat(("##slider" + violinDrawlistPlots[i].attributeNames[j]).c_str(), &violinDrawlistPlots[i].attributeScalings[j], 0, 1)) {
						for (int jj = 0; jj < violinDrawlistPlots[i].drawLists.size(); ++jj) {
							std::vector<std::pair<uint32_t, float>> area;
							HistogramManager::Histogram& hist = histogramManager->getHistogram(violinDrawlistPlots[i].drawLists[jj]);
							(renderOrderDLConsider && ((jj == 0) || (!renderOrderBasedOnFirstDL))) ? violinDrawlistPlots[i].attributeOrder[jj] = sortHistogram(hist, violinDrawlistPlots[i], renderOrderDLConsider, renderOrderDLReverse) : violinDrawlistPlots[i].attributeOrder[jj] = violinDrawlistPlots[i].attributeOrder[0];
						}
                        updateHistogramComparisonDL(i);
					}
					ImGui::NextColumn();
					if (ImGui::Checkbox(("##log" + std::to_string(j)).c_str(), &histogramManager->logScale[j])) {
						histogramManager->updateSmoothedValues();
						updateAllViolinPlotMaxValues(renderOrderBasedOnFirstDL);
						for (int jj = 0; jj < violinDrawlistPlots[i].drawLists.size(); ++jj) {
							HistogramManager::Histogram& hist = histogramManager->getHistogram(violinDrawlistPlots[i].drawLists[jj]);
							(renderOrderDLConsider && ((jj == 0) || (!renderOrderBasedOnFirstDL))) ? violinDrawlistPlots[i].attributeOrder[jj] = sortHistogram(hist, violinDrawlistPlots[i], renderOrderDLConsider, renderOrderDLReverse) : violinDrawlistPlots[i].attributeOrder[jj] = violinDrawlistPlots[i].attributeOrder[0];
						}
                        updateHistogramComparisonDL(i);
                    }
					ImGui::NextColumn();
					ImGui::ColorEdit4(("##Line Col" + std::to_string(j)).c_str(), &violinDrawlistPlots[i].attributeLineColors[j].x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
					ImGui::NextColumn();
					ImGui::ColorEdit4(("##Fill Col" + std::to_string(j)).c_str(), &violinDrawlistPlots[i].attributeFillColors[j].x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
					ImGui::NextColumn();
					ImGui::Separator();
				}

				ImGui::Columns(1);
				// Draw everything to load Colorbrewer Colorpalettes
				if (violinDrawlistPlots[i].attributeNames.size() > 0) {
					includeColorbrewerToViolinPlot((violinDrawlistPlots[i].colorPaletteManager),
						&(violinDrawlistPlots[i].attributeLineColors),
						&(violinDrawlistPlots[i].attributeFillColors));
				}






				ImGui::Columns(6);
				if (ImGui::DragInt2(("Matrix dimensions##" + std::to_string(i)).c_str(), (int*)&violinDrawlistPlots[i].matrixSize.first, .01f, 1, 10)) {
					violinDrawlistPlots[i].drawListOrder.resize(violinDrawlistPlots[i].matrixSize.first* violinDrawlistPlots[i].matrixSize.second, 0xffffffff);
				}
				ImGui::NextColumn();

				if ((ImGui::Button("Optimize sides <right/left>")) || (violinPlotDLReplaceNonStop)) {
					if (violinDrawlistPlots[i].drawLists.size() != 0) {
						// Only compute the order for the first histogram in the list (the first one in the matrix. Is that the same?)
						violinAdaptSidesAutoObj.vdlp = &(violinDrawlistPlots[i]);
						violinAdaptSidesAutoObj.optimizeSidesNowDL = true;
					}
				}
				ImGui::NextColumn();
				ImGui::Checkbox("", &violinPlotDLInsertCustomColors);
				ImGui::SameLine(50);
				if (ImGui::BeginMenu("Apply colors of Dark2YellowSplit")) {
					std::vector<std::string> *availablePalettes =
						violinDrawlistPlots[i].colorPaletteManager->colorPalette->getQualPaletteNames();

					std::vector<const char*>  vc = convertStringVecToConstChar(availablePalettes);

					if (vc.size() > 0) {
						//static char* violinYs[] = { "Standard","Local brush","Global brush","All brushes" };
						if (ImGui::BeginCombo("Line Palette", vc[violinPlotDLAutoColorAssignLine])) {
							for (int v = 0; v < vc.size(); ++v) {
								if (ImGui::MenuItem(vc[v])) {
									violinDrawlistPlots[i].colorPaletteManager->chosenAutoColorPaletteLine =
										(*availablePalettes)[v];
									violinPlotDLAutoColorAssignLine = v;
								}
							}
							ImGui::EndCombo();
						}
						if (ImGui::BeginCombo("Fill Palette", vc[violinPlotDLAutoColorAssignFill])) {
							for (int v = 0; v < vc.size(); ++v) {
								if (ImGui::MenuItem(vc[v])) {
									violinDrawlistPlots[i].colorPaletteManager->chosenAutoColorPaletteFill =
										(*availablePalettes)[v];
									violinPlotDLAutoColorAssignFill = v;
								}
							}
							ImGui::EndCombo();
						}
					}
					ImGui::EndMenu();
				}


				ImGui::NextColumn();
				ImGui::Checkbox("Re-place constantly", &violinPlotDLReplaceNonStop);
				ImGui::NextColumn();
				ImGui::Checkbox("Consider blending order", &violinPlotDLConsiderBlendingOrder);
				ImGui::NextColumn();
				if (ImGui::Checkbox("Reverse color pallette", &violinPlotDLReverseColorPallette))
				{
					violinDrawlistPlots[i].colorPaletteManager->setReverseColorOrder(violinPlotDLReverseColorPallette);
				}
                ImGui::Columns(2);
				if (ImGui::Button("Fix order and colors"))
				{
					violinPlotDLReplaceNonStop = false;
					violinDrawlistPlots[i].colorPaletteManager->useColorPalette = false;
					renderOrderDLConsiderNonStop = false;
                    renderOrderDLConsider = false;
				}
                ImGui::NextColumn();

                if (ImGui::Button("Order MPVPs according wrt HistDist")){
                    if (violinPlotDLIdxInListForHistComparison[i] != -1){
                        std::sort(violinDrawlistPlots[i].drawListOrder.begin(), violinDrawlistPlots[i].drawListOrder.end(),
                                  [&]//[&(violinDrawlistPlots[i])]
                                  (uint32_t &a, uint32_t &b)
                        {return sortMPVPWHistMeasure(a, b, violinDrawlistPlots[i].histDistToRepresentative);});

                    }
                }


				ImGui::Columns(1);

				//drawing the setttings for the drawlists
				for (int j = 0; j < violinDrawlistPlots[i].drawLists.size(); ++j) {
                    if (j != 0)ImGui::SameLine();
                    // String of the draggable button to drag in a dl into one position of the violin matrix
                    ImGui::Button(violinDrawlistPlots[i].drawLists[j].c_str());
					if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
						int p[] = { -1,j };		//holding the index in the pcAttriOrd array and the value of it
						ImGui::SetDragDropPayload("ViolinDrawlist", p, sizeof(p));
                        // Name shown during drag&drop event
                        ImGui::Text("%s", violinDrawlistPlots[i].drawLists[j].c_str());
						ImGui::EndDragDropSource();
					}

                    // Rightclick on the name to set it as representative to compare the histograms to
                    if (ImGui::IsItemClicked(1))
                    {
                        (violinPlotDLIdxInListForHistComparison[i] == j) ? violinPlotDLIdxInListForHistComparison[i] = -1 : violinPlotDLIdxInListForHistComparison[i] = j;
                        updateHistogramComparisonDL(i);

                        if (false){
                            std::string currDlDist = "TODOO";
                            // Only draw the histogram distance measure if it is computed
                            if (violinPlotDLIdxInListForHistComparison[i] >= -1){
                                ImGui::Text("TODOOO");
                                         //violinDrawlistPlots[i].histDistToRepresentative
                            }
                        }
                    }

				}

				// Drawing the violin plots
				ImVec2 leftUpperCorner = ImGui::GetCursorScreenPos();
				ImVec2 leftUpperCornerStart = leftUpperCorner;
				ImVec2 size((ImGui::GetWindowContentRegionWidth() - (violinDrawlistPlots[i].matrixSize.second - 1) * violinPlotXSpacing) / violinDrawlistPlots[i].matrixSize.second, (ImGui::GetWindowContentRegionMax().y - leftUpperCorner.y + ImGui::GetWindowPos().y - violinDrawlistPlots[i].matrixSize.first * ImGui::GetFrameHeightWithSpacing()) / (float)violinDrawlistPlots[i].matrixSize.first);
				ViolinDrawState drawState = (violinPlotOverlayLines) ? ViolinDrawStateArea : ViolinDrawStateAll;
				bool done = false;
				while (!done) {
					leftUpperCorner = leftUpperCornerStart;
					for (int x = 0; x < violinDrawlistPlots[i].matrixSize.first; ++x) {	//Drawing the plots per matrix entry
						for (int y = 0; y < violinDrawlistPlots[i].matrixSize.second; ++y) {
							int j = violinDrawlistPlots[i].drawListOrder[x * violinDrawlistPlots[i].matrixSize.second + y];

							ImVec2 framePos = leftUpperCorner;
							framePos.y += ImGui::GetFrameHeightWithSpacing();
							if(drawState == ViolinDrawStateAll || drawState == ViolinDrawStateArea) ImGui::RenderFrame(framePos, framePos + size, ImGui::GetColorU32(violinBackgroundColor), true, ImGui::GetStyle().FrameRounding);
							ImGui::SetCursorScreenPos(framePos);
							if (size.x > 0 && size.y > 0) {	//safety check. ImGui crahes when button size is 0
								if (io.KeyCtrl) {
									ImGui::PushStyleColor(ImGuiCol_Button, { 0,0,0,0 });
									if (ImGui::Button(("##invBut" + std::to_string(x * violinDrawlistPlots[i].matrixSize.second + y)).c_str(), size) && j!= 0xffffffff) {
										if (violinDrawlistPlots[i].selectedDrawlists.find(j) == violinDrawlistPlots[i].selectedDrawlists.end()) {
											violinDrawlistPlots[i].selectedDrawlists.insert(j);
											if (enableIsoSurfaceWindow && violinPlotDLSendToIso) {
												uint32_t w = (isoSurfaceRegularGrid) ? isoSurfaceRegularGridDim[0] : SpacialData::rlatSize;
												uint32_t h = (isoSurfaceRegularGrid) ? isoSurfaceRegularGridDim[1] : SpacialData::altitudeSize + 22;
												uint32_t d = (isoSurfaceRegularGrid) ? isoSurfaceRegularGridDim[2] : SpacialData::rlonSize;
												std::vector<std::pair<float, float>> posBounds(3);
												for (int i = 0; i < 3; ++i) {
													posBounds[i].first = pcAttributes[posIndices[i]].min;
													posBounds[i].second = pcAttributes[posIndices[i]].max;
												}
												if (!isoSurfaceRegularGrid) {
													posBounds[0].first = SpacialData::rlat[0];
													posBounds[0].second = SpacialData::rlat[SpacialData::rlatSize - 1];
													posBounds[1].first = SpacialData::altitude[0];
													posBounds[1].second = SpacialData::altitude[SpacialData::altitudeSize - 1];
													posBounds[2].first = SpacialData::rlon[0];
													posBounds[2].second = SpacialData::rlon[SpacialData::rlonSize - 1];
												}
												DrawList* dl;
												for (auto& draw : g_PcPlotDrawLists) {
													if (violinDrawlistPlots[i].drawLists[j] == draw.name) {
														dl = &draw;
														break;
													}
												}
												std::vector<float*>* data;
												for (auto& ds : g_PcPlotDataSets) {
													if (ds.name == dl->parentDataSet) {
														data = &ds.data;
													}
												}
												int index = -1;
												for (int in = 0; in < isoSurfaceRenderer->drawlistBrushes.size(); ++in) {
													if (isoSurfaceRenderer->drawlistBrushes[in].drawlist == dl->name && isoSurfaceRenderer->drawlistBrushes[in].brush == "") {
														index = in;
														break;
													}
												}
												if (index == -1) {
													uint32_t wi = (isoSurfaceRegularGrid) ? isoSurfaceRegularGridDim[0] : SpacialData::rlatSize;
													uint32_t he = (isoSurfaceRegularGrid) ? isoSurfaceRegularGridDim[1] : SpacialData::altitudeSize + 22;
													uint32_t de = (isoSurfaceRegularGrid) ? isoSurfaceRegularGridDim[2] : SpacialData::rlonSize;
													glm::vec4 isoColor;
													(isoSurfaceRenderer->drawlistBrushes.size() == 0) ? isoColor = { 0,1,0, 0.627 } : isoColor = { 1,0,1,0.627 };
													isoSurfaceRenderer->drawlistBrushes.push_back({ dl->name, "",isoColor, {wi, he, de} });
												}
												isoSurfaceRenderer->update3dBinaryVolume(w, h, d, &posIndices.x, posBounds, pcAttributes.size(), data->size(), dl->buffer, dl->activeIndicesBufferView, dl->indices.size(), dl->indicesBuffer, isoSurfaceRegularGrid, index);
												isoSurfaceRenderer->render();
											}
										}
										else {
											violinDrawlistPlots[i].selectedDrawlists.erase(j);
											int index = -1;
											for (int in = 0; in < isoSurfaceRenderer->drawlistBrushes.size(); ++in) {
												if (isoSurfaceRenderer->drawlistBrushes[in].drawlist == violinDrawlistPlots[i].drawLists[j] && isoSurfaceRenderer->drawlistBrushes[in].brush == "") {
													index = in;
													break;
												}
											}
											if (index != -1) {
												isoSurfaceRenderer->deleteBinaryVolume(index);
												isoSurfaceRenderer->render();
											}
										}
									}
									ImGui::PopStyleColor();
								}
								else {
									ImGui::InvisibleButton(("invBut" + std::to_string(x * violinDrawlistPlots[i].matrixSize.second + y)).c_str(), size);
								}
							}
							if (ImGui::IsItemClicked(1)) {
								violinDrawlistPlots[i].drawListOrder[x * violinDrawlistPlots[i].matrixSize.second + y] = 0xffffffff;
								updateMaxHistogramValues(violinDrawlistPlots[i]);
							}
							if ((drawState == ViolinDrawStateAll || drawState == ViolinDrawStateLine) && j != 0xffffffff && ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
								int p[] = { (int)(x * violinDrawlistPlots[i].matrixSize.second + y),j };		//holding the index in the pcAttriOrd array and the value of it
								ImGui::SetDragDropPayload("ViolinDrawlist", p, sizeof(p));
                                ImGui::Text("%s", violinDrawlistPlots[i].drawLists[j].c_str());
								ImGui::EndDragDropSource();
							}
							if ((drawState == ViolinDrawStateAll || drawState == ViolinDrawStateLine) && ImGui::BeginDragDropTarget()) {
								if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ViolinDrawlist")) {
									int* other = (int*)payload->Data;
									if (other[0] == -1) {
										violinDrawlistPlots[i].drawListOrder[x * violinDrawlistPlots[i].matrixSize.second + y] = other[1];
									}
									else {
										violinDrawlistPlots[i].drawListOrder[other[0]] = j;
										violinDrawlistPlots[i].drawListOrder[x * violinDrawlistPlots[i].matrixSize.second + y] = other[1];
									}
									updateMaxHistogramValues(violinDrawlistPlots[i]);
								}
								ImGui::EndDragDropTarget();
							}
							// we also support to drop a drawlist directly into a gridspace
							if ((drawState == ViolinDrawStateAll || drawState == ViolinDrawStateLine) && ImGui::BeginDragDropTarget()) {
								if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Drawlist")) {
									DrawList* dl = *((DrawList**)payload->Data);
									//check if the drawlist was already added to this plot
									if (std::find(violinDrawlistPlots[i].drawLists.begin(), violinDrawlistPlots[i].drawLists.end(), dl->name) == violinDrawlistPlots[i].drawLists.end()) {
										if (!violinDrawlistPlots[i].attributeNames.size()) {	//creating all needed resources e.g. attribute components
											violinDrawlistPlots[i].activeAttributes = new bool[pcAttributes.size()];
											violinDrawlistPlots[i].maxGlobalValue = 0;
											int j = 0;
											for (Attribute& a : pcAttributes) {
												violinDrawlistPlots[i].attributeNames.push_back(a.name);
												violinDrawlistPlots[i].activeAttributes[j] = true;
												violinDrawlistPlots[i].attributeLineColors.push_back({ 0,0,0,1 });
												violinDrawlistPlots[i].attributeFillColors.push_back({ .5f,.5f,.5f,.5f });
												violinDrawlistPlots[i].attributePlacements.push_back((j % 2) ? ViolinMiddleLeft : ViolinMiddleRight);
												violinDrawlistPlots[i].attributeScalings.push_back(1);
												violinDrawlistPlots[i].violinScalesX.push_back(ViolinScaleGlobalAttribute);
												violinDrawlistPlots[i].maxValues.push_back(0);
												++j;
											}
										}

										std::vector<std::pair<float, float>> minMax;
										for (Attribute& a : pcAttributes) {
											minMax.push_back({ a.min,a.max });
										}
										DataSet* ds;
										for (DataSet& d : g_PcPlotDataSets) {
											if (d.name == dl->parentDataSet) {
												ds = &d;
												break;
											}
										}
										exeComputeHistogram(dl->name, minMax, dl->buffer, ds->data.size(), dl->indicesBuffer, dl->indices.size(), dl->activeIndicesBufferView);
                                        updateHistogramComparisonDL(i);
										//histogramManager->computeHistogramm(dl->name, minMax, dl->buffer, ds->data.size(), dl->indicesBuffer, dl->indices.size(), dl->activeIndicesBufferView);
										HistogramManager::Histogram& hist = histogramManager->getHistogram(dl->name);
										// ToDo:  Check, whether the ordering here should also be adjusted if  'renderOrderBasedOnFirstDL = true'
										violinDrawlistPlots[i].attributeOrder.push_back({});
										violinDrawlistPlots[i].attributeOrder.back() = sortHistogram(hist, violinDrawlistPlots[i], renderOrderDLConsider, renderOrderDLReverse);
										//std::vector<std::pair<uint32_t, float>> area;
                                        for (int j = 0; j < hist.maxCount.size(); ++j) {
											if (hist.maxCount[j] > violinDrawlistPlots[i].maxValues[j]) {
												violinDrawlistPlots[i].maxValues[j] = hist.maxCount[j];
											}
											if (hist.maxCount[j] > violinDrawlistPlots[i].maxGlobalValue) {
												violinDrawlistPlots[i].maxGlobalValue = hist.maxCount[j];
											}
										}

										violinDrawlistPlots[i].drawLists.push_back(dl->name);
										//violinDrawlistPlots[i].drawListOrder.push_back(violinDrawlistPlots[i].drawListOrder.size());
										//std::sort(area.begin(), area.end(), [](std::pair<uint32_t, float>& a, std::pair<uint32_t, float>& b) {return a.second > b.second; });
										//for (int j = 0; j < pcAttributes.size(); ++j)violinDrawlistPlots[i].attributeOrder.back().push_back(area[j].first);
									}
									else {
										updateMaxHistogramValues(violinDrawlistPlots[i]);
                                        updateHistogramComparisonDL(i);
									}
									violinDrawlistPlots[i].drawListOrder[x * violinDrawlistPlots[i].matrixSize.second + y] = violinDrawlistPlots[i].drawLists.size() - 1;
									
								}
								ImGui::EndDragDropTarget();
							}

							// if the current violin plot is selected draw a rect around it
							if (violinDrawlistPlots[i].selectedDrawlists.find(j) != violinDrawlistPlots[i].selectedDrawlists.end()) {
								ImGui::GetWindowDrawList()->AddRect(framePos, framePos + size, IM_COL32(255,200,0,255), ImGui::GetStyle().FrameRounding,ImDrawCornerFlags_All,5);
							}
							if (j == 0xffffffff) {
								leftUpperCorner.x += size.x + violinPlotXSpacing;
								continue;
							}
							ImVec2 textPos = framePos;
							textPos.y -= ImGui::GetTextLineHeight();
                            if (violinPlotDLIdxInListForHistComparison[i] != -1){textPos.y -= 1.1*ImGui::GetTextLineHeight();}
							ImGui::SetCursorScreenPos(textPos);

                            // Here, the text above each MPVP is written.


							ImGui::Text(violinDrawlistPlots[i].drawLists[j].c_str());
                            if (violinPlotDLIdxInListForHistComparison[i] != -1){
                                ImVec2 textPosCurr = textPos;
                                textPosCurr.y += 1.1*ImGui::GetTextLineHeight();
                                ImGui::SetCursorScreenPos(textPosCurr);
                                ImGui::Text(std::to_string(violinDrawlistPlots[i].histDistToRepresentative[j]).c_str());
                            }

							ImGui::PushClipRect(framePos, framePos + size, false);
							HistogramManager::Histogram& hist = histogramManager->getHistogram(violinDrawlistPlots[i].drawLists[j]);
							DrawList* dl = nullptr;
							if (violinYScale == ViolinYScaleLocalBrush || violinYScale == ViolinYScaleBrushes) {
								for (DrawList& draw : g_PcPlotDrawLists) {
									if (draw.name == violinDrawlistPlots[i].drawLists[j]) {
										dl = &draw;
									}
								}
							}
							//std::vector<std::pair<float, float>> localMinMax = std::vector<std::pair<float,float>>(pcAttributes.size(), { std::numeric_limits<float>().max(),std::numeric_limits<float>().min() });
							if (violinYScale == ViolinYScaleLocalBrush || violinYScale == ViolinYScaleBrushes) {
								for (int k = 0; k < pcAttributes.size(); ++k) {
									for (int mi = 0; mi < dl->brushes[k].size(); ++mi) {
										if (dl->brushes[k][mi].minMax.first < localMinMax[k].first) localMinMax[k].first = dl->brushes[k][mi].minMax.first;
										if (dl->brushes[k][mi].minMax.second > localMinMax[k].second) localMinMax[k].second = dl->brushes[k][mi].minMax.second;
									}
								}
								for (int k = 0; k < pcAttributes.size(); ++k) {
									if (localMinMax[k].first == std::numeric_limits<float>().max()) {
										localMinMax[k].first = pcAttributes[k].min;
										localMinMax[k].second = pcAttributes[k].max;
									}
								}
							}
							for (int k : violinDrawlistPlots[i].attributeOrder[j]) {
								if (!violinDrawlistPlots[i].activeAttributes[k]) continue;

								float histYStart;
								float histYEnd;
								switch (violinYScale) {
								case ViolinYScaleStandard:
									histYStart = 0;
									histYEnd = size.y;
									break;
								case ViolinYScaleLocalBrush:
									histYStart = ((hist.ranges[k].second - localMinMax[k].second) / (localMinMax[k].first - localMinMax[k].second) * size.y);
									histYEnd = ((hist.ranges[k].first - localMinMax[k].second) / (localMinMax[k].first - localMinMax[k].second) * size.y);
									break;
								case ViolinYScaleGlobalBrush:
									histYStart = ((hist.ranges[k].second - globalMinMax[k].second) / (globalMinMax[k].first - globalMinMax[k].second) * size.y);
									histYEnd = ((hist.ranges[k].first - globalMinMax[k].second) / (globalMinMax[k].first - globalMinMax[k].second) * size.y);
									break;
								case ViolinYScaleBrushes:
									float min = (localMinMax[k].first < globalMinMax[k].first) ? localMinMax[k].first : globalMinMax[k].first;
									float max = (localMinMax[k].second > globalMinMax[k].second) ? localMinMax[k].second : globalMinMax[k].second;
									histYStart = ((hist.ranges[k].second - max) / (min - max) * size.y);
									histYEnd = ((hist.ranges[k].first - max) / (min - max) * size.y);
									break;
								}
								//if (dl && dl->brushes[k].size()) {
								//	float max = dl->brushes[k][0].minMax.second;
								//	float min = dl->brushes[k][0].minMax.first;
								//	for (int mi = 1; mi < dl->brushes[k].size(); ++mi) {
								//		if (dl->brushes[k][mi].minMax.first < min) min = dl->brushes[k][mi].minMax.first;
								//		if (dl->brushes[k][mi].minMax.second > max) max = dl->brushes[k][mi].minMax.second;
								//	}
								//	histYStart = ((hist.ranges[k].second - max) / (min - max) * size.y);
								//	histYEnd = ((hist.ranges[k].first - max) / (min - max) * size.y);
								//}
								//else {
								//	histYStart = ((hist.ranges[k].second - pcAttributes[k].max) / (pcAttributes[k].min - pcAttributes[k].max) * size.y);
								//	histYEnd = ((hist.ranges[k].first - pcAttributes[k].max) / (pcAttributes[k].min - pcAttributes[k].max) * size.y);
								//}
								//if (!yScaleToCurrenMax) {
								//	histYStart = 0;
								//	histYEnd = size.y;
								//}
								float histYFillStart = (histYStart < 0) ? 0 : histYStart;
								float histYFillEnd = (histYEnd > size.y) ? size.y : histYEnd;
								float histYLineStart = histYStart + framePos.y;
								float histYLineEnd = histYEnd + framePos.y;
								float histYLineDiff = histYLineEnd - histYLineStart;

								float div = 0;
								switch (violinDrawlistPlots[i].violinScalesX[k]) {
								case ViolinScaleSelf:
									div = hist.maxCount[k];
									break;
								case ViolinScaleLocal:
									//div = hist.maxGlobalCount;
									div = determineViolinScaleLocalDiv(hist.maxCount, &(violinDrawlistPlots[i].activeAttributes), violinDrawlistPlots[i].attributeScalings);
									break;
								case ViolinScaleGlobal:
									div = violinDrawlistPlots[i].maxGlobalValue;
									break;
								case ViolinScaleGlobalAttribute:
									div = violinDrawlistPlots[i].maxValues[k];
									break;
								}

								div /= violinDrawlistPlots[i].attributeScalings[k];

								switch (violinDrawlistPlots[i].attributePlacements[k]) {
								case ViolinLeftHalf:
								{
									div *= 2;
								}
								case ViolinLeft:
								{
									//filling
									if (drawState == ViolinDrawStateArea || drawState == ViolinDrawStateAll) {
										hist.binsRendered[k].clear();
										hist.areaRendered[k] = 0;
										for (int p = histYFillStart; p < histYFillEnd; ++p) {
											ImVec2 a(framePos.x, framePos.y + p);
											float v = getBinVal(((1 - (p + .5f) + histYEnd) / (histYEnd - histYStart)), hist.bins[k]);
											ImVec2 b(framePos.x + v / div * size.x, framePos.y + p + 1);
											hist.binsRendered[k].push_back(std::abs(b.x - a.x));
											hist.areaRendered[k] += std::abs(b.x - a.x);
											if (b.x - a.x >= 1)
												ImGui::GetWindowDrawList()->AddRectFilled(a, b, ImColor(violinDrawlistPlots[i].attributeFillColors[k]));
										}
									}
									//outline
									if (drawState == ViolinDrawStateLine || drawState == ViolinDrawStateAll) {
										ImGui::GetWindowDrawList()->AddLine(ImVec2(framePos.x, histYLineEnd),
											ImVec2(framePos.x + hist.bins[k][0] / div * size.x, histYLineEnd), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
										for (int l = 1; l < hist.bins[k].size(); ++l) {
											ImGui::GetWindowDrawList()->AddLine(ImVec2(framePos.x + hist.bins[k][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[k].size() - 1) * histYLineDiff),
												ImVec2(framePos.x + hist.bins[k][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[k].size() - 1) * histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
										}
										ImGui::GetWindowDrawList()->AddLine(ImVec2(framePos.x, histYLineEnd - histYLineDiff),
											ImVec2(framePos.x + hist.bins[k][hist.bins[k].size() - 1] / div * size.x, histYLineEnd - histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
									}
									break;
								}
								case ViolinRightHalf:
								{
									div *= 2;
								}
								case ViolinRight:
								{
									//filling
									if (drawState == ViolinDrawStateArea || drawState == ViolinDrawStateAll) {
										hist.binsRendered[k].clear();
										hist.areaRendered[k] = 0;
										for (int p = histYFillStart; p < histYFillEnd; ++p) {
											ImVec2 a(framePos.x + size.x, framePos.y + p);
											float v = getBinVal(((1 - (p + .5f) + histYEnd) / (histYEnd - histYStart)), hist.bins[k]);
											ImVec2 b(framePos.x + size.x - v / div * size.x, framePos.y + p + 1);
											hist.binsRendered[k].push_back(std::abs(b.x - a.x));
											hist.areaRendered[k] += std::abs(b.x - a.x);
											if (a.x - b.x >= 1)
												ImGui::GetWindowDrawList()->AddRectFilled(a, b, ImColor(violinDrawlistPlots[i].attributeFillColors[k]));
										}
									}
									//outline
									if (drawState == ViolinDrawStateLine || drawState == ViolinDrawStateAll) {
										ImGui::GetWindowDrawList()->AddLine(ImVec2(framePos.x + size.x, histYLineEnd),
											ImVec2(framePos.x + size.x - hist.bins[k][0] / div * size.x, histYLineEnd), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
										for (int l = 1; l < hist.bins[k].size(); ++l) {
											ImGui::GetWindowDrawList()->AddLine(ImVec2(framePos.x + size.x - hist.bins[k][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[k].size() - 1) * histYLineDiff),
												ImVec2(framePos.x + size.x - hist.bins[k][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[k].size() - 1) * histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
										}
										ImGui::GetWindowDrawList()->AddLine(ImVec2(framePos.x + size.x, histYLineEnd - histYLineDiff),
											ImVec2(framePos.x + size.x - hist.bins[k][hist.bins[k].size() - 1] / div * size.x, histYLineEnd - histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
									}
									break;
								}
								case ViolinMiddle:
								{
									float xBase = framePos.x + .5f * size.x;
									//filling
									if (drawState == ViolinDrawStateArea || drawState == ViolinDrawStateAll) {
										hist.binsRendered[k].clear();
										hist.areaRendered[k] = 0;
										for (int p = histYFillStart; p < histYFillEnd; ++p) {
											float v = getBinVal(((1 - (p + .5f) + histYEnd) / (histYEnd - histYStart)), hist.bins[k]);
											ImVec2 a(xBase - .5f * v / div * size.x, framePos.y + p);
											ImVec2 b(xBase + .5f * v / div * size.x, framePos.y + p + 1);
											hist.binsRendered[k].push_back(std::abs(b.x - a.x));
											hist.areaRendered[k] += std::abs(b.x - a.x);
											if (b.x - a.x >= 1)
												ImGui::GetWindowDrawList()->AddRectFilled(a, b, ImColor(violinDrawlistPlots[i].attributeFillColors[k]));
										}
									}
									if (drawState == ViolinDrawStateLine || drawState == ViolinDrawStateAll) {
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase + hist.bins[k][0] / div * size.x / 2, histYLineEnd),
											ImVec2(xBase - hist.bins[k][0] / div * size.x / 2, histYLineEnd), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
										for (int l = 1; l < hist.bins[k].size(); ++l) {
											//left Line
											ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase - .5f * hist.bins[k][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[k].size() - 1) * histYLineDiff),
												ImVec2(xBase - .5f * hist.bins[k][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[k].size() - 1) * histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
											//right Line
											ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase + .5f * hist.bins[k][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[k].size() - 1) * histYLineDiff),
												ImVec2(xBase + .5f * hist.bins[k][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[k].size() - 1) * histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
										}
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase + hist.bins[k][hist.bins[k].size() - 1] / div * size.x / 2, histYLineEnd - histYLineDiff),
											ImVec2(xBase - hist.bins[k][hist.bins[k].size() - 1] / div * size.x / 2, histYLineEnd - histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
									}
									break;
								}
								case ViolinMiddleLeft:
								{
									float xBase = framePos.x + .5f * size.x;
									//filling
									if (drawState == ViolinDrawStateArea || drawState == ViolinDrawStateAll) {
										hist.binsRendered[k].clear();
										hist.areaRendered[k] = 0;
										for (int p = histYFillStart; p < histYFillEnd; ++p) {
											float v = getBinVal(((1 - (p + .5f) + histYEnd) / (histYEnd - histYStart)), hist.bins[k]);
											ImVec2 a(xBase - .5f * v / div * size.x, framePos.y + p);
											ImVec2 b(xBase, framePos.y + p + 1);
											hist.binsRendered[k].push_back(std::abs(b.x - a.x));
											hist.areaRendered[k] += std::abs(b.x - a.x);
											if (b.x - a.x >= 1)
												ImGui::GetWindowDrawList()->AddRectFilled(a, b, ImColor(violinDrawlistPlots[i].attributeFillColors[k]));
										}
									}
									if (drawState == ViolinDrawStateLine || drawState == ViolinDrawStateAll) {
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase, histYLineEnd),
											ImVec2(xBase - hist.bins[k][0] / div * size.x / 2, histYLineEnd), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
										for (int l = 1; l < hist.bins[k].size(); ++l) {
											//left Line
											ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase - .5f * hist.bins[k][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[k].size() - 1) * histYLineDiff),
												ImVec2(xBase - .5f * hist.bins[k][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[k].size() - 1) * histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
										}
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase, histYLineEnd - histYLineDiff),
											ImVec2(xBase - hist.bins[k][hist.bins[k].size() - 1] / div * size.x / 2, histYLineEnd - histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
										//right Line
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase, framePos.y), ImVec2(xBase, framePos.y + size.y), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
									}
									break;
								}
								case ViolinMiddleRight:
								{
									float xBase = framePos.x + .5f * size.x;
									//filling
									if (drawState == ViolinDrawStateArea || drawState == ViolinDrawStateAll) {
										hist.binsRendered[k].clear();
										hist.areaRendered[k] = 0;
										for (int p = histYFillStart; p < histYFillEnd; ++p) {
											float v = getBinVal(((1 - (p + .5f) + histYEnd) / (histYEnd - histYStart)), hist.bins[k]);
											ImVec2 a(xBase, framePos.y + p);
											ImVec2 b(xBase + .5f * v / div * size.x, framePos.y + p + 1);
											hist.binsRendered[k].push_back(std::abs(b.x - a.x));
											hist.areaRendered[k] += std::abs(b.x - a.x);
											if (b.x - a.x >= 1)
												ImGui::GetWindowDrawList()->AddRectFilled(a, b, ImColor(violinDrawlistPlots[i].attributeFillColors[k]));
										}
									}
									if (drawState == ViolinDrawStateLine || drawState == ViolinDrawStateAll) {
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase + hist.bins[k][0] / div * size.x / 2, histYLineEnd),
											ImVec2(xBase, histYLineEnd), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
										for (int l = 1; l < hist.bins[k].size(); ++l) {
											//right Line
											ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase + .5f * hist.bins[k][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[k].size() - 1) * histYLineDiff),
												ImVec2(xBase + .5f * hist.bins[k][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[k].size() - 1) * histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
										}
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase + hist.bins[k][hist.bins[k].size() - 1] / div * size.x / 2, histYLineEnd - histYLineDiff),
											ImVec2(xBase, histYLineEnd - histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
										//left Line
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase, framePos.y), ImVec2(xBase, framePos.y + size.y), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotThickness);
									}
									break;
								}
								}
							}
							optimizeViolinSidesAndAssignCustColors();
							leftUpperCorner.x += size.x + violinPlotXSpacing;
							ImGui::PopClipRect();
						}
						leftUpperCorner.x = leftUpperCornerStart.x;
						leftUpperCorner.y += size.y + ImGui::GetFrameHeightWithSpacing();
					}
					
					if (drawState == ViolinDrawStateAll || drawState == ViolinDrawStateLine) done = true;
					if (drawState == ViolinDrawStateArea) drawState = ViolinDrawStateLine;
				}
				ImGui::PopItemWidth();
				ImGui::EndChild();
				//drag and drop drawlists onto this plot child to add it to this violin plot
				if (ImGui::BeginDragDropTarget()) {
					if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Drawlist")) {
						DrawList* dl = *((DrawList**)payload->Data);
						//check if the drawlist was already added to this plot
						if (std::find(violinDrawlistPlots[i].drawLists.begin(), violinDrawlistPlots[i].drawLists.end(), dl->name) == violinDrawlistPlots[i].drawLists.end()) {
							if (!violinDrawlistPlots[i].attributeNames.size()) {	//creating all needed resources e.g. attribute components
								violinDrawlistPlots[i].activeAttributes = new bool[pcAttributes.size()];
								violinDrawlistPlots[i].maxGlobalValue = 0;
								int j = 0;
								for (Attribute& a : pcAttributes) {
									violinDrawlistPlots[i].attributeNames.push_back(a.name);
									violinDrawlistPlots[i].activeAttributes[j] = true;
									violinDrawlistPlots[i].attributeLineColors.push_back({ 0,0,0,1 });
									violinDrawlistPlots[i].attributeFillColors.push_back({ .5f,.5f,.5f,.5f });
									violinDrawlistPlots[i].attributePlacements.push_back((j % 2) ? ViolinMiddleLeft : ViolinMiddleRight);
									violinDrawlistPlots[i].attributeScalings.push_back(1);
									violinDrawlistPlots[i].violinScalesX.push_back(ViolinScaleGlobalAttribute);
									violinDrawlistPlots[i].maxValues.push_back(0);
									++j;
								}
							}

							std::vector<std::pair<float, float>> minMax;
							for (Attribute& a : pcAttributes) {
								minMax.push_back({ a.min,a.max });
							}
							DataSet* ds;
							for (DataSet& d : g_PcPlotDataSets) {
								if (d.name == dl->parentDataSet) {
									ds = &d;
									break;
								}
							}
							exeComputeHistogram(dl->name, minMax, dl->buffer, ds->data.size(), dl->indicesBuffer, dl->indices.size(), dl->activeIndicesBufferView);
                            updateHistogramComparisonDL(i);
							//histogramManager->computeHistogramm(dl->name, minMax, dl->buffer, ds->data.size(), dl->indicesBuffer, dl->indices.size(), dl->activeIndicesBufferView);
							HistogramManager::Histogram& hist = histogramManager->getHistogram(dl->name);
							std::vector<std::pair<uint32_t, float>> area;
							for (int j = 0; j < hist.maxCount.size(); ++j) {
								if (hist.maxCount[j] > violinDrawlistPlots[i].maxValues[j]) {
									violinDrawlistPlots[i].maxValues[j] = hist.maxCount[j];
								}
								if (hist.maxCount[j] > violinDrawlistPlots[i].maxGlobalValue) {
									violinDrawlistPlots[i].maxGlobalValue = hist.maxCount[j];
								}
								area.push_back({ j, violinDrawlistPlots[i].attributeScalings[j] / hist.maxCount[j] });
							}

							violinDrawlistPlots[i].drawLists.push_back(dl->name);
							//violinDrawlistPlots[i].drawListOrder.push_back(violinDrawlistPlots[i].drawListOrder.size());
							violinDrawlistPlots[i].attributeOrder.push_back({});
							if (renderOrderDLConsider) {

								violinDrawlistPlots[i].attributeOrder.back() = sortHistogram(hist, violinDrawlistPlots[i], renderOrderDLConsider, renderOrderDLReverse);

								/*if (!renderOrderDLReverse) {
									std::sort(area.begin(), area.end(), [](std::pair<uint32_t, float>& a, std::pair<uint32_t, float>& b) {return sortDescPair(a, b); });
								}
								else
								{
									std::sort(area.begin(), area.end(), [](std::pair<uint32_t, float>& a, std::pair<uint32_t, float>& b) {return sortAscPair(a, b); });
								}*/
							}
							else {
								for (int j = 0; j < pcAttributes.size(); ++j)violinDrawlistPlots[i].attributeOrder.back().push_back(area[j].first);
							}
						}
					}
					ImGui::EndDragDropTarget();
				}
			}
			
			//adding new Plots
			ImGui::SetCursorPosX(ImGui::GetWindowWidth() / 2 - plusWidth / 2);
			if (ImGui::Button("+", ImVec2(plusWidth, 0))) {
				//ViolinDrawlistPlot *currVPDLP = new ViolinDrawlistPlot();
				violinDrawlistPlots.emplace_back();//*currVPDLP);
				//currVPDLP = nullptr;
				//operator delete(currVPDLP);
				violinDrawlistPlots.back().matrixSize = { 1,5 };
				violinDrawlistPlots.back().drawListOrder = std::vector<uint32_t>(5, 0xffffffff);
                violinPlotDLIdxInListForHistComparison.push_back(-1);
			}
			ImGui::End();
		}

		if (animationItemsDisabled) {
			ImGui::PopItemFlag();
			animationItemsDisabled = false;
		}

		// Rendering
		ImGui::Render();
		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault();
		memcpy(&wd->ClearValue.color.float32[0], &clear_color, 4 * sizeof(float));
		FrameRender(wd);

		//FramePresent(wd);
		rescaleTableColumns = false;
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
	for (ViolinPlot& vp : violinAttributePlots) {
		if (vp.activeAttributes) delete[] vp.activeAttributes;
	}
	for (ViolinDrawlistPlot& vp : violinDrawlistPlots) {
		if (vp.activeAttributes) delete[] vp.activeAttributes;
	}


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
#ifdef BUBBLEVIEW
		delete bubblePlotter;
#endif
		delete brushIsoSurfaceRenderer;
		delete isoSurfaceRenderer;
		delete settingsManager;
		delete gpuBrusher;
		delete histogramManager;

		for (GlobalBrush& gb : globalBrushes) {
			if (gb.kdTree) delete gb.kdTree;
		}
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
