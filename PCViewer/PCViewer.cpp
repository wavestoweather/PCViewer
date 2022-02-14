/*
This program is written and maintained by Josef Stumpfegger (josefstumpfegger@outlook.de) and Alexander Kumpf (alexander.kumpf@tum.de)
As this program originally was not written with the intend of being published, the code-basis is not the most beautiful one, we are sorry.
Should you find errors, problems, speed up ideas or anything else, dont be shy and contact either of us!
Other than that, we wish you a beautiful day and a lot of fun with this program.
*/

//memory leak detection
#ifdef _DEBUG
#define DETECTMEMLEAK
#endif

//enable this define to print the time needed to render the pc Plot
//#define PRINTRENDERTIME
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
#include "Structures.hpp"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl.h"
#include "imgui/imgui_impl_vulkan.h"
#include "imgui/imgui_internal.h"
#include "cimg/CImg.h"
#ifdef Success
#undef Success
#endif
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
#include "TransferFunctionEditor.h"
#include "Data.hpp"
#include "DrawlistColorPalette.hpp"
#include "LineBundles.hpp"
#include "ClusteringWorkbench.hpp"
#include "ScatterplotWorkbench.hpp"
#include "CorrelationMatrixWorkbench.hpp"
#include "GpuRadixSorter.hpp"
#include "PCRenderer.hpp"
#include "compression/CompressionWorkbench.hpp"
#include "compression/HierarchyImportManager.hpp"

#include "ColorPalette.h"
#include "ColorMaps.hpp"

#include <stdio.h>          // printf, fprintf
#include <stdlib.h>         // abort
#include <SDL.h>
#include <SDL_vulkan.h>
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
#include <iomanip>
#include <sstream>
#include <utility>
#include <netcdf.h>
#include <filesystem>
#include <memory>
#include <optional>
	

#ifdef DETECTMEMLEAK
#define new new( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#endif

//important vector holding all supported file formats
std::vector<std::string> supportedDataFormats{ ".nc", ".csv", ".idxf", ".dlf" };

// debug print settings
std::vector<std::string> debugLevels{"NoInfos", "Error", "ErrorWarning", "ErrorWarningInfo"};
static int debugLevel = 3;

//defines for key ids
#define KEYW 26
#define KEYA 4
#define KEYS 22
#define KEYD 7
#define KEYQ 20
#define KEYE 8
#define KEYP 19
#define KEYENTER 40
#define KEYESC 41

//defines for the medians
#define MEDIANCOUNT 3
#define MEDIAN 0
#define ARITHMEDIAN 1
#define GOEMEDIAN 2

#define BRUSHWIDTH 20
#define EDGEHOVERDIST 5
#define DRAGTHRESH .02f

#define SCROLLSPEED .04f

//defines the amount of fractures per axis
#define FRACTUREDEPTH 15

//marked line thickness in violin plots
#define LINEDISTANCE 5
#define LINEMULTIPLIER 2

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

//#define IMGUI_UNLIMITED_FRAME_RATE
#define _DEBUG
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
static ImGui_ImplVulkanH_Frame  g_ExportWindowFrame;
static VkRenderPass				g_ExportWindowRenderPass;
static VkDeviceMemory			g_ExportWindowMemory;
static int                      g_MinImageCount = 2;
static bool                     g_SwapChainRebuild = false;
static int                      g_SwapChainResizeWidth = 0;
static int                      g_SwapChainResizeHeight = 0;
static float					g_ExportScale = 2.0f;
static int						g_ExportImageWidth = 1280 * g_ExportScale;
static int						g_ExportImageHeight = 720 * g_ExportScale;
static int						g_ExportCountDown = -1;					//amount of frames until export is done (used to let menu bars close, plots render ...), -1 when disabled
static int						g_ExportViewportNumber = 0;
static char						g_ExportPath[200] = "export.png";
static long						g_MaxStorageBufferSize = 0;

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

static std::vector<QueryAttribute> queryAttributes;
static bool queryAttributesCsv;		//indicates if the quried attributes are from a csv file (Only subsampling is available, as the table size is not known prior to read out)

struct Vertex {			//currently holds just the y coordinate. The x computed in the vertex shader via the index
	float y;
};

struct RectVertex {		//struct which describes the vertecies for the rects
	Vec4 pos;
	Vec4 col;
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
	std::string nam;
	//std::string name;									//the name of a global brush describes the template list it was created from and more...
	std::string id;										//id of the brush
	std::map<std::string, int> lineRatios;			//contains the ratio of still active lines per drawlist
	std::map<int, std::vector<std::pair<unsigned int, std::pair<float, float>>>> brushes;	//for every brush that exists, one entry in this map exists, where the key is the index of the Attribute in the pcAttributes vector and the pair describes the minMax values
};

struct TemplateBrush {
	std::string name;									//identifier for the template brush
	TemplateList* parent;
	DataSet* parentDataSet;
	std::map<int, std::pair<float, float>> brushes;
};



enum PCViewerState {
	Normal,
	AnimateDrawlists,
	AnimateDrawlistsExport,
	AnimateGlobalBrush,
	AnimateGlobalBrushExport
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

struct ViolinPlot {
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
	std::vector<std::vector<float>> summedBins;		//summed bin values
	std::vector<float> maxSummedValues;				//max value for summed bins

    ColorPaletteManager *colorPaletteManager;
};

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

static VkDeviceMemory			g_PcPlotMem = VK_NULL_HANDLE;
static VkImage					g_PcPlot = VK_NULL_HANDLE;
static VkImageView				g_PcPlotView = VK_NULL_HANDLE;
static VkSampler				g_PcPlotSampler = VK_NULL_HANDLE;
static VkDescriptorSet			g_PcPlotImageDescriptorSet = VK_NULL_HANDLE;
static VkRenderPass				g_PcPlotRenderPass = VK_NULL_HANDLE;		//contains the render pass for the pc
static VkRenderPass				g_PcPlotRenderPass_noClear = VK_NULL_HANDLE;
static VkDescriptorSetLayout	g_PcPlotDescriptorLayout = VK_NULL_HANDLE;
static VkDescriptorSetLayout	g_PcPlotDataSetLayout = VK_NULL_HANDLE;
static VkDescriptorPool			g_PcPlotDescriptorPool = VK_NULL_HANDLE;
static VkDescriptorSet			g_PcPlotDescriptorSet = VK_NULL_HANDLE;
static VkBuffer					g_PcPlotDescriptorBuffer = VK_NULL_HANDLE;
static VkDeviceMemory			g_PcPlotDescriptorBufferMemory = VK_NULL_HANDLE;
static VkPipelineLayout			g_PcPlotPipelineLayout = VK_NULL_HANDLE;	//contains the pipeline which is used to assign global shader variables
static VkPipeline				g_PcPlotPipeline = VK_NULL_HANDLE;			//contains the graphics pipeline for the pc
//variables for spline pipeline
static VkPipelineLayout			g_PcPlotSplinePipelineLayout = VK_NULL_HANDLE;
static VkPipeline				g_PcPlotSplinePipeline = VK_NULL_HANDLE;
static VkPipelineLayout			g_PcPlotSplinePipelineLayout_noClear = VK_NULL_HANDLE;
static VkPipeline				g_PcPlotSplinePipeline_noClear = VK_NULL_HANDLE;
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
static VkFramebuffer			g_PcPlotFramebuffer_noClear = VK_NULL_HANDLE;
static VkCommandPool			g_PcPlotCommandPool = VK_NULL_HANDLE;
static VkCommandBuffer			g_PcPlotCommandBuffer = VK_NULL_HANDLE;
static VkFence					g_PcPlotRenderFence = VK_NULL_HANDLE;
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

static bool* pcAttributeEnabled = NULL;											//Contains whether a specific attribute is enabled
static std::vector<Attribute> pcAttributes = std::vector<Attribute>();			//Contains the attributes and its bounds	
static std::vector<int> pcAttributesSorted;
static std::vector<int> pcAttrOrd = std::vector<int>();							//Contains the ordering of the attributes	
static std::vector<std::string> droppedPaths = std::vector<std::string>();
static std::vector<std::string> recentFiles;
static int recentFilesAmt = 10;
static std::vector<uint8_t> droppedPathActive;
static bool* createDLForDrop = NULL;
static bool pathDropped = false;
static std::default_random_engine engine;
static std::uniform_int_distribution<int> distribution(0, 35);
static std::vector<int> pcPlotSelectedDrawList;									//Contains the index of the drawlist that is currently selected
static std::map<std::string, std::pair<bool,std::vector<float>>> dimensionValues;//Contains the dimension values for each dimension. Dimension values SHOULD be read using the getDimensionValues method to secure that the dimension values for an attribute is available

static bool atomicGpuFloatAddAvailable{};

static PCViewerState pcViewerState = PCViewerState::Normal;
struct PCSettings {
	bool autoAlpha = true;
	float alphaDrawLists = .5f;
	Vec4 PcPlotBackCol = { 0,0,0,1 };
	bool enableAxisLines = true;
	bool enableZeroTick = true;
	int axisTickAmount = 10;
	int axisTickWidth = 5;
	bool createDefaultOnLoad = true;
	bool rescaleTableColumns = true;

	//variables for the histogramm
	float histogrammWidth = .1f;
	bool drawHistogramm = false;
	bool normaliseHistogramm = false;
	bool adustHistogrammByActiveLines = true;
	bool computeRatioPtsInDLvsIn1axbrushedParent = false;
	bool histogrammDensity = true;
	bool pcPlotDensity = false;
	float densityRadius = .005f;
	bool enableDensityMapping = true;
	bool enableDensityGreyscale = false;
	bool calculateMedians = true;
	bool mapDensity = true;
	int histogrammDrawListComparison = -1;
	Vec4 histogrammBackCol = { .2f,.2f,.2,1 };
	Vec4 densityBackCol = { 0,0,0,1 };
	float medianLineWidth = 1.0f;
	bool enableBrushing = false;

	//variables for brush templates
	bool brushTemplatesEnabled = true;
	bool showCsvTemplates = false;
	bool updateBrushTemplates = false;
	bool drawListForTemplateBrush = false;
	int liveBrushThreshold = 5e5;
	int lineBatchSize = 2e6;

	//variables for global brushes
	bool toggleGlobalBrushes = true;
	int brushCombination = 0;				//How global brushes should be combined. 0->OR, 1->AND
	float brushMuFactor = .001f;				//factor to add a mu to the bounds of a brush

	//variables for fractions
	int maxFractionDepth = 24;
	int outlierRank = 11;					//min rank(amount ofdatapoints in a kd tree node) needed to not be an outlier node
	int boundsBehaviour = 2;
	int splitBehaviour = 1;
	int maxRenderDepth = 13;
	float fractionBoxWidth = BRUSHWIDTH;
	int fractionBoxLineWidth = 3;
	float multivariateStdDivThresh = 1.0f;

	bool renderSplines = true;

	//variables for animation
	float animationDuration = 2.0f;		//time for every active brush to show in seconds
	bool animationExport = true;
	int animationSteps = 2;

	//variables for band rendering
	float haloWidth = .1f;

	//variables for hierarchical data
	uint32_t maxHierarchyLines = 1000000;
}static pcSettings;

//variables for brush templates
static bool* brushTemplateAttrEnabled = NULL;
static int selectedTemplateBrush = -1;
static std::vector<TemplateBrush> templateBrushes;

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

//variables for priority rendering
static int priorityAttribute = -1;
static float priorityAttributeCenterValue = 0;
static bool prioritySelectAttribute = false;
static bool priorityReorder = false;
static int priorityListIndex = 0;

//variables for the 3d views
struct View3dSettings {
	bool enabled = false;
	int activeAttribute = -1;
	uint32_t posIndices[3]{ 1,0,2 };
}static view3dSettings;
static View3d* view3d;

struct BubbleWindowSettings {
	bool enabled = false;
	bool coupleToBrushing = true;
	int regularGridDim[3]{ 51,30,81 };
	glm::uvec3 posIndices{ 1,0,2 };
}static bubbleWindowSettings;

static BubblePlotter* bubblePlotter;

static SettingsManager* settingsManager;
static DrawlistColorPalette* drawListColorPalette;

static GpuBrusher* gpuBrusher;

static HistogramManager* histogramManager;

static IsoSurfRenderer* isoSurfaceRenderer = nullptr;
static BrushIsoSurfRenderer* brushIsoSurfaceRenderer = nullptr;
struct IsoSettings {
	bool enabled = false;
	bool coupleIsoSurfaceRenderer = true;
	bool coupleBrushIsoSurfaceRenderer = true;
	glm::uvec3 posIndices{ 1,0,2 };
}static brushIsoSurfSettings;
static IsoSettings isoSurfSettings;

static ScatterplotWorkbench* scatterplotWorkbench;

static std::unique_ptr<CorrelationMatrixWorkbench> correlationMatrixWorkbench;

//variables for animation
static std::chrono::steady_clock::time_point animationStart(std::chrono::duration<int>(0));
static bool* animationActiveDatasets = nullptr;
static bool animationItemsDisabled = false;
static int animationCurrentDrawList = -1;
static char animationExportPath[200] = "export/test%d.png";
static int animationBrush = -1;
static int animationCurrentStep = -1;
static int animationAttribute = -1;
static std::vector<std::pair<unsigned int, std::pair<float, float>>> animationAttributeBrush;

typedef struct {
	bool optimizeSidesNowAttr = false;
	bool optimizeSidesNowDL = false;
	ViolinDrawlistPlot *vdlp = nullptr;
	ViolinPlot *vp = nullptr;
} AdaptViolinSidesAutoStruct;

//variables for violin plots
static float violinPlotBinsSize = 150;
struct ViolinSettings {
	int violinPlotHeight = 1000;//550;
	int violinPlotXSpacing = 15;
	int violinPlotAttrStacking = 0;
	bool enabled = false;
	float violinPlotThickness = 1;
	ImVec4 violinBackgroundColor = { 1,1,1,1 };
	bool coupleViolinPlots = true;
	bool showViolinPlotsMinMax = true;
	bool violinPlotDLSendToIso = true;
	bool violinPlotDLInsertCustomColors = true;
	bool violinPlotAttrInsertCustomColors = true;
	bool violinPlotAttrReplaceNonStop = false;
	bool violinPlotAttrConsiderBlendingOrder = true;
	bool violinPlotDLConsiderBlendingOrder = true;
	bool violinPlotDLReplaceNonStop = false;
	bool violinPlotAttrReverseColorPallette = false;
	bool violinPlotDLReverseColorPallette = false;
	int autoColorAssingFill = 4;
	int autoColorAssingLine = 4;

	bool yScaleToCurrenMax = false;
	bool violinPlotOverlayLines = true;
	bool renderOrderBasedOnFirstAtt = true;
	bool renderOrderBasedOnFirstDL = true;
	bool renderOrderAttConsider = true;
	bool renderOrderDLConsider = true;
	bool renderOrderAttConsiderNonStop = true;
	bool renderOrderDLConsiderNonStop = true;

	bool renderOrderDLReverse = false;
	bool logScaleDLGlobal = false;
	ViolinYScale violinYScale = ViolinYScaleStandard;
}static violinPlotAttributeSettings;
static ViolinSettings violinPlotDrawlistSettings;

static std::vector<int> violinPlotDLIdxInListForHistComparison;
static bool violinPlotDLUseRenderedBinsForHistComp = false;
static std::string ttempStr = "abc";

std::vector<ViolinPlot> violinAttributePlots;
std::vector<ViolinDrawlistPlot> violinDrawlistPlots;
AdaptViolinSidesAutoStruct violinAdaptSidesAutoObj;

static TransferFunctionEditor* transferFunctionEditor;
static std::shared_ptr<ClusteringWorkbench> clusteringWorkbench;
static std::shared_ptr<PCRenderer> pcRenderer;
static std::shared_ptr<CompressionWorkbench> compressionWorkbench;

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

static float intArrayGetter(void* data, int idx)
{
    int * arr = (int*)data;
    return float(arr[idx]);
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

	VkUtil::createPipeline(g_Device, &vertexInputInfo, g_PcPlotWidth, g_PcPlotHeight, dynamicStates, shaderModules, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &g_PcPlotRenderPass, &g_PcPlotHistoPipelineLayout, &g_PcPlotHistoPipeline);

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

	VkUtil::createPipeline(g_Device, &vertexInputInfo, g_PcPlotWidth, g_PcPlotHeight, dynamicStates, shaderModules, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &g_PcPlotRenderPass, &g_PcPlotHistoPipelineAdditiveLayout, &g_PcPlotHistoAdditivePipeline);

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
	VkUtil::createImage(g_Device, sizeof(heatmap) / sizeof(*heatmap) / 4, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &g_PcPlotDensityIronMap);
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
	VkUtil::createBuffer(g_Device, sizeof(heatmap), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &stagingBuffer);

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
	vkMapMemory(g_Device, stagingBufferMemory, 0, sizeof(heatmap), 0, &d);
	memcpy(d, heatmap, sizeof(heatmap));
	vkUnmapMemory(g_Device, stagingBufferMemory);

	VkCommandBuffer stagingCommandBuffer;
	VkUtil::createCommandBuffer(g_Device, g_PcPlotCommandPool, &stagingCommandBuffer);

	VkUtil::transitionImageLayout(stagingCommandBuffer, g_PcPlotDensityIronMap, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	VkUtil::copyBufferToImage(stagingCommandBuffer, stagingBuffer, g_PcPlotDensityIronMap, sizeof(heatmap) / sizeof(*heatmap) / 4, 1);
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

static void cleanupExportWindow() {
	//ImGui_ImplVulkanH_DestroyFrame(g_Device, &g_ExportWindowFrame, g_Allocator);
	vkFreeMemory(g_Device, g_ExportWindowMemory, g_Allocator);
	vkFreeCommandBuffers(g_Device, g_ExportWindowFrame.CommandPool, 1, &g_ExportWindowFrame.CommandBuffer);
	vkDestroyCommandPool(g_Device, g_ExportWindowFrame.CommandPool, g_Allocator);

	vkDestroyImageView(g_Device, g_ExportWindowFrame.BackbufferView, g_Allocator);
	vkDestroyImage(g_Device, g_ExportWindowFrame.Backbuffer, g_Allocator);
	vkDestroyFramebuffer(g_Device, g_ExportWindowFrame.Framebuffer, g_Allocator);
	vkDestroyRenderPass(g_Device, g_ExportWindowRenderPass, g_Allocator);
}

static void recreateExportWindow() {
	if (g_ExportWindowFrame.Backbuffer) {	//destroy old resources
		cleanupExportWindow();
	}

	VkUtil::createRenderPass(g_Device, VkUtil::PASS_TYPE_COLOR_EXPORT, &g_ExportWindowRenderPass);

	VkUtil::createImage(g_Device, g_ExportImageWidth, g_ExportImageHeight, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &g_ExportWindowFrame.Backbuffer);
	VkMemoryRequirements memReq{};
	vkGetImageMemoryRequirements(g_Device, g_ExportWindowFrame.Backbuffer, &memReq);
	VkMemoryAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	allocInfo.allocationSize = memReq.size;
	allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, 0);
	vkAllocateMemory(g_Device, &allocInfo, g_Allocator, &g_ExportWindowMemory);
	vkBindImageMemory(g_Device, g_ExportWindowFrame.Backbuffer, g_ExportWindowMemory, 0);

	VkUtil::createImageView(g_Device, g_ExportWindowFrame.Backbuffer, VK_FORMAT_B8G8R8A8_UNORM, 1, VK_IMAGE_ASPECT_COLOR_BIT, &g_ExportWindowFrame.BackbufferView);

	std::vector<VkImageView> views{ g_ExportWindowFrame.BackbufferView };
	VkUtil::createFrameBuffer(g_Device, g_ExportWindowRenderPass, views, g_ExportImageWidth, g_ExportImageHeight, &g_ExportWindowFrame.Framebuffer);

	VkResult err;
	{
		VkCommandPoolCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		info.queueFamilyIndex = g_QueueFamily;
		err = vkCreateCommandPool(g_Device, &info, g_Allocator, &g_ExportWindowFrame.CommandPool);
		check_vk_result(err);
	}
	{
		VkCommandBufferAllocateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		info.commandPool = g_ExportWindowFrame.CommandPool;
		info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		info.commandBufferCount = 1;
		err = vkAllocateCommandBuffers(g_Device, &info, &g_ExportWindowFrame.CommandBuffer);
		check_vk_result(err);
	}
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
	//vertexInputInfo.vertexBindingDescriptionCount = 1;
	//vertexInputInfo.pVertexBindingDescriptions = &bindingDescripiton;
	//vertexInputInfo.vertexAttributeDescriptionCount = 1;
	//vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

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
	uboLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	uboLayoutBindings[0].descriptorCount = 1;
	uboLayoutBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

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

	uboLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	layoutInfo.bindingCount = 1;
	err = vkCreateDescriptorSetLayout(g_Device, &layoutInfo, nullptr, &g_PcPlotDataSetLayout);
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
	poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	poolInfo.maxSets = 100;

	err = vkCreateDescriptorPool(g_Device, &poolInfo, nullptr, &g_PcPlotDescriptorPool);
	check_vk_result(err);

	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = g_PcPlotDescriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &g_PcPlotDescriptorLayout;

	err = vkAllocateDescriptorSets(g_Device, &allocInfo, &g_PcPlotDescriptorSet);
	check_vk_result(err);

	VkDescriptorSetLayout layouts[2]{g_PcPlotDescriptorLayout, g_PcPlotDataSetLayout};
	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 2;
	pipelineLayoutInfo.pSetLayouts = layouts;
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
	attributeDescription.format = VK_FORMAT_UNDEFINED;
	attributeDescription.offset = 0;

	vertexInputInfo = {};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	//vertexInputInfo.vertexBindingDescriptionCount = 1;
	//vertexInputInfo.pVertexBindingDescriptions = &bindingDescripiton;
	//vertexInputInfo.vertexAttributeDescriptionCount = 1;
	//vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

	VkDescriptorSetLayoutBinding uboLayoutBinding = {};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;

	VkUtil::BlendInfo blendInfo;
	blendInfo.blendAttachment = colorBlendAttachment;
	blendInfo.createInfo = colorBlending;

	std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
	descriptorSetLayouts.push_back(g_PcPlotDescriptorLayout);
	descriptorSetLayouts.push_back(g_PcPlotDataSetLayout);

	std::vector<VkDynamicState> dynamicStateVec;
	dynamicStateVec.push_back(VK_DYNAMIC_STATE_LINE_WIDTH);

	VkUtil::createPipeline(g_Device, &vertexInputInfo, g_PcPlotWidth, g_PcPlotHeight, dynamicStateVec, shaderModules, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &g_PcPlotRenderPass, &g_PcPlotSplinePipelineLayout, &g_PcPlotSplinePipeline);

	//----------------------------------------------------------------------------------------------
	//creating the pipeline for spline rendering without clear values
	//----------------------------------------------------------------------------------------------
	vertexBytes = PCUtil::readByteFile(g_vertShaderPath);
	shaderModules[0] = VkUtil::createShaderModule(g_Device, vertexBytes);
	geometryBytes = PCUtil::readByteFile(g_geomShaderPath);
	shaderModules[3] = VkUtil::createShaderModule(g_Device, geometryBytes);
	fragmentBytes = PCUtil::readByteFile(g_fragShaderPath);
	shaderModules[4] = VkUtil::createShaderModule(g_Device, fragmentBytes);

	VkUtil::createPipeline(g_Device, &vertexInputInfo, g_PcPlotWidth, g_PcPlotHeight, dynamicStateVec, shaderModules, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY, &rasterizer, &multisampling, nullptr, &blendInfo, descriptorSetLayouts, &g_PcPlotRenderPass_noClear, &g_PcPlotSplinePipelineLayout_noClear, &g_PcPlotSplinePipeline_noClear);

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
	vkDestroyDescriptorSetLayout(g_Device, g_PcPlotDataSetLayout, nullptr);
	vkDestroyPipelineLayout(g_Device, g_PcPlotPipelineLayout, nullptr);
	vkDestroyPipeline(g_Device, g_PcPlotPipeline, nullptr);
	vkDestroyPipelineLayout(g_Device, g_PcPlotSplinePipelineLayout, nullptr);
	vkDestroyPipeline(g_Device, g_PcPlotSplinePipeline, nullptr);
	vkDestroyPipelineLayout(g_Device, c_IndexPipelineLayout, nullptr);
	vkDestroyPipeline(g_Device, c_IndexPipeline, nullptr);
	vkDestroyDescriptorSetLayout(g_Device, c_IndexPipelineDescSetLayout, nullptr);
	vkDestroyPipeline(g_Device, g_PcPlotSplinePipeline_noClear, nullptr);
	vkDestroyPipelineLayout(g_Device, g_PcPlotSplinePipelineLayout_noClear, nullptr);
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

	VkUtil::createRenderPass(g_Device, VkUtil::PASS_TYPE_COLOR16_OFFLINE_NO_CLEAR, &g_PcPlotRenderPass_noClear);
}

static void cleanupPcPlotRenderPass() {
	vkDestroyRenderPass(g_Device, g_PcPlotRenderPass, nullptr);
	vkDestroyRenderPass(g_Device, g_PcPlotRenderPass_noClear, nullptr);
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

	//creating the no clear framebuffer for spline rendering
	VkUtil::createFrameBuffer(g_Device, g_PcPlotRenderPass_noClear, attachments, g_PcPlotWidth, g_PcPlotHeight, &g_PcPlotFramebuffer_noClear);
}

static void cleanupPcPlotFramebuffer() {
	vkDestroyFramebuffer(g_Device, g_PcPlotFramebuffer, nullptr);
	vkDestroyFramebuffer(g_Device, g_PcPlotDensityFrameBuffer, nullptr);
	vkDestroyFramebuffer(g_Device, g_PcPlotFramebuffer_noClear, nullptr);
}

static void createPcPlotCommandPool() {
	VkResult err;

	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = g_QueueFamily;

	err = vkCreateCommandPool(g_Device, &poolInfo, nullptr, &g_PcPlotCommandPool);
	check_vk_result(err);

	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	err = vkCreateFence(g_Device, &fenceInfo, nullptr, &g_PcPlotRenderFence);
	vkResetFences(g_Device, 1, &g_PcPlotRenderFence);
}

static void cleanupPcPlotCommandPool() {
	vkDestroyCommandPool(g_Device, g_PcPlotCommandPool, nullptr);
	vkDestroyFence(g_Device, g_PcPlotRenderFence, nullptr);
}

static void fillVertexBuffer(Buffer vertexBuffer, const Data& data){
	uint32_t bufferSize = data.packedByteSize();
	Buffer stagingBuffer;
	VkUtil::createBuffer(g_Device, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,&stagingBuffer.buffer);
	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(g_Device, stagingBuffer.buffer, &memRequirements);
	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	VkResult err = vkAllocateMemory(g_Device, &allocInfo, nullptr, &stagingBuffer.memory);
	check_vk_result(err);
	vkBindBufferMemory(g_Device, stagingBuffer.buffer, stagingBuffer.memory, 0);

	void* mem;
	vkMapMemory(g_Device, stagingBuffer.memory, 0, bufferSize, 0, &mem);
	data.packData(mem);
	vkUnmapMemory(g_Device, stagingBuffer.memory);

	VkCommandBuffer copyComm;
	VkUtil::createCommandBuffer(g_Device, g_PcPlotCommandPool, &copyComm);
	VkUtil::copyBuffer(copyComm, stagingBuffer.buffer, vertexBuffer.buffer, bufferSize, 0, 0);
	VkUtil::commitCommandBuffer(g_Queue, copyComm);
	check_vk_result(vkQueueWaitIdle(g_Queue));
	vkFreeCommandBuffers(g_Device, g_PcPlotCommandPool, 1, &copyComm);
	vkDestroyBuffer(g_Device, stagingBuffer.buffer, nullptr);
	vkFreeMemory(g_Device, stagingBuffer.memory, nullptr);
}

static void createPcPlotVertexBuffer(const std::vector<Attribute>& Attributes, const Data& data, const std::optional<VertexBufferCreateInfo> info = {}) {
	VkResult err;

	//creating the command buffer as its needed to do all the operations in here
	//createPcPlotCommandBuffer();

	Buffer vertexBuffer;

	uint64_t bufferSize{};
	if(info){
		// standard data -> standard data byte size
		if(info->dataType == DataType::Continuous || info->dataType == DataType::ContinuousDlf){
			bufferSize = data.packedByteSize();
		}
		// hierarchical data -> reserving memory according to size given in info with 20% overhead for safety
		else if(info->dataType == DataType::Hierarchichal){
			bufferSize = (Attributes.size() + info->additionalAttributeStorage) * (info->maxLines) * 1.2 * sizeof(float);
		}
	}
	else{
		bufferSize = data.packedByteSize();
	}

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = bufferSize;
	bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &vertexBuffer.buffer);
	check_vk_result(err);

	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(g_Device, vertexBuffer.buffer, &memRequirements);

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	err = vkAllocateMemory(g_Device, &allocInfo, nullptr, &vertexBuffer.memory);
	check_vk_result(err);

	vkBindBufferMemory(g_Device, vertexBuffer.buffer, vertexBuffer.memory, 0);

	//filling the Vertex Buffer with all Datapoints
	if(!info || info->dataType == DataType::Continuous || info->dataType == DataType::ContinuousDlf){
		fillVertexBuffer(vertexBuffer, data);
	}

	std::vector<VkDescriptorSetLayout> layouts{g_PcPlotDataSetLayout};
	VkUtil::createDescriptorSets(g_Device, layouts, g_PcPlotDescriptorPool, &vertexBuffer.descriptorSet);

	VkUtil::updateDescriptorSet(g_Device, vertexBuffer.buffer, bufferSize, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, vertexBuffer.descriptorSet);

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
	if (buffer.descriptorSet){
		vkFreeDescriptorSets(g_Device, g_PcPlotDescriptorPool, 1, &buffer.descriptorSet);
	}

	g_PcPlotVertexBuffers.erase(it);
}

static void exeComputeHistogram(std::string& name, std::vector<std::pair<float, float>>& minMax, VkBuffer data, uint32_t amtOfData, VkBuffer indices, uint32_t amtOfIndices, VkBufferView indicesActivations, bool callForviolinAttributePlots = false);

static void createPcPlotDrawList(TemplateList& tl, const DataSet& ds, const char* listName) {
	VkResult err;

	DrawList dl = {};
	dl.parentTemplateList = &tl;
	dl.data = &ds.data;
	dl.attributes = &pcAttributes;
	UniformBufferObject ubo;
	ubo.vertTransformations.resize(pcAttributes.size());
	//uniformBuffer for pcPlot Drawing
	Buffer uboBuffer;

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = ubo.size();
	bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
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
	bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
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
	bufferInfo.size = ubo.size();
	bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &dl.medianUbo);
	check_vk_result(err);

	dl.medianUboOffset = allocInfo.allocationSize;

	vkGetBufferMemoryRequirements(g_Device, dl.medianUbo, &memRequirements);
	memRequirements.size = (memRequirements.size % memRequirements.alignment) ? memRequirements.size + (memRequirements.alignment - (memRequirements.size % memRequirements.alignment)) : memRequirements.size;
	allocInfo.allocationSize += memRequirements.size;

	//Median Buffer for Median Lines
	bufferInfo.size = MEDIANCOUNT * pcAttributes.size() * sizeof(float);
	bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &dl.medianBuffer);
	check_vk_result(err);

	dl.medianBufferOffset = allocInfo.allocationSize;

	vkGetBufferMemoryRequirements(g_Device, dl.medianBuffer, &memRequirements);
	allocInfo.allocationSize += memRequirements.size;
	memTypeBits |= memRequirements.memoryTypeBits;

	//Indexbuffer
	if(ds.dataType == DataType::Continuous || ds.dataType == DataType::ContinuousDlf)
		bufferInfo.size = tl.indices.size() * (pcAttributes.size() + 3) * sizeof(uint32_t);
	else
		bufferInfo.size = pcSettings.maxHierarchyLines * (pcAttributes.size() + 3) * sizeof(uint32_t);
	bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &dl.indexBuffer);
	check_vk_result(err);

	dl.indexBufferOffset = 0;
	vkGetBufferMemoryRequirements(g_Device, dl.indexBuffer, &memRequirements);
	VkMemoryAllocateInfo allIn{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	allIn.allocationSize = memRequirements.size;
	allIn.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	vkAllocateMemory(g_Device, &allIn, nullptr, &dl.indexBufferMemory);
	vkBindBufferMemory(g_Device, dl.indexBuffer, dl.indexBufferMemory, 0);

	//priority rendering color buffer
	if(ds.dataType == DataType::Continuous || ds.dataType == DataType::ContinuousDlf)
		bufferInfo.size = ds.data.size() * sizeof(float);
	else
		bufferInfo.size = pcSettings.maxHierarchyLines * sizeof(float);
	bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	err = vkCreateBuffer(g_Device, &bufferInfo, nullptr, &dl.priorityColorBuffer);
	check_vk_result(err);

	dl.priorityColorBufferOffset = allocInfo.allocationSize;
	vkGetBufferMemoryRequirements(g_Device, dl.priorityColorBuffer, &memRequirements);
	allocInfo.allocationSize += memRequirements.size;

	memTypeBits |= memRequirements.memoryTypeBits;

	//active indices buffer
	if(ds.dataType == DataType::Continuous || ds.dataType == DataType::ContinuousDlf)
		VkUtil::createBuffer(g_Device, ds.data.size() * sizeof(bool), VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT, &dl.activeIndicesBuffer);
	else
		VkUtil::createBuffer(g_Device, pcSettings.maxHierarchyLines * sizeof(bool), VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT, &dl.activeIndicesBuffer);

	dl.activeIndicesBufferOffset = allocInfo.allocationSize;
	vkGetBufferMemoryRequirements(g_Device, dl.activeIndicesBuffer, &memRequirements);
	allocInfo.allocationSize += memRequirements.size;
	memTypeBits |= memRequirements.memoryTypeBits;

	//indices buffer
	if(ds.dataType == DataType::Continuous || ds.dataType == DataType::ContinuousDlf)	
		VkUtil::createBuffer(g_Device, tl.indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &dl.indicesBuffer);
	else
		VkUtil::createBuffer(g_Device, pcSettings.maxHierarchyLines * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &dl.indicesBuffer);

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
	uint32_t offset = ubo.size();
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
	VkUtil::updateDescriptorSet(g_Device, dl.medianUbo, ubo.size(), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, dl.medianUboDescSet);

	//creating and uploading the indexbuffer data
	//uint32_t* indBuffer = new uint32_t[tl.indices.size() * 2];
	//for (int i = 0; i < tl.indices.size(); i++) {
	//	indBuffer[2 * i] = tl.indices[i] * pcAttributes.size();
	//	indBuffer[2 * i + 1] = tl.indices[i] * pcAttributes.size();
	//}
	//void* d;
	//vkMapMemory(g_Device, dl.dlMem, offset, tl.indices.size() * sizeof(uint32_t) * 2, 0, &d);
	//memcpy(d, indBuffer, tl.indices.size() * sizeof(uint32_t) * 2);
	//vkUnmapMemory(g_Device, dl.dlMem);
	//delete[] indBuffer;


	//binding the medianBuffer
	vkBindBufferMemory(g_Device, dl.medianBuffer, dl.dlMem, dl.medianBufferOffset);
	layouts = {g_PcPlotDataSetLayout};
	VkUtil::createDescriptorSets(g_Device, layouts, g_PcPlotDescriptorPool, &dl.medianBufferSet);
	VkUtil::updateDescriptorSet(g_Device, dl.medianBuffer, MEDIANCOUNT * pcAttributes.size() * sizeof(float), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, dl.medianBufferSet);

	//binding the indexBuffer
	//vkBindBufferMemory(g_Device, dl.indexBuffer, dl.dlMem, dl.indexBufferOffset);

	//binding the  priority rendering buffer
	vkBindBufferMemory(g_Device, dl.priorityColorBuffer, dl.dlMem, dl.priorityColorBufferOffset);

	//binding the active indices buffer, creating the buffer view and uploading the correct indices to the graphicscard
	vkBindBufferMemory(g_Device, dl.activeIndicesBuffer, dl.dlMem, dl.activeIndicesBufferOffset);
	std::vector<uint8_t> actives(ds.data.size(), 1);			//vector with 0 initialized everywhere
	if(ds.dataType == DataType::Hierarchichal){
		actives.resize(pcSettings.maxHierarchyLines, 1);
	}
	VkUtil::createBufferView(g_Device, dl.activeIndicesBuffer, VK_FORMAT_R8_SNORM, 0, actives.size() * sizeof(bool), &dl.activeIndicesBufferView);
	
	VkUtil::uploadData(g_Device, dl.dlMem, dl.activeIndicesBufferOffset, actives.size() * sizeof(bool), actives.data());

	//binding indices buffer and uploading the indices
	vkBindBufferMemory(g_Device, dl.indicesBuffer, dl.dlMem, dl.indicesBufferOffset);
	if(ds.dataType == DataType::Continuous || ds.dataType == DataType::ContinuousDlf)	
		VkUtil::uploadData(g_Device, dl.dlMem, dl.indicesBufferOffset, tl.indices.size() * sizeof(uint32_t), tl.indices.data());
	else{
		std::vector<uint32_t> tI(pcSettings.maxHierarchyLines);
		std::iota(tI.begin(), tI.end(), 0);
		VkUtil::uploadData(g_Device, dl.dlMem, dl.indicesBufferOffset, tI.size() * sizeof(uint32_t), tI.data());
	}

	//creating the Descriptor sets for the histogramm uniform buffers
	layouts = std::vector<VkDescriptorSetLayout>(dl.histogramUbos.size());
	for (auto& l : layouts) {
		l = g_PcPlotHistoDescriptorSetLayout;
	}

	dl.histogrammDescSets = std::vector<VkDescriptorSet>(layouts.size());
	VkUtil::createDescriptorSets(g_Device, layouts, g_DescriptorPool, dl.histogrammDescSets.data());

	size_t dataByteSize = ds.data.packedByteSize();
	if(ds.dataType == DataType::Hierarchichal){
		dataByteSize = (pcAttributes.size() + 1) * (pcSettings.maxHierarchyLines) * 1.2 * sizeof(float);
	}
	//updating the descriptor sets
	for (int i = 0; i < layouts.size(); i++) {
		VkUtil::updateDescriptorSet(g_Device, dl.histogramUbos[i], sizeof(HistogramUniformBuffer), 0, dl.histogrammDescSets[i]);
		VkUtil::updateTexelBufferDescriptorSet(g_Device, dl.activeIndicesBufferView, 1, dl.histogrammDescSets[i]);
		VkUtil::updateDescriptorSet(g_Device, tl.buffer, dataByteSize, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, dl.histogrammDescSets[i]);
	}

	//specifying the uniform buffer location
	VkDescriptorBufferInfo desBufferInfos[1] = {};
	desBufferInfos[0].buffer = dl.ubo;
	desBufferInfos[0].offset = 0;
	desBufferInfos[0].range = ubo.size();

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
	descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pBufferInfo = desBufferInfos;

	vkUpdateDescriptorSets(g_Device, 1, &descriptorWrite, 0, nullptr);
	if(ds.dataType == DataType::Hierarchichal)
		VkUtil::updateDescriptorSet(g_Device, dl.priorityColorBuffer, pcSettings.maxHierarchyLines * sizeof(float), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, dl.uboDescSet);
	else
		VkUtil::updateDescriptorSet(g_Device, dl.priorityColorBuffer, ds.data.size() * sizeof(float), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, dl.uboDescSet);
	VkUtil::updateImageDescriptorSet(g_Device, g_PcPlotDensityIronMapSampler, g_PcPLotDensityIronMapView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2, dl.uboDescSet);

	if(ds.dataType == DataType::Hierarchichal)
		VkUtil::updateDescriptorSet(g_Device, dl.priorityColorBuffer, pcSettings.maxHierarchyLines * sizeof(float), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, dl.medianUboDescSet);
	else
		VkUtil::updateDescriptorSet(g_Device, dl.priorityColorBuffer, ds.data.size() * sizeof(float), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, dl.medianUboDescSet);
	VkUtil::updateImageDescriptorSet(g_Device, g_PcPlotDensityIronMapSampler, g_PcPLotDensityIronMapView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2, dl.medianUboDescSet);

	rgb col = drawListColorPalette->getNextColor();

	dl.name = std::string(listName);
	dl.buffer = tl.buffer;
	dl.dataDescriptorSet = ds.buffer.descriptorSet;
	dl.color = { (float)col.r,(float)col.g,(float)col.b,pcSettings.autoAlpha ? std::clamp(1.0f/ (tl.indices.size() * .001f),.004f, 1.f) : pcSettings.alphaDrawLists };
	dl.prefColor = dl.color;
	dl.show = true;
	dl.showHistogramm = true;
	dl.parentDataSet = ds.name;
	
	dl.brushedRatioToParent = std::vector<float>(pcAttributes.size(), 1);
	if(ds.dataType == DataType::Hierarchichal){
		dl.inheritanceFlags = InheritanceFlags::hierarchical;
		dl.indices = {};		//nothing yet loaded
		std::string_view hierarchy(reinterpret_cast<const char*>(ds.additionalData.data()), ds.additionalData.size());
		dl.hierarchyImportManager= std::make_shared<HierarchyImportManager>(hierarchy, pcSettings.maxHierarchyLines);
	}
	else{
		dl.indices = std::vector<uint32_t>(tl.indices);
	}

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
	std::string name = drawList.name;
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
			if (it->medianBufferSet){
				vkFreeDescriptorSets(g_Device, g_PcPlotDescriptorPool, 1, &it->medianBufferSet);
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
			if (it->indexBufferMemory){
				vkFreeMemory(g_Device, it->indexBufferMemory, nullptr);
				it->indexBufferMemory = VK_NULL_HANDLE;
			}
			if (it->lineBundles) delete it->lineBundles;
			if (it->clusterBundles) delete it->clusterBundles;
			g_PcPlotDrawLists.erase(it);
			break;
		}
	}
	correlationMatrixWorkbench->updateCorrelationScores(g_PcPlotDrawLists, {name});
}

static void removePcPlotDrawLists(DataSet dataSet) {
	for (auto it = g_PcPlotDrawLists.begin(); it != g_PcPlotDrawLists.end(); ) {
		if (it->parentDataSet == dataSet.name) {
			removePcPlotDrawList(*it);
			it = g_PcPlotDrawLists.begin(); //resetting the iterator, as the old one is now not valid anymore
		}
		else {
			it++;
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

	pcSettings.updateBrushTemplates = true;

	g_PcPlotDataSets.erase(it);

	//if this was the last data set reset the ofther buffer too
	//Attributes also have to be deleted
	if (g_PcPlotDataSets.size() == 0) {
		cleanupPcPlotVertexBuffer();

		pcAttributes.clear();
		pcAttrOrd.clear();
		pcAttributesSorted.clear();
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
	}

	g_PcPlotDataSets.clear();
	cleanupPcPlotVertexBuffer();
}

static void createPcPlotCommandBuffer(bool batching) {
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
	renderPassInfo.renderPass = batching ? g_PcPlotRenderPass_noClear: g_PcPlotRenderPass;
	renderPassInfo.framebuffer = batching ? g_PcPlotFramebuffer_noClear: g_PcPlotFramebuffer;
	renderPassInfo.renderArea.offset = { 0,0 };
	renderPassInfo.renderArea.extent = { g_PcPlotWidth,g_PcPlotHeight };

	VkClearValue clearColor = { pcSettings.PcPlotBackCol.x,pcSettings.PcPlotBackCol.y,pcSettings.PcPlotBackCol.z,pcSettings.PcPlotBackCol.w };//{ 0.0f,0.0f,0.0f,1.0f };

	renderPassInfo.clearValueCount = !batching;
	renderPassInfo.pClearValues = &clearColor;

	vkCmdBeginRenderPass(g_PcPlotCommandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	if (batching) {
		vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipeline_noClear);
		return;
	}
	if (pcSettings.renderSplines)
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

	err = vkQueueSubmit(g_Queue, 1, &submitInfo, g_PcPlotRenderFence);
	check_vk_result(err);

	err = vkWaitForFences(g_Device, 1, &g_PcPlotRenderFence, true, 60e9);	//timeout of 60 seconds
	check_vk_result(err);

	vkResetFences(g_Device, 1, &g_PcPlotRenderFence);

	vkFreeCommandBuffers(g_Device, g_PcPlotCommandPool, 1, &g_PcPlotCommandBuffer);
}

//getting the dimension values array. Creates the dimensin values array if it is not yet created
static std::pair<bool,std::vector<float>>& getDimensionValues(const DataSet& ds, int dimension) {
	// checking for the dimension values array and creating it if it doesnt exist
	bool arraysExist = true;
	if (dimensionValues.find(pcAttributes[dimension].name) == dimensionValues.end()) {
		arraysExist = false;
	}
	if (!arraysExist) {
		dimensionValues[pcAttributes[dimension].name] = {};
		std::set<float> used;
		for (int i = 0; i < ds.data.columns[dimension].size(); ++i) {
			float d = ds.data.columns[dimension][i];		//directly qurying the column to avoid copying
			if (used.find(d) == used.end()) {
				dimensionValues[pcAttributes[dimension].name].second.push_back(d);
				used.insert(d);
			}
		}
		//sort dimension values and check for linearity
		std::vector<float>& ref = dimensionValues[pcAttributes[dimension].name].second;
		std::sort(ref.begin(), ref.end());
		float diff = ref[1] - ref[0];
		dimensionValues[pcAttributes[dimension].name].first = true;
		for (int i = 2; i < ref.size(); ++i) {
			float d = std::abs(ref[i] - ref[i - 1] - diff);
			//check if the stepdifference is larger than 1%
			if (d / diff > .01f) {
				dimensionValues[pcAttributes[dimension].name].first = false;
				break;
			}
		}
	}
	return dimensionValues[pcAttributes[dimension].name];
}

// This function assumes that only indices of active attributes are passed. 
static int placeOfInd(int ind, bool countDisabled = false) {
	int place = 0;
	for (int i : pcAttrOrd) {
		if (i == ind)
			break;
		if (pcAttributeEnabled[i] || countDisabled)
			place++;
	}
	return place;
}

//returns the attribute index corresponding to the n-ths place in the plot
static int attributeOfPlace(int place) {
	int ind = 0;
	for (int i = 0; i < pcAttrOrd.size(); ++i) {
		ind = pcAttrOrd[i];
		if (pcAttributeEnabled[pcAttrOrd[i]]) --place;
		if (place < 0) break;
	}
	return ind;
}

static void drawPcPlot(const std::vector<Attribute>& attributes, const std::vector<int>& attributeOrder, const bool* attributeEnabled, const ImGui_ImplVulkanH_Window* wd) {
#ifdef PRINTRENDERTIME
	uint32_t amtOfLines = 0;
#endif
	//PCRenderer::GlobalPCSettings settings{pcAttributes, pcAttributeEnabled, pcAttrOrd, pcSettings.renderSplines, pcSettings.medianLineWidth};
	//pcRenderer->renderPCPlots(g_PcPlotDrawLists, settings);
	//PCUtil::Stopwatch stopwatch(std::cout, "drawPcPlot(...)");

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
	ubo.vertTransformations.resize(ubo.amtOfAttributes);
	if(ubo.amtOfAttributes)
		ubo.vertTransformations[0].w = (priorityAttribute != -1) ? 1.f : 0;
	if (pcSettings.drawHistogramm) {
		ubo.padding = pcSettings.histogrammWidth / 2;
	}
	else {
		ubo.padding = 0;
	}

	int c = 0;

	for (int i : attributeOrder) {
		ubo.vertTransformations[i].x = c;
		if (attributeEnabled[i])
			c++;
		ubo.vertTransformations[i].y = attributes[i].min;
		ubo.vertTransformations[i].z = attributes[i].max;
	}

	std::vector<std::pair<int, int>> order;
	for (int i = 0; i < pcAttributes.size(); i++) {
		if (pcAttributeEnabled[i]) {
			order.push_back(std::pair<int, int>(i, placeOfInd(i)));
		}
	}

	std::sort(order.begin(), order.end(), [](std::pair<int, int>a, std::pair<int, int>b) {return a.second < b.second; });

	//filling the indexbuffer with the used indeces
	uint16_t* ind = new uint16_t[amtOfIndeces + ((pcSettings.renderSplines) ? 2 : 0)];			//contains all indeces to copy
	for (int i = 0; i < order.size(); i++) {
		ind[i + ((pcSettings.renderSplines) ? 1 : 0)] = order[i].first;
	}
	if (pcSettings.renderSplines && pcAttributes.size()) {
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
		int copyAmount = sizeof(uint16_t) * (attributes.size() + ((pcSettings.renderSplines) ? 2 : 0));
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
		uint32_t uboSize = sizeof(UniformBufferObject) - sizeof(UniformBufferObject::vertTransformations);
		uint32_t trafoSize = sizeof(ubo.vertTransformations[0]) * ubo.vertTransformations.size();
		std::vector<uint8_t> bits(uboSize + trafoSize);
		ubo.vertTransformations[0].w = (priorityAttribute != -1 && c == priorityListIndex) ? 1.f : 0;
		ubo.color = ds.color;
		std::copy_n(reinterpret_cast<uint8_t*>(&ubo), uboSize, bits.data());
		std::copy_n(reinterpret_cast<uint8_t*>(ubo.vertTransformations.data()), trafoSize, bits.data() + uboSize);
		vkMapMemory(g_Device, ds.dlMem, 0, bits.size(), 0, &da);
		memcpy(da, bits.data(), bits.size());
		vkUnmapMemory(g_Device, ds.dlMem);

		ubo.vertTransformations[0].w = 0;
		ubo.color = ds.medianColor;
		std::copy_n(reinterpret_cast<uint8_t*>(&ubo), uboSize, bits.data());
		std::copy_n(reinterpret_cast<uint8_t*>(ubo.vertTransformations.data()), trafoSize, bits.data() + uboSize);
		vkMapMemory(g_Device, ds.dlMem, ds.medianUboOffset, bits.size(), 0, &da);
		memcpy(da, bits.data(), bits.size());
		vkUnmapMemory(g_Device, ds.dlMem);

		c++;
	}

	//vector of command buffers for batch rendering
	std::vector<VkCommandBuffer> line_batch_commands;
	int max_amt_of_lines = 0;
	for (auto drawList = g_PcPlotDrawLists.begin(); drawList != g_PcPlotDrawLists.end(); ++drawList) {
		if(drawList->show)
			max_amt_of_lines += drawList->indices.size();
	}
	bool batching = max_amt_of_lines > pcSettings.lineBatchSize && pcSettings.renderSplines;		//if more lines could be rendererd than the set batch size use batched rendering(only activated for spline rendering)
	int curIndex = 0;
	int batchSizeLeft = pcSettings.lineBatchSize;

	//starting the pcPlotCommandBuffer
	createPcPlotCommandBuffer(batching);

	//counting the amount of active drawLists for histogramm rendering
	int activeDrawLists = 0;

	//now drawing for every draw list in g_pcPlotdrawlists
	if (batching) {
		//creating the standard batch command buffer
		line_batch_commands.push_back({});
		VkUtil::createCommandBuffer(g_Device, g_PcPlotCommandPool, &line_batch_commands[0]);
		std::vector<VkClearValue> clearValues{ { pcSettings.PcPlotBackCol.x,pcSettings.PcPlotBackCol.y,pcSettings.PcPlotBackCol.z,pcSettings.PcPlotBackCol.w } };
		VkUtil::beginRenderPass(line_batch_commands[0], clearValues, g_PcPlotRenderPass, g_PcPlotFramebuffer, { g_PcPlotWidth, g_PcPlotHeight });
		vkCmdBindPipeline(line_batch_commands[0], VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipeline);

		//binding the all needed things
		vkCmdBindDescriptorSets(line_batch_commands.back(), VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipelineLayout, 0, 1, &g_PcPlotDescriptorSet, 0, nullptr);

		if (pcAttributes.size())
			vkCmdBindIndexBuffer(line_batch_commands.back(), g_PcPlotIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

		for (auto drawList = g_PcPlotDrawLists.rbegin(); g_PcPlotDrawLists.rend() != drawList; ++drawList) {
			if (!drawList->show)
				continue;
			if (drawList->renderBundles){
				drawList->lineBundles->setAxisInfosBuffer(drawList->ubo, ubo.size());
				vkCmdEndRenderPass(line_batch_commands.back());
				std::copy(&pcSettings.PcPlotBackCol.x, &pcSettings.PcPlotBackCol.x + 4, drawList->lineBundles->haloColor);
				drawList->lineBundles->haloWidth = pcSettings.haloWidth;
				drawList->lineBundles->recordDrawBundles(line_batch_commands.back());
				VkUtil::beginRenderPass(line_batch_commands.back(), clearValues, g_PcPlotRenderPass_noClear, g_PcPlotFramebuffer_noClear, { g_PcPlotWidth, g_PcPlotHeight });
				vkCmdBindPipeline(line_batch_commands[0], VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipeline);
				continue;
			}
			else if(drawList->renderClusterBundles){
				drawList->clusterBundles->setAxisInfosBuffer(drawList->ubo, ubo.size());
				vkCmdEndRenderPass(line_batch_commands.back());
				std::copy(&pcSettings.PcPlotBackCol.x, &pcSettings.PcPlotBackCol.x + 4, drawList->clusterBundles->haloColor);
				drawList->clusterBundles->haloWidth = pcSettings.haloWidth;
				drawList->clusterBundles->recordDrawBundles(line_batch_commands.back());
				VkUtil::beginRenderPass(line_batch_commands.back(), clearValues, g_PcPlotRenderPass_noClear, g_PcPlotFramebuffer_noClear, { g_PcPlotWidth, g_PcPlotHeight });
				vkCmdBindPipeline(line_batch_commands[0], VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipeline);
				continue;
			}
			do {
				VkDeviceSize offsets[] = { 0 };
				//vkCmdBindVertexBuffers(line_batch_commands.back(), 0, 1, &drawList->buffer, offsets);
				vkCmdBindIndexBuffer(line_batch_commands.back(), drawList->indexBuffer, 0, VK_INDEX_TYPE_UINT32);

				//binding the right ubo
				VkDescriptorSet descSets[2]{drawList->uboDescSet, drawList->dataDescriptorSet};
				if (pcSettings.renderSplines){
					vkCmdBindDescriptorSets(line_batch_commands.back(), VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipelineLayout, 0, 2, descSets, 0, nullptr);
				}
				else{
					vkCmdBindDescriptorSets(line_batch_commands.back(), VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotPipelineLayout, 0, 2, descSets, 0, nullptr);
				}

				vkCmdSetLineWidth(line_batch_commands.back(), 1.0f);

				//ready to draw with draw indexed
				int indices_count;
				if (drawList->indices.size() - curIndex <= batchSizeLeft) {
					indices_count = drawList->indices.size() - curIndex;
					batchSizeLeft -= indices_count;
				}
				else {
					indices_count = batchSizeLeft;
					batchSizeLeft = 0;
				}
				uint32_t amtOfI = indices_count * (order.size() + 3);
				uint32_t iOffset = curIndex * (order.size() + 1 + ((pcSettings.renderSplines) ? 2 : 0));
				vkCmdDrawIndexed(line_batch_commands.back(), amtOfI, 1, iOffset, 0, 0);

				curIndex += indices_count;

				//draw the Median Line
				if (drawList->activeMedian != 0 && curIndex == drawList->indices.size()) {
					vkCmdSetLineWidth(line_batch_commands.back(), pcSettings.medianLineWidth);
					//vkCmdBindVertexBuffers(line_batch_commands.back(), 0, 1, &drawList->medianBuffer, offsets);
					vkCmdBindIndexBuffer(line_batch_commands.back(), g_PcPlotIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

					VkDescriptorSet medianDescSets[2]{drawList->medianUboDescSet, drawList->medianBufferSet};
					if (pcSettings.renderSplines)
						vkCmdBindDescriptorSets(line_batch_commands.back(), VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipelineLayout, 0, 2, medianDescSets, 0, nullptr);
					else
						vkCmdBindDescriptorSets(line_batch_commands.back(), VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotPipelineLayout, 0, 2, medianDescSets, 0, nullptr);

					vkCmdDrawIndexed(line_batch_commands.back(), amtOfIndeces + ((pcSettings.renderSplines) ? 2 : 0), 1, 0, (drawList->activeMedian - 1) * pcAttributes.size(), 0);

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

				if (batchSizeLeft == 0) {
					vkCmdEndRenderPass(line_batch_commands.back());
					line_batch_commands.push_back({});
					VkUtil::createCommandBuffer(g_Device, g_PcPlotCommandPool, &line_batch_commands.back());
					//clearValues.clear();
					VkUtil::beginRenderPass(line_batch_commands.back(), clearValues, g_PcPlotRenderPass_noClear, g_PcPlotFramebuffer_noClear, { g_PcPlotWidth, g_PcPlotHeight });

					vkCmdBindPipeline(line_batch_commands.back(), VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipeline_noClear);

					//binding the all needed things
					vkCmdBindDescriptorSets(line_batch_commands.back(), VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipelineLayout_noClear, 0, 1, &g_PcPlotDescriptorSet, 0, nullptr);

					if (pcAttributes.size())
						vkCmdBindIndexBuffer(line_batch_commands.back(), g_PcPlotIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

					batchSizeLeft = pcSettings.lineBatchSize;
				}
			} while (curIndex < drawList->indices.size());
			curIndex = 0;
		}
		//rendering all batches
		vkCmdEndRenderPass(line_batch_commands.back());
		for (VkCommandBuffer b : line_batch_commands) {
			VkUtil::commitCommandBuffer(g_Queue, b);
			err = vkQueueWaitIdle(g_Queue); check_vk_result(err);
		}

		vkFreeCommandBuffers(g_Device, g_PcPlotCommandPool, line_batch_commands.size(), line_batch_commands.data());
	}
	else {
		//binding the all needed things
		if (pcSettings.renderSplines)
			vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipelineLayout, 0, 1, &g_PcPlotDescriptorSet, 0, nullptr);
		else
			vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotPipelineLayout, 0, 1, &g_PcPlotDescriptorSet, 0, nullptr);

		if (pcAttributes.size())
			vkCmdBindIndexBuffer(g_PcPlotCommandBuffer, g_PcPlotIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

		for (auto drawList = g_PcPlotDrawLists.rbegin(); g_PcPlotDrawLists.rend() != drawList; ++drawList) {
			if (!drawList->show)
				continue;

			if (drawList->renderBundles){
				drawList->lineBundles->setAxisInfosBuffer(drawList->ubo, ubo.size());
				vkCmdEndRenderPass(g_PcPlotCommandBuffer);
				std::copy(&pcSettings.PcPlotBackCol.x, &pcSettings.PcPlotBackCol.x + 4, drawList->lineBundles->haloColor);
				drawList->lineBundles->haloWidth = pcSettings.haloWidth;
				drawList->lineBundles->recordDrawBundles(g_PcPlotCommandBuffer);
				std::vector<VkClearValue> clearValues{ { pcSettings.PcPlotBackCol.x,pcSettings.PcPlotBackCol.y,pcSettings.PcPlotBackCol.z,pcSettings.PcPlotBackCol.w } };
				VkUtil::beginRenderPass(g_PcPlotCommandBuffer, clearValues, g_PcPlotRenderPass_noClear, g_PcPlotFramebuffer_noClear, { g_PcPlotWidth, g_PcPlotHeight });
				vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipeline);
				if (pcSettings.renderSplines)
					vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipeline);
				else
					vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotPipeline);
				continue;
			}
			if (drawList->renderClusterBundles){
				drawList->clusterBundles->setAxisInfosBuffer(drawList->ubo, ubo.size());
				vkCmdEndRenderPass(g_PcPlotCommandBuffer);
				std::copy(&pcSettings.PcPlotBackCol.x, &pcSettings.PcPlotBackCol.x + 4, drawList->clusterBundles->haloColor);
				drawList->clusterBundles->haloWidth = pcSettings.haloWidth;
				drawList->clusterBundles->recordDrawBundles(g_PcPlotCommandBuffer);
				std::vector<VkClearValue> clearValues{ { pcSettings.PcPlotBackCol.x,pcSettings.PcPlotBackCol.y,pcSettings.PcPlotBackCol.z,pcSettings.PcPlotBackCol.w } };
				VkUtil::beginRenderPass(g_PcPlotCommandBuffer, clearValues, g_PcPlotRenderPass_noClear, g_PcPlotFramebuffer_noClear, { g_PcPlotWidth, g_PcPlotHeight });
				vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipeline);
				if (pcSettings.renderSplines)
					vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipeline);
				else
					vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotPipeline);
				continue;
			}

			VkDeviceSize offsets[] = { 0 };
			//vkCmdBindVertexBuffers(g_PcPlotCommandBuffer, 0, 1, &drawList->buffer, offsets);
			vkCmdBindIndexBuffer(g_PcPlotCommandBuffer, drawList->indexBuffer, 0, VK_INDEX_TYPE_UINT32);

			//binding the right ubo
			VkDescriptorSet descSets[2]{drawList->uboDescSet, drawList->dataDescriptorSet};
			if (pcSettings.renderSplines)
				vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipelineLayout, 0, 2, descSets, 0, nullptr);
			else
				vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotPipelineLayout, 0, 2, descSets, 0, nullptr);

			vkCmdSetLineWidth(g_PcPlotCommandBuffer, 1.0f);

			//ready to draw with draw indexed
			uint32_t amtOfI = drawList->indices.size() * (order.size() + 3);
			vkCmdDrawIndexed(g_PcPlotCommandBuffer, amtOfI, 1, 0, 0, 0);

			//draw the Median Line
			if (drawList->activeMedian != 0) {
				vkCmdSetLineWidth(g_PcPlotCommandBuffer, pcSettings.medianLineWidth);
				vkCmdBindVertexBuffers(g_PcPlotCommandBuffer, 0, 1, &drawList->medianBuffer, offsets);
				vkCmdBindIndexBuffer(g_PcPlotCommandBuffer, g_PcPlotIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

				VkDescriptorSet medianDescSets[2]{drawList->medianUboDescSet, drawList->medianBufferSet};
				if (pcSettings.renderSplines)
					vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotSplinePipelineLayout, 0, 2, medianDescSets, 0, nullptr);
				else
					vkCmdBindDescriptorSets(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotPipelineLayout, 0, 2, medianDescSets, 0, nullptr);

				vkCmdDrawIndexed(g_PcPlotCommandBuffer, amtOfIndeces + ((pcSettings.renderSplines) ? 2 : 0), 1, 0, (drawList->activeMedian - 1) * pcAttributes.size(), 0);

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
	}

	delete[] ind;

	if (pcSettings.pcPlotDensity && pcAttributes.size() > 0) {
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

	if (pcSettings.drawHistogramm && pcAttributes.size() > 0) {
		//drawing the histogramm background
		RectVertex* rects = new RectVertex[pcAttributes.size() * 4];
		float x = -1;
		for (int i = 0; i < pcAttributes.size(); i++) {
			if (pcAttributeEnabled[i]) {
				RectVertex vert;
				vert.pos = { x,1,0,0 };
				vert.col = pcSettings.histogrammDensity ? pcSettings.densityBackCol : pcSettings.histogrammBackCol;
				rects[i * 4] = vert;
				vert.pos.y = -1;
				rects[i * 4 + 1] = vert;
				vert.pos.x += pcSettings.histogrammWidth;
				rects[i * 4 + 2] = vert;
				vert.pos.y = 1;
				rects[i * 4 + 3] = vert;
				x += (2 - pcSettings.histogrammWidth) / (amtOfIndeces - 1);
			}
			else {
				RectVertex vert;
				vert.pos = { -2,-2,0,0 };
				vert.col = pcSettings.histogrammBackCol;
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
		if (pcSettings.histogrammDensity && pcSettings.enableDensityMapping) {
			vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotHistoAdditivePipeline);
		}
		else {
			vkCmdBindPipeline(g_PcPlotCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_PcPlotHistoPipeline);
		}

		//the offset which has to be added to draw the histogramms next to one another
		uint32_t amtOfHisto = 0;
		for (auto& dl : g_PcPlotDrawLists) {
			if (dl.showHistogramm)
				amtOfHisto++;
		}
		if (amtOfHisto != 0) {
			HistogramUniformBuffer hubo = {};
			float gap = (2 - pcSettings.histogrammWidth) / (amtOfIndeces - 1);
			float xOffset = .0f;
			float width = pcSettings.histogrammWidth / amtOfHisto;
			for (auto drawList = g_PcPlotDrawLists.begin(); g_PcPlotDrawLists.end() != drawList; ++drawList) {
				//ignore drawLists which are disabled
				if (!drawList->showHistogramm)
					continue;

				//setting the color in the hubo to copy
				hubo.color = drawList->color;
				if (pcSettings.adustHistogrammByActiveLines && pcSettings.histogrammDensity && pcSettings.enableDensityMapping) hubo.color.w /= activeBrushRatios[drawList->name] + FLT_EPSILON;
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
					if (pcSettings.histogrammDensity && pcSettings.enableDensityMapping) {
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

		if (pcSettings.histogrammDensity) {
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
			float gap = (2 - pcSettings.histogrammWidth) / (amtOfIndeces - 1);
			for (int i = 0; i < amtOfIndeces; i++) {
				verts[i * 4] = { gap * i - 1,1,0,0 };
				verts[i * 4 + 1] = { gap * i - 1,-1,0,0 };
				verts[i * 4 + 2] = { gap * i + pcSettings.histogrammWidth - 1,-1,0,0 };
				verts[i * 4 + 3] = { gap * i + pcSettings.histogrammWidth - 1,1,0,0 };
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
#ifdef PRINTRENDERTIME
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif

	//when cleaning up the command buffer all data is drawn
	cleanupPcPlotCommandBuffer();

	//err = vkQueueWaitIdle(g_Queue);
	//check_vk_result(err);

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
		VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
		appInfo.apiVersion = VK_API_VERSION_1_1;
		appInfo.pApplicationName = "PCViewer";

		VkInstanceCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		create_info.enabledExtensionCount = extensions_count;
		create_info.ppEnabledExtensionNames = extensions;
		create_info.pApplicationInfo = &appInfo;

#ifdef IMGUI_VULKAN_DEBUG_REPORT
		// Enabling multiple validation layers grouped as LunarG standard validation
		const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
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
		VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
		VkPhysicalDeviceShaderAtomicFloatFeaturesEXT floatFeat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
		feat2.pNext = &floatFeat;
		vkGetPhysicalDeviceFeatures2(gpus[0], &feat2);
		atomicGpuFloatAddAvailable = feat.shaderFloat64 && floatFeat.shaderBufferFloat32AtomicAdd && floatFeat.shaderBufferFloat64AtomicAdd;
		if(!floatFeat.shaderImageFloat32AtomicAdd){
			std::cout << "Gpu does not support float64 atomic add -> gpu accelerated correlation calculations disabled" << std::endl;
		}

#ifdef _DEBUG
		std::cout << "Gometry shader usable:" << feat.geometryShader << std::endl;
		std::cout << "Wide lines usable:" << feat.wideLines << std::endl;
#endif

		// If a number >1 of GPUs got reported, you should find the best fit GPU for your purpose
		// e.g. VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU if available, or with the greatest memory available, etc.
		// for sake of simplicity we'll just take the first one, assuming it has a graphics queue family.
		g_PhysicalDevice = gpus[0];

		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(g_PhysicalDevice, &props);
		g_MaxStorageBufferSize = props.limits.maxStorageBufferRange;

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
		int device_extension_count = 3;
		if(atomicGpuFloatAddAvailable) device_extension_count = 4;
		const char* device_extensions[] = { "VK_KHR_swapchain", "VK_KHR_maintenance3", "VK_EXT_descriptor_indexing", VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME };

		VkPhysicalDeviceShaderAtomicFloatFeaturesEXT floatFeat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
		floatFeat.shaderBufferFloat32AtomicAdd = VK_TRUE;
		floatFeat.shaderBufferFloat64AtomicAdd = VK_TRUE;

		VkPhysicalDeviceDescriptorIndexingFeaturesEXT indexingFeatures{};
		indexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
		if(atomicGpuFloatAddAvailable)
			indexingFeatures.pNext = &floatFeat;
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
		if(atomicGpuFloatAddAvailable)
			deviceFeatures.shaderFloat64 = VK_TRUE;
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
	ImGui_ImplVulkanH_CreateOrResizeWindow(g_Instance, g_PhysicalDevice, g_Device, wd, g_QueueFamily, g_Allocator, width, height, g_MinImageCount);
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

static void FrameRender(ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data)
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

	// Record dear imgui primitives into command buffer
	ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);

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

static void FrameRenderExport(ImGui_ImplVulkanH_Frame* fd, ImDrawData* draw_data, const VkClearValue& clearColor) {
	VkResult err;
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
		info.renderPass = g_ExportWindowRenderPass;
		info.framebuffer = fd->Framebuffer;
		info.renderArea.extent.width = g_ExportImageWidth;
		info.renderArea.extent.height = g_ExportImageHeight;
		info.clearValueCount = 1;
		info.pClearValues = &clearColor;
		vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	// Record dear imgui primitives into command buffer
	ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);

	// Submit command buffer
	vkCmdEndRenderPass(fd->CommandBuffer);
	{
		VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSubmitInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		info.waitSemaphoreCount = 0;
		info.pWaitSemaphores = NULL;
		info.pWaitDstStageMask = &wait_stage;
		info.commandBufferCount = 1;
		info.pCommandBuffers = &fd->CommandBuffer;
		info.signalSemaphoreCount = 0;
		info.pSignalSemaphores = NULL;

		err = vkEndCommandBuffer(fd->CommandBuffer);
		check_vk_result(err);
		err = vkQueueSubmit(g_Queue, 1, &info, VK_NULL_HANDLE);
		check_vk_result(err);
	}
}

static void FramePresent(ImGui_ImplVulkanH_Window* wd, SDL_Window* window)
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
    if (err == VK_ERROR_OUT_OF_DATE_KHR)
    {
        SDL_GetWindowSize(window, &g_SwapChainResizeWidth, &g_SwapChainResizeHeight);;
        g_SwapChainRebuild = true;
        return;
    }
    check_vk_result(err);
    wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->ImageCount; // Now we can use the next set of semaphores
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

static bool openHierarchy(const char* filename, const char* attributeInfo){
	//opening the info file to get attribute information
	std::vector<Attribute> infoAttributes;				//note that there is indeed one extra attribute available for the cluster counts which is not inside this vector
	std::vector<std::string> infoAttributeNames;
	std::ifstream info(attributeInfo, std::ios::binary);
	if(!info){
		std::cout << "The .info file for the hierarchical dataset could not be opened";
		return false;
	}
	while(info.good() && !info.eof()){
		infoAttributes.push_back({});
		info >> infoAttributes.back().name >> infoAttributes.back().min >> infoAttributes.back().max;
		infoAttributes.back().originalName = infoAttributes.back().name;
		infoAttributeNames.push_back(infoAttributes.back().name);
		info.get();		//jumping over the newline character to get proper eof notification
	}

	//checking attributes correctnes
	auto permutation = checkAttriubtes(infoAttributeNames);	//note: permutation is currently not used, as its thought that the dataest will always have the same structure
	if(!pcAttributes.empty() && permutation.empty()){
		std::cout << "The attributes of the hierarchical data set are not the same as the ones already loaded in the program." << std::endl;
		return false;
	}
	if(pcAttributes.empty()){
		pcAttributes = infoAttributes;

		//setting up the boolarray and setting all the attributes to true
		pcAttributeEnabled = new bool[pcAttributes.size()];
		activeBrushAttributes = new bool[pcAttributes.size()];
		for (int i = 0; i < pcAttributes.size(); i++) {
			pcAttributeEnabled[i] = true;
			activeBrushAttributes[i] = false;
			pcAttrOrd.push_back(i);
		}
	}

	DataSet ds{};			//the data variable of the dataset will stay empty
	std::string sFilename(filename);
	ds.name = sFilename.substr(sFilename.find_last_of("/\\") + 1);
	if(ds.name.empty()){
		sFilename.pop_back();
		ds.name = sFilename.substr(sFilename.find_last_of("/\\") + 1);
	}
	ds.dataType = DataType::Hierarchichal;	//setting the hierarchical type to indicate hierarchical data
	VertexBufferCreateInfo cI{};
	cI.maxLines = pcSettings.maxHierarchyLines;
	cI.additionalAttributeStorage = 1;		//TODO: change to a variable size
	cI.dataType = DataType::Hierarchichal;
	createPcPlotVertexBuffer(pcAttributes, ds.data, cI);
	ds.buffer = g_PcPlotVertexBuffers.back();
	std::string_view fileView(filename);
	ds.additionalData.insert(ds.additionalData.begin(), reinterpret_cast<const uint8_t*>(fileView.begin()), reinterpret_cast<const uint8_t*>(fileView.end()));

	// adding the default template list
	TemplateList tl = {};
	tl.buffer = g_PcPlotVertexBuffers.back().buffer;
	tl.name = "Default Templatelist";
	tl.isIndexRange = true;
	tl.indices = {0,0};
	for(auto& a: infoAttributes)
		tl.minMax.push_back({a.min, a.max});
	tl.parentDataSetName = ds.name;
	ds.drawLists.push_back(tl);

	g_PcPlotDataSets.push_back(ds);
	return true;
}

static bool openCsv(const char* filename) {

	std::ifstream f(filename, std::ios::in | std::ios::binary);
	std::stringstream input;
	input << f.rdbuf();

	if (!f.is_open()) {
		std::cout << "The given file was not found" << std::endl;
		return false;
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
	std::vector<float> categorieFloats;
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

			int count = 0;
			while ((pos = line.find(delimiter)) != std::string::npos) {
				cur = line.substr(0, pos);
				line.erase(0, pos + delimiter.length());
				if(!queryAttributes[count++].active) continue;		//ignore deactivated attributes
				tmp.push_back({ cur, cur,{},{},std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() });
				attributes.push_back(tmp.back().name);
			}
			//adding the last item which wasn't recognized
			line = line.substr(0, line.find("\r"));
			if(queryAttributes[count++].active){
				tmp.push_back({ line, line,{},{},std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() });
				attributes.push_back(tmp.back().name);
			}

			//checking if the Attributes are correct
			permutation = checkAttriubtes(attributes);
			if (pcAttributes.size() != 0) {
				if (tmp.size() != pcAttributes.size()) {
					std::cout << "The Amount of Attributes of the .csv file is not compatible with the currently loaded datasets" << std::endl;
					f.close();
					return false;
				}

				if (!permutation.size()) {
					std::cout << "The attributes of the .csv data are not the same as the ones already loaded in the program." << std::endl;
					return false;
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

			categorieFloats.resize(pcAttributes.size(), 0);
			firstLine = false;
		}

		//parsing the data which follows the attribute declaration
		else {
			ds.data.columns.resize(pcAttributes.size());
			ds.data.columnDimensions.resize(pcAttributes.size(), {0});		//all columns only depend on the first dimension, the index dimension
			size_t attr = 0;
			float curF = 0;
			int count = 0;
			while ((pos = line.find(delimiter)) != std::string::npos) {
				cur = line.substr(0, pos);
				if(cur[0] == '\"'){
					line.erase(0, pos);
					cur.erase(0,1);
					if(cur.back() == '\"'){
						cur.pop_back();
					}
					else{
						while(line[0] != '\"'){
							cur += line[0];
							line.erase(0, 1);
						}
					}
					// deleting all things to the next comma
					pos = line.find(delimiter);
					line.erase(0, pos + delimiter.length());
				}
				else
					line.erase(0, pos + delimiter.length());
				if(!queryAttributes[count++].active) continue;
				//checking for an overrunning attribute counter
				if (attr == pcAttributes.size()) {
					std::cerr << "The dataset to open is not consitent!" << std::endl;
					f.close();
					return false;
				}

				if (cur.empty()) curF = 0;
				else {
					char* ptr;
					curF = strtof(cur.c_str(), &ptr);
					if ((*ptr) != '\0') {	//cur is not a number
						if (pcAttributes[attr].categories.find(cur) != pcAttributes[attr].categories.end()) {	//the categorie is already in the set
							curF = pcAttributes[attr].categories[cur];
						}
						else {	//add new categorie with a new float for this categorie
							curF = categorieFloats[attr];
							categorieFloats[attr] = categorieFloats[attr] + 1;
							pcAttributes[attr].categories[cur] = curF;
						}
					}
				}

				//updating the bounds if a new highest value was found in the current data.
				if (curF > pcAttributes[permutation[attr]].max)
					pcAttributes[permutation[attr]].max = curF;
				if (curF < pcAttributes[permutation[attr]].min)
					pcAttributes[permutation[attr]].min = curF;

				ds.data.columns[permutation[attr++]].push_back(curF);
			}
			if(!queryAttributes[count++].active) continue;
			if (attr == pcAttributes.size()) {
				std::cerr << "The dataset to open is not consitent!" << std::endl;
				f.close();
				return false;
			}

			//adding the last item which wasn't recognized
			cur = line;//.substr(0, line.size() - 1);
			cur.erase(std::remove(cur.begin(), cur.end(), '\"'), cur.end());
			if (cur.empty()) curF = 0;
			else {
				char* ptr;
				curF = strtof(cur.c_str(), &ptr);
				if ((*ptr) != '\0') {	//cur is not a number
					if (pcAttributes[attr].categories.find(cur) != pcAttributes[attr].categories.end()) {	//the categorie is already in the set
						curF = pcAttributes[attr].categories[cur];
					}
					else {																					//add new categorie with a new float for this categorie
						curF = categorieFloats[attr];
						categorieFloats[attr] = categorieFloats[attr] + 1;
						pcAttributes[attr].categories[cur] = curF;
					}
				}
			}

			//updating the bounds if a new highest value was found in the current data.
			if (curF > pcAttributes[permutation[attr]].max)
				pcAttributes[permutation[attr]].max = curF;
			if (curF < pcAttributes[permutation[attr]].min)
				pcAttributes[permutation[attr]].min = curF;
			ds.data.columns[permutation[attr]].push_back(curF);
		}
	}
	//setting the dataset index dimension size
	ds.data.dimensionSizes = {(uint32_t) ds.data.columns[0].size()};
	ds.data.subsampleTrim({(uint32_t)queryAttributes.back().dimensionSubsample}, {{0, ds.data.dimensionSizes[0]}});
	ds.data.compress();
#ifdef _DEBUG	//debug check for same length columns
	uint32_t columnsSize = 0;
	for(int i = 0; i < ds.data.columns.size(); ++i){
		if(i == 0)
			columnsSize = ds.data.columns[i].size();
		else
			assert(columnsSize == ds.data.columns[i].size());
	}
#endif

	//lexicografically ordering labeled data
	std::vector<std::vector<std::pair<int,int>>> lexicon;	//first has the index of the categorie, second has its corresponding numeric value
	//building the lexicon
	for (uint32_t k = 0; k < pcAttributes.size(); ++k) {
		lexicon.push_back({});
		if (pcAttributes[k].categories.size()) {
			int c = 0;
			for (auto& categorie : pcAttributes[k].categories) {
				lexicon[k].push_back({ c, int(categorie.second) });
				categorie.second = c++;
			}
			std::sort(lexicon[k].begin(), lexicon[k].end(), [](auto& a, auto& b) {return a.second < b.second; });	//after sorting the seconds are ordererd and first then corresponds to the index which should be written instead of second
		}
	}
	//now ordering
	for (int i = 0; i < ds.data.size(); ++i) {
		for (int k = 0; k < pcAttributes.size(); ++k) {
			if (lexicon[k].size()) {
				ds.data(i, k) = lexicon[k][int(ds.data(i, k))].first;
			}
		}
	}

    for (unsigned int k = 0; k < pcAttributes.size(); ++k){
		if (pcAttributes[k].categories.size()) {
			pcAttributes[k].categories_ordered = std::vector<std::pair<std::string, float>>(pcAttributes[k].categories.begin(), pcAttributes[k].categories.end());
			std::sort(pcAttributes[k].categories_ordered.begin(), pcAttributes[k].categories_ordered.end(), [](auto& first, auto& second) {return first.second < second.second; });
			float diff = (pcAttributes[k].categories_ordered.back().second - pcAttributes[k].categories_ordered.front().second) * 0.05f;
			pcAttributes[k].min = pcAttributes[k].categories_ordered.front().second - diff;
			pcAttributes[k].max = pcAttributes[k].categories_ordered.back().second + diff;
		}	
        if (pcAttributes[k].max == pcAttributes[k].min)   {
            pcAttributes[k].max += minRangeEps;
			pcAttributes[k].min -= minRangeEps;
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
			if (ds.data(i,j) < tl.minMax[j].first)
				tl.minMax[j].first = ds.data(i,j);
			if (ds.data(i,j) > tl.minMax[j].second)
				tl.minMax[j].second = ds.data(i,j);
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
	for (int d = 0; d < ds.data.size(); ++d) {
		for (int i = 0; i < pcAttributes.size(); i++) {
			std::cout << ds.data(d, i) << " , ";
		}
		std::cout << std::endl;
		if (dc++ > 10)
			break;
	}
#endif
    return true;
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

static bool openDlf(const char* filename) {
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
				return false;
			}
			else {
				file >> amtOfPoints;
			}
			file >> tmp;
			//checking for the variables section
			if (tmp != std::string("Attributes:")) {
				std::cout << "Attributes section not found. Got " << tmp << " instead" << std::endl;
				return false;
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
						return false;
					}

#ifdef _DEBUG
					std::cout << "The Attribute check was successful" << std::endl;
#endif
				}

				//reading in new values
				else {
					for (int i = 0; i < attributes.size(); i++) {
						pcAttributes.push_back({ attributes[i], attributes[i],{},{},std::numeric_limits<float>::max(),std::numeric_limits<float>::min() - 1 });
					}

					//check for attributes overflow
					if (pcAttributes.size() == 100) {
						std::cout << "Too much attributes found, or Datablock not detected." << std::endl;
						pcAttributes.clear();
						return false;
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
			ds.dataType = DataType::ContinuousDlf;
			if (tmp != std::string("Data:")) {
				std::cout << "Data Section not found. Got " << tmp << " instead." << std::endl;
				pcAttributes.clear();
				return false;
			}
			//reading the data
			else {
				ds.data.dimensionSizes = {(uint32_t)amtOfPoints};
				ds.data.columns.resize(pcAttributes.size(), std::vector<float>(amtOfPoints));
				ds.data.columnDimensions.resize(pcAttributes.size(), {0});
				std::string fname(filename);
				int offset = (fname.find_last_of("/") < fname.find_last_of("\\")) ? fname.find_last_of("/") : fname.find_last_of("\\");
				ds.name = fname.substr(offset + 1);

				file >> tmp;

				int a = 0;
				for (int i = 0; i < amtOfPoints * pcAttributes.size() && tmp != std::string("Drawlists:"); file >> tmp, i++) {
					int datum = i / pcAttributes.size();
					int index = i % pcAttributes.size();
					//index = datum * pcAttributes.size() + permutation[index];
					float d = std::stof(tmp);
					ds.data.columns[permutation[index]][datum] = d;
					if (pcAttributes[a].min > d) {
						pcAttributes[a].min = d;
					}
					if (pcAttributes[a].max < d) {
						pcAttributes[a].max = d;
					}
					a = (a + 1) % pcAttributes.size();
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
					if (ds.data(i,j) < defaultT.minMax[j].first)
						defaultT.minMax[j].first = ds.data(i,j);
					if (ds.data(i,j) > defaultT.minMax[j].second)
						defaultT.minMax[j].second = ds.data(i,j);
				}
			}
			ds.drawLists.push_back(defaultT);

			//reading the draw lists
			if (tmp != std::string("Drawlists:")) {
				std::cout << "Missing Draw lists section. Got " << tmp << " instead" << std::endl;
				pcAttributes.clear();
				return false;
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
							if (ds.data(i,j) < tl.minMax[j].first)
								tl.minMax[j].first = ds.data(i,j);
							if (ds.data(i,j) > tl.minMax[j].second)
								tl.minMax[j].second = ds.data(i,j);
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
        return true;
	}
	else {
		std::cout << "The dlf File could not be opened." << std::endl;
        return false;
	}
}

static std::vector<QueryAttribute> queryCSV(const char* filename) {
	std::ifstream file(filename, std::ios::in);
	std::string firstLine;
	std::getline(file, firstLine);
	std::stringstream lineStream(firstLine);
	
	std::vector<QueryAttribute> ret;
	std::string name;
	while(std::getline(lineStream, name, ',')){
		ret.push_back(QueryAttribute{ std::string(name), 0, 1, 1, 0, 0, -1, true, false });
	}
	ret.push_back(QueryAttribute{ "Index", 1, 0, 1, 0, 0, -1, true, false });
	return ret;
}

static std::vector<QueryAttribute> queryNetCDF(const char* filename) {
	int fileId, retval;
	if ((retval = nc_open(filename, NC_NOWRITE, &fileId))) {
		std::cout << "Error opening the file" << std::endl;
		nc_close(fileId);
		return {};
	}

	int ndims, nvars, ngatts, unlimdimid;
	if ((retval = nc_inq(fileId, &ndims, &nvars, &ngatts, &unlimdimid))) {
		std::cout << "Error reading out viariable information" << std::endl;
		nc_close(fileId);
		return {};
	}

	std::vector<QueryAttribute> out;
	//getting all dimensions to distinguish the size for the data arrays
	uint32_t data_size = 0;
	std::vector<int> dimSizes(ndims);
	std::vector<bool> dimIsStringLenght(ndims);
	for (int i = 0; i < ndims; ++i) {
		size_t dim_size;
		if ((retval = nc_inq_dimlen(fileId, i, &dim_size))) {
			std::cout << "Error reading out dimension size" << std::endl;
			nc_close(fileId);
			return {};
		}
		//check for string length
		bool isStringLength = true;
		for(int j = 0; j < nvars; ++j){
			int dimensions;
			if((retval = nc_inq_varndims(fileId, j, &dimensions))){
				std::cout << "Error at getting variable dimensions(string length check)" << std::endl;
				nc_close(fileId);
				return {};
			}
			std::vector<int> dims(dimensions);
			if((retval = nc_inq_vardimid(fileId, j, dims.data()))){
				std::cout << "ERror at getting varialbe dimensions(string length check)" << std::endl;
				nc_close(fileId);
				return {};
			}
			//only check if current dimension is used
			if(std::find(dims.begin(), dims.end(), i) != dims.end()){
				nc_type varType;
				if((retval = nc_inq_vartype(fileId, j, &varType))){
					std::cout << "Error at getting variable type(string length check)" << std::endl;
					nc_close(fileId);
					return {};
				}
				if(varType != NC_CHAR){
					isStringLength = false;
					break;
				}
			}
		}
		if(!isStringLength){
			if (data_size == 0) data_size = dim_size;
			else data_size *= dim_size;
		}
		//else{
		//	std::cout << "Dim " << i << " is a stringlength dimension" << std::endl;
		//}
		dimSizes[i] = dim_size;
		dimIsStringLenght[i] = isStringLength;
	}
	//std::cout << "netCDF data size: " << data_size << std::endl;
	//for (int i = 0; i < data.size(); ++i) {
	//	data[i].resize(data_size);
	//}

	std::vector<std::vector<int>> attribute_dims;

	char vName[NC_MAX_NAME];
	for (int i = 0; i < nvars; ++i) {
		if ((retval = nc_inq_varname(fileId, i, vName))) {
			std::cout << "Error at reading variables" << std::endl;
			nc_close(fileId);
			return {};
		}
		out.push_back({ std::string(vName), 0, 1, -1, 0, 0, 0, true, false });
		if ((retval = nc_inq_varndims(fileId, i, &out.back().dimensionality))) {
			std::cout << "Error at getting variable dimensions" << std::endl;
			nc_close(fileId);
			return {};
		}
	}

	//creating the indices of the dimensions. Fastest varying is the last of the dimmensions
	std::vector<size_t> iter_indices(ndims), iter_stops(ndims);
	std::vector<int> dimension_variable_indices(ndims);
	for (int i = 0; i < ndims; ++i) {
		char dimName[NC_MAX_NAME];
		if ((retval = nc_inq_dim(fileId, i, dimName, &iter_stops[i]))) {
			std::cout << "Error at reading dimensions 2" << std::endl;
			nc_close(fileId);
			return {};
		}
		//dimensions are not attributes, they can however appear as an attribute too in the list
		//check if dimension is a string length that should not be shown
		if(dimIsStringLenght[i])
			out.push_back(QueryAttribute{ std::string(dimName), -dimSizes[i], 1, 1, 0, 0, dimSizes[i], true, false });
		else
			out.push_back(QueryAttribute{ std::string(dimName), dimSizes[i], 1, 1, 0, 0, dimSizes[i], true, false });
		//if ((retval = nc_inq_varid(fileId, dimName, &dimension_variable_indices[i]))) {
		//	// dimensions can exist as dimensions exclusiveley
		//	std::cout << "Error at getting variable id of dimension" << std::endl;
		//	nc_close(fileId);
		//	return {};
		//}
		//std:: cout << "Dimension " << dimName << " at index " << dimension_variable_indices[i] << " with lenght" << iter_stops[i] << std::endl;
	}
	//int c = 0;
	//for (int i : dimension_variable_indices) {
	//	out[i].dimensionSize = iter_stops[c++];
	//	out[i].trimIndices[1] = out[i].dimensionSize;
	//}

	//everything needed was red, so colosing the file
	nc_close(fileId);

	return out;
}

std::vector<QueryAttribute> queryFileAttributes(const char* filename){
	std::string file(filename);
	if (file.substr(file.find_last_of(".") + 1) == "csv") {
		queryAttributesCsv = true;
		return queryCSV(filename);
	}
	if (file.substr(file.find_last_of(".") + 1) == "nc") {
		queryAttributesCsv = false;
		return queryNetCDF(filename);
	}
	return {};	//default return empty vector
}

static bool openNetCDF(const char* filename){
    int fileId, retval;
    if((retval = nc_open(filename, NC_NOWRITE,&fileId))){
        std::cout << "Error opening the file" << std::endl;
        nc_close(fileId);
        return false;
    }
    
    int ndims, nvars, ngatts, unlimdimid;
    if((retval = nc_inq(fileId, &ndims, &nvars, &ngatts, &unlimdimid))){
        std::cout << "Error at reading out viariable information" << std::endl;
        nc_close(fileId);
        return false;
    }
    
    //attribute check
    std::vector<Attribute> tmp;
    std::vector<std::string> attributes;
    std::vector<std::vector<int>> attribute_dims;
    std::vector<std::vector<int>> variable_dims;
	std::vector<int> attr_to_var;
    
    char vName[NC_MAX_NAME];
    for(int i = 0; i < nvars; ++i){
        if((retval = nc_inq_varname(fileId, i, vName))){
            std::cout << "Error at reading variables" << std::endl;
            nc_close(fileId);
            return false;
        }
		int ndims;
		if ((retval = nc_inq_varndims(fileId, i, &ndims))) {
			std::cout << "Error at getting variable dimensions" << std::endl;
			nc_close(fileId);
			return false;
		}
		variable_dims.push_back(std::vector<int>(ndims));
		if ((retval = nc_inq_vardimid(fileId, i, variable_dims.back().data()))) {
			std::cout << "Error at getting variable dimension array" << std::endl;
			nc_close(fileId);
			return false;
		}
		if (queryAttributes[i].active) {
			tmp.push_back({ vName, vName,{},{},std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() });
			attributes.push_back(tmp.back().name);
			attr_to_var.push_back(i);
			attribute_dims.push_back(variable_dims[i]);
		}
        //std::cout << vName << "(";
        //for(int dim: attribute_dims.back()){
        //    std::cout << dim << ", ";
        //}
        //std::cout << "\b\b)" << std::endl;
    }
    
    //creating the indices of the dimensions. Fastest varying is the last of the dimmensions
    std::vector<size_t> iter_indices(ndims), iter_stops(ndims), iter_increments(ndims, 1), iter_starts(ndims, 0);
	std::vector<size_t> dim_sizes(ndims);
    //std::vector<int> dimension_variable_indices(ndims);
	std::vector<bool> dimension_is_stringsize(ndims);
    for(int i = 0; i < ndims; ++i){
        char dimName[NC_MAX_NAME];
        if((retval = nc_inq_dim(fileId, i, dimName, &dim_sizes[i]))){
            std::cout << "Error at reading dimensions 2" << std::endl;
            nc_close(fileId);
            return false;
        }
        //if((retval = nc_inq_varid(fileId, dimName, &dimension_variable_indices[i]))){
        //    std::cout << "Error at getting variable id of dimension" << std::endl;
        //    nc_close(fileId);
        //    return false;
        //}
		dimension_is_stringsize[i] = queryAttributes[queryAttributes.size() - ndims + i].dimensionSize < 0;
		iter_increments[i] = queryAttributes[queryAttributes.size() - ndims + i].dimensionSubsample;
		iter_starts[i] = queryAttributes[queryAttributes.size() - ndims + i].trimIndices[0];
		iter_stops[i] = queryAttributes[queryAttributes.size() - ndims + i].trimIndices[1];
		iter_indices[i] = iter_starts[i];
		if (!queryAttributes[queryAttributes.size() - ndims + i].active) {
			iter_indices[i] = queryAttributes[queryAttributes.size() - ndims + i].dimensionSlice;
			iter_starts[i] = iter_indices[i];
			iter_stops[i] = iter_indices[i] + 1;
		}
        //std:: cout << "Dimension " << dimName << " at index " << dimension_variable_indices[i] << " with lenght" << iter_stops[i] << std::endl;
    }

	std::vector<float> fill_values(nvars);
	std::vector<float> has_fill_value(nvars);
	std::vector<std::vector<std::string>> categories(nvars);
	//getting all dimensions to distinguish the size for the data arrays
	uint32_t data_size = 0;
	uint32_t reduced_data_size = 1;
	for (int i = 0; i < ndims; ++i) {
		size_t dim_size;
		if ((retval = nc_inq_dimlen(fileId, i, &dim_size))) {
			std::cout << "Error at reading out dimension size" << std::endl;
			nc_close(fileId);
			return false;
		}
		if (data_size == 0) data_size = dim_size;
		else data_size *= dim_size;
		if (!dimension_is_stringsize[i]) {
			dim_size = queryAttributes[queryAttributes.size() - ndims + i].trimIndices[1] - queryAttributes[queryAttributes.size() - ndims + i].trimIndices[0];
			dim_size = dim_size / queryAttributes[queryAttributes.size() - ndims + i].dimensionSubsample + ((dim_size % queryAttributes[queryAttributes.size() - ndims + i].dimensionSubsample) ? 1 : 0);
			reduced_data_size *= dim_size;
		}
	}

	//attribute check
	//checking if the Attributes are correct
	std::vector<int> permutation = checkAttriubtes(attributes);
    if (pcAttributes.size() != 0) {
        if (tmp.size() != pcAttributes.size()) {
            std::cout << "The Amount of Attributes of the .nc file is not compatible with the currently loaded datasets" << std::endl;
            return false;
        }

        if (!permutation.size()) {
            std::cout << "The attributes of the .nc data are not the same as the ones already loaded in the program." << std::endl;
            return false;
        }
	}
	//if this is the first Dataset to be loaded, fill the pcAttributes vector
	else {
		pcAttributes = tmp;

		//setting up the boolarray and setting all the attributes to true
		pcAttributeEnabled = new bool[pcAttributes.size()];
		activeBrushAttributes = new bool[pcAttributes.size()];
		for (int i = 0; i < pcAttributes.size(); i++) {
			pcAttributeEnabled[i] = true;
			activeBrushAttributes[i] = false;
			pcAttrOrd.push_back(i);
		}

		//setting up the categorical datastruct
		for(int i = 0; i < pcAttributes.size(); ++i){
			if(categories[attr_to_var[i]].size()){
				//we do have categorical data
				std::vector<std::pair<std::string,int>> lexicon;
				int c = 0;
				for (auto& categorie : categories[attr_to_var[i]]) {
					lexicon.push_back({ categorie, c });
					c++;
				}
				std::sort(lexicon.begin(), lexicon.end(), [](auto& a, auto& b) {return a.first < b.first; });	//after sorting the names are ordered in lexigraphical order, the seconds are the original indices
		
				for(int j = 0; j < categories[attr_to_var[i]].size() ; ++j){
					pcAttributes[i].categories[lexicon[j].first] = j;
					pcAttributes[i].categories_ordered.push_back({lexicon[j].first, j});
				}
			}
		}
	}

	//preparing the data member of dataset to hold the data
	DataSet ds{};
	ds.data.dimensionSizes = std::vector<uint32_t>(dim_sizes.begin(), dim_sizes.end());
	ds.data.columnDimensions.resize(attribute_dims.size());
	ds.data.columns.resize(tmp.size());
	ds.data.columnDimensions.resize(tmp.size());
	for(int i = 0; i < ds.data.columns.size(); ++i){
		int var = attr_to_var[permutation[i]];
		ds.data.columnDimensions[permutation[i]] = std::vector<uint32_t>(attribute_dims[i].begin(), attribute_dims[i].end());
		int columnSize = 1;
		for (int dim : attribute_dims[i]) {
			columnSize *= ds.data.dimensionSizes[dim];
		}
		ds.data.columns[permutation[i]].resize(columnSize);
	}
	for(int i = 0; i < dim_sizes.size(); ++i){
		if(dimension_is_stringsize[i])
			ds.data.removeDim(i, 0);
	}

	//reading out all data from the netCDF file(including conversion)
	for (int i = 0; i < ds.data.columns.size(); ++i) {
		int var = attr_to_var[permutation[i]];
		nc_type type;
		if ((retval = nc_inq_vartype(fileId, var, &type))){
			std::cout << "Error at reading data type" << std::endl;
			nc_close(fileId);
			return false;
		}
		int hasFill = 0; // if 1 no fill
		float fillValue = 0;
		switch(type){
			case NC_FLOAT:
			if ((retval = nc_get_var_float(fileId, var, ds.data.columns[i].data()))) {
				std::cout << "Error at reading data" << std::endl;
				nc_close(fileId);
				return false;
			}
			if ((retval = nc_inq_var_fill(fileId, var, &hasFill, &fillValue))) {
				std::cout << "Error at reading fill value" << std::endl;
				nc_close(fileId);
				return false;
			}
			break;
			case NC_DOUBLE:
			{
			auto d = std::vector<double>(ds.data.columns[i].size());
			if ((retval = nc_get_var_double(fileId, var, d.data()))) {
				std::cout << "Error at reading data" << std::endl;
				nc_close(fileId);
				return false;
			}
			ds.data.columns[i] = std::vector<float>(d.begin(), d.end());
			double f = 0;
			if ((retval = nc_inq_var_fill(fileId, var, &hasFill, &f))) {
				std::cout << "Error at reading fill value" << std::endl;
				nc_close(fileId);
				return false;
			}
			fillValue = f;
			break;
			}
			case NC_INT:
			{
			auto d = std::vector<int>(ds.data.columns[i].size());
			if ((retval = nc_get_var_int(fileId, var, d.data()))) {
				std::cout << "Error at reading data" << std::endl;
				nc_close(fileId);
				return false;
			}
			ds.data.columns[i] = std::vector<float>(d.begin(), d.end());
			int f = 0;
			if ((retval = nc_inq_var_fill(fileId, var, &hasFill, &f))) {
				std::cout << "Error at reading fill value" << std::endl;
				nc_close(fileId);
				return false;
			}
			fillValue = f;
			break;
			}
			case NC_UINT:
			{
			auto d = std::vector<uint32_t>(ds.data.columns[i].size());
			if ((retval = nc_get_var_uint(fileId, var, d.data()))) {
				std::cout << "Error at reading data" << std::endl;
				nc_close(fileId);
				return false;
			}
			ds.data.columns[i] = std::vector<float>(d.begin(), d.end());
			uint32_t f = 0;
			if ((retval = nc_inq_var_fill(fileId, var, &hasFill, &f))) {
				std::cout << "Error at reading fill value" << std::endl;
				nc_close(fileId);
				return false;
			}
			fillValue = f;
			break;
			}
			case NC_UINT64:{
				auto d = std::vector<unsigned long long>(ds.data.columns[i].size());
				if((retval = nc_get_var_ulonglong(fileId, var, d.data()))){
					std::cout << "Error at reading fill value" << std::endl;
					nc_close(fileId);
					return false;
				}
				ds.data.columns[i] = std::vector<float>(d.begin(), d.end());
				unsigned long long f = 0;
				if ((retval = nc_inq_var_fill(fileId, var, &hasFill, &f))) {
					std::cout << "Error at reading fill value" << std::endl;
					nc_close(fileId);
					return false;
				}
				fillValue = f;
				break;
			}
			case NC_INT64:{
				auto d = std::vector<long long>(ds.data.columns[i].size());
				if((retval = nc_get_var_longlong(fileId, var, d.data()))){
					std::cout << "Error at reading fill value" << std::endl;
					nc_close(fileId);
					return false;
				}
				ds.data.columns[i] = std::vector<float>(d.begin(), d.end());
				long long f = 0;
				if ((retval = nc_inq_var_fill(fileId, var, &hasFill, &f))) {
					std::cout << "Error at reading fill value" << std::endl;
					nc_close(fileId);
					return false;
				}
				fillValue = f;
				break;
			}
			case NC_UBYTE:
			case NC_BYTE:{
				auto d = std::vector<uint8_t>(ds.data.columns[i].size());
				if((retval = nc_get_var_ubyte(fileId, var, d.data()))){
					std::cout << "Error at reading fill value" << std::endl;
					nc_close(fileId);
					return false;
				}
				ds.data.columns[i] = std::vector<float>(d.begin(), d.end());
				uint8_t f = 0;
				if ((retval = nc_inq_var_fill(fileId, var, &hasFill, &f))) {
					std::cout << "Error at reading fill value" << std::endl;
					nc_close(fileId);
					return false;
				}
				fillValue = f;
				break;
			}
			case NC_CHAR:
			//categorical data
			{
				int dataSize = 1;
				int amtOfDims;
				if((retval = nc_inq_varndims(fileId, var, &amtOfDims))){
					std::cout << "Error at reading dimsizes for categorical data" << std::endl;
					nc_close(fileId);
					return false;
				}
				std::vector<int> dims(amtOfDims);
				if((retval = nc_inq_vardimid(fileId, var, dims.data()))){
					std::cout << "Error at reading dims for categorical data" << std::endl;
					nc_close(fileId);
					return false;
				}
				int wordlen = 0;
				for(auto dim: dims){
					dataSize *= std::abs(queryAttributes[queryAttributes.size() - ndims + dim].dimensionSize);
					if(queryAttributes[queryAttributes.size() - ndims + dim].dimensionSize < 0){
						wordlen = -queryAttributes[queryAttributes.size() - ndims + dim].dimensionSize;
					}
				}
				std::vector<char> names(dataSize);
				if((retval = nc_get_var_text(fileId, i, names.data()))){
					std::cout << "Error at reading categorical data" << std::endl;
					nc_close(fileId);
					return false;
				}
				int c = 0;
				for(int offset = 0; offset < dataSize; offset += wordlen){
					categories[i].push_back(std::string(&names[offset], &names[offset] + wordlen));
					pcAttributes[i].categories[categories[i].back()] = c++;
					pcAttributes[i].categories_ordered.push_back({categories[i].back(), float(pcAttributes[i].categories[categories[i].back()])});
					ds.data.columns[i][offset / wordlen] = pcAttributes[i].categories[categories[i].back()];
				}
				std::sort(pcAttributes[i].categories_ordered.begin(), pcAttributes[i].categories_ordered.end(), [&](auto& left, auto& right){return left.second < right.second;});
			}
			break;
			default:
				std::cout << "The variable type " << type << " can not be handled correctly!" << std::endl;
		}
		if (hasFill != 1) {
			fill_values[i] = fillValue;
			has_fill_value[i] = true;
		}
	}
    
    //everything needed was red, so colosing the file
    nc_close(fileId);

	//linearizing a column if wanted
	for (int i = 0, c = 0; i < queryAttributes.size(); ++i) {
		if (!queryAttributes[i].active) continue;
		if (queryAttributes[i].linearize) {
			ds.data.linearizeColumn(permutation[c]);
		}
		++c;
	}
    
	std::string fname(filename);
	int offset = (fname.find_last_of("/") < fname.find_last_of("\\")) ? fname.find_last_of("/") : fname.find_last_of("\\");
	ds.name = fname.substr(offset + 1);
    
	//reducing the dataset
	std::vector<uint32_t> samplingRates(ds.data.dimensionSizes.size());
	std::vector<std::pair<uint32_t, uint32_t>> trimIndices(samplingRates.size());
	int c = 0;
	for(int i = 0; i < iter_starts.size(); ++i){
		if(dimension_is_stringsize[i]) continue;
		samplingRates[c] = iter_increments[i];
		trimIndices[c].first = iter_starts[i];
		trimIndices[c].second = iter_stops[i];
		++c;
	}
	ds.data.subsampleTrim(samplingRates, trimIndices);
	ds.data.compress();
	ds.reducedDataSetSize = ds.data.size();

	uint64_t packedSize = ds.data.packedByteSize();
	if(ds.data.packedByteSize() > g_MaxStorageBufferSize){		//The data has to be split up.
		std::cout << "The byte size needed for the GPU buffer is too large!" << std::endl;
		std::cout << "In total " << (ds.data.packedByteSize() >> 20) << "MByte is needed, but only " << (g_MaxStorageBufferSize >> 20) << "MByte are allowed." << std::endl;
		std::cout << "Reduce the data size by " << 100.0f - 100.0f * (float(g_MaxStorageBufferSize) / ds.data.packedByteSize()) << "% to load it" << std::endl;
		return false;
	}

	TemplateList tl = {};
	tl.name = "Default Drawlist";
	for (int i = 0; i < ds.data.size(); i++) {
		tl.indices.push_back(i);
	}
	tl.pointRatio = tl.indices.size() / (float)ds.data.size();

	//getting the minimum and maximum values for all attributes. This will later be used for brush creation
	for (int i = 0; i < pcAttributes.size(); i++) {
		tl.minMax.push_back(std::pair<float, float>(std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()));
	}
	for (int j = 0; j < pcAttributes.size(); j++) {
		for (float f: ds.data.columns[j]) {
			//ignoring fill values
			if (has_fill_value[j] && f == fill_values[j] && pcAttributes[j].categories.empty())
				continue;
			if (f < tl.minMax[j].first)
				tl.minMax[j].first = f;
			if (f > tl.minMax[j].second)
				tl.minMax[j].second = f;
            //updating pcAttributes minmax if needed
            if(tl.minMax[j].first < pcAttributes[j].min)
                pcAttributes[j].min = tl.minMax[j].first;
            if(tl.minMax[j].second > pcAttributes[j].max)
                pcAttributes[j].max = tl.minMax[j].second;
		}
	}

	for (int j = 0; j < pcAttributes.size(); ++j) {
		if (pcAttributes[j].min == pcAttributes[j].max) {
			pcAttributes[j].min -= .1;
			pcAttributes[j].max += .1;
		}
		else if (std::isinf(pcAttributes[j].min) && std::isinf(pcAttributes[j].max)) {
			pcAttributes[j].min = -.1;
			pcAttributes[j].max = .1;
		}
	}
	for (int j = 0; j < pcAttributes.size(); j++) {
		for (float& f: ds.data.columns[j]) {
			//replace the fill values with better suited ones.
			if (has_fill_value[j] && f == fill_values[j] && pcAttributes[j].categories.empty())
				f = 2 * pcAttributes[j].max - pcAttributes[j].min;
		}
	}

	createPcPlotVertexBuffer(pcAttributes, ds.data);

	ds.buffer = g_PcPlotVertexBuffers.back();
	tl.buffer = g_PcPlotVertexBuffers.back().buffer;
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
	for (int d = 0; d < ds.data.size(); ++d) {
		for (int i = 0; i < pcAttributes.size(); i++) {
			std::cout << ds.data(d, i) << " , ";
		}
		std::cout << std::endl;
		if (dc++ > 10)
			break;
	}
    #endif
    return true;
}

static void sortAttributes() {
	pcAttributesSorted.resize(pcAttributes.size());
	for (int i = 0; i < pcAttributesSorted.size(); ++i) pcAttributesSorted[i] = i;
	std::sort(pcAttributesSorted.begin(), pcAttributesSorted.end(), [&](int lhs, int rhs) {return pcAttributes[lhs].name < pcAttributes[rhs].name; }); //now one can print sorted by iteratin through the vector and printing the name of the attribute at the current iterator index
}

static void saveRecentFiles() {
	int const stringLength = 250;
	SettingsManager::Setting s{};
	s.id = "RecentFiles";
	s.type = "RecentFiles";
	s.byteLength = stringLength * recentFiles.size();
	s.data = new char[stringLength * recentFiles.size()];
	char* cur = (char*)s.data;
	for (auto& f : recentFiles) {
		strcpy(cur, f.c_str());
		cur += stringLength;
	}
	settingsManager->addSetting(s);
	delete[] static_cast<char*>(s.data);
}

static void loadRecentFiles() {
	int const stringLength = 250;
	SettingsManager::Setting& s = settingsManager->getSetting("RecentFiles");
	recentFiles = std::vector<std::string>(s.byteLength / stringLength);
	char* cur = (char*)s.data;
	for (int i = 0; i < recentFiles.size(); ++i, cur += stringLength) {
		recentFiles[i] = cur;
	}
}

static bool openDataset(const char* filename) {
	//checking the datatype and calling the according method
	std::string file = filename;
    bool opened = false;
	if(std::string_view(file).substr(file.find_last_of("/\\")).find_last_of(".") == std::string_view::npos){	//no file but a folder
		std::string hierarchyInfo;
		for(const auto& entry: std::filesystem::directory_iterator(filename)){
			//auto ext = entry.path().extension();
			//if(ext.string().size())
			//	std::cout << ext.string() << std::endl;
			if(entry.is_regular_file() && entry.path().extension() == ".info"){
				hierarchyInfo = entry.path();
				break;
			}
		}
		if(hierarchyInfo.size()){	//opening hierarchy dataset
			opened = openHierarchy(filename, hierarchyInfo.c_str());
		}
	}
	else if (file.substr(file.find_last_of(".") + 1) == "csv") {
		opened = openCsv(filename);
	}
	else if (file.substr(file.find_last_of(".") + 1) == "dlf") {
		opened = openDlf(filename);
	}
    else if (file.substr(file.find_last_of(".") + 1) == "nc"){
        opened = openNetCDF(filename);
    }
	else {
		std::cout << "The given type of the file is not supported by this programm" << std::endl;
		return false;
	}
    if(!opened){
		pcAttributes.clear();
		pcAttrOrd.clear();
		pcAttributesSorted.clear();
		return false;
	}
	//printing Amount of data loaded
	std::cout << "Amount of data loaded: " << g_PcPlotDataSets.back().data.size() << std::endl;

	//adding path to the recent files list if not yet added and resizing the list if too large
	if (std::find(recentFiles.begin(), recentFiles.end(), std::string(filename)) == recentFiles.end()) {
		recentFiles.push_back(filename);
		if (recentFiles.size() > recentFilesAmt) {
			recentFiles = std::vector<std::string>(recentFiles.begin() + 1, recentFiles.end());
		}
		saveRecentFiles();
	}

	//standard things which should be done on loading of a dataset
	//adding a standard attributes saving
	pcSettings.histogrammWidth = 1.0f / (pcAttributes.size() * 5);
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

	sortAttributes();		//sorting the attributes to display in alphabetical order

	delete[] static_cast<char*>(s.data);
    return true;
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
				if (ds.data(i,j) < tl.minMax[j].first)
					tl.minMax[j].first = ds.data(i,j);
				if (ds.data(i,j) > tl.minMax[j].second)
					tl.minMax[j].second = ds.data(i,j);
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

	if (violinPlotDrawlistSettings.violinYScale == ViolinYScaleLocalBrush || violinPlotDrawlistSettings.violinYScale == ViolinYScaleBrushes) {
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

	if (violinPlotDrawlistSettings.violinYScale == ViolinYScaleGlobalBrush || violinPlotDrawlistSettings.violinYScale == ViolinYScaleBrushes) {
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

	if (violinPlotDrawlistSettings.violinYScale == ViolinYScaleStandard)
	{
		for (int k = 0; k < pcAttributes.size(); ++k) {
			// Find the attribute in the PC plot to determine min and max values.
			std::string currAttributeName = violinDrawlistPlot.attributeNames[k];
			auto it = std::find_if(pcAttributes.begin(), pcAttributes.end(),
				[&violinDrawlistPlot, k](const Attribute& currObj) {return currObj.name == violinDrawlistPlot.attributeNames[k];  });

			violinMinMax[k] = std::pair<float,float>(it->min, it->max);

		}
		return;
		//return violinMinMax;
	}


	DrawList* dl = nullptr;
	if (violinPlotDrawlistSettings.violinYScale == ViolinYScaleLocalBrush || violinPlotDrawlistSettings.violinYScale == ViolinYScaleBrushes) {
		for (DrawList& draw : g_PcPlotDrawLists) {
			if (draw.name == violinDrawlistPlot.drawLists[dlNr]) {
				dl = &draw;
			}
		}
	}

	switch (violinPlotDrawlistSettings.violinYScale) {
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

    if (violinPlotDrawlistSettings.violinYScale == ViolinYScaleStandard)
    {
        for (int k = 0; k < pcAttributes.size(); ++k) {
            // Find the attribute in the PC plot to determine min and max values.
            std::string currAttributeName = violinAttrPlot.attributeNames[k];
            auto it = std::find_if(pcAttributes.begin(), pcAttributes.end(),
                [&violinAttrPlot, k](const Attribute& currObj) {return currObj.name == violinAttrPlot.attributeNames[k];  });

            violinMinMax[k] = std::pair<float, float>(it->min, it->max);

        }
        return;
        //return violinMinMax;
    }


    DrawList* dl = nullptr;
    if (violinPlotDrawlistSettings.violinYScale == ViolinYScaleLocalBrush || violinPlotDrawlistSettings.violinYScale == ViolinYScaleBrushes) {
        for (DrawList& draw : g_PcPlotDrawLists) {
            if (draw.name == violinAttrPlot.drawLists[dlNr].name) {
                dl = &draw;
            }
        }
    }

    switch (violinPlotDrawlistSettings.violinYScale) {
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
static void exeComputeHistogram(std::string& name, std::vector<std::pair<float, float>>& minMax, VkBuffer data, uint32_t amtOfData, VkBuffer indices, uint32_t amtOfIndices, VkBufferView indicesActivations, bool callForviolinAttributePlots) {
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

	//updateing drawlist line bundles
	if(dl.lineBundles){
		std::vector<std::pair<uint32_t, bool>> ord(pcAttributes.size());
		for(int i = 0; i < pcAttributes.size(); ++i){
			ord[i] = {pcAttrOrd[i], pcAttributeEnabled[pcAttrOrd[i]]};
		}
		dl.lineBundles->updateAttributeOrdering(ord);
	}

	//updating drawlist cluster bundles
	if(dl.clusterBundles){
		std::vector<std::pair<uint32_t, bool>> ord(pcAttributes.size());
		for(int i = 0; i < pcAttributes.size(); ++i){
			ord[i] = {pcAttrOrd[i], pcAttributeEnabled[pcAttrOrd[i]]};
		}
		dl.clusterBundles->updateAttributeOrdering(ord);
	}

	//ordering active indices if priority rendering is enabled
	if (priorityReorder) {
		priorityReorder = false;
		Data* data;
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
		std::sort(dl.indices.begin(), dl.indices.end(), [data, p](int a, int b) {return fabs((*data)(a,p) - priorityAttributeCenterValue) > fabs((*data)(b,p) - priorityAttributeCenterValue); });
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

	Data* data;
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
		color[i] = 1 - .9f * (fabs((*data)(i,priorityAttribute) - priorityAttributeCenterValue) / denom);
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
	if ((ImGui::IsMouseDown(0) && pcSettings.liveBrushThreshold < amtOfLines) || !isoSurfSettings.coupleIsoSurfaceRenderer) return;
	if (brushIsoSurfSettings.coupleBrushIsoSurfaceRenderer && brushIsoSurfSettings.enabled) {
		if (brushIsoSurfaceRenderer->brushColors.find(gb.id) != brushIsoSurfaceRenderer->brushColors.end()) {
			std::vector<std::vector<std::pair<float, float>>> minMax(pcAttributes.size());
			for (auto& axis : gb.brushes) {
				for (auto& m : axis.second) {
					minMax[axis.first].push_back(m.second);
				}
			}
			brushIsoSurfaceRenderer->updateBrush(gb.id, minMax);
		}
	}
	int index = -1;
	for (auto& db : isoSurfaceRenderer->drawlistBrushes) {
		++index;
		if (gb.id != db.brush) continue;
		DrawList* dl = nullptr;
		for (DrawList& draw : g_PcPlotDrawLists) {
			if (draw.name == db.drawlist) {
				dl = &draw;
				break;
			}
		}
		if (!dl) continue;
		DataSet* ds = nullptr;
		for (DataSet& d : g_PcPlotDataSets) {
			if (dl->parentDataSet == d.name) {
				ds = &d;
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

		auto& xDim = getDimensionValues(*ds, isoSurfSettings.posIndices.x), yDim = getDimensionValues(*ds, isoSurfSettings.posIndices.y), zDim = getDimensionValues(*ds, isoSurfSettings.posIndices.z);
		uint32_t w = xDim.second.size();
		uint32_t h = yDim.second.size();
		uint32_t d = zDim.second.size();
		bool regularGrid[3]{ xDim.first, yDim.first, zDim.first };

		isoSurfaceRenderer->update3dBinaryVolume(xDim.second, yDim.second, zDim.second, pcAttributes.size(), brushIndices, minMax, posIndices, dl->buffer, ds->data.size() * pcAttributes.size() * sizeof(float), dl->indicesBuffer, dl->indices.size(), miMa, index);
	}
}

static void updateIsoSurface(DrawList& dl) {
	int amtOfLines = 0;
	for (auto& dl : g_PcPlotDrawLists) amtOfLines += dl.indices.size();
	if ((ImGui::IsMouseDown(0) && pcSettings.liveBrushThreshold < amtOfLines) || !isoSurfSettings.coupleIsoSurfaceRenderer) return;

	int index = -1;
	for (auto& db : isoSurfaceRenderer->drawlistBrushes) {
		++index;
		if (dl.name != db.drawlist || db.brush.size()) continue;
		uint32_t posIndices[3];
		isoSurfaceRenderer->getPosIndices(index, posIndices);
		std::vector<std::pair<float, float>> posBounds(3);
		for (int i = 0; i < 3; ++i) {
			posBounds[i].first = pcAttributes[posIndices[i]].min;
			posBounds[i].second = pcAttributes[posIndices[i]].max;
		}
		DataSet* ds;
		for (DataSet& d : g_PcPlotDataSets) {
			if (d.name == dl.parentDataSet) {
				ds = &d;
				break;
			}
		}
		auto& xDim = getDimensionValues(*ds, isoSurfSettings.posIndices.x), yDim = getDimensionValues(*ds, isoSurfSettings.posIndices.y), zDim = getDimensionValues(*ds, isoSurfSettings.posIndices.z);
		uint32_t w = xDim.second.size();
		uint32_t h = yDim.second.size();
		uint32_t d = zDim.second.size();
		bool regularGrid[4]{ xDim.first, yDim.first, zDim.first };
		isoSurfaceRenderer->update3dBinaryVolume(xDim.second, yDim.second, zDim.second, posIndices, posBounds, pcAttributes.size(), ds->data.size(), dl.buffer, dl.activeIndicesBufferView, dl.indices.size(), dl.indicesBuffer, regularGrid, index);
	}
}

static void updateWorkbenchRenderings(DrawList& dl){
	//rendering the updated active points in the bubble plotter
	if (bubbleWindowSettings.coupleToBrushing) {
		bubblePlotter->render();
	}

	if ((violinPlotDrawlistSettings.coupleViolinPlots || violinPlotAttributeSettings.coupleViolinPlots) && histogramManager->containsHistogram(dl.name)) {
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
		updateAllViolinPlotMaxValues(violinPlotDrawlistSettings.renderOrderBasedOnFirstDL);
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

	if (isoSurfSettings.coupleIsoSurfaceRenderer && isoSurfSettings.enabled) {
		updateIsoSurface(dl);
	}

	if (scatterplotWorkbench->active){
		std::vector<int> indices(pcAttributes.size());
		for(int i = 0; i < pcAttributes.size(); ++i) indices[i] = i;
		scatterplotWorkbench->updateRenders(indices);
	}

	if (correlationMatrixWorkbench->active){
		correlationMatrixWorkbench->updateCorrelationScores(g_PcPlotDrawLists, {dl.name});
	}
}

static bool updateActiveIndices(DrawList& dl) {
	if(dl.data->size() == 0) return false;		// can happen for hierarchy files with delayed loading
	//safety check to avoid updates of large drawlists. Update only occurs when mouse was released
	if (dl.indices.size() > pcSettings.liveBrushThreshold) {
		if (ImGui::GetIO().MouseDown[0] && !ImGui::IsMouseDoubleClicked(0)) return false;
	}

	//getting the parent dataset data
	Data* data;
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
		GpuBrusher::ReturnStruct res = gpuBrusher->brushIndices(brush, data->size(), dl.buffer, dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView, pcAttributes.size(), true, pcSettings.brushCombination == 1, globalBrushes.size() == 0);
		globalRemainingLines = res.activeLines;
		firstBrush = false;
	}
	
	//apply global brushes
	std::vector<int> globalIndices;
	bool globalBrushesActive = false;
	if (pcSettings.toggleGlobalBrushes && !dl.immuneToGlobalBrushes) {
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
				GpuBrusher::ReturnStruct res;// = gpuBrusher->brushIndices(gb.fractions, gb.attributes, data->size(), dl.buffer, dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView, pcAttributes.size(), firstBrush, brushCombination == 1, c == globalBrushes.size());
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
					res = gpuBrusher->brushIndices(gb.multivariates, gb.kdTree->getOriginalBounds(), gb.attributes, data->size(), dl.buffer, dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView, pcAttributes.size(), firstBrush, pcSettings.brushCombination == 1, c == globalBrushes.size(), pcSettings.multivariateStdDivThresh);
				}
				else {
					res = gpuBrusher->brushIndices(gb.fractions, gb.attributes, data->size(), dl.buffer, dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView, pcAttributes.size(), firstBrush, pcSettings.brushCombination == 1, c == globalBrushes.size());
				}
				gb.lineRatios[dl.name] = res.singleBrushActiveLines;
				globalRemainingLines = res.activeLines;
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
				GpuBrusher::ReturnStruct res = gpuBrusher->brushIndices(brush, data->size(), dl.buffer, dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView, pcAttributes.size(), firstBrush, pcSettings.brushCombination == 1, c == globalBrushes.size());
				gb.lineRatios[dl.name] = res.singleBrushActiveLines;
				globalRemainingLines = res.activeLines;
				firstBrush = false;
				++c;
			}
		}
	}

	//apply lasso brushes
	bool lassoBrushExists = scatterplotWorkbench->lassoSelections.find(dl.name) != scatterplotWorkbench->lassoSelections.end();
	if(lassoBrushExists){
		GpuBrusher::ReturnStruct res = gpuBrusher->brushIndices(scatterplotWorkbench->lassoSelections[dl.name], data->size(), dl.buffer, dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView, pcAttributes.size(), firstBrush, pcSettings.brushCombination == 1, true);
		firstBrush = false;
		globalRemainingLines = res.activeLines;
	}

	//if no brush is active, reset the active indices
	if (!brush.size() && !globalBrushesActive && !lassoBrushExists) {
		Data* data;
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
	dl.activeLinesAmt = globalRemainingLines;

	// Computing ratios for the pie charts
	if (pcSettings.computeRatioPtsInDLvsIn1axbrushedParent && pcSettings.drawHistogramm) {
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
					GpuBrusher::ReturnStruct res = gpuBrusher->brushIndices(brush, data->size(), dl.buffer, dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView, pcAttributes.size());
					if(res.singleBrushActiveLines)
						dl.brushedRatioToParent[b.first] = float(globalRemainingLines) / res.singleBrushActiveLines;
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
							for(int i = 0; i < parentDS->data.size(); ++i)
							{
								iVal++;
								float v = parentDS->data(i,ax);
								if ((v >= currBrMin) && (v <= currBrMax))
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
							GpuBrusher::ReturnStruct res = gpuBrusher->brushIndices(brush, data->size(), dl.buffer, dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView, pcAttributes.size());

							std::cout << res.singleBrushActiveLines << "\n";

							float currBrMin = currBr.second.first;
							float currBrMax = currBr.second.second;
							for(int i = 0; i < parentDS->data.size(); ++i)
							{
								float v = parentDS->data(i,iax);
								if ((v >= currBrMin) && (v <= currBrMax))
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

	updateWorkbenchRenderings(dl);

	//setting the median to no median to enforce median recalculation
	dl.activeMedian = 0;
	return true;
}

//This method does the same as updataActiveIndices, only for ALL drawlists
//whenever possible use updataActiveIndices, not updateAllActiveIndicess
//The return value indicates if brushing was performed (The method checks for live update)
static bool updateAllActiveIndices() {
	bool ret = false;
	for (DrawList& dl : g_PcPlotDrawLists) {
		ret = updateActiveIndices(dl);
	}
	return ret;
}

static void uploadDensityUiformBuffer() {
	DensityUniformBuffer ubo = {};
	ubo.enableMapping = pcSettings.enableDensityMapping | ((uint8_t)(pcSettings.histogrammDensity && pcSettings.enableDensityMapping)) * 2 | uint32_t(pcSettings.enableDensityGreyscale)<<2;
	ubo.gaussRange = pcSettings.densityRadius;
	ubo.imageHeight = g_PcPlotHeight;
	int amtOfIndices = 0;
	for (int i = 0; i < pcAttributes.size(); i++) {
		if (pcAttributeEnabled[i]) amtOfIndices++;
	}
	ubo.gap = (1 - pcSettings.histogrammWidth / 2) / (amtOfIndices - 1);
	if (pcSettings.histogrammDrawListComparison != -1) {
		float offset = 0;
		int activeHists = 0;
		int c = 0;
		for (auto it = g_PcPlotDrawLists.begin(); it != g_PcPlotDrawLists.end(); ++it, c++) {
			if (it->showHistogramm) {
				activeHists++;
				if (c == pcSettings.histogrammDrawListComparison) {
					offset = activeHists;
				}
			}
			else if (c == pcSettings.histogrammDrawListComparison) {
				std::cout << "Histogramm to compare to is not active." << std::endl;
			}
		}
		ubo.compare = (offset / activeHists - (1 / (2.0f * activeHists))) * pcSettings.histogrammWidth / 2;
	}
	else {
		ubo.compare = -1;
	}
	void* d;
	vkMapMemory(g_Device, g_PcPlotIndexBufferMemory, g_PcPLotDensityUboOffset, sizeof(DensityUniformBuffer), 0, &d);
	memcpy(d, &ubo, sizeof(DensityUniformBuffer));
	vkUnmapMemory(g_Device, g_PcPlotIndexBufferMemory);
}

static void uploadDrawListTo3dView(DrawList& dl, int attribute) {
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

	auto& xDim = getDimensionValues(*parent, view3dSettings.posIndices[0]), yDim = getDimensionValues(*parent, view3dSettings.posIndices[1]), zDim = getDimensionValues(*parent, view3dSettings.posIndices[2]);
	bool linDims[3]{ xDim.first, yDim.first, zDim.first };
	float minMax[2]{ pcAttributes[attribute].min, pcAttributes[attribute].max };

	view3d->update3dImage(xDim.second, yDim.second, zDim.second, linDims, view3dSettings.posIndices, attribute, minMax, parent->buffer.buffer, parent->data.size() * pcAttributes.size() * sizeof(float), dl.indicesBuffer, dl.indices.size(), pcAttributes.size());
}

static void exportBrushAsCsv(DrawList& dl, const  char* filepath) {
	std::string path(filepath);
	if (path.substr(path.find_last_of('.')) != ".csv") {
#ifdef _DEBUG
		std::cout << "The filepath with filename given was not a .csv file. Instead " << path.substr(path.find_last_of('.')) << " was found." << std::endl;
#endif
		return;
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
			file << ds->data(i,j);
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
#ifdef _DEBUG
		std::cout << "The filepath with filename given was not a .idxf file. Instead " << path.substr(path.find_last_of('.')) << " was found." << std::endl;
#endif
		return;
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


static void exportTemplateListAsCsv(TemplateList& tl, const char* filepath){
	std::string path(filepath);
	if (path.substr(path.find_last_of('.')) != ".csv") {
#ifdef _DEBUG
		std::cout << "The filepath with filename given was not a .csv file. Instead " << path.substr(path.find_last_of('.')) << " was found." << std::endl;
#endif
		return;
	}

	auto ds = std::find_if(g_PcPlotDataSets.begin(), g_PcPlotDataSets.end(), [&](DataSet& d){return d.name == tl.parentDataSetName;});

	std::ofstream file(filepath);
	//adding the attributes
	for (int i = 0; i < pcAttributes.size(); i++) {
		file << pcAttributes[i].name;
		if (i != pcAttributes.size() - 1)
			file << ",";
	}
	file << "\n";
	//adding the data;
	for (int i : tl.indices) {
		for (int j = 0; j < pcAttributes.size(); j++) {
			file << ds->data(i,j);
			if (j != pcAttributes.size() - 1)
				file << ",";
		}
		file << "\n";
	}
}

static void exportTemplateListAsIdxf(TemplateList& tl, const char* filepath){
	std::string path(filepath);
	if (path.substr(path.find_last_of('.')) != ".idxf") {
#ifdef _DEBUG
		std::cout << "The filepath with filename given was not a .idxf file. Instead " << path.substr(path.find_last_of('.')) << " was found." << std::endl;
#endif
		return;
	}
	std::ofstream file(filepath);
	for (int i : tl.indices) {
		file << i << "\n";
	}
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
	if (!pcSettings.calculateMedians)
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
		std::sort(dataCpy.begin(), dataCpy.end(), [i, ds](int a, int b) {return ds->data(a,i) > ds->data(b,i); });
		medianArr[MEDIAN * pcAttributes.size() + i] = ds->data(dataCpy[dataCpy.size() >> 1],i);
	}

	//arithmetic median calculation
	for (int i = 0; i < actIndices.size(); i++) {
		for (int j = 0; j < pcAttributes.size(); j++) {
			if (i == 0)
				medianArr[ARITHMEDIAN * pcAttributes.size() + j] = ds->data(actIndices[i],j) / actIndices.size();
			medianArr[ARITHMEDIAN * pcAttributes.size() + j] += ds->data(actIndices[i],j) / actIndices.size();
		}
	}
	//for (int i = 0; i < pcAttributes.size(); i++) {
	//	medianArr[ARITHMEDIAN * pcAttributes.size() + i] /= actIndices.size();
	//}

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
		if (!violinPlotDrawlistSettings.renderOrderDLConsider) {
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
				if (violinPlotDrawlistSettings.renderOrderDLConsider && ((drawL == 0) || (!violinPlotDrawlistSettings.renderOrderBasedOnFirstDL))) {
					drawListPlot.attributeOrder[drawL] = sortHistogram(hist, drawListPlot, violinPlotDrawlistSettings.renderOrderDLConsider, violinPlotDrawlistSettings.renderOrderDLReverse);
				}
				else if ((violinPlotDrawlistSettings.renderOrderBasedOnFirstDL && drawL > 0)) {
					break;
				}
			}
			if ((violinPlotDrawlistSettings.renderOrderBasedOnFirstDL && drawL > 0)) {
				break;
			}
		}
	}
	else if (option == "attr") {
		std::cout << "Automatic non-stop reordering of attribute violins is not implemented yet. \n";
		return;
	}


}

static void updateSummedBins(ViolinPlot& plot) {
	//updating summed histograms
	plot.summedBins = std::vector<std::vector<float>>(plot.maxValues.size(), std::vector<float>(violinPlotBinsSize));	//nulling summed array
	plot.maxSummedValues.resize(plot.summedBins.size());
	for (auto& dl : plot.drawLists) {
		if (!dl.activated) continue;
		HistogramManager::Histogram& histogram = histogramManager->getHistogram(dl.name);
		for (int attribute = 0; attribute < histogram.bins.size(); ++attribute) {
			for (int bin = 0; bin < violinPlotBinsSize; ++bin) {
				plot.summedBins[attribute][bin] += histogram.bins[attribute][bin];
			}
		}
		for (int attribute = 0; attribute < plot.summedBins.size(); ++attribute) {
			plot.maxSummedValues[attribute] = *std::max_element(plot.summedBins[attribute].begin(), plot.summedBins[attribute].end());
		}
	}
}

inline void updateAllViolinPlotMaxValues(bool renderOrderBasedOnFirst) {
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

			if (violinPlotDrawlistSettings.renderOrderDLConsider && ((drawL == 0) || (!renderOrderBasedOnFirst)))
			{
				drawListPlot.attributeOrder[drawL] = sortHistogram(hist, drawListPlot, violinPlotDrawlistSettings.renderOrderDLConsider, violinPlotDrawlistSettings.renderOrderDLReverse);
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
	for (auto& drawListPlot : violinAttributePlots) {
		updateSummedBins(drawListPlot);
		drawListPlot.maxGlobalValue = 0;
		for (int j = 0; j < drawListPlot.maxValues.size(); ++j) {
			drawListPlot.maxValues[j] = 0;
		}
		int drawL = 0;
		for (auto& drawList : drawListPlot.drawLists) {
			HistogramManager::Histogram& hist = histogramManager->getHistogram(drawList.name);
			for (int j = 0; j < hist.maxCount.size(); ++j) {
				if (hist.maxCount[j] > drawListPlot.maxValues[j]) {
					drawListPlot.maxValues[j] = hist.maxCount[j];
				}
				if (hist.maxCount[j] > drawListPlot.maxGlobalValue) {
					drawListPlot.maxGlobalValue = hist.maxCount[j];
				}
			}
		}
	}
}


static void optimizeViolinSidesAndAssignCustColors() {
	if (violinAdaptSidesAutoObj.optimizeSidesNowAttr)
	{
		auto& hist = histogramManager->getHistogram(violinAdaptSidesAutoObj.vp->drawLists[0].name);
		histogramManager->determineSideHist(hist, &(violinAdaptSidesAutoObj.vp->activeAttributes), violinPlotAttributeSettings.violinPlotAttrConsiderBlendingOrder);

		violinAdaptSidesAutoObj.vp->violinPlacements.clear();
		for (int j = 0; j < violinAdaptSidesAutoObj.vp->attributeNames.size(); ++j) {
			violinAdaptSidesAutoObj.vp->violinPlacements.push_back((hist.side[j] % 2) ? ViolinLeft : ViolinRight);


		}
		if (violinPlotAttributeSettings.violinPlotAttrInsertCustomColors || violinPlotAttributeSettings.violinPlotAttrConsiderBlendingOrder) {
			changeColorsToCustomAlternatingColors((violinAdaptSidesAutoObj.vp->colorPaletteManager), violinAdaptSidesAutoObj.vp->attributeNames.size(), &(violinAdaptSidesAutoObj.vp->drawListLineColors), &(violinAdaptSidesAutoObj.vp->drawListFillColors),
				hist, &(violinAdaptSidesAutoObj.vp->activeAttributes), violinPlotAttributeSettings.violinPlotAttrInsertCustomColors);
		}
		violinAdaptSidesAutoObj.optimizeSidesNowAttr = false;
	}
	///


	if (violinAdaptSidesAutoObj.optimizeSidesNowDL)
	{
		auto& hist = histogramManager->getHistogram(violinAdaptSidesAutoObj.vdlp->drawLists[0]);
		histogramManager->determineSideHist(hist, &(violinAdaptSidesAutoObj.vdlp->activeAttributes), violinPlotDrawlistSettings.violinPlotDLConsiderBlendingOrder);

		violinAdaptSidesAutoObj.vdlp->attributePlacements.clear();
		for (int j = 0; j < violinAdaptSidesAutoObj.vdlp->attributeNames.size(); ++j) {
			violinAdaptSidesAutoObj.vdlp->attributePlacements.push_back((hist.side[j] % 2) ? ViolinMiddleLeft : ViolinMiddleRight);
		}

		if (violinPlotDrawlistSettings.violinPlotDLInsertCustomColors || violinPlotDrawlistSettings.violinPlotDLConsiderBlendingOrder) {
			changeColorsToCustomAlternatingColors((violinAdaptSidesAutoObj.vdlp->colorPaletteManager), violinAdaptSidesAutoObj.vdlp->attributeNames.size(), &(violinAdaptSidesAutoObj.vdlp->attributeLineColors), &(violinAdaptSidesAutoObj.vdlp->attributeFillColors),
				hist, &(violinAdaptSidesAutoObj.vdlp->activeAttributes), violinPlotDrawlistSettings.violinPlotDLInsertCustomColors);
		}
		violinAdaptSidesAutoObj.optimizeSidesNowDL = false;
	}
}

void violinAttributePlotAddDrawList(ViolinPlot& plot, DrawList& dl, uint32_t i) {
	std::vector<std::string> attrNames;
	std::vector<std::pair<float, float>> minMax;
	for (Attribute& a : pcAttributes) {
		minMax.push_back({ a.min,a.max });
		attrNames.push_back(a.name);
	}

	if (violinAttributePlots[i].attributeNames.size()) {		//checking if the attributes of the dataset to be added are the same as the already existing attributes in this violin plot
		bool attributeCheckFail = false;
		if (violinAttributePlots[i].attributeNames.size() != pcAttributes.size()) return;
		for (int l = 0; l < pcAttributes.size(); ++l) {
			if (pcAttributes[l].name != violinAttributePlots[i].attributeNames[l]) {
				attributeCheckFail = true;
				break;
			}
		}
		if (attributeCheckFail) {
#ifdef _DEBUG
			std::cout << "The attribute check for the drawlist to add failed." << std::endl;
#endif
			return;
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
		if (ds.name == dl.parentDataSet)
			parent = &ds;
	}
	exeComputeHistogram(dl.name, minMax, dl.buffer, parent->data.size(), dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView, true);
	//histogramManager->computeHistogramm(k->name, minMax, k->buffer, parent->data.size(), k->indicesBuffer, k->indices.size(), k->activeIndicesBufferView);
	bool datasetIncluded = false;
	for (int j = 0; j < violinAttributePlots[i].drawLists.size(); ++j) {
		if (dl.name == violinAttributePlots[i].drawLists[j].name) {
			datasetIncluded = true;
			break;
		}
	}
	if (!datasetIncluded) {
		violinAttributePlots[i].drawLists.push_back({ dl.name, true });
		violinAttributePlots[i].violinPlacements.push_back(ViolinLeft);
		violinAttributePlots[i].violinScalesX.push_back(ViolinScaleGlobalAttribute);
		violinAttributePlots[i].drawListLineColors.push_back({ 0,0,0,1 });
		violinAttributePlots[i].drawListFillColors.push_back({ 0,0,0,0 });
	}
	HistogramManager::Histogram& h = histogramManager->getHistogram(dl.name);
	for (int l = 0; l < h.maxCount.size(); ++l) {
		if (violinAttributePlots[i].maxValues[l] < h.maxCount[l]) {
			violinAttributePlots[i].maxValues[l] = h.maxCount[l];
		}
	}
	if (h.maxGlobalCount > violinAttributePlots[i].maxGlobalValue) {
		violinAttributePlots[i].maxGlobalValue = h.maxGlobalCount;
	}
	updateSummedBins(violinAttributePlots[i]);
}

void violinDrawListPlotAddDrawList(ViolinDrawlistPlot& drawPlot, DrawList& dl, uint32_t i) {
	//check if the drawlist was already added to this plot
	if (std::find(drawPlot.drawLists.begin(), drawPlot.drawLists.end(), dl.name) == drawPlot.drawLists.end()) {
		if (!drawPlot.attributeNames.size()) {	//creating all needed resources e.g. attribute components
			drawPlot.activeAttributes = new bool[pcAttributes.size()];
			drawPlot.maxGlobalValue = 0;
			int j = 0;
			for (Attribute& a : pcAttributes) {
				drawPlot.attributeNames.push_back(a.name);
				drawPlot.activeAttributes[j] = true;
				drawPlot.attributeLineColors.push_back({ 0,0,0,1 });
				drawPlot.attributeFillColors.push_back({ .5f,.5f,.5f,.5f });
				drawPlot.attributePlacements.push_back((j % 2) ? ViolinMiddleLeft : ViolinMiddleRight);
				drawPlot.attributeScalings.push_back(1);
				drawPlot.violinScalesX.push_back(ViolinScaleGlobalAttribute);
				drawPlot.maxValues.push_back(0);
				++j;
			}
		}

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
		updateHistogramComparisonDL(i);
		//histogramManager->computeHistogramm(dl->name, minMax, dl->buffer, ds->data.size(), dl->indicesBuffer, dl->indices.size(), dl->activeIndicesBufferView);
		HistogramManager::Histogram& hist = histogramManager->getHistogram(dl.name);
		std::vector<std::pair<uint32_t, float>> area;
		for (int j = 0; j < hist.maxCount.size(); ++j) {
			if (hist.maxCount[j] > drawPlot.maxValues[j]) {
				drawPlot.maxValues[j] = hist.maxCount[j];
			}
			if (hist.maxCount[j] > drawPlot.maxGlobalValue) {
				drawPlot.maxGlobalValue = hist.maxCount[j];
			}
			area.push_back({ j, drawPlot.attributeScalings[j] / hist.maxCount[j] });
		}

		drawPlot.drawLists.push_back(dl.name);
		//violinDrawlistPlots[i].drawListOrder.push_back(violinDrawlistPlots[i].drawListOrder.size());
		drawPlot.attributeOrder.push_back({});
		if (violinPlotDrawlistSettings.renderOrderDLConsider) {

			drawPlot.attributeOrder.back() = sortHistogram(hist, drawPlot, violinPlotDrawlistSettings.renderOrderDLConsider, violinPlotDrawlistSettings.renderOrderDLReverse);

			/*if (!renderOrderDLReverse) {
				std::sort(area.begin(), area.end(), [](std::pair<uint32_t, float>& a, std::pair<uint32_t, float>& b) {return sortDescPair(a, b); });
			}
			else
			{
				std::sort(area.begin(), area.end(), [](std::pair<uint32_t, float>& a, std::pair<uint32_t, float>& b) {return sortAscPair(a, b); });
			}*/
		}
		else {
			for (int j = 0; j < pcAttributes.size(); ++j)drawPlot.attributeOrder.back().push_back(area[j].first);
		}
	}
}

void addExportMenu() {
	if (ImGui::BeginMenu("Export Image")) {
		ImGui::DragFloat("Size muliplicator", &g_ExportScale, .5f, .5f, 20);
		ImVec2 size = ImGui::GetWindowViewport()->GetWorkSize();
		ImGui::Text("Resulting size: {%d, %d}", (int)(size.x * g_ExportScale), (int)(size.y * g_ExportScale));
		ImGui::InputText("Export file(including filepath)", g_ExportPath, 200);
		if (ImGui::MenuItem("Export")) {
			ImGuiID id = ImGui::GetWindowViewport()->ID;
			for (int i = 0; i < ImGui::GetCurrentContext()->Viewports.Size; i++) {
				if (ImGui::GetCurrentContext()->Viewports[i]->ID == id) {
					g_ExportViewportNumber = i;
					break;
				}
			}
			if (g_ExportImageWidth != (int)size.x * g_ExportScale || g_ExportImageHeight != (int)size.y * g_ExportScale) {
				g_ExportImageWidth = (int)size.x * g_ExportScale;
				g_ExportImageHeight = (int)size.y * g_ExportScale;
				recreateExportWindow();
			}
			g_ExportCountDown = 1;
		}
		ImGui::EndMenu();
	}
}

template<typename T>
void addSaveSettingsMenu(T* settingStruct, const std::string& settingName, const std::string& settingType) {
	if (ImGui::BeginMenu(("Load " + settingName).c_str())) {
		if (ImGui:: MenuItem("Default")) {
			T def;
			*settingStruct = def; 
		}
		for (SettingsManager::Setting* savedStyle : *settingsManager->getSettingsType(settingType)) {
			if (ImGui::MenuItem(savedStyle->id.c_str())) {
				memcpy(settingStruct, savedStyle->data, sizeof(T));
			}
		}
		ImGui::EndMenu();
		}
	if (ImGui::BeginMenu(("Save/Remove " + settingName).c_str())) {
		static char styleName[200]{};
		ImGui::InputText((settingName + "name").c_str(), styleName, 200);
		if (ImGui::MenuItem("Save")) {
			SettingsManager::Setting s{};
			s.id = styleName;
			s.type = settingType;
			s.byteLength = sizeof(T);
			s.data = settingStruct;
			settingsManager->addSetting(s);
		}
		ImGui::Separator();
		ImGui::Text("Click to delete:");
		std::string del;
		for (SettingsManager::Setting* savedStyle : *settingsManager->getSettingsType(settingType)) {
			if (ImGui::MenuItem(savedStyle->id.c_str())) {
				del = savedStyle->id;
			}
		}
		if (del.size()) settingsManager->deleteSetting(del);
		ImGui::EndMenu();
	}
	if (ImGui::BeginMenu(("Set Default " + settingName).c_str())) {
		int selection = -1;
		if (settingsManager->getSetting(("default" + settingName).c_str()).id != "settingnotfound")
			selection = *((int*)settingsManager->getSetting(("default" + settingName).c_str()).data);
		if (ImGui::MenuItem("Default", "", selection == -1)) settingsManager->deleteSetting("default" + settingName);
		int c = 0;
		for (SettingsManager::Setting* savedStyle : *settingsManager->getSettingsType(settingType)) {

			if (ImGui::MenuItem(savedStyle->id.c_str(), "", selection == c)) {
				SettingsManager::Setting s{};
				s.id = "default" + settingName;
				s.data = &c;
				s.byteLength = sizeof(c);
				s.type = "default" + settingType;
				settingsManager->addSetting(s);
			}
			c++;
		}
		ImGui::EndMenu();
	}
}

bool loadAttributeSettings(const std::string setting, int attribute) {
	SettingsManager::Setting s = settingsManager->getSetting(setting);
	if (((int*)(s.data))[0] != pcAttributes.size()) {
		return false;
	}

	std::vector<Attribute> savedAttr;
	char* d = (char*)s.data + sizeof(int);
	bool cont = false;
	for (int i = 0; i < ((int*)s.data)[0]; i++) {
		Attribute a = {};
		a.name = std::string(d);
		d += a.name.size() + 1;
		a.min = *(float*)d;
		d += sizeof(float);
		a.max = *(float*)d;
		d += sizeof(float);
		savedAttr.push_back(a);
		if (pcAttributes[i].name != savedAttr[i].name) {
			return false;
		}
	}

	int* o = (int*)d;
	bool* act = (bool*)(d + pcAttributes.size() * sizeof(int));
	if (attribute < 0) {
		for (int i = 0; i < pcAttributes.size(); i++) {
			pcAttributes[i] = savedAttr[i];
			pcAttrOrd[i] = o[i];
			pcAttributeEnabled[i] = act[i];
		}
	}
	else { // only reset min, max
		pcAttributes[attribute] = savedAttr[attribute];
	}
	return true;
}

int main(int, char**)
{
#ifdef DETECTMEMLEAK
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
	engine.seed(15);

	//std::vector<float> numbers(100);
	//std::iota(numbers.begin(), numbers.end(), 0);
	//std::ofstream tes("/run/media/lachei/3d02119e-bc93-4969-9fc5-523f06321708/test/temp/1");
	//tes << 5 << " " << numbers.size() << "\n";
	//tes.write(reinterpret_cast<char*>(numbers.data()), numbers.size() * sizeof(numbers[0]));
	//tes.close();

	//test of multivariate gauss calculations
	//float determinant;
	//std::vector<std::vector<double>> X{ {10,0,-3,10}, {-2,-4,1,.5},{3,0,2,7},{-3,5,9,0} };
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
	int pcPlotPreviousSlectedDrawList = -1;									//Index of the previously selected drawlist
	bool addIndeces = false;

    // Setup SDL
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
    {
        printf("Error: %s\n", SDL_GetError());
        return -1;
    }
    // Setup window
    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_MAXIMIZED);
    SDL_Window* window = SDL_CreateWindow("PCViewer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
	g_SwapChainResizeWidth = 1280;
	g_SwapChainResizeHeight = 720;

	// Setup Drag and drop callback
	SDL_EventState(SDL_DROPFILE, SDL_ENABLE);

	// Setup Vulkan
	uint32_t extensions_count = 0;
    SDL_Vulkan_GetInstanceExtensions(window, &extensions_count, NULL);
    const char** extensions = new const char*[extensions_count + 1];
    SDL_Vulkan_GetInstanceExtensions(window, &extensions_count, extensions);
    extensions[extensions_count] = "VK_KHR_get_physical_device_properties2";
    SetupVulkan(extensions, extensions_count + 1);
    delete[] extensions;

	// Create Window Surface
    VkSurfaceKHR surface;
    VkResult err;
    if (SDL_Vulkan_CreateSurface(window, g_Instance, &surface) == 0)
    {
        printf("Failed to create Vulkan surface.\n");
        return 1;
    }

    // Create Framebuffers
    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    ImGui_ImplVulkanH_Window* wd = &g_MainWindowData;
    SetupVulkanWindow(wd, surface, w, h);

	// Setup image exportbuffer
	recreateExportWindow();

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

	// Setup Platform/Renderer bindings
	ImGui_ImplSDL2_InitForVulkan(window);
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
		createPcPlotFramebuffer();
		createPcPlotPipeline();

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
		drawListColorPalette = new DrawlistColorPalette(settingsManager);
	}

	{//brushing gpu
		gpuBrusher = new GpuBrusher(g_Device, g_PhysicalDevice, g_PcPlotCommandPool, g_Queue, g_DescriptorPool);
	}

	{//histogram manager
		histogramManager = new HistogramManager(g_Device, g_PhysicalDevice, g_PcPlotCommandPool, g_Queue, g_DescriptorPool, violinPlotBinsSize);
	}

	{// scatterplot workbench
		scatterplotWorkbench = new ScatterplotWorkbench({{0,0}, g_PhysicalDevice, g_Device, g_DescriptorPool, g_PcPlotCommandPool, g_Queue}, pcAttributes);
	}

	{// correlation matrix workbench
		VkUtil::Context c{0, 0, g_PhysicalDevice, g_Device, g_DescriptorPool, g_PcPlotCommandPool, g_Queue};
		if(!atomicGpuFloatAddAvailable){
			c.screenSize[0] = 0xffffffff;
		}
		correlationMatrixWorkbench = std::make_unique<CorrelationMatrixWorkbench>(c);
	}

	{// clustering workbench
		clusteringWorkbench = std::make_shared<ClusteringWorkbench>(g_Device, pcAttributes, g_PcPlotDataSets, g_PcPlotDrawLists);
	}

	//{// testing sorter
	//	GpuRadixSorter sorter({{0,0}, g_PhysicalDevice, g_Device, g_DescriptorPool, g_PcPlotCommandPool, g_Queue});
	//	std::vector<uint32_t> nums(1e8);
	//	std::iota(nums.rbegin(), nums.rend(), 0);
	//	for(int i = 0; i < 10; ++i){
	//		std::cout << "Iteration: " << i << std::endl;
	//		sorter.sort(nums);
	//	}
	//	
	//	bool ok = true;
	//}

	{
		//pcRenderer = std::make_shared<PCRenderer>(VkUtil::Context{{0,0}, g_PhysicalDevice, g_Device, g_DescriptorPool, g_PcPlotCommandPool, g_Queue}, g_PcPlotWidth, g_PcPlotHeight, g_PcPlotDescriptorLayout, g_PcPlotDataSetLayout);
	}

	{
		compressionWorkbench = std::make_shared<CompressionWorkbench>();
	}
	
	io.ConfigWindowsMoveFromTitleBarOnly = true;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
	transferFunctionEditor = new TransferFunctionEditor(g_Device, g_PhysicalDevice, g_PcPlotCommandPool, g_Queue, g_DescriptorPool);
	view3d->setTransferFunctionImage(transferFunctionEditor->getTransferImageView());

	{//Set imgui style and load all settings
		// Setup Dear ImGui style
		if (settingsManager->getSetting("defaultstyle").id != "settingnotfound") {
			auto styles = settingsManager->getSettingsType("style");
			int index = *((int*)settingsManager->getSetting("defaultstyle").data);
			memcpy(&ImGui::GetStyle(), (*styles)[index]->data, sizeof(ImGuiStyle));
		}
		else {
			ImGui::StyleColorsDark();
			ImGuiStyle& style = ImGui::GetStyle();
			style.ChildRounding = 5;
			style.FrameRounding = 3;
			style.GrabRounding = 3;
			style.WindowRounding = 0;
			style.PopupRounding = 3;
		}
		if (settingsManager->getSetting("defaultPCSettings").id != "settingnotfound") {
			auto set = settingsManager->getSettingsType("pcsettings");
			int index = *((int*)settingsManager->getSetting("defaultPCSettings").data);
			memcpy(&pcSettings, (*set)[index]->data, sizeof(PCSettings));
		}
		if (settingsManager->getSetting("defaultBubbleSettings").id != "settingnotfound") {
			auto set = settingsManager->getSettingsType("bubblesettings");
			int index = *((int*)settingsManager->getSetting("defaultBubbleSettings").data);
			memcpy(&bubbleWindowSettings, (*set)[index]->data, sizeof(BubbleWindowSettings));
		}
		if (settingsManager->getSetting("defaultIsoSettings").id != "settingnotfound") {
			auto set = settingsManager->getSettingsType("isosettingss");
			int index = *((int*)settingsManager->getSetting("defaultIsoSettings").data);
			memcpy(&isoSurfSettings, (*set)[index]->data, sizeof(IsoSettings));
		}
		if (settingsManager->getSetting("defaultBrushIsoSettings").id != "settingnotfound") {
			auto set = settingsManager->getSettingsType("brushisosettingss");
			int index = *((int*)settingsManager->getSetting("defaultBrushIsoSettings").data);
			memcpy(&brushIsoSurfSettings, (*set)[index]->data, sizeof(IsoSettings));
		}
		if (settingsManager->getSetting("defaultViolinAttribute").id != "settingnotfound") {
			auto set = settingsManager->getSettingsType("violinattribute");
			int index = *((int*)settingsManager->getSetting("defaultViolinAttribute").data);
			memcpy(&violinPlotAttributeSettings, (*set)[index]->data, sizeof(ViolinSettings));
		}
		if (settingsManager->getSetting("defaultViolinDrawlist").id != "settingnotfound") {
			auto set = settingsManager->getSettingsType("violidrawlist");
			int index = *((int*)settingsManager->getSetting("defaultViolinDrawlist").data);
			memcpy(&violinPlotAttributeSettings, (*set)[index]->data, sizeof(ViolinSettings));
		}

		//loading recent files
		loadRecentFiles();
	}

	// Main loop
    bool done = false;
	while (!done)
	{
		// Poll and handle events (inputs, window resize, etc.)
		// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
		// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
		// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
		// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
		SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT)
                done = true;
            else if(event.type == SDL_DROPFILE) {       // In case if dropped file
                droppedPaths.push_back(std::string(event.drop.file));
				droppedPathActive.push_back(1);
                pathDropped = true;
				std::string file(event.drop.file);
				if (droppedPaths.size() == 1) {
					queryAttributes = queryFileAttributes(event.drop.file);
				}
                SDL_free(event.drop.file);              // Free dropped_filedir memory;
            }
        }
        if(droppedPaths.size() && !createDLForDrop){
            createDLForDrop = new bool[droppedPaths.size()];
            for(int i = 0 ; i< droppedPaths.size(); ++i) createDLForDrop[i] = true;
        }

		if (g_SwapChainRebuild && g_SwapChainResizeWidth > 0 && g_SwapChainResizeHeight > 0)
		{
			g_SwapChainRebuild = false;
			ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
			ImGui_ImplVulkanH_CreateOrResizeWindow(g_Instance, g_PhysicalDevice, g_Device, &g_MainWindowData, g_QueueFamily, g_Allocator, g_SwapChainResizeWidth, g_SwapChainResizeHeight, g_MinImageCount);
			g_MainWindowData.FrameIndex = 0;
		}

		//disable keyboard navigation if brushes are active
		if (brushDragIds.size() && (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_NavEnableKeyboard)) {
			ImGui::GetIO().ConfigFlags ^= ImGuiConfigFlags_NavEnableKeyboard; //deactivate keyboard navigation
		}
		if (brushDragIds.empty() && !(ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_NavEnableKeyboard)) {
			ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; //enable keyboard navigation
		}

		// Start the Dear ImGui frame
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplSDL2_NewFrame(window);
		ImGui::NewFrame();

		if (pcViewerState >= PCViewerState::AnimateDrawlists && pcViewerState <= PCViewerState::AnimateGlobalBrushExport) {
			//disabling inputs when animating
			animationItemsDisabled = true;
			switch (pcViewerState)
			{
			case PCViewerState::AnimateDrawlists:
			case PCViewerState::AnimateDrawlistsExport:
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
				if (animationCurrentDrawList != (int)(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - animationStart).count() / pcSettings.animationDuration)) {
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
				break;

			case PCViewerState::AnimateGlobalBrush:
			case PCViewerState::AnimateGlobalBrushExport:
				//advance global brush
				if (animationCurrentStep != (int)(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - animationStart).count() / pcSettings.animationDuration) && g_ExportCountDown < 0) { //also waits for image export
					++animationCurrentStep;
					if (animationCurrentStep >= pcSettings.animationSteps) {
						//reset brush on axis
						globalBrushes[animationBrush].brushes[animationAttribute] = animationAttributeBrush;
						updateAllActiveIndices();
						pcPlotRender = true;

						//reset PCViewer state
						pcViewerState = PCViewerState::Normal;
					}
					else {
						//advancing the brush
						float d = (pcAttributes[animationAttribute].max - pcAttributes[animationAttribute].min) / pcSettings.animationSteps / 2;
						float a = float(animationCurrentStep) / (pcSettings.animationSteps - 1);
						float v = a * pcAttributes[animationAttribute].max + (1 - a) * pcAttributes[animationAttribute].min;
						globalBrushes[animationBrush].brushes[animationAttribute][0].second = { v - d, v + d };
						pcPlotRender = true;
						updateAllActiveIndices();
						if (pcViewerState == PCViewerState::AnimateGlobalBrushExport) {
							//enable animation export (wait one frame for drawlist rendering)
							sprintf(g_ExportPath, animationExportPath, animationCurrentStep);
							g_ExportViewportNumber = 0;
							g_ExportCountDown = 1;
						}
					}
				}
				break;
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

		//check if a path was dropped in the application
		if (pathDropped && !addIndeces) {
			ImGui::OpenPopup("OPENDATASET");
			ImGui::SetNextWindowFocus();
			ImGui::SetNextWindowPos(ImGui::GetWindowPos() + ImVec2(400,100), ImGuiCond_Appearing);
			if (ImGui::BeginPopupModal("OPENDATASET", NULL, ImGuiWindowFlags_AlwaysAutoResize))
			{
				if (ImGui::CollapsingHeader("Attribute activations")) {
					ImGui::Text("Attribute Query");
					static bool allActive;
					if(ImGui::Checkbox("activate/deactivate all", &allActive)){
						for(auto& a : queryAttributes)
							a.active = allActive;
					}
					bool prefAtt = true;
					int c = 0;
					for (auto& a : queryAttributes) {
						if(a.dimensionSize < 0) continue;	//ignore stringlength dimensions
						if(prefAtt && a.dimensionSize != 0){	//split line for dimensions
							prefAtt = false;
							ImGui::Separator();
						}
						ImGui::Text("%s", a.name.c_str());
						ImGui::SameLine(100);
						ImGui::Text("%d, %s", a.dimensionality, ((a.dimensionSize > 0) ? "Dim" : " "));
						ImGui::SameLine(150);
						ImGui::Checkbox(("active##" + std::to_string(c)).c_str(), &a.active);
						if (a.dimensionSize > 0) {
							ImGui::PushItemWidth(100);
							if (a.active) {
								ImGui::SameLine();
								ImGui::InputInt(("sampleFrequency##" + std::to_string(c)).c_str(), &a.dimensionSubsample);
								if (a.dimensionSubsample < 1) a.dimensionSubsample = 1;
								if(!queryAttributesCsv){
									ImGui::SameLine();
									ImGui::InputInt2(("Trim Indices##" + std::to_string(c)).c_str(), a.trimIndices);
								}
							}
							else {
								ImGui::SameLine();
								ImGui::InputInt(("slice Index##" + std::to_string(c)).c_str(), &a.dimensionSlice);
								a.dimensionSlice = std::clamp(a.dimensionSlice, 0, a.dimensionSize - 1);
							}
							ImGui::PopItemWidth();
						}
						else{
							ImGui::SameLine();
							ImGui::Checkbox(("Linearize##" + std::to_string(c)).c_str(), &a.linearize);
						}
						++c;
					}
				}
				ImGui::Text("Do you really want to open these Datasets?");
				int c = 0;
				for (std::string& s : droppedPaths) {
					ImGui::Checkbox(s.c_str(), (bool*)&droppedPathActive[c++]);
				}
				ImGui::Separator();

				if (ImGui::Button("Open", ImVec2(120, 0)) || ImGui::IsKeyPressed(KEYENTER)) {
					ImGui::CloseCurrentPopup();
					c = 0;
					for (std::string& s : droppedPaths) {
						if (!droppedPathActive[c++]) continue;
						bool success = openDataset(s.c_str());
						if (success && pcSettings.createDefaultOnLoad) {
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

		if (animationItemsDisabled) {
			ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
		}

		bool openSave = ImGui::GetIO().KeyCtrl && ImGui::IsKeyDown(22), openAttributesManager = false, saveColor = false, openColorManager = false, openUserDoc = false;
		float color[4];
		if (ImGui::BeginMenuBar()) {
			if (ImGui::BeginMenu("File")) {
				if (ImGui::BeginMenu("Open")) {
					bool open = ImGui::InputText("Directory Path", pcFilePath, 200, ImGuiInputTextFlags_EnterReturnsTrue);
					if (ImGui::IsItemHovered()) {
						ImGui::BeginTooltip();
						ImGui::Text("Enter either a file including filepath,\nOr a folder (division with /) and all datasets in the folder will be loaded\nOr drag and drop files to load onto application.");
						ImGui::EndTooltip();
					}

					ImGui::SameLine();

					//Opening a new Dataset into the Viewer
					if (ImGui::Button("Open") || open) {
						std::string f = pcFilePath;
						std::string fileExtension = f.substr(f.find_last_of("/\\") + 1);
						size_t pos = fileExtension.find_last_of(".");
						if (pos != std::string::npos) {		//entered discrete file
							bool success = openDataset(pcFilePath);
							if (success && pcSettings.createDefaultOnLoad) {
								//pcPlotRender = true;
								createPcPlotDrawList(g_PcPlotDataSets.back().drawLists.front(), g_PcPlotDataSets.back(), g_PcPlotDataSets.back().name.c_str());
								pcPlotRender = updateActiveIndices(g_PcPlotDrawLists.back());
							}
						}
						else {					//entered folder -> open open dataset dialogue
							for (const auto& entry : std::filesystem::directory_iterator(f)) {
								if (entry.is_regular_file()) {	//only process normal enties
									fileExtension = entry.path().u8string().substr(entry.path().u8string().find_last_of("."));
									if (std::find(supportedDataFormats.begin(), supportedDataFormats.end(), fileExtension) == supportedDataFormats.end()) continue;	//ignore unsupported file formats
									droppedPaths.emplace_back(entry.path().u8string());
									droppedPathActive.emplace_back(1);
									pathDropped = true;
									f = entry.path().u8string();
									queryAttributes = queryFileAttributes((entry.path().u8string()).c_str());
								}
							}
						}
					}
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Open recent")) {
					for (auto& f : recentFiles) {
						if (ImGui::MenuItem(f.c_str())) {
							droppedPaths.push_back(f);
							droppedPathActive.push_back(1);
							pathDropped = true;
							queryAttributes = queryFileAttributes(f.c_str());
						}
					}
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Export")) {
					if (ImGui::MenuItem("Png")) {
						ImGuiID id = ImGui::GetWindowViewport()->ID;
						ImVec2 size = ImGui::GetWindowViewport()->GetWorkSize();
						for (int i = 0; i < ImGui::GetCurrentContext()->Viewports.Size; i++) {
							if (ImGui::GetCurrentContext()->Viewports[i]->ID == id) {
								g_ExportViewportNumber = i;
								break;
							}
						}
						if (g_ExportImageWidth != (int)size.x * g_ExportScale || g_ExportImageHeight != (int)size.y * g_ExportScale) {
							g_ExportImageWidth = (int)size.x * g_ExportScale;
							g_ExportImageHeight = (int)size.y * g_ExportScale;
							recreateExportWindow();
						}
						g_ExportCountDown = 1;
					}
					ImGui::EndMenu();
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Edit")) {
				if (ImGui::BeginMenu("Load style")) {
					for (SettingsManager::Setting* savedStyle : *settingsManager->getSettingsType("style")) {
						if (ImGui::MenuItem(savedStyle->id.c_str())) {
							memcpy(&ImGui::GetStyle(), savedStyle->data, sizeof(ImGuiStyle));
						}
					}
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Save/Remove style")) {
					static char styleName[200]{};
					ImGui::InputText("Stylename", styleName, 200);
					if (ImGui::MenuItem("Save")) {
						SettingsManager::Setting s{};
						s.id = styleName;
						s.type = "style";
						s.byteLength = sizeof(ImGuiStyle);
						s.data = &ImGui::GetStyle();
						settingsManager->addSetting(s);
					}
					ImGui::Separator();
					ImGui::Text("Click to delete:");
					std::string del;
					for (SettingsManager::Setting* savedStyle : *settingsManager->getSettingsType("style")) {
						if (ImGui::MenuItem(savedStyle->id.c_str())) {
							del = savedStyle->id;
						}
					}
					if (del.size()) settingsManager->deleteSetting(del);
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Set Default style")) {
					int selection = -1;
					if (settingsManager->getSetting("defaultstyle").id != "settingnotfound")
						selection = *((int*)settingsManager->getSetting("defaultstyle").data);
					if (ImGui::MenuItem("Default", "", selection == -1)) settingsManager->deleteSetting("defaultstyle");
					int c = 0;
					for (SettingsManager::Setting* savedStyle : *settingsManager->getSettingsType("style")) {

						if (ImGui::MenuItem(savedStyle->id.c_str(), "", selection == c)) {
							SettingsManager::Setting s{};
							s.id = "defaultstyle";
							s.data = &c;
							s.byteLength = sizeof(c);
							s.type = "defaultstyle";
							settingsManager->addSetting(s);
						}
						c++;
					}
					ImGui::EndMenu();
				}
				ImGui::ShowStyleEditor();
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("View")) {
				ImGui::MenuItem("Bubbleplot workbench", "", &bubbleWindowSettings.enabled);
				ImGui::MenuItem("3d View", "", &view3dSettings.enabled);
				if (ImGui::BeginMenu("Iso surface workbenches")) {
					ImGui::MenuItem("Iso surface workbench", "", &isoSurfSettings.enabled);
					ImGui::MenuItem("Direct iso surface workbench", "", &brushIsoSurfSettings.enabled);
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Violinplot workbenches")) {
					ImGui::MenuItem("Violin attribute major", "", &violinPlotAttributeSettings.enabled);
					ImGui::MenuItem("Violin drawlist major", "", &violinPlotDrawlistSettings.enabled);
					ImGui::EndMenu();
				}
				ImGui::MenuItem("Clustering workbench", "", &clusteringWorkbench->active);
				ImGui::MenuItem("Scatterplot workbench", "", &scatterplotWorkbench->active);
				ImGui::MenuItem("Correlation matrix workbench", "", &correlationMatrixWorkbench->active);
				ImGui::MenuItem("Compression workbench", "", &compressionWorkbench->active);
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Options")) {
				addSaveSettingsMenu<PCSettings>(&pcSettings, "PCSettings", "pcsettings");
				ImGui::Separator();
				if (ImGui::MenuItem("Activate Global Brushing", "", &pcSettings.toggleGlobalBrushes) && !pcSettings.toggleGlobalBrushes) {
					pcPlotRender = updateAllActiveIndices();
				}

				if (ImGui::BeginMenu("Brush Combination")) {
					static char const* combinations[] = { "OR","AND" };
					if (ImGui::Combo("brushCombination", &pcSettings.brushCombination, combinations, sizeof(combinations) / sizeof(*combinations))) {
						pcPlotRender = updateAllActiveIndices();
					}

					ImGui::EndMenu();
				}
				ImGui::InputFloat("Mu add factor", &pcSettings.brushMuFactor, 0.000001, 0.001, 10);

				ImGui::Separator();
				if (ImGui::InputInt("Max fraction depth", &pcSettings.maxFractionDepth, 1, 1)) {
					if (pcSettings.maxFractionDepth < 1) pcSettings.maxFractionDepth = 1;
					if (pcSettings.maxFractionDepth > 30)pcSettings.maxFractionDepth = 30;
				}

				if (ImGui::InputInt("Outlier rank", &pcSettings.outlierRank, 1, 1)) {
					if (pcSettings.outlierRank < 1) pcSettings.outlierRank = 1;
				}

				static char const* boundsTypes[] = { "No adjustment","Pull in outside", "Pull in both sides" };
				if (ImGui::BeginCombo("Bounds behaviour", boundsTypes[pcSettings.boundsBehaviour])) {
					for (int i = 0; i < 3; i++) {
						if (ImGui::MenuItem(boundsTypes[i])) pcSettings.boundsBehaviour = i;
					}
					ImGui::EndCombo();
				}

				static char const* splitTypes[] = { "Split half","SAH" };
				if (ImGui::BeginCombo("Split behaviour", splitTypes[pcSettings.splitBehaviour])) {
					for (int i = 0; i < 2; ++i) {
						if (ImGui::MenuItem(splitTypes[i])) pcSettings.splitBehaviour = i;
					}
					ImGui::EndCombo();
				}

				ImGui::DragFloat("Fractionbox width", &pcSettings.fractionBoxWidth, 1, 0, 100);

				if (ImGui::InputInt("Fractionbox linewidth", &pcSettings.fractionBoxLineWidth, 1, 1)) {
					if (pcSettings.fractionBoxLineWidth < 1) pcSettings.maxFractionDepth = 1;
					if (pcSettings.fractionBoxLineWidth > 30) pcSettings.maxFractionDepth = 30;
				}

				ImGui::SliderFloat("Multivariate std dev thresh", &pcSettings.multivariateStdDivThresh, .01f, 5);

				if (ImGui::IsItemDeactivatedAfterEdit()) {
					pcPlotRender = updateAllActiveIndices();
				}

				ImGui::Separator();

				ImGui::SliderFloat("Animation duration per step", &pcSettings.animationDuration, .1f, 10);
				ImGui::Checkbox("Export animation steps", &pcSettings.animationExport);
				ImGui::InputText("Export path(including file-name/ending)", animationExportPath, 200);
				ImGui::Separator();
				if (ImGui::MenuItem("Start drawlist animation")) {
					pcViewerState = pcSettings.animationSteps ? PCViewerState::AnimateDrawlistsExport : PCViewerState::AnimateDrawlists;
					animationStart = std::chrono::steady_clock::now();
				}
				ImGui::Separator();
				if (ImGui::BeginCombo("Brush to animate", animationBrush == -1 ? "Select" : globalBrushes[animationBrush].nam.c_str())) {
					for (int i = 0; i < globalBrushes.size(); ++i) {
						if (ImGui::MenuItem(globalBrushes[i].nam.c_str())) {
							animationBrush = i;
						}
					}
					ImGui::EndCombo();
				}
				if (ImGui::BeginCombo("Attribute to animate", animationAttribute == -1 ? "Select" : pcAttributes[animationAttribute].name.c_str())) {
					for (int i = 0; i < pcAttributes.size(); ++i) {
						if (ImGui::MenuItem(pcAttributes[i].name.c_str())) {
							animationAttribute = i;
						}
					}
					ImGui::EndCombo();
				}
				ImGui::DragInt("Steps amount", &pcSettings.animationSteps, 1, 2, 1e6);
				if (ImGui::MenuItem("Start global brush animation") && animationBrush >= 0 && animationAttribute >= 0) {
					pcViewerState = pcSettings.animationExport ? PCViewerState::AnimateGlobalBrushExport : PCViewerState::AnimateGlobalBrush;
					animationStart = std::chrono::steady_clock::now();
					animationAttributeBrush = globalBrushes[animationBrush].brushes[animationAttribute];
					float d = (pcAttributes[animationAttribute].max - pcAttributes[animationAttribute].min) / pcSettings.animationSteps / 2;
					globalBrushes[animationBrush].brushes[animationAttribute] = { { currentBrushId++,{pcAttributes[animationAttribute].min - d, pcAttributes[animationAttribute].min + d} } };
					animationCurrentStep = -1;
				}
				ImGui::Separator();
				ImGui::DragFloat("Size muliplicator", &g_ExportScale, .5f, .5f, 20);
				ImVec2 size = ImGui::GetWindowViewport()->GetWorkSize();
				ImGui::Text("Resulting size: {%d, %d}", (int)(size.x* g_ExportScale), (int)(size.y* g_ExportScale));
				ImGui::InputText("Export file(including filepath)", g_ExportPath, 200);
				if (ImGui::MenuItem("Export")) {
					ImGuiID id = ImGui::GetWindowViewport()->ID;
					for (int i = 0; i < ImGui::GetCurrentContext()->Viewports.Size; i++) {
						if (ImGui::GetCurrentContext()->Viewports[i]->ID == id) {
							g_ExportViewportNumber = i;
							break;
						}
					}
					if (g_ExportImageWidth != (int)size.x * g_ExportScale || g_ExportImageHeight != (int)size.y * g_ExportScale) {
						g_ExportImageWidth = (int)size.x * g_ExportScale;
						g_ExportImageHeight = (int)size.y * g_ExportScale;
						recreateExportWindow();
					}
					g_ExportCountDown = 1;
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
			//addExportMenu();
			if (ImGui::BeginMenu("Help")) {
				if (ImGui::MenuItem("User Documentation")) {
					openUserDoc = true;
				}
				ImGui::EndMenu();
			}

			//if (ImGui::BeginMenu("Global brush")) {
			//	if (ImGui::MenuItem("Activate Global Brushing", "", &pcSettings.toggleGlobalBrushes) && !pcSettings.toggleGlobalBrushes) {
			//		pcPlotRender = updateAllActiveIndices();
			//	}
			//
			//	if (ImGui::BeginMenu("Brush Combination")) {
			//		static char* const combinations[] = { "OR","AND" };
			//		if (ImGui::Combo("brushCombination", &pcSettings.brushCombination, combinations, sizeof(combinations) / sizeof(*combinations))) {
			//			pcPlotRender = updateAllActiveIndices();
			//		}
			//
			//		ImGui::EndMenu();
			//	}
			//	ImGui::InputFloat("Mu add factor", &pcSettings.brushMuFactor, 0.000001, 0.001,10);
			//
			//	ImGui::EndMenu();
			//}
			//if (ImGui::BeginMenu("Fractioning")) {
			//	if (ImGui::InputInt("Max fraction depth", &pcSettings.maxFractionDepth, 1, 1)) {
			//		if (pcSettings.maxFractionDepth < 1) pcSettings.maxFractionDepth = 1;
			//		if (pcSettings.maxFractionDepth > 30)pcSettings.maxFractionDepth = 30;
			//	}
			//
			//	if (ImGui::InputInt("Outlier rank", &pcSettings.outlierRank, 1, 1)) {
			//		if (pcSettings.outlierRank < 1) pcSettings.outlierRank = 1;
			//	}
			//
			//	static char* boundsTypes[] = { "No adjustment","Pull in outside", "Pull in both sides" };
			//	if (ImGui::BeginCombo("Bounds behaviour", boundsTypes[pcSettings.boundsBehaviour])) {
			//		for (int i = 0; i < 3; i++) {
			//			if (ImGui::MenuItem(boundsTypes[i])) pcSettings.boundsBehaviour = i;
			//		}
			//		ImGui::EndCombo();
			//	}
			//
			//	static char* splitTypes[] = { "Split half","SAH" };
			//	if (ImGui::BeginCombo("Split behaviour", splitTypes[pcSettings.splitBehaviour])) {
			//		for (int i = 0; i < 2; ++i) {
			//			if (ImGui::MenuItem(splitTypes[i])) pcSettings.splitBehaviour = i;
			//		}
			//		ImGui::EndCombo();
			//	}
			//
			//	ImGui::DragFloat("Fractionbox width", &pcSettings.fractionBoxWidth, 1, 0, 100);
			//
			//	if (ImGui::InputInt("Fractionbox linewidth", &pcSettings.fractionBoxLineWidth, 1, 1)) {
			//		if (pcSettings.fractionBoxLineWidth < 1) pcSettings.maxFractionDepth = 1;
			//		if (pcSettings.fractionBoxLineWidth > 30) pcSettings.maxFractionDepth = 30;
			//	}
			//	
			//	ImGui::SliderFloat("Multivariate std dev thresh", &pcSettings.multivariateStdDivThresh, .01f, 5);
			//	
			//	if (ImGui::IsItemDeactivatedAfterEdit()) {
			//		pcPlotRender = updateAllActiveIndices();
			//	}
			//
			//	ImGui::EndMenu();
			//}
			//if (ImGui::BeginMenu("Animation")) {
			//	ImGui::SliderFloat("Animation duration per step", &pcSettings.animationDuration, .1f, 10);
			//	ImGui::Checkbox("Export animation steps", &pcSettings.animationExport);
			//	ImGui::InputText("Export path(including file-name/ending)", animationExportPath, 200);
			//	ImGui::Separator();
			//	if (ImGui::MenuItem("Start drawlist animation")) {
			//		pcViewerState = pcSettings.animationSteps ? PCViewerState::AnimateDrawlistsExport : PCViewerState::AnimateDrawlists;
			//		animationStart = std::chrono::steady_clock::now();
			//	}
			//	ImGui::Separator();
			//	if (ImGui::BeginCombo("Brush to animate", animationBrush == -1 ? "Select" : globalBrushes[animationBrush].name.c_str())) {
			//		for (int i = 0; i < globalBrushes.size(); ++i) {
			//			if (ImGui::MenuItem(globalBrushes[i].name.c_str())) {
			//				animationBrush = i;
			//			}
			//		}
			//		ImGui::EndCombo();
			//	}
			//	if (ImGui::BeginCombo("Attribute to animate", animationAttribute == -1 ? "Select" : pcAttributes[animationAttribute].name.c_str())) {
			//		for (int i = 0; i < pcAttributes.size(); ++i) {
			//			if (ImGui::MenuItem(pcAttributes[i].name.c_str())) {
			//				animationAttribute = i;
			//			}
			//		}
			//		ImGui::EndCombo();
			//	}
			//	ImGui::DragInt("Steps amount", &pcSettings.animationSteps, 1, 2, 1e6);
			//	if (ImGui::MenuItem("Start global brush animation") && animationBrush >= 0 && animationAttribute >= 0) {
			//		pcViewerState = pcSettings.animationExport ? PCViewerState::AnimateGlobalBrushExport : PCViewerState::AnimateGlobalBrush;
			//		animationStart = std::chrono::steady_clock::now();
			//		animationAttributeBrush = globalBrushes[animationBrush].brushes[animationAttribute];
			//		float d = (pcAttributes[animationAttribute].max - pcAttributes[animationAttribute].min) / pcSettings.animationSteps / 2;
			//		globalBrushes[animationBrush].brushes[animationAttribute] = { { currentBrushId++,{pcAttributes[animationAttribute].min - d, pcAttributes[animationAttribute].min + d} } };
			//		animationCurrentStep = -1;
			//	}
			//	ImGui::EndMenu();
			//}
			//if (ImGui::BeginMenu("Workbenches")) {
			//	ImGui::MenuItem("Bubbleplot workbench", "", &bubbleWindowSettings.enabled);
			//	ImGui::MenuItem("3d View", "", &view3dSettings.enabled);
			//	if(ImGui::BeginMenu("Iso surface workbenches")) {
			//		ImGui::MenuItem("Iso surface workbench", "", &isoSurfSettings.enabled);
			//		ImGui::MenuItem("Direct iso surface workbench", "", &brushIsoSurfSettings.enabled);
			//		ImGui::EndMenu();
			//	}
			//	if (ImGui::BeginMenu("Violinplot workbenches")) {
			//		ImGui::MenuItem("Violin attribute major", "", &violinPlotAttributeSettings.enabled);
			//		ImGui::MenuItem("Violin drawlist major", "", &violinPlotDrawlistSettings.enabled);
			//		ImGui::EndMenu();
			//	}
			//	ImGui::EndMenu();
			//}
			ImGui::EndMenuBar();
		}
		//popup for saving a new Attribute Setting
		if (openSave) {
			ImGui::OpenPopup("Save attribute setting");
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
		if (openUserDoc) {
			ImGui::OpenPopup("UserDoc");
		}
		if (ImGui::BeginPopup("UserDoc", ImGuiWindowFlags_AlwaysAutoResize)) {
			ImGui::Text("For the user documentation go onto the website");
			char website[] = "https://github.com/wavestoweather/PCViewer/blob/master/doc/overview.md";
			ImGui::SetNextItemWidth(500);
			ImGui::InputText("##userdoc", website, 71, ImGuiInputTextFlags_ReadOnly);
			ImGui::EndPopup();
		}
		if (ImGui::BeginPopupModal("Manage attribute settings", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
			std::string del;
			for (SettingsManager::Setting* s : *settingsManager->getSettingsType("AttributeSetting")) {
				ImGui::Text("%s", s->id.c_str());
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

				delete[] static_cast<char*>(s.data);
			}
			ImGui::SetItemDefaultFocus();
			ImGui::SameLine();
			if (ImGui::Button("Cancel", ImVec2(120, 0))) {
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

#ifdef _DEBUG
		ImGui::ShowDemoWindow(NULL);
#endif
		
		//Parallel coordinates plot ----------------------------------------------------------------------------------------
		ImVec2 picPos;
		bool picHovered;
		bool histogramHovered = false;
		bool anyHover = false;
		size_t amtOfLabels = 0;
		if (ImGui::Begin("Parallel coordinates", NULL)) {
			float windowW = ImGui::GetWindowWidth();
			// Labels for the titels of the attributes
			// Position calculation for each of the Label
			for (int i = 0; i < pcAttributes.size(); i++)
				if (pcAttributeEnabled[i])
					amtOfLabels++;

			size_t paddingSide = 10;			//padding from left and right screen border
			size_t gap = (windowW - 2 * paddingSide) / (amtOfLabels - 1);
			ImVec2 buttonSize = ImVec2(70, 20);
			size_t offset = 0;

			//drawing the buttons which can be changed via drag and drop + showing labels for categorie data
			int c = 0;		//describing the position of the element in the AttrOrd vector
			int c1 = 0;
			for (auto i : pcAttrOrd) {
				//not creating button for unused Attributes
				if (!pcAttributeEnabled[i]) {
					c++;
					continue;
				}

				std::string name = pcAttributes[i].name;
				static int editAttributeName = -1;
				static char newAttributeName[256];
				if (c1 != 0)
					ImGui::SameLine(offset - c1 * (buttonSize.x / amtOfLabels));
				if (editAttributeName == i) {
					ImGui::SetNextItemWidth(100);
					ImGui::InputText("##newName", newAttributeName, ImGuiInputTextFlags_AutoSelectAll);
					if ((ImGui::IsKeyPressedMap(ImGuiKey_Enter) || ImGui::IsKeyPressedMap(ImGuiKey_KeyPadEnter))) {
						pcAttributes[i].name = std::string(newAttributeName);
						sortAttributes();		//resorting attributes
						editAttributeName = -1;
					}
					if (!ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
						editAttributeName = -1;
					}
				}
				else {
					int textSize = ImGui::CalcTextSize(name.c_str()).x;
					if (textSize >= buttonSize.x) {
						//add ellipsis at the end of the text
						bool tooLong = true;
						std::string curSubstring = name.substr(0, name.size() - 4) + "...";
						while (tooLong) {
							textSize = ImGui::CalcTextSize(curSubstring.c_str()).x;
							tooLong = textSize > buttonSize.x;
							curSubstring = curSubstring.substr(0, curSubstring.size() - 4) + "...";
						}
						name = curSubstring;
					}
					ImGui::Button(name.c_str(), buttonSize);
					if (name != pcAttributes[i].originalName && ImGui::IsItemHovered()) {
						ImGui::BeginTooltip();
						ImGui::Text("%s", pcAttributes[i].originalName.c_str());
						ImGui::Text("Drag and drop to switch axes, hold ctrl to shuffle");
						ImGui::EndTooltip();
					}
					if (name == pcAttributes[i].originalName && ImGui::IsItemHovered()) {
						ImGui::BeginTooltip();
						ImGui::Text("Drag and drop to switch axes, hold ctrl to shuffle");
						ImGui::EndTooltip();
					}
					if (ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {
						editAttributeName = i;
						strcpy(newAttributeName, pcAttributes[i].originalName.c_str());
					}
				}

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
			static int popupMinMax = -1;
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

				if (ImGui::DragFloat(name.c_str(), &pcAttributes[i].max, (pcAttributes[i].max - pcAttributes[i].min) * .001f, 0.0f, 0.0f, "%6.4g")) {
					pcPlotRender = true;
					pcPlotPreviousSlectedDrawList = -1;
				}
				if (ImGui::IsItemClicked(1)) {
					ImGui::OpenPopup("MinMaxPopup");
					popupMinMax = i;
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
				if (ImGui::DragFloat(name.c_str(), &pcAttributes[i].min, (pcAttributes[i].max - pcAttributes[i].min) * .001f, .0f, .0f, "%6.4g")) {
					pcPlotRender = true;
					pcPlotPreviousSlectedDrawList = -1;
				}
				if (ImGui::IsItemClicked(1)) {
					ImGui::OpenPopup("MinMaxPopup");
					popupMinMax = i;
				}
				ImGui::PopItemWidth();

				c++;
				c1++;
				offset += gap;
			}
			if (ImGui::BeginPopup("MinMaxPopup")) {
				if (ImGui::MenuItem("Swap min/max")) {
					std::swap(pcAttributes[popupMinMax].min, pcAttributes[popupMinMax].max);
					pcPlotRender = true;
				}
				if (ImGui::MenuItem("Reset min/max")) {
					std::string resetName = g_PcPlotDrawLists.front().name;
					bool update = loadAttributeSettings(resetName, popupMinMax);
					if (update) {
						updateAllDrawListIndexBuffer();
						pcPlotRender = true;
					}
				}
				ImGui::EndPopup();
			}

			ImVec2 picSize(ImGui::GetWindowWidth() - 2 * paddingSide + 5, io.DisplaySize.y * 2 / 5);
			bool brushRightClickMenu = false;
			bool isBrushRightClickMenuOpen = ImGui::IsPopupOpen("BrushMenu");
			if (pcSettings.toggleGlobalBrushes) {
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
					if (ImGui::Checkbox(name.c_str(), &brushTemplateAttrEnabled[i]) || pcSettings.updateBrushTemplates) {
						pcSettings.updateBrushTemplates = false;
						if (selectedTemplateBrush != -1) {
							if (pcSettings.drawListForTemplateBrush) {
								removePcPlotDrawList(g_PcPlotDrawLists.back());
								pcSettings.drawListForTemplateBrush = false;
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
							if (ds.dataType == DataType::ContinuousDlf) {			//oneData indicates a .dlf data -> template brushes are available
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
							else if (pcSettings.showCsvTemplates && ds.dataType != DataType::ContinuousDlf) {
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
				if (ImGui::Checkbox("Show .idxf brush templates", &pcSettings.showCsvTemplates)) {
					pcSettings.updateBrushTemplates = true;
				}

				ImGui::SameLine(250);
				if (ImGui::Button("Combine active global brushes")) {
					GlobalBrush combo;
					combo.nam = "Combined(";
					combo.id = "comb(";
					bool any = false;
					for (auto& brush : globalBrushes) {
						if (!brush.active)
							continue;
						
						any = true;
						for (auto& br : brush.brushes) {
							combo.brushes[br.first].insert(combo.brushes[br.first].end(), br.second.begin(), br.second.end());
						}
						brush.active = false;
						combo.nam += brush.nam.substr(std::min(brush.nam.length() ,(size_t)5)) + "|";
						combo.id += brush.id.substr(std::min(brush.id.length() ,(size_t)5)) + "|";
					}
					combo.active = true;
					combo.edited = true;
					combo.nam += ")";
					combo.id += ")";
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
					pcSettings.updateBrushTemplates = true;
				}

				//drawing the list for brush templates
				ImGui::BeginChild("brushTemplates", ImVec2(400, 200), true, ImGuiWindowFlags_HorizontalScrollbar);
				ImGui::Text("Brush Templates");
				ImGui::Separator();
				for (int i = 0; i < templateBrushes.size(); i++) {
					if (ImGui::Selectable(templateBrushes[i].name.c_str(), selectedTemplateBrush == i)) {
						selectedGlobalBrush = -1;
						pcPlotSelectedDrawList.clear();
						if (selectedTemplateBrush != i) {
							if (selectedTemplateBrush != -1) {
								if (pcSettings.drawListForTemplateBrush) {
									removePcPlotDrawList(g_PcPlotDrawLists.back());
									pcSettings.drawListForTemplateBrush = false;
								}
								if (globalBrushes.back().kdTree) delete globalBrushes.back().kdTree;
								globalBrushes.pop_back();
							}
							selectedTemplateBrush = i;
							GlobalBrush preview{};
							preview.active = true;
							preview.edited = false;
							preview.useMultivariate = false;
							preview.nam = templateBrushes[i].name;
							preview.id = templateBrushes[i].name;
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
								pcSettings.drawListForTemplateBrush = true;
								createPcPlotDrawList(preview.parentDataset->drawLists.front(), *templateBrushes[i].parentDataSet, preview.parent->name.c_str());
							}
							pcPlotRender = updateAllActiveIndices();
						}
						else {
							selectedTemplateBrush = -1;
							if (pcSettings.drawListForTemplateBrush) {
								removePcPlotDrawList(g_PcPlotDrawLists.back());
								pcSettings.drawListForTemplateBrush = false;
							}
							if (globalBrushes.back().kdTree) delete globalBrushes.back().kdTree;
							globalBrushes.pop_back();
							pcPlotRender = updateAllActiveIndices();
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
					if (ImGui::Selectable(globalBrushes[i].nam.c_str(), selectedGlobalBrush == i, ImGuiSelectableFlags_None, ImVec2(350, 0))) {
						pcPlotSelectedDrawList.clear();
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
						ImGui::Text("%s", brush->id.c_str());
						ImGui::EndDragDropSource();
					}
					static char newBrushName[50]{};
					if (ImGui::IsItemClicked(1)) {
						ImGui::OpenPopup(("GlobalBrushPopup##" + globalBrushes[i].id).c_str());
						strcpy(newBrushName, globalBrushes[i].nam.c_str());
					}
					if (ImGui::BeginPopup(("GlobalBrushPopup##" + globalBrushes[i].id).c_str(), ImGuiWindowFlags_AlwaysAutoResize)) {
						ImGui::SetNextItemWidth(100);
						if (ImGui::InputText("##newBrushName", newBrushName, 50, ImGuiInputTextFlags_EnterReturnsTrue)) {
							globalBrushes[i].nam = newBrushName;
							ImGui::CloseCurrentPopup();
						}
						ImGui::SameLine();
						if (ImGui::MenuItem("Rename")) {
							globalBrushes[i].nam = newBrushName;
						}
						if (globalBrushes[i].kdTree) {
							if (ImGui::BeginCombo("Fracture depth", std::to_string(globalBrushes[i].fractureDepth).c_str())) {
								for (int j = 0; j < pcSettings.maxFractionDepth; j++) {
									if (ImGui::Selectable(std::to_string(j).c_str())) {
										globalBrushes[i].fractureDepth = j;
										globalBrushes[i].fractions = globalBrushes[i].kdTree->getBounds(j, pcSettings.outlierRank);
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
									globalBrushes[i].kdTree = new KdTree(globalBrushes[i].parentDataset->drawLists.front().indices, globalBrushes[i].parentDataset->data, globalBrushes[i].attributes, bounds, pcSettings.maxFractionDepth, (KdTree::BoundsBehaviour) pcSettings.boundsBehaviour, (KdTree::SplitBehaviour)pcSettings.splitBehaviour);
								else
									globalBrushes[i].kdTree = new KdTree(globalBrushes[i].parent->indices, globalBrushes[i].parentDataset->data, globalBrushes[i].attributes, bounds, pcSettings.maxFractionDepth, (KdTree::BoundsBehaviour) pcSettings.boundsBehaviour, (KdTree::SplitBehaviour) pcSettings.splitBehaviour);
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
								if(!globalBrushes[i].parentDataset){
									std::cout << "Parent dataset has to be set for global brush." << std::endl;
								}
								else{
#ifdef _DEBUG
									std::cout << "Starting to build the kd tree for fracturing." << std::endl;
#endif
									if(globalBrushes[i].edited)
										globalBrushes[i].kdTree = new KdTree(globalBrushes[i].parentDataset->drawLists.front().indices, globalBrushes[i].parentDataset->data, globalBrushes[i].attributes, bounds, pcSettings.maxFractionDepth, (KdTree::BoundsBehaviour)pcSettings.boundsBehaviour, (KdTree::SplitBehaviour)pcSettings.splitBehaviour);
									else
										globalBrushes[i].kdTree = new KdTree(globalBrushes[i].parent->indices, globalBrushes[i].parentDataset->data, globalBrushes[i].attributes, bounds, pcSettings.maxFractionDepth, (KdTree::BoundsBehaviour)pcSettings.boundsBehaviour, (KdTree::SplitBehaviour)pcSettings.splitBehaviour);
#ifdef _DEBUG
									std::cout << "Kd tree done." << std::endl;
#endif
								}
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
									if (pcSettings.drawListForTemplateBrush) {
										removePcPlotDrawList(g_PcPlotDrawLists.back());
										pcSettings.drawListForTemplateBrush = false;
									}
									if (globalBrushes.back().kdTree) delete globalBrushes.back().kdTree;
									globalBrushes.pop_back();
								}
								else {
									globalBrushes[i] = globalBrushes[globalBrushes.size() - 2];
									globalBrushes[globalBrushes.size() - 2] = globalBrushes[globalBrushes.size() - 1];
									if (pcSettings.drawListForTemplateBrush) {
										removePcPlotDrawList(g_PcPlotDrawLists.back());
										pcSettings.drawListForTemplateBrush = false;
									}
									if (globalBrushes.back().kdTree) delete globalBrushes.back().kdTree;
									globalBrushes.pop_back();
								}
							}
							else {
								globalBrushes[i] = globalBrushes[globalBrushes.size() - 1];
								if (pcSettings.drawListForTemplateBrush) {
									removePcPlotDrawList(g_PcPlotDrawLists.back());
									pcSettings.drawListForTemplateBrush = false;
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
					if (i < globalBrushes.size() && ImGui::Checkbox(("##cbgb_" + globalBrushes[i].id).c_str(), &globalBrushes[i].active)) {
						pcPlotRender = updateAllActiveIndices();
					}
				}
				
				static int globalBrushCreateCount = 0;
				if (ImGui::Button("+##globalBrush", {100,0})) {
					globalBrushes.push_back({});
					globalBrushes.back().active = true;
					globalBrushes.back().nam = std::to_string(globalBrushCreateCount++);
					globalBrushes.back().id = globalBrushes.back().nam;
					for (int i = 0; i < pcAttributes.size(); ++i) {
						globalBrushes.back().brushes[i] = {};
					}
				}

				if (popEnd) {
					if (pcSettings.drawListForTemplateBrush) {
						removePcPlotDrawList(g_PcPlotDrawLists.back());
						pcSettings.drawListForTemplateBrush = false;
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
						gb.nam = dl->name;
						gb.id = dl->name;
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
					ImGui::BeginChild(("##brushStat" + brush.id).c_str(), ImVec2(400, 0), true);
					ImGui::Text("%s", brush.nam.c_str());
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
								ImGui::Text("%s", ratio.first.c_str());
								ImGui::EndTooltip();
							}
						}
						else {
							ImGui::Text("%s", ratio.first.c_str());
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
                        // This variable is used to generate individual tooltips.
                        int brushCase = -1;
						if (brush.parent != nullptr && brush.lineRatios.find(brush.parent->name) != brush.lineRatios.end()) {
							static const float width = 180;
							ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(screenCursorPos.x + xOffset, screenCursorPos.y), ImVec2(screenCursorPos.x + xOffset + width, screenCursorPos.y + lineHeight - 1), ImGui::ColorConvertFloat4ToU32(ImGui::GetStyle().Colors[ImGuiCol_FrameBg]), ImGui::GetStyle().FrameRounding);
							float linepos = width / 2;
							if (brush.parent->name == dl->name) {	//identity dataset
								// It cannot move past the middle line, since once all points of the idx-list are contained, there are no more to add.
                                // Ratio: if (br/ds2 > cl/ds)  -> ( 1 -  (cl/ds) // (br/ds))  otherwise -(1 - (br/ds)//(cl/ds)). Since ds1 =ds2, it's basically if br > cl (which is never the case), then move br/cl to the left
								linepos += (brush.lineRatios[brush.parent->name] / (float)ds->data.size() > brush.parent->pointRatio) ? 
									(1 - (brush.parent->pointRatio / (brush.lineRatios[brush.parent->name] / (float)ds->data.size()))) * linepos : 
									-(1 - ((brush.lineRatios[brush.parent->name] / (float)ds->data.size()) / brush.parent->pointRatio)) * linepos;

                                brushCase = 0;
								//linepos += (dl->activeInd.size()/(float)ds->data.size() > brush.parent->pointRatio) ? (1 - (brush.par ent->pointRatio / (dl->activeInd.size() / (float)ds->data.size()))) * linepos : -(1 - ((dl->activeInd.size() / (float)ds->data.size()) / brush.parent->pointRatio)) * linepos;
								//linepos += (brush.lineRatios[brush.parent->name] > brush.parent->pointRatio) ? (1 - (brush.parent->pointRatio / (brush.lineRatios[brush.parent->name]))) * linepos : -(1 - ((brush.lineRatios[brush.parent->name]) / brush.parent->pointRatio)) * linepos;
							}
                            else{


                                if (brush.parent->parentDataSetName == ""){
                                    DataSet* parentDS = nullptr;
                                    // DataSet* currParentDataSet = nullptr;
                                    // Determine parent drawlist
                                    for (auto& ds : g_PcPlotDataSets)
                                    {
                                        for (auto& currdl : ds.drawLists)
                                        {
                                            // Checking the buffer Reference should be enough, nevertheless, we check all 3 conditions.
                                            if ((currdl.name == brush.parent->name) && (currdl.indices.size() == brush.parent->indices.size())  && (&currdl.buffer == &(brush.parent->buffer) ) )
                                            {
                                                parentDS = &ds;
                                                brush.parent->parentDataSetName = ds.name;
                                                std::cout << "setting brush parent data set name to: " << brush.parent->parentDataSetName << "\n";
                                                break;

                                            }
                                        }
                                        if (parentDS != nullptr) { break; }
                                    }
                                }

                                if (brush.parent->parentDataSetName  == dl->name)
                                {
                                    //linepos += (dl->activeInd.size()/(float)ds->data.size() > brush.parent->pointRatio) ? (1 - (brush.parent->pointRatio / (dl->activeInd.size() / (float)ds->data.size()))) * linepos : -(1 - ((dl->activeInd.size() / (float)ds->data.size()) / brush.parent->pointRatio)) * linepos;
                                    //linepos += (ratio.second > brush.lineRatios[brush.parent->name]) ? (1 - (brush.lineRatios[brush.parent->name] / ratio.second)) * linepos : -(1 - (ratio.second / brush.lineRatios[brush.parent->name])) * linepos;
                                    //linepos += (dl->activeInd.size()/(float)ds->data.size() > brush.lineRatios[brush.parent->name]) ? (1 - (brush.lineRatios[brush.parent->name] / (dl->activeInd.size() / (float)ds->data.size()))) * linepos : -(1 - ((dl->activeInd.size() / (float)ds->data.size()) / brush.lineRatios[brush.parent->name])) * linepos;
                                    // Ratio: if ( br2 / ds2 > br/ds ) -> 1- (br/ds)//(br2/ds2) otherwise the other way round
                                    //
                                    linepos += (ratio.second / (float)ds->data.size() > (brush.lineRatios[brush.parent->name] / (float)brush.parentDataset->data.size())) ? (1 - ((brush.lineRatios[brush.parent->name] / (float)brush.parentDataset->data.size()) / (ratio.second / (float)ds->data.size()))) * linepos : -(1 - ((ratio.second / (float)ds->data.size()) / (brush.lineRatios[brush.parent->name] / (float)brush.parentDataset->data.size()))) * linepos;

                                    brushCase = 1;

                                }
                                else{
                                    float brRatioRep = brush.lineRatios[brush.parent->parentDataSetName] /  (float)brush.parentDataset->data.size();
                                    float brRatioCurrentMember = ratio.second /  (float)ds->data.size();
                                    linepos += (brRatioCurrentMember < brRatioRep) ? (1 - (brRatioCurrentMember / brRatioRep))* linepos : (-(1 - (brRatioRep/brRatioCurrentMember))) * linepos;

                                    brushCase = 2;
                                    //linepos += (ratio.second / (float)ds->data.size() > (brush.lineRatios[brush.parent->name] / (float)brush.parentDataset->data.size())) ? (1 - ((brush.lineRatios[brush.parent->name] / (float)brush.parentDataset->data.size()) / (ratio.second / (float)ds->data.size()))) * linepos : -(1 - ((ratio.second / (float)ds->data.size()) / (brush.lineRatios[brush.parent->name] / (float)brush.parentDataset->data.size()))) * linepos;
                                }
                            }


							ImGui::GetWindowDrawList()->AddLine(ImVec2(screenCursorPos.x + xOffset + linepos, screenCursorPos.y), ImVec2(screenCursorPos.x + xOffset + linepos, screenCursorPos.y + lineHeight - 1), IM_COL32(255, 0, 0, 255), 5);
							ImGui::GetWindowDrawList()->AddLine(ImVec2(screenCursorPos.x + xOffset + width / 2, screenCursorPos.y), ImVec2(screenCursorPos.x + xOffset + width / 2, screenCursorPos.y + lineHeight - 1), IM_COL32(255, 255, 255, 255));

							if (ImGui::IsMouseHoveringRect(ImVec2(screenCursorPos.x + xOffset, screenCursorPos.y), ImVec2(screenCursorPos.x + xOffset + width, screenCursorPos.y + lineHeight - 1))) {
								ImGui::BeginTooltip();
                                std::string caseDependentString = "";
                                switch(brushCase){
                                case 0:
                                    caseDependentString = "Remaining lines in cluster after brush (in percent)\n";
                                    break;
                                case 1:
                                    caseDependentString = "Ratio #br/#cl\n";
                                    break;
                                case 2:
                                    caseDependentString = "Ratio (% brushed in Rep) / (% brushed in this data set) \n";
                                    break;
                                }



                                if (linepos < width / 2) { // y = w/2 - (1-x)*(w/2)
                                    std::stringstream sstr;
                                    sstr << std::fixed << std::setprecision(2) << (((linepos / (width / 2))) * 100);
                                    caseDependentString +=  "Ratio is  " + sstr.str() + "%. Less points were selected.";

//                                    ImGui::Text(std::strcat(caseDependentString.c_str(), "Ratio is  %2.1f%%. Less points were selected", ((linepos / (width / 2))) * 100));
                                    ImGui::Text("%s", caseDependentString.c_str());
								}
                                else {// y = w/2 - (1-x)*w/2

                                    std::stringstream sstr;
                                    sstr << std::fixed << std::setprecision(2) << (1.0/ ((1 - ((linepos - width / 2) / (width / 2)))));
                                    caseDependentString +=  "Ratio is  " + sstr.str() + ".  More points were selected.";

//                                    ImGui::Text(cDS + "Ratio is  %2.1f% to 1. More points were selected.", 1.0/ ((1 - ((linepos - width / 2) / (width / 2)))));

                                    ImGui::Text("%s", caseDependentString.c_str());
								}
								ImGui::EndTooltip();
							}
						}

						screenCursorPos.y += lineHeight;
						cursorPos.y += lineHeight;
					}
					ImGui::SetCursorPos(defaultCursorPos);
					int hover = ImGui::PlotHistogramVertical(("##histo" + brush.id).c_str(), ratios.data(), ratios.size(), 0, NULL, 0, 1.0f, ImVec2(75, lineHeight * ratios.size()));
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
						ImGui::Text("%s", ratio.first.c_str());
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

				gap = (picSize.x - ((pcSettings.drawHistogramm) ? pcSettings.histogrammWidth / 2.0f * picSize.x : 0)) - 4;
				//drawing axis lines
				if (pcSettings.enableAxisLines) {
					for (int i = 0; i < amtOfLabels; i++) {
						float x = picPos.x + i * gap / (amtOfLabels - 1) + ((pcSettings.drawHistogramm) ? (pcSettings.histogrammWidth / 4.0 * picSize.x) : 0);
						ImVec2 a(x, picPos.y);
						ImVec2 b(x, picPos.y + picSize.y - 1);
						ImGui::GetWindowDrawList()->AddLine(a, b, IM_COL32((1 - pcSettings.PcPlotBackCol.x) * 255, (1 - pcSettings.PcPlotBackCol.y) * 255, (1 - pcSettings.PcPlotBackCol.z) * 255, 255), 1);
						//drawing axis sections
						int attrib = attributeOfPlace(i);
						if (pcSettings.axisTickAmount > 0 && pcAttributes[attrib].categories.empty()) {
							float lineHeight = ImGui::GetTextLineHeight() / 2;
							ImVec2 oldCoursor = ImGui::GetCursorPos();
							int ticksAmount = pcSettings.axisTickAmount;
							if (pcSettings.enableZeroTick) {
								--ticksAmount;
								Vec2 bounds{ std::min(pcAttributes[attrib].min, pcAttributes[attrib].max), std::max(pcAttributes[attrib].min, pcAttributes[attrib].max) };
								if (bounds.u <= 0 && bounds.v > 0) {	//only draw the 0 tick if 0 is in the range
									float y = picPos.y + picSize.y - picSize.y * (-pcAttributes[attrib].min / (pcAttributes[attrib].max - pcAttributes[attrib].min)) - 1;
									a = ImVec2(x, y);
									if (i == amtOfLabels - 1) b = ImVec2(x - pcSettings.axisTickWidth, y);
									else b = ImVec2(x + pcSettings.axisTickWidth, y);
									ImGui::GetWindowDrawList()->AddLine(a, b, IM_COL32((1 - pcSettings.PcPlotBackCol.x) * 255, (1 - pcSettings.PcPlotBackCol.y) * 255, (1 - pcSettings.PcPlotBackCol.z) * 255, 255), 1);
									b.y -= lineHeight;
									if (i == amtOfLabels - 1) b.x -= ImGui::CalcTextSize("0").x;
									ImGui::SetCursorScreenPos(b);
									ImGui::Text("0");
								}
							}
							//if (ticksAmount == 1) ++ticksAmount;

							for (int tick = 0; tick < ticksAmount; ++tick) {
								float y = (tick + 1) / float(ticksAmount + 1);
								float yval = y * (pcAttributes[attrib].max - pcAttributes[attrib].min) + pcAttributes[attrib].min;
								y *= picSize.y;
								y = picPos.y + picSize.y - y - 1;
								a = ImVec2(x, y);
								b = (i == amtOfLabels - 1) ? ImVec2(x - pcSettings.axisTickWidth, y): ImVec2(x + pcSettings.axisTickWidth, y);
								ImGui::GetWindowDrawList()->AddLine(a, b, IM_COL32((1 - pcSettings.PcPlotBackCol.x) * 255, (1 - pcSettings.PcPlotBackCol.y) * 255, (1 - pcSettings.PcPlotBackCol.z) * 255, 255), 1);
								static char str[20];
								sprintf(str, "%g", yval);
								b.y -= lineHeight;
								if (i == amtOfLabels - 1) b.x -= ImGui::CalcTextSize(str).x;
								ImGui::SetCursorScreenPos(b);
								ImGui::Text("%s", str);
							}

							ImGui::SetCursorPos(oldCoursor);
						}
					}
				}

				//drawing pie chart for the first drawlist
				if (pcSettings.computeRatioPtsInDLvsIn1axbrushedParent && pcSettings.drawHistogramm) {
					// Count, how many histograms are drawn
					int nrActiveHists = 0;
					for (auto &currdl : g_PcPlotDrawLists)
					{
						nrActiveHists += int(currdl.showHistogramm);
					}
					float xStartOffset = -pcSettings.histogrammWidth / 4.0 * picSize.x;
					float xOffsetPerAttr = (pcSettings.histogrammWidth * picSize.x) / (2 * nrActiveHists);
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



							float x = picPos.x + iActAttr * gap / (amtOfLabels - 1) + ((pcSettings.drawHistogramm) ? (pcSettings.histogrammWidth / 4.0 * picSize.x) : 0);
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

				int ind = 0;
				ImVec2 mousePos = ImGui::GetMousePos();
				for (int i : pcAttrOrd) {
					if (!pcAttributeEnabled[i]) continue;
					float x = picPos.x + float(ind) / (amtOfLabels - 1) * picSize.x; 
					//enable zoom + scroll on attribute axes
					bool hover_axis = mousePos.y >= picPos.y && mousePos.y <= picPos.y + picSize.y && mousePos.x >= x - BRUSHWIDTH / 2 && mousePos.x <= x + BRUSHWIDTH / 2;
					// axis zoom
					if (hover_axis && ImGui::GetIO().KeyCtrl && ImGui::GetIO().MouseWheel) {
						float diff = pcAttributes[i].max - pcAttributes[i].min;
						pcAttributes[i].max -= ImGui::GetIO().MouseWheel * diff * SCROLLSPEED;
						pcAttributes[i].min += ImGui::GetIO().MouseWheel * diff * SCROLLSPEED;
						pcPlotRender = true;
					}
					// axis scroll
					if (hover_axis && ImGui::GetIO().KeyAlt && ImGui::GetIO().MouseWheel) {
						float diff = pcAttributes[i].max - pcAttributes[i].min;
						pcAttributes[i].max -= ImGui::GetIO().MouseWheel * diff * SCROLLSPEED;
						pcAttributes[i].min -= ImGui::GetIO().MouseWheel * diff * SCROLLSPEED;
						pcPlotRender = true;
					}

					//drawing the categorie boxes
					if (pcAttributes[i].categories.size()) {
                        float prev_y = picPos.y + 1.2f * picSize.y;
						for (auto categorie : pcAttributes[i].categories_ordered) {
							float xAnchor = .5f;
							if (ind == 0) xAnchor = 0;
							if (ind == amtOfLabels - 1) xAnchor = 1;
							float y = (categorie.second - pcAttributes[i].min) / (pcAttributes[i].max - pcAttributes[i].min);
							if (y < 0 || y > 1) continue;		//label not seeable
							y = picPos.y + (1 - y) * picSize.y;
                            if (y + 1.2f * ImGui::GetTextLineHeightWithSpacing() > prev_y) continue;
							
							ImVec2 textSize = ImGui::CalcTextSize(categorie.first.c_str());
							ImGui::SetNextWindowPos({ x ,y }, 0, { xAnchor,.5f });
							ImGui::SetNextWindowBgAlpha(ImGui::GetStyle().Colors[ImGuiCol_PopupBg].w * 0.60f);
							ImGuiWindowFlags flags = ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking;
							ImGui::Begin(("Tooltip Categorie" + categorie.first + pcAttributes[i].originalName).c_str(), NULL, flags);
							ImGui::Text("%s", categorie.first.c_str());
							ImGui::End();
                            prev_y = y;
						}
					}
					ind++;
				}

				//clearing the dragged brushes if ctrl key is released
				if (!ImGui::GetIO().MouseDown && !ImGui::GetIO().KeyCtrl) {
					brushDragIds.clear();
				}

				//drawing the global brush
				if (selectedGlobalBrush != -1) {
					if (globalBrushes[selectedGlobalBrush].fractureDepth) {
						GlobalBrush& globalBrush = globalBrushes[selectedGlobalBrush];
						for (int i = 0; i < globalBrush.fractions.size(); i++) {
							for (int j = 0; j < globalBrush.fractions[i].size(); j++) {
								int axis = globalBrush.attributes[j];
								float x = gap * placeOfInd(axis) / (amtOfLabels - 1) + picPos.x - pcSettings.fractionBoxWidth / 2 + ((pcSettings.drawHistogramm) ? (pcSettings.histogrammWidth / 4.0 * picSize.x) : 0);
								float width = pcSettings.fractionBoxWidth;
								float y = ((globalBrush.fractions[i][j].second - pcAttributes[axis].max) / (pcAttributes[axis].min - pcAttributes[axis].max)) * picSize.y + picPos.y;
								float height = (globalBrush.fractions[i][j].second - globalBrush.fractions[i][j].first) / (pcAttributes[axis].max - pcAttributes[axis].min) * picSize.y;
								if (i < pow(2, pcSettings.maxRenderDepth))
									ImGui::GetWindowDrawList()->AddRect(ImVec2(x, y), ImVec2(x + width, y + height), IM_COL32(0, 230, 100, 255), 2, ImDrawCornerFlags_All, pcSettings.fractionBoxLineWidth);
							}
						}
					}
					else {
						bool anyHover = false;
						static bool newBrush = false;
						std::set<int> brushDelete;
						for (auto& brush : globalBrushes[selectedGlobalBrush].brushes) {
							if (!pcAttributeEnabled[brush.first])
								continue;

							ImVec2 mousePos = ImGui::GetIO().MousePos;
							float x = gap * placeOfInd(brush.first) / (amtOfLabels - 1) + picPos.x - BRUSHWIDTH / 2 + ((pcSettings.drawHistogramm) ? (pcSettings.histogrammWidth / 4.0 * picSize.x) : 0);
							float width = BRUSHWIDTH;

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
								edgeHover |= mousePos.x > x&& mousePos.x<x + width && mousePos.y>y - EDGEHOVERDIST + height && mousePos.y < y + EDGEHOVERDIST + height ? 2 : edgeHover;
								int r = (brushDragIds.find(br.first) != brushDragIds.end()) ? 200 : 30;
                                // Determine color for brush box highlight etc.
                                ImU32 brushBoxColor = IM_COL32(r, 0, 200, 255);
                                if (r == 30){brushBoxColor = IM_COL32(r, 230, 100, 255);}

                                ImGui::GetWindowDrawList()->AddRect(ImVec2(x, y), ImVec2(x + width, y + height), brushBoxColor, 1, ImDrawCornerFlags_All, 5);
								brushHover |= hover || edgeHover;
								//set mouse cursor
								if (edgeHover) {
									ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
								}
								if (hover) {
									ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
								}
								//activate dragging of edge
								if (edgeHover &&  ImGui::GetIO().MouseClicked[0]) {
									if (!ImGui::GetIO().KeyCtrl) {
										brushDragIds.clear();
									}
									brushDragIds.insert(br.first);
									brushDragMode = edgeHover;
								}
								if (hover && ImGui::GetIO().MouseClicked[0]) {
									if (!ImGui::GetIO().KeyCtrl) {
										brushDragIds.clear();
									}
									brushDragIds.insert(br.first);
									brushDragMode = 0;
								}
								//drag edge
								if (brushDragIds.find(br.first) != brushDragIds.end() && (ImGui::GetIO().MouseDown[0] || ImGui::IsKeyPressed(82) || ImGui::IsKeyPressed(81))) {
									globalBrushes[selectedGlobalBrush].edited = true;
									if (brushDragMode == 0 || ImGui::IsKeyPressed(82) || ImGui::IsKeyPressed(81)) {
										float delta = ImGui::GetIO().MouseDelta.y / picSize.y * (pcAttributes[brush.first].max - pcAttributes[brush.first].min);
										if (ImGui::IsKeyPressed(82))
											delta = (pcAttributes[brush.first].max - pcAttributes[brush.first].min) * -.01;
										if( ImGui::IsKeyPressed(81))
											delta = (pcAttributes[brush.first].max - pcAttributes[brush.first].min) * .01;
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

								//check for deletion of brush
								if (ImGui::GetIO().MouseClicked[1] && hover) {
									brushRightClickMenu = true;
									//brushDragIds.clear();
								}

								if (ImGui::IsKeyPressed(76, false) && brushDragIds.size()) {
									brushDelete = brushDragIds;
									brushDragIds.clear();
								}

								if (ImGui::IsMouseDoubleClicked(0) && brushHover) {
									brushDelete = { (int)br.first };
									brushDragIds.clear();
								}

								//adjusting the bounds of the brush by a mu
								if (brushHover && ImGui::GetIO().MouseWheel) {
									if (ImGui::GetIO().MouseWheel > 0) {
										br.second.first += ImGui::GetIO().MouseWheel * (pcAttributes[brush.first].max - pcAttributes[brush.first].min) * pcSettings.brushMuFactor;
									}
									else {
										br.second.second += ImGui::GetIO().MouseWheel * (pcAttributes[brush.first].max - pcAttributes[brush.first].min) * pcSettings.brushMuFactor;
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
									ImGui::Begin(("Tooltip brush max##" + std::to_string(br.first)).c_str(), NULL, flags);
									ImGui::Text("%f", br.second.second);
									ImGui::End();

									ImGui::SetNextWindowPos({ x + width / 2, y + height }, 0, { xAnchor,0 });
									ImGui::SetNextWindowBgAlpha(ImGui::GetStyle().Colors[ImGuiCol_PopupBg].w * 0.60f);
									ImGui::Begin(("Tooltip brush min##" + std::to_string(br.first)).c_str(), NULL, flags);
									ImGui::Text("%f", br.second.first);
									ImGui::End();
								}

								ind++;
							}

							//create a new brush
							bool axisHover = mousePos.x > x&& mousePos.x < x + BRUSHWIDTH && mousePos.y > picPos.y&& mousePos.y < picPos.y + picSize.y;
							anyHover |= brushHover;
							if (!brushHover && axisHover && brushDragIds.empty()) {
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
								newBrush = true;
							}

						}
						//deleting a brush
						if (brushDelete.size()) {
							globalBrushes[selectedGlobalBrush].edited = true;
							for (auto& gb : globalBrushes[selectedGlobalBrush].brushes) {
								for (int br = 0; br < gb.second.size(); ++br) {
									if (brushDelete.find(gb.second[br].first) != brushDelete.end()) {
										gb.second[br] = gb.second[gb.second.size() - 1];
										gb.second.pop_back();
										--br;
									}
								}
							}
							brushDelete.clear();
							pcPlotRender = updateAllActiveIndices();
							updateIsoSurface(globalBrushes[selectedGlobalBrush]);
						}

						//release edge
						//if (brushDragIds.find(br.first) != brushDragIds.end() && ImGui::GetIO().MouseReleased[0] && !ImGui::GetIO().KeyCtrl) {
						if (!anyHover && brushDragIds.size() && (ImGui::GetIO().MouseReleased[0] || (!newBrush && ImGui::GetIO().MouseClicked[0])) && !ImGui::GetIO().KeyCtrl) {
							newBrush = false;
							brushDragIds.clear();
							pcPlotRender = updateAllActiveIndices();
							updateIsoSurface(globalBrushes[selectedGlobalBrush]);
						}
						if (anyHover && (ImGui::GetIO().MouseReleased[0]) || (brushDragIds.size() && ( ImGui::IsKeyPressed(82) || ImGui::IsKeyPressed(81)))) {
							pcPlotRender = updateAllActiveIndices();
							updateIsoSurface(globalBrushes[selectedGlobalBrush]);
						}
						//if (edgeHover && !ImGui::GetIO().MouseDown[0])
						//	edgeHover = 0;
					}
				}

				//drawing the template brush, these are not changeable
				if (selectedTemplateBrush != -1) {
					for (const auto& brush : globalBrushes.back().brushes) {
						if (!pcAttributeEnabled[brush.first] || !brush.second.size())
							continue;

						float x = gap * placeOfInd(brush.first) / (amtOfLabels - 1) + picPos.x - BRUSHWIDTH / 2 + ((pcSettings.drawHistogramm) ? (pcSettings.histogrammWidth / 4.0 * picSize.x) : 0);
						float y = ((brush.second[0].second.second - pcAttributes[brush.first].max) / (pcAttributes[brush.first].min - pcAttributes[brush.first].max)) * picSize.y + picPos.y;
						float width = BRUSHWIDTH;
						float height = (brush.second[0].second.second - brush.second[0].second.first) / (pcAttributes[brush.first].max - pcAttributes[brush.first].min) * picSize.y;
						ImGui::GetWindowDrawList()->AddRect(ImVec2(x, y), ImVec2(x + width, y + height), IM_COL32(30, 0, 200, 150), 1, ImDrawCornerFlags_All, 5);
					}
				}
			}

			//drawing the brush windows
			if (pcPlotSelectedDrawList.size()) {
				//getting the drawlist;
				DrawList* dl = 0;
				uint32_t c = 0;
				for (DrawList& d : g_PcPlotDrawLists) {
					if (c == pcPlotSelectedDrawList[0]) {
						dl = &d;
						break;
					}
					c++;
				}

				static bool newBrush = false;
				bool brushDelete = false;

				for (int i = 0; i < pcAttributes.size(); i++) {
					if (!pcAttributeEnabled[i])
						continue;

					int del = -1;
					int ind = 0;
					bool brushHover = false;

					ImVec2 mousePos = ImGui::GetIO().MousePos;
					float x = gap * placeOfInd(i) / (amtOfLabels - 1) + picPos.x - BRUSHWIDTH / 2 + ((pcSettings.drawHistogramm) ? (pcSettings.histogrammWidth / 4.0 * picSize.x) : 0);
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

						anyHover |= brushHover;

						//set mouse cursor
						if (edgeHover) {
							ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
						}
						if (hover) {
							ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
						}
						//activate dragging of edge
						if (edgeHover && ImGui::GetIO().MouseClicked[0]) {
							if (!ImGui::GetIO().KeyCtrl) {
								brushDragIds.clear();
							}
							brushDragIds.insert(b.id);
							brushDragMode = edgeHover;
						}
						if (hover && ImGui::GetIO().MouseClicked[0]) {
							if (!ImGui::GetIO().KeyCtrl) {
								brushDragIds.clear();
							}
							brushDragIds.insert(b.id);
							brushDragMode = 0;
						}
						//drag edge
						if (brushDragIds.find(b.id) != brushDragIds.end() && (ImGui::GetIO().MouseDown[0] || ImGui::IsKeyPressed(82) || ImGui::IsKeyPressed(81))) {
							if (brushDragMode == 0 || ImGui::IsKeyPressed(82) || ImGui::IsKeyPressed(81)) {
								float delta = ImGui::GetIO().MouseDelta.y / picSize.y * (pcAttributes[i].max - pcAttributes[i].min);
								if (ImGui::IsKeyPressed(82))
									delta = (pcAttributes[i].max - pcAttributes[i].min) * -.01;
								if (ImGui::IsKeyPressed(81))
									delta = (pcAttributes[i].max - pcAttributes[i].min) * .01;
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

						//check right click
						if (ImGui::GetIO().MouseClicked[1] && hover) {
							brushRightClickMenu = true;
						}

						if (ImGui::IsKeyPressed(76, false) && brushDragIds.size()) {
							brushDelete = true;
						}

						if (ImGui::IsMouseDoubleClicked(0) && brushHover) {
							brushDelete = true;
							brushDragIds = { b.id };
						}

						//draw tooltip on hover for min and max value
						if (hover || edgeHover || brushDragIds.find(b.id) != brushDragIds.end()) {
							float xAnchor = .5f;
							if (pcAttrOrd[i] == 0) xAnchor = 0;
							if (pcAttrOrd[i] == pcAttributes.size() - 1) xAnchor = 1;

							ImGui::SetNextWindowPos({ x + width / 2,y }, 0, { xAnchor,1 });
							ImGui::SetNextWindowBgAlpha(ImGui::GetStyle().Colors[ImGuiCol_PopupBg].w * 0.60f);
							ImGuiWindowFlags flags = ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking;
							ImGui::Begin(("Tooltip brush max##" + std::to_string(b.id)).c_str(), NULL, flags);
							ImGui::Text("%f", b.minMax.second);
							ImGui::End();

							ImGui::SetNextWindowPos({ x + width / 2, y + height }, 0, { xAnchor,0 });
							ImGui::SetNextWindowBgAlpha(ImGui::GetStyle().Colors[ImGuiCol_PopupBg].w * 0.60f);
							ImGui::Begin(("Tooltip brush min##" + std::to_string(b.id)).c_str(), NULL, flags);
							ImGui::Text("%f", b.minMax.first);
							ImGui::End();
						}

						ind++;
					}

					//create a new brush
					bool axisHover = mousePos.x > x&& mousePos.x < x + BRUSHWIDTH && mousePos.y > picPos.y&& mousePos.y < picPos.y + picSize.y;
					if (!brushHover && axisHover && brushDragIds.size() == 0) {
						ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

						if (ImGui::GetIO().MouseClicked[0]) {
							newBrush = true;
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
				//deleting a brush
				if (brushDelete) {
					for (auto& gb : dl->brushes) {
						for (int br = 0; br < gb.size(); ++br) {
							if (brushDragIds.find(gb[br].id) != brushDragIds.end()) {
								gb[br] = gb[gb.size() - 1];
								gb.pop_back();
								--br;
							}
						}
					}
					brushDragIds.clear();
					pcPlotRender = updateActiveIndices(*dl);
				}
				
				//release edge
				//if (brushDragIds.find(br.first) != brushDragIds.end() && ImGui::GetIO().MouseReleased[0] && !ImGui::GetIO().KeyCtrl) {
				if (!anyHover && brushDragIds.size() && (ImGui::GetIO().MouseReleased[0] || (!newBrush && ImGui::GetIO().MouseClicked[0])) && !ImGui::GetIO().KeyCtrl) {
					newBrush = false;
					brushDragIds.clear();
					pcPlotRender = updateActiveIndices(*dl);
					updateIsoSurface(*dl);
				}
				if (anyHover && (ImGui::GetIO().MouseReleased[0]) || brushDragIds.size() && (ImGui::IsKeyPressed(82) || ImGui::IsKeyPressed(81)) && selectedGlobalBrush < 0) {
					pcPlotRender = updateActiveIndices(*dl);
					updateIsoSurface(*dl);
				}
			}

			//brush right click menu
			static std::set<int> brushIdsCopy;
			if (brushRightClickMenu) {
				ImGui::OpenPopup("BrushMenu");
				brushIdsCopy = brushDragIds;
			}
			if (ImGui::BeginPopup("BrushMenu")) {
				brushDragIds = brushIdsCopy;
				ImGui::PushItemWidth(100);
				if (ImGui::MenuItem("Delete", "", false, (bool)brushDragIds.size())) {
					if (selectedGlobalBrush >= 0) {
						globalBrushes[selectedGlobalBrush].edited = true;
						for (auto& gb : globalBrushes[selectedGlobalBrush].brushes) {
							for (int br = 0; br < gb.second.size(); ++br) {
								if (brushDragIds.find(gb.second[br].first) != brushDragIds.end()) {
									gb.second[br] = gb.second[gb.second.size() - 1];
									gb.second.pop_back();
									--br;
								}
							}
						}
						brushDragIds.clear();
						pcPlotRender = updateAllActiveIndices();
						updateIsoSurface(globalBrushes[selectedGlobalBrush]);
					}
					else {
						DrawList* dl = 0;
						uint32_t c = 0;
						for (DrawList& d : g_PcPlotDrawLists) {
							if (c == pcPlotSelectedDrawList[0]) {
								dl = &d;
								break;
							}
							c++;
						}
						for (auto& gb : dl->brushes) {
							for (int br = 0; br < gb.size(); ++br) {
								if (brushDragIds.find(gb[br].id) != brushDragIds.end()) {
									gb[br] = gb[gb.size() - 1];
									gb.pop_back();
									--br;
								}
							}
						}
						brushDragIds.clear();
						pcPlotRender = updateActiveIndices(*dl);
						updateIsoSurface(*dl);
					}
				}
				ImGui::DragInt("LiveBrushThreshold", &pcSettings.liveBrushThreshold, 1000, 0, 10000000);

				ImGui::EndPopup();
			}

			//plot right click menu
			for (int i = 0; i < amtOfLabels; ++i) {
				auto mousePos = ImGui::GetIO().MousePos;
				float histWidth = pcSettings.histogrammWidth * picSize.x * .5f;
				float x = gap * i / (amtOfLabels - 1) + picPos.x - histWidth / 2 + ((pcSettings.drawHistogramm) ? (pcSettings.histogrammWidth / 4.0 * picSize.x) : 0);
				histogramHovered |= mousePos.x > x && mousePos.x < x + histWidth && mousePos.y > picPos.y && mousePos.y < picPos.y + picSize.y;
				histogramHovered &= pcSettings.drawHistogramm;
			}
			if (picHovered && !anyHover && !histogramHovered && ImGui::GetIO().MouseClicked[1]) {
				ImGui::OpenPopup("PCPMenu");
			}
			if (ImGui::BeginPopup("PCPMenu")) {
				ImGui::PushItemWidth(100);
				if (ImGui::MenuItem("DrawHistogram", "", &pcSettings.drawHistogramm))
					pcPlotRender = true;
				if (ImGui::MenuItem("Show Pc Plot Density", "", &pcSettings.pcPlotDensity))
					pcPlotRender = true;
				if (ImGui::MenuItem("Density Mapping", "", &pcSettings.enableDensityMapping))
					pcPlotRender = true;
				if (ImGui::MenuItem("Greyscale denisty", "", &pcSettings.enableDensityGreyscale)) {
					if (pcAttributes.size()) {
						uploadDensityUiformBuffer();
						pcPlotRender = true;
					}
				}
				ImGui::MenuItem("Enable Medina Calc", "", &pcSettings.calculateMedians);
				if (ImGui::MenuItem("Enable brushing", "", &pcSettings.enableBrushing)) {
					updateAllActiveIndices();
					pcPlotRender = true;
				}
				if (ImGui::SliderFloat("Median line width", &pcSettings.medianLineWidth, 1, 20))
					pcPlotRender = true;
				if (ImGui::ColorEdit4("Plot background", &pcSettings.PcPlotBackCol.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar))
					pcPlotRender = true;
				if (ImGui::MenuItem("Render splines", "", &pcSettings.renderSplines))
					pcPlotRender = true;
				ImGui::MenuItem("Enable axis lines", "", &pcSettings.enableAxisLines);
				ImGui::MenuItem("Always show 0 tick", "", &pcSettings.enableZeroTick);
				ImGui::DragInt("Tick amount", &pcSettings.axisTickAmount, 0, 1000);
				
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

				//if (ImGui::IsKeyPressed(KEYP) && !ImGui::IsAnyItemActive()) {
				//	if (prioritySelectAttribute) {
				//		prioritySelectAttribute = false;
				//	}
				//	else {
				//		prioritySelectAttribute = true;
				//	}
				//}


				if (ImGui::MenuItem("Set Priority center")) {
					if (ImGui::IsItemHovered()) {
						ImGui::SetTooltip("or press 'P' to set a priority rendering center");
					}

					prioritySelectAttribute = true;
				}

				auto histComp = g_PcPlotDrawLists.begin();
				if (pcSettings.histogrammDrawListComparison != -1) std::advance(histComp, pcSettings.histogrammDrawListComparison);
				if (ImGui::BeginCombo("Histogramm Comparison", (pcSettings.histogrammDrawListComparison == -1) ? "Off" : histComp->name.c_str())) {
					if (ImGui::MenuItem("Off")) {
						pcSettings.histogrammDrawListComparison = -1;
						uploadDensityUiformBuffer();
						if (pcSettings.drawHistogramm) {
							pcPlotRender = true;
						}
					}
					auto it = g_PcPlotDrawLists.begin();
					for (int i = 0; i < g_PcPlotDrawLists.size(); i++, ++it) {
						if (ImGui::MenuItem(it->name.c_str())) {
							pcSettings.histogrammDrawListComparison = i;
							uploadDensityUiformBuffer();
							if (pcSettings.drawHistogramm) {
								pcPlotRender = true;
							}
						}
					}

					ImGui::EndCombo();
				}

				ImGui::MenuItem("Default drawlist on load", "", &pcSettings.createDefaultOnLoad);
				ImGui::SliderInt("Line batch size", &pcSettings.lineBatchSize, 1e5, 1e7);

				ImGui::PopItemWidth();
				ImGui::EndPopup();
			}
			
			if (!anyHover && histogramHovered && ImGui::GetIO().MouseClicked[1]) {
				ImGui::OpenPopup("HistMenu");
			}
			if (ImGui::BeginPopup("HistMenu")) {
				ImGui::PushItemWidth(100);
				if (ImGui::MenuItem("Draw Pie-Ratio", "", &pcSettings.computeRatioPtsInDLvsIn1axbrushedParent))
					updateAllActiveIndices();
				if (ImGui::SliderFloat("Histogram wdith", &pcSettings.histogrammWidth, 0, 2.f / (amtOfLabels)))
					pcPlotRender = true;
				if (ImGui::ColorEdit4("Histogram background", &pcSettings.histogrammBackCol.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar))
					pcPlotRender = true;
				if (ImGui::MenuItem("Show Density", "", &pcSettings.histogrammDensity))
					pcPlotRender = true;
				if (ImGui::ColorEdit4("Density background", &pcSettings.densityBackCol.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar))
					pcPlotRender = true;
				if (ImGui::SliderFloat("Blur radius", &pcSettings.densityRadius, .001f, .5f)) {
					uploadDensityUiformBuffer();
					pcPlotRender = true;
				}
				if (ImGui::MenuItem("Adjust density by line count", "", &pcSettings.adustHistogrammByActiveLines)) {
					uploadDensityUiformBuffer();
					pcPlotRender = true;
				}
				if (ImGui::BeginMenu("Add reference histogram")) {
					for (auto& ds : g_PcPlotDataSets) {
						if (ImGui::MenuItem(ds.name.c_str())) {
							//create new drawlist from the default drawlist, set it immune to global brushes and deaktivate standard rendering
							createPcPlotDrawList(ds.drawLists.front(), ds, ("Reference_" + ds.name).c_str());
							g_PcPlotDrawLists.back().immuneToGlobalBrushes = true;
							g_PcPlotDrawLists.back().show = false;
							pcPlotRender = updateActiveIndices(g_PcPlotDrawLists.back());
						}
					}
					ImGui::EndMenu();
				}

				ImGui::EndPopup();
			}

			//handling priority selection
			if (prioritySelectAttribute) {
				pcPlotSelectedDrawList.clear();
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
					float x = gap * placeOfInd(i) / (amtOfLabels - 1) + picPos.x - BRUSHWIDTH / 2 + ((pcSettings.drawHistogramm) ? (pcSettings.histogrammWidth / 4.0 * picSize.x) : 0);
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
			if (pcPlotRender && violinPlotDrawlistSettings.renderOrderDLConsider && violinPlotDrawlistSettings.renderOrderDLConsiderNonStop && violinPlotDrawlistSettings.enabled) {
				sortAllHistograms(std::string("dl"));
			}
			if (pcPlotRender && violinPlotAttributeSettings.renderOrderAttConsider && violinPlotAttributeSettings.renderOrderAttConsiderNonStop && violinPlotAttributeSettings.enabled) {
				sortAllHistograms(std::string("attr"));
			}



			//Settings section
			ImGui::BeginChild("Settings", ImVec2(500, -1), true);
			ImGui::Text("Settings");

			ImGui::Separator();

			ImGui::Text("Coordinate Settings:");

			static int popupAttribute = -1;
			static char newAttributeName[50]{};
			for (int i : pcAttributesSorted) {
				if (ImGui::Checkbox(pcAttributes[i].name.c_str(), &pcAttributeEnabled[i])) {
					updateAllDrawListIndexBuffer();
					pcPlotRender = true;
				}
				if (ImGui::IsItemClicked(1)){ 
					ImGui::OpenPopup("AttributePopup"); 
					popupAttribute = i;
					strcpy(newAttributeName, pcAttributes[i].originalName.c_str());
				}
				ImGui::SameLine(200);
				if(ImGui::ArrowButton(("##attributeright" + std::to_string(i)).c_str(), ImGuiDir_Left)) {
					//switch left
					if (pcAttributeEnabled[i]) {
						int ownPlace = placeOfInd(i, true);
						int prefInd = ownPlace - 1;
						for (;;) {
							if (prefInd < 0 || pcAttributeEnabled[pcAttrOrd[prefInd]]) break;
							else --prefInd;
						}
						if (prefInd >= 0) {
							switchAttributes(prefInd, ownPlace, io.KeyCtrl);
							updateAllDrawListIndexBuffer();
							pcPlotRender = true;
						}
					}
				}
				if (ImGui::IsItemHovered()) {
					ImGui::BeginTooltip();
					ImGui::Text("Switch with attribute left");
					ImGui::Text("Alternative: drag and drop axis labels)");
					ImGui::EndTooltip();
				}
				ImGui::SameLine();
				if (ImGui::ArrowButton(("##attributeleft" + std::to_string(i)).c_str(), ImGuiDir_Right)) {
					//switch right
					if (pcAttributeEnabled[i]) {
						int ownPlace = placeOfInd(i, true);
						int nextInd = ownPlace + 1;
						for (;;) {
							if (nextInd >= pcAttributes.size() || pcAttributeEnabled[pcAttrOrd[nextInd]]) break;
							else ++nextInd;
						}
						if (nextInd < pcAttributes.size()) {
							switchAttributes(nextInd, ownPlace, io.KeyCtrl);
							updateAllDrawListIndexBuffer();
							pcPlotRender = true;
						}
					}
				}
				if (ImGui::IsItemHovered()) {
					ImGui::BeginTooltip();
					ImGui::Text("Switch with attribute right");
					ImGui::Text("Alternative: drag and drop axis labels)");
					ImGui::EndTooltip();
				}
			}

			if (ImGui::BeginPopup("AttributePopup")) {
				ImGui::SetNextItemWidth(100);
				if (ImGui::InputText("##newName", newAttributeName, 50, ImGuiInputTextFlags_EnterReturnsTrue)) {
					pcAttributes[popupAttribute].name = newAttributeName;
					sortAttributes();
					ImGui::CloseCurrentPopup();
				}
				ImGui::SameLine();
				if (ImGui::MenuItem("Rename")) {
					pcAttributes[popupAttribute].name = newAttributeName;
					sortAttributes();
				}
				if (ImGui::MenuItem("Reset min/max")) {
					std::string resetName = g_PcPlotDrawLists.front().name;
					bool update = loadAttributeSettings(resetName, popupAttribute);
					if (update) {
						updateAllDrawListIndexBuffer();
						pcPlotRender = true;
					}
				}
				if (ImGui::MenuItem("Swap min/max")) {
					std::swap(pcAttributes[popupAttribute].min, pcAttributes[popupAttribute].max);
					pcPlotRender = true;
				}

				ImGui::EndPopup();
			}

			ImGui::Separator();
			bool openLoad = false;
			if (ImGui::CollapsingHeader("Saved Attributes settings")){
				for (SettingsManager::Setting* s : *settingsManager->getSettingsType("AttributeSetting")) {
					if (ImGui::Button(s->id.c_str())) {
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
						updateAllDrawListIndexBuffer();
						pcPlotRender = true;
					}
				}
			}
			if (openLoad) {
				ImGui::OpenPopup("Load error");
			}
			//popup for loading error
			if (ImGui::BeginPopupModal("Load error")) {
				ImGui::Text("Error at loading the current setting");
				if ((ImGui::Button("Close", ImVec2(120, 0))) || ImGui::IsKeyPressed(KEYESC)) {
					ImGui::CloseCurrentPopup();
				}
				ImGui::EndPopup();
			}

			ImGui::Separator();

			if (ImGui::CollapsingHeader("PCP/Hitogram settings")) {
				ImGui::Text("Histogram Settings:");
				ImGui::Columns(2);
				if (ImGui::Checkbox("Draw Histogram", &pcSettings.drawHistogramm)) {
					pcPlotRender = true;
					if (pcSettings.computeRatioPtsInDLvsIn1axbrushedParent)
					{
						pcPlotRender = updateAllActiveIndices();
					}
				}
				ImGui::NextColumn();
				if (ImGui::Checkbox("Draw Pie-Ratio", &pcSettings.computeRatioPtsInDLvsIn1axbrushedParent)) {
					if (pcSettings.drawHistogramm) {
						pcPlotRender = updateAllActiveIndices();
					}
				}


				ImGui::Columns(1);
				if (ImGui::SliderFloat("Histogram Width", &pcSettings.histogrammWidth, 0, .5) && pcSettings.drawHistogramm) {
					if (pcSettings.histogrammDrawListComparison != -1) {
						uploadDensityUiformBuffer();
					}
					pcPlotRender = true;
				}
				if (ImGui::ColorEdit4("Histogram Background", &pcSettings.histogrammBackCol.x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar) && pcSettings.drawHistogramm) {
					pcPlotRender = true;
				}
				if (ImGui::Checkbox("Show Density", &pcSettings.histogrammDensity) && pcSettings.drawHistogramm) {
					pcPlotRender = true;
				}
				if (ImGui::ColorEdit4("Density Background", &pcSettings.densityBackCol.x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar) && pcSettings.drawHistogramm) {
					pcPlotRender = true;
				}
				if (ImGui::SliderFloat("Blur radius", &pcSettings.densityRadius, .001f, .5f)) {
					uploadDensityUiformBuffer();
					pcPlotRender = true;
				}
				if (ImGui::Checkbox("Adjust density by active lines", &pcSettings.adustHistogrammByActiveLines)) {
					pcPlotRender = true;
				}
				ImGui::Separator();

				ImGui::Text("Parallel Coordinates Settings:");

				if (ImGui::Checkbox("Show PcPlot Density", &pcSettings.pcPlotDensity)) {
					pcPlotRender = true;
				}

				if (ImGui::Checkbox("Enable density mapping", &pcSettings.enableDensityMapping)) {
					if (pcAttributes.size()) {
						uploadDensityUiformBuffer();
						pcPlotRender = true;
					}
				}

				if (ImGui::Checkbox("Enable grayscale density", &pcSettings.enableDensityGreyscale)) {
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

				if (ImGui::Checkbox("Enable median calc", &pcSettings.calculateMedians)) {
					for (DrawList& dl : g_PcPlotDrawLists) {
						dl.activeMedian = 0;
					}
				}

				if (ImGui::Checkbox("Enable brushing", &pcSettings.enableBrushing)) {
					pcPlotRender = updateAllActiveIndices();
				}

				if (ImGui::SliderFloat("Median line width", &pcSettings.medianLineWidth, .5f, 20.0f)) {
					pcPlotRender = true;
				}

				if (ImGui::ColorEdit4("Plot Background Color", &pcSettings.PcPlotBackCol.x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar)) {
					pcPlotRender = true;
				}

				if (ImGui::Checkbox("Render Splines", &pcSettings.renderSplines)) {
					updateAllDrawListIndexBuffer();
					pcPlotRender = true;
				}

				if (ImGui::Checkbox("Enable Axis Lines", &pcSettings.enableAxisLines)) {
				}

				if (ImGui::Checkbox("Always show 0 tick", &pcSettings.enableZeroTick)) {}

				if (ImGui::DragInt("Amout of ticks", &pcSettings.axisTickAmount, 1, 0, 100)) {}

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

				if (ImGui::IsKeyPressed(KEYP) && !ImGui::IsAnyItemActive()) {
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

				auto histComp = g_PcPlotDrawLists.begin();
				if (pcSettings.histogrammDrawListComparison != -1) std::advance(histComp, pcSettings.histogrammDrawListComparison);
				if (ImGui::BeginCombo("Histogramm Comparison", (pcSettings.histogrammDrawListComparison == -1) ? "Off" : histComp->name.c_str())) {
					if (ImGui::MenuItem("Off")) {
						pcSettings.histogrammDrawListComparison = -1;
						uploadDensityUiformBuffer();
						if (pcSettings.drawHistogramm) {
							pcPlotRender = true;
						}
					}
					auto it = g_PcPlotDrawLists.begin();
					for (int i = 0; i < g_PcPlotDrawLists.size(); i++, ++it) {
						if (ImGui::MenuItem(it->name.c_str())) {
							pcSettings.histogrammDrawListComparison = i;
							uploadDensityUiformBuffer();
							if (pcSettings.drawHistogramm) {
								pcPlotRender = true;
							}
						}
					}

					ImGui::EndCombo();
				}

				ImGui::Checkbox("Create default drawlist on load", &pcSettings.createDefaultOnLoad);

				ImGui::DragInt("Live brush threshold", &pcSettings.liveBrushThreshold, 1000);

				ImGui::SliderInt("Max line batch size", &pcSettings.lineBatchSize, 30000, 1e7);
			}

			ImGui::EndChild();

			//DataSets, from which draw lists can be created
			ImGui::SameLine();

			ImGui::BeginChild("DataSets", ImVec2((ImGui::GetWindowWidth() - 500) / 2, -1), true, ImGuiWindowFlags_HorizontalScrollbar);

			DataSet* destroySet = NULL;
			bool destroy = false;

			ImGui::Text("Datasets:");
			bool open = ImGui::InputText("Directory Path", pcFilePath, 200, ImGuiInputTextFlags_EnterReturnsTrue);
			if (ImGui::IsItemHovered()) {
				ImGui::BeginTooltip();
				ImGui::Text("Enter either a file including filepath,\nOr a folder (division with /) and all datasets in the folder will be loaded\nOr drag and drop files to load onto application.");
				ImGui::EndTooltip();
			}

			ImGui::SameLine();

			//Opening a new Dataset into the Viewer
			if (ImGui::Button("Open") || open) {
				std::string f = pcFilePath;
				std::string fileExtension = f.substr(f.find_last_of("/\\") + 1);
				size_t pos = fileExtension.find_last_of(".");
				if (pos != std::string::npos) {		//entered discrete file
					bool success = openDataset(pcFilePath);
					if (success && pcSettings.createDefaultOnLoad) {
						//pcPlotRender = true;
						createPcPlotDrawList(g_PcPlotDataSets.back().drawLists.front(), g_PcPlotDataSets.back(), g_PcPlotDataSets.back().name.c_str());
						pcPlotRender = updateActiveIndices(g_PcPlotDrawLists.back());
					}
				}
				else {					//entered folder -> open open dataset dialogue
					for (const auto& entry : std::filesystem::directory_iterator(f)) {
						if (entry.is_regular_file()) {	//only process normal enties
							fileExtension = entry.path().u8string().substr(entry.path().u8string().find_last_of("."));
							if (std::find(supportedDataFormats.begin(), supportedDataFormats.end(), fileExtension) == supportedDataFormats.end()) continue;	//ignore unsupported file formats
							droppedPaths.emplace_back(entry.path().u8string());
							droppedPathActive.emplace_back(1);
							pathDropped = true;
							f = entry.path().u8string();
							queryAttributes = queryFileAttributes((entry.path().u8string()).c_str());
						}
					}
				}
			}
			ImGui::Separator();
			for (DataSet& ds : g_PcPlotDataSets) {
				if (ImGui::TreeNode(ds.name.c_str())) {
					static TemplateList* convert = nullptr;
					int c = 0;		//counter to reduce the amount of template lists being drawn
					for (TemplateList& tl : ds.drawLists) {
						static int subsample = 1, trim[2];
						if (c++ > 10000)break;
						if (ImGui::Button(tl.name.c_str())) {
							ImGui::OpenPopup(tl.name.c_str());
							trim[0] = 0;
							trim[1] = tl.indices.size();
							strcpy(pcDrawListName, tl.name.c_str());
						}
						if (ImGui::IsItemClicked(1)) {
							ImGui::OpenPopup("CONVERTTOBRUSH");
							convert = &tl;
						}
						if (ImGui::BeginPopupModal(tl.name.c_str(), NULL, ImGuiWindowFlags_AlwaysAutoResize))
						{
							int destination = 0;
							if(ImGui::BeginTabBar("Destination")){
								if(ImGui::BeginTabItem("Drawlist")){
									destination = 0;
									ImGui::Text("%s", (std::string("Creating a DRAWLIST list from ") + tl.name).c_str());
									ImGui::EndTabItem();
								}
								if(ImGui::BeginTabItem("TemplateList")){
									destination = 1;
									ImGui::Text("%s", (std::string("Creating a TEMPLATELIST from ") + tl.name).c_str());
									ImGui::EndTabItem();
								}
								ImGui::EndTabBar();
							}
							ImGui::Separator();
							ImGui::InputText("Drawlist Name", pcDrawListName, 200);
							if(ImGui::CollapsingHeader("Subsample/Trim")){
								if(ImGui::InputInt("Subsampling Rate", &subsample)) subsample = std::max(subsample, 1);
								if(ImGui::InputInt2("Trim indcies",trim)){
									trim[0] = std::clamp(trim[0], 0, trim[1] - 1);
									trim[1] = std::clamp(trim[1], trim[0] + 1, int(tl.indices.size()));
								}
							}

							if ((ImGui::Button("Create", ImVec2(120, 0))) || ImGui::IsKeyPressed(KEYENTER))
							{
								ImGui::CloseCurrentPopup();
								if(destination == 0){
									auto tmp = tl.indices;
									tl.indices.resize(int(ceilf(1.f * (trim[1] - trim[0]) / subsample)));
									int ind = 0;
									for(int i = trim[0]; i < trim[1]; i += subsample) tl.indices[ind++] = tmp[i];
									createPcPlotDrawList(tl, ds, pcDrawListName);
									tl.indices = tmp;
									pcPlotRender = updateActiveIndices(g_PcPlotDrawLists.back());
								}
								else{
									auto found = std::find_if(ds.drawLists.begin(), ds.drawLists.end(), [&](TemplateList& tl){return tl.name == pcDrawListName;});
									if(found == ds.drawLists.end()){
										ds.drawLists.push_back(tl);
										ds.drawLists.back().name = pcDrawListName;
										ds.drawLists.back().indices.resize(int(ceilf(1.f * (trim[1] - trim[0]) / subsample)));
										int ind = 0;
										for(int i = trim[0]; i < trim[1]; i += subsample) ds.drawLists.back().indices[ind++] = tl.indices[i];
									}
									else{
										if(debugLevel >= 1)
											std::cout << "A template list with the same name is already existing! No template list is produced." << std::endl;
									}
								}
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
					bool exportIdxf = false;
					bool exportCsv = false;
					if (ImGui::BeginPopup("CONVERTTOBRUSH")) {
						if (ImGui::MenuItem("Convert to global brush")) {
							convertToGlobalBrush = true;
							ImGui::CloseCurrentPopup();
						}
						if (ImGui::MenuItem("Convert to lokal brush")) {
							convertToLokalBrush = true;
							ImGui::CloseCurrentPopup();
						}
						if(ImGui::MenuItem("Safe as .Idxf")){
							exportIdxf = true;
							ImGui::CloseCurrentPopup();
						}
						if(ImGui::MenuItem("Safe as .csv")){
							exportCsv = true;
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
							if(std::string(n).empty()){
								if(debugLevel >= 1)
									std::cout << "The name for the new Brush musn't be empyt!" << std::endl;
							}
							else{
								GlobalBrush brush = {};
								brush.nam = std::string(n);
								brush.id = std::string(n);
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
							}
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

					if(exportIdxf){
						ImGui::OpenPopup("ExportIdxf");
					}
					if(ImGui::BeginPopupModal("ExportIdxf")){
						static char exportFilename[300];
						ImGui::InputText("Filename (has to include .idxf)", exportFilename, 300);
						if(ImGui::Button("Export") || ImGui::IsKeyPressedMap(ImGuiKey_Enter)){
							exportTemplateListAsIdxf(*convert, exportFilename);
							ImGui::CloseCurrentPopup();
						}
						ImGui::SameLine();
						if(ImGui::Button("Cancel") || ImGui::IsKeyPressedMap(ImGuiKey_Escape)){
							ImGui::CloseCurrentPopup();
						}
						ImGui::EndPopup();
					}

					if(exportCsv){
						ImGui::OpenPopup("ExportCsv");
					}
					if(ImGui::BeginPopupModal("ExportCsv")){
						static char exportFilename[300];
						ImGui::InputText("Filename (has to include .csv)", exportFilename, 300);
						if(ImGui::Button("Export") || ImGui::IsKeyPressedMap(ImGuiKey_Enter)){
							exportTemplateListAsCsv(*convert, exportFilename);
							ImGui::CloseCurrentPopup();
						}
						ImGui::SameLine();
						if(ImGui::Button("Cancel") || ImGui::IsKeyPressedMap(ImGuiKey_Escape)){
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
							ImGui::SliderFloat("Default Alpha Value", &pcSettings.alphaDrawLists, .0f, 1.0f);
						}

						for (int i = 0; i < droppedPaths.size(); i++) {
							ImGui::Text("%s", droppedPaths[i].c_str());
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
                    
                    //Popup for member split
                    if(ImGui::Button("Split dataset")){
                        ImGui::OpenPopup("SPLITDATASET");
                    }
                    if(ImGui::BeginPopupModal("SPLITDATASET", NULL, ImGuiWindowFlags_AlwaysAutoResize)){
						static int destination = 0; //0 corresponds to draw list as target, 1 is template lsits as target
						if(ImGui::BeginTabBar("Destination")){
							if(ImGui::BeginTabItem("Drawlist")){
								destination = 0;
								ImGui::Text("The resulting splits are delivered as Drawlists(instantly drawn)");
								ImGui::EndTabItem();
							}
							if(ImGui::BeginTabItem("Templatelist")){
								destination = 1;
								ImGui::Text("The resulting splits are delivered as Templatelists(Are only list of indices)");
								ImGui::EndTabItem();
							}
							ImGui::EndTabBar();
						}
                        static int selectedAtt = 0;
                        static int amtOfGroups = 100;
						static int splitType = 0;
						static std::vector<float> quantiles{0, 1.0f};
						static std::vector<float> values{pcAttributes[selectedAtt].min, pcAttributes[selectedAtt].max};
						if(ImGui::BeginTabBar("SplitTab")){
							if(ImGui::BeginTabItem("Uniform Value Split")){
								splitType = 0;
                        		if(ImGui::BeginCombo("Split axis", pcAttributes[selectedAtt].name.c_str())){
                        		    for(int att = 0; att < pcAttributes.size(); ++att){
                        		        if(ImGui::MenuItem(pcAttributes[att].name.c_str())) selectedAtt = att;
                        		    }
                        		    ImGui::EndCombo();
                        		}
                        		ImGui::InputInt("Amount of split groups", &amtOfGroups);
								ImGui::EndTabItem();
							}
							if(ImGui::BeginTabItem("Value Split")){
								splitType = 3;
								int addItem = -1;
								int deleteItem = -1;
								if(ImGui::BeginCombo("Split axis", pcAttributes[selectedAtt].name.c_str())){
                        		    for(int att = 0; att < pcAttributes.size(); ++att){
                        		        if(ImGui::MenuItem(pcAttributes[att].name.c_str())){
											selectedAtt = att;
											values.front() = pcAttributes[att].min;
											values.back() = pcAttributes[att].max;
										}
                        		    }
                        		    ImGui::EndCombo();
                        		}
								ImGui::Text("Split values:");
								for(int i = 0; i < values.size(); ++i){
									float min = pcAttributes[selectedAtt].min, max = pcAttributes[selectedAtt].max, speed = .01f;
									if(i == 0) speed = 0.0000000001;
									else if(i == values.size() - 1) speed = 0.000000001;
									else {min = values[i - 1], max = values[i + 1]; speed = (max - min) / 500;}
									ImGui::DragFloat(("##quantile" + std::to_string(i)).c_str(), values.data() + i, speed, min, max);
									if(i != 0 && i != values.size()-1){
										ImGui::SameLine();
										if(ImGui::Button(("X##deleteQuant" + std::to_string(i)).c_str())) deleteItem = i;
									}
									if(i < values.size() - 1){
										static float buttonHeight = 10;
										static float space = 5;
										float prevCursorPosY = ImGui::GetCursorPosY();
										ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetTextLineHeightWithSpacing() / 2.0f + space);
										if(ImGui::Button(("##addButton" + std::to_string(i)).c_str(), ImVec2(250,buttonHeight))){
											addItem = i;
										}
										ImGui::SetCursorPosY(prevCursorPosY + space);
									}
								}
								if(addItem >= 0) values.insert(values.begin() + addItem + 1, (values[addItem] + values[addItem + 1]) / 2.0f);
								if(deleteItem >= 0) values.erase(values.begin() + deleteItem);
							
								ImGui::EndTabItem();
							}
							if(ImGui::BeginTabItem("Quantiles Split")){
								splitType = 1;
								int addItem = -1;
								int deleteItem = -1;
								if(ImGui::BeginCombo("Split axis", pcAttributes[selectedAtt].name.c_str())){
                        		    for(int att = 0; att < pcAttributes.size(); ++att){
                        		        if(ImGui::MenuItem(pcAttributes[att].name.c_str())) selectedAtt = att;
                        		    }
                        		    ImGui::EndCombo();
                        		}
								ImGui::Text("Split quantiles:");
								for(int i = 0; i < quantiles.size(); ++i){
									float min = 0, max = 1, speed = .01f;
									if(i == 0) speed = 0.0000000001;
									else if(i == quantiles.size() - 1) speed = 0.000000001;
									else {min = quantiles[i - 1], max = quantiles[i + 1];}
									ImGui::DragFloat(("##quantile" + std::to_string(i)).c_str(), quantiles.data() + i, speed, min, max);
									if(i != 0 && i != quantiles.size()-1){
										ImGui::SameLine();
										if(ImGui::Button(("X##deleteQuant" + std::to_string(i)).c_str())) deleteItem = i;
									}
									if(i < quantiles.size() - 1){
										static float buttonHeight = 10;
										static float space = 5;
										float prevCursorPosY = ImGui::GetCursorPosY();
										ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetTextLineHeightWithSpacing() / 2.0f + space);
										if(ImGui::Button(("##addButton" + std::to_string(i)).c_str(), ImVec2(250,buttonHeight))){
											addItem = i;
										}
										ImGui::SetCursorPosY(prevCursorPosY + space);
									}
								}
								if(addItem >= 0) quantiles.insert(quantiles.begin() + addItem + 1, (quantiles[addItem] + quantiles[addItem + 1]) / 2.0f);
								if(deleteItem >= 0) quantiles.erase(quantiles.begin() + deleteItem);

								ImGui::EndTabItem();
							}
							if(ImGui::BeginTabItem("Automatic Split")){
								splitType = 2;
								ImGui::Text("Only select variables with discrete values.\n If too much values wrt data size are found, no split will be performed.");
								if(ImGui::BeginCombo("Split axis", pcAttributes[selectedAtt].name.c_str())){
                        		    for(int att = 0; att < pcAttributes.size(); ++att){
                        		        if(ImGui::MenuItem(pcAttributes[att].name.c_str())) selectedAtt = att;
                        		    }
                        		    ImGui::EndCombo();
                        		}
								ImGui::EndTabItem();
							}

							ImGui::EndTabBar();
						}
                        if(ImGui::Button("Split")){
							std::vector<std::vector<uint32_t>> indices(amtOfGroups);
                            std::vector<uint32_t> sta = ds.drawLists.front().indices;
							switch(splitType){
							case 0: //linear block split
                            
                            	//assigning each datum to one of the groups
								{
                            	float attMin = pcAttributes[selectedAtt].min, attMax = pcAttributes[selectedAtt].max;
                            	float attDiff = attMax - attMin;
                            	for(int datum = 0; datum < ds.data.size(); ++datum){
                            	    float da = ds.data(datum,selectedAtt);
                            	    int index = int(((da - attMin) / attDiff) * (amtOfGroups - 1) + .5f);
                            	    indices[index].push_back(datum);
                            	}
	
                            	//safe standard indexlist of default list
                            	for(int group = 0; group < amtOfGroups; ++group){
                            	    std::string t_name = ds.name + "_" + std::to_string(group);
                            	    ds.drawLists.front().indices = indices[group];
									if(destination == 0){
                            	    	createPcPlotDrawList(ds.drawLists.front(), ds, t_name.c_str());
										updateActiveIndices(g_PcPlotDrawLists.back());
									}
									else{
										ds.drawLists.push_back(ds.drawLists.front());
										ds.drawLists.back().name = t_name;
									}
                            	}
								}
                            	
								break;
							case 1:		//quantile split
								//ordering the indices in a copy fo the default drawlist
								{
								auto ordered = sta;
								std::sort(ordered.begin(), ordered.end(), [&](uint32_t left, uint32_t right){return ds.data(left, selectedAtt) < ds.data(right, selectedAtt);});
								quantiles.front() = 0; quantiles.back() = 1;
								for(int i = 0; i < quantiles.size() - 1; ++i){
									std::vector<uint32_t> quant(ordered.begin() + ordered.size() * quantiles[i], ordered.begin() + ordered.size() * quantiles[i + 1]);
									if(quant.emplace_back()) continue; //ignore empty quantiles
									ds.drawLists.front().indices = quant;
									std::string t_name = ds.name + "_" + std::to_string(i);
									if(destination == 0){
										createPcPlotDrawList(ds.drawLists.front(), ds, t_name.c_str());
										updateActiveIndices(g_PcPlotDrawLists.back());
									}
									else{
										ds.drawLists.push_back(ds.drawLists.front());
										ds.drawLists.back().name = t_name;
									}
								}
								}
								break;
							case 2:		//automatic split behaviour
								{
									const auto& dimensionValues =  getDimensionValues(ds, selectedAtt).second;
									if(dimensionValues.size() * 10 < ds.data.size()){	//the dimension values size has to be 10 times smaller than the whole data size to count for an axis with group values
										indices.resize(dimensionValues.size());
										for(uint32_t d = 0; d < ds.data.size(); ++d){
											float data = ds.data(d, selectedAtt);
											indices[std::lower_bound(dimensionValues.begin(), dimensionValues.end(), data) - dimensionValues.begin()].push_back(d);
										}
										std::remove_if(indices.begin(), indices.end(), [&](std::vector<uint32_t>& v){return v.empty();}); //ignore empty groups
										for(int group = 0; group < indices.size(); ++group){
                            	    		std::string t_name = ds.name + "_" + std::to_string(group);
                            	    		ds.drawLists.front().indices = indices[group];
											if(destination == 0){
                            	    			createPcPlotDrawList(ds.drawLists.front(), ds, t_name.c_str());
												updateActiveIndices(g_PcPlotDrawLists.back());
											}
											else{
												ds.drawLists.push_back(ds.drawLists.front());
												ds.drawLists.back().name = t_name;
											}
                            			}
									}
									else{
										std::cout << "The selected attribute seems to not bundle the data, splitting is aborted" << std::endl;
									}
								}
								break;
							case 3:
								{	//value split
									quantiles.front() = pcAttributes[selectedAtt].min, quantiles.back() = pcAttributes[selectedAtt].max;
									indices.resize(values.size());
									for(uint32_t d = 0; d < ds.data.size(); ++d){
										float data = ds.data(d, selectedAtt);
										indices[std::upper_bound(values.begin(), values.end(), data) - values.begin()].push_back(d);
									}
									indices.erase(std::remove_if(indices.begin(), indices.end(), [&](std::vector<uint32_t>& v){return v.empty();}), indices.end()); //ignore empty groups
									for(int group = 0; group < indices.size(); ++group){
                            	    	std::string t_name = ds.name + "_" + std::to_string(group);
                            	    	ds.drawLists.front().indices = indices[group];
										if(destination == 0){
                            	    		createPcPlotDrawList(ds.drawLists.front(), ds, t_name.c_str());
											updateActiveIndices(g_PcPlotDrawLists.back());
										}
										else{
											ds.drawLists.push_back(ds.drawLists.front());
											ds.drawLists.back().name = t_name;
										}
                            		}
								}
								break;
							}
							ds.drawLists.front().indices = sta;
							if(destination == 0)
								pcPlotRender = true;
                            
                            ImGui::CloseCurrentPopup();
                        }
                        ImGui::SameLine();
                        if(ImGui::Button("Cancel")){
                            ImGui::CloseCurrentPopup();
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
			ImGui::SameLine();
			static float uniform_alpha = .5f;
			if (ImGui::SliderFloat("Set uniform alpha", &uniform_alpha, 0.003f, 1.0f)) {	//when changed set alpha for each dl
				for (DrawList& dl : g_PcPlotDrawLists) {
					dl.color.w = uniform_alpha;
				}
				pcPlotRender = true;
			}
			ImGui::SameLine();
			if(ImGui::Button("Edit drawlist color palette")){
				drawListColorPalette->openColorPaletteEditor();
			}
			drawListColorPalette->drawColorPaletteEditor();
			int count = 0;

			ImGui::Columns(9, "Columns", true);
			if (pcSettings.rescaleTableColumns) {
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
			ImGui::Separator();

			//showing texts to describe whats in the corresponding column
			ImGui::Text("Drawlist Name");
			ImGui::NextColumn();
			ImGui::Text("Draw");
			ImGui::NextColumn();
			ImGui::Text(" ");
			ImGui::NextColumn();
			ImGui::Text(" ");
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
				bool contained = pcPlotSelectedDrawList.end() != std::find(pcPlotSelectedDrawList.begin(), pcPlotSelectedDrawList.end(), count);
				if (ImGui::Selectable(dl.name.c_str(), contained)) {
					selectedGlobalBrush = -1;
					if (contained && io.KeyCtrl) {
						int place = 0;
						for (; place < pcPlotSelectedDrawList.size(); ++place) if (count == pcPlotSelectedDrawList[place]) break;
						pcPlotSelectedDrawList[place] = pcPlotSelectedDrawList.size() > 1 ? pcPlotSelectedDrawList[pcPlotSelectedDrawList.size() - 1] : 0;
						pcPlotSelectedDrawList.pop_back();
					}
					else if (contained)
						pcPlotSelectedDrawList.clear();
					else if (io.KeyShift) {
						if (pcPlotSelectedDrawList.back() < count)
							for (int addition = pcPlotSelectedDrawList.back() + 1; addition <= count; ++addition) pcPlotSelectedDrawList.push_back(addition);
						else
							for (int addition = pcPlotSelectedDrawList.back() - 1; addition >= count; --addition) pcPlotSelectedDrawList.push_back(addition);
					}
					else if (io.KeyCtrl)
						pcPlotSelectedDrawList.push_back(count);
					else {
						pcPlotSelectedDrawList.clear();
						pcPlotSelectedDrawList.push_back(count);
					}
				}
				if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
					DrawList* point = &dl;
					if (!contained) pcPlotSelectedDrawList.push_back(count);
					ImGui::SetDragDropPayload("Drawlist", &point, sizeof(DrawList*));
					auto cur_draw = g_PcPlotDrawLists.begin();
					for (int cur_d = 0; cur_d < g_PcPlotDrawLists.size(); ++cur_d, ++cur_draw) {
						if (pcPlotSelectedDrawList.end() != std::find(pcPlotSelectedDrawList.begin(), pcPlotSelectedDrawList.end(), cur_d)) {
							ImGui::Text("%s", cur_draw->name.c_str());
						}
					}
					ImGui::EndDragDropSource();
				}
				if (ImGui::IsItemHovered() && io.MouseClicked[1]) {
					ImGui::OpenPopup(("drawListMenu" + dl.name).c_str());
				}
				bool openClusterSelection = false;
				bool openDrawlistInfos = false;
				if (ImGui::BeginPopup(("drawListMenu" + dl.name).c_str())) {
					if(ImGui::MenuItem("Drawlist statistics")){
						openDrawlistInfos = true;
					}
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
					//if (ImGui::MenuItem("Send to Bubble plotter")) {
					//	DataSet* parent;
					//	for (auto it = g_PcPlotDataSets.begin(); it != g_PcPlotDataSets.end(); ++it) {
					//		if (it->name == dl.parentDataSet) {
					//			parent = &(*it);
					//		}
					//	}
					//	std::vector<uint32_t> ids;
					//	std::vector<std::string> attributeNames;
					//	std::vector<std::pair<float, float>> attributeMinMax;
					//	for (int i = 0; i < pcAttributes.size(); ++i) {
					//		attributeNames.push_back(pcAttributes[i].name);
					//		attributeMinMax.push_back({ pcAttributes[i].min,pcAttributes[i].max });
					//	}
					//	bubblePlotter->setBubbleData(dl.indices, attributeNames, attributeMinMax, parent->data, dl.buffer, dl.activeIndicesBufferView, attributeNames.size(), parent->data.size());

					//	//Debugging of histogramms
					//	//histogramManager->setNumberOfBins(100);
					//	//histogramManager->computeHistogramm(dl.name, dl.activeInd, attributeMinMax, parent->buffer.buffer, parent->data.size());
					//	//for (auto& i : histogramManager->getHistogram(dl.name).bins)
					//	//{
					//	//	std::for_each(i.begin(), i.end(), [](uint32_t a) {std::cout << a << ","; });
					//	//	std::cout << std::endl;
					//	//}
					//}
					//ImGui::Separator();
					static float bundlesAmt = 1;
					static bool changed = true;
					changed |= ImGui::SliderFloat("Bundles amount", &bundlesAmt, .01f, 10.0f);
					if(ImGui::MenuItem("Render as Bands", "", &dl.renderBundles)){
						pcPlotRender = true;
						if(dl.renderBundles && (changed || !dl.lineBundles)){
							if(dl.lineBundles) delete dl.lineBundles;
							dl.color.w = .1f;
							auto parent = std::find(g_PcPlotDataSets.begin(), g_PcPlotDataSets.end(), DataSet{dl.parentDataSet});
							assert(parent!=g_PcPlotDataSets.end());
							{
								std::vector<std::pair<float,float>> histMinmax(pcAttributes.size());
								for(int i = 0; i < pcAttributes.size(); ++i) histMinmax[i] = {pcAttributes[i].min, pcAttributes[i].max};
								exeComputeHistogram(dl.name, histMinmax, dl.buffer, parent->data.size(), dl.indicesBuffer, dl.indices.size(), dl.activeIndicesBufferView);
							}
							float prevStdDev = 0;
							if(changed){
								prevStdDev = histogramManager->stdDev;
								histogramManager->setSmoothingKernelSize(10.0 / bundlesAmt);
							}

							VkUtil::Context vkContext{g_PcPlotWidth, g_PcPlotHeight, g_PhysicalDevice, g_Device, g_DescriptorPool, g_PcPlotCommandPool, g_Queue};
							std::vector<std::pair<std::string, std::pair<float, float>>> attributes(pcAttributes.size());
							std::vector<std::pair<uint32_t, bool>> attributeOrder(pcAttributes.size());
							for(int i = 0; i < pcAttributes.size(); ++i) {
								attributes[i] = {pcAttributes[i].name, {pcAttributes[i].min, pcAttributes[i].max}};
								attributeOrder[i] = {pcAttrOrd[i], pcAttributeEnabled[pcAttrOrd[i]]};
							}
							dl.lineBundles = new LineBundles(vkContext, g_PcPlotRenderPass_noClear, g_PcPlotFramebuffer_noClear, dl.name, &parent->data, histogramManager, attributes, attributeOrder, &dl.color.x);
							
							if(changed){
								changed = false;
								histogramManager->setSmoothingKernelSize(prevStdDev);
							}
						}
					}
					if(ImGui::MenuItem("Render cluster bands", "", &dl.renderClusterBundles)){
						openClusterSelection = true;
						pcPlotRender = true;
					}
					dl.renderClusterBundles &= bool(dl.clusterBundles);
					if(ImGui::SliderFloat("Halo size", &pcSettings.haloWidth, 0, 1.01f)){
						pcPlotRender = true;
					}

					ImGui::Separator();
					static char templateListName[200];
					ImGui::InputText("##templateListName", templateListName, 200);
					ImGui::SameLine();
					if(ImGui::MenuItem("Convert to Template List")){
						std::string listName(templateListName);
						if(listName.empty()){
							if(debugLevel >= 1)
								std::cout << "No name entered!" << std::endl;
						}
						else{
							auto ds = std::find_if(g_PcPlotDataSets.begin(), g_PcPlotDataSets.end(), [&] (DataSet& s){return s.name == dl.parentDataSet;});
							auto tl = std::find_if(ds->drawLists.begin(), ds->drawLists.end(), [&](TemplateList& t){return t.name == listName;});
							if(tl != ds->drawLists.end()){
								if(debugLevel >= 1)
									std::cout << "Template list name already exists for the derived Data Set!" << std::endl;
							}
							else{
								//now converting to template list
								ds->drawLists.push_back({});
								auto& curTl = ds->drawLists.back();
								curTl.name = listName;
								curTl.buffer = ds->buffer.buffer;
								curTl.parentDataSetName = ds->name;
								curTl.pointRatio = 1;
								curTl.minMax = std::vector<std::pair<float, float>>(pcAttributes.size(), {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()});
								std::vector<uint8_t> activeIndices(ds->data.size());
								VkUtil::downloadData(g_Device, dl.dlMem, dl.activeIndicesBufferOffset, activeIndices.size(), activeIndices.data());
								for(uint32_t i = 0; i < activeIndices.size(); ++i){
									if(activeIndices[i]){
										curTl.indices.push_back(i);
										for(int a = 0; a < pcAttributes.size(); ++a){
											float curVal = ds->data(i, a);
											if(curVal < curTl.minMax[a].first){
												curTl.minMax[a].first = curVal;
											}
											if(curVal > curTl.minMax[a].second){
												curTl.minMax[a].second = curVal;
											}
										}
									}
								}
								if(debugLevel >= 3){
									std::cout << "Drawlist to Templatelist conversion done" << std::endl;
								}
							}
						}
					}

					ImGui::EndPopup();
				}
				if(openClusterSelection)
					ImGui::OpenPopup("ClusterSelection");
				if(ImGui::BeginPopupModal("ClusterSelection")){
					auto parent = std::find(g_PcPlotDataSets.begin(), g_PcPlotDataSets.end(), DataSet{dl.parentDataSet});
					assert(parent!=g_PcPlotDataSets.end());
					static std::vector<uint8_t> selectedTl;
					selectedTl.resize(parent->drawLists.size(), 1);
					static char filterText[200];
					if(ImGui::InputText("TemplateList filter(Upper/Lowercase matters)", filterText, 200)){
						int c = 0;
						for(auto& tl: parent->drawLists){
							if(tl.name.find(filterText)==tl.name.npos){
								selectedTl[c] = 0;		//setting not filtered lists to false
							}
							++c;
						}
					}
					ImGui::BeginChild("TemplateLists",{400, 200}, true);
					int c = 0;
					for(auto& tl: parent->drawLists){
						if(tl.name.find(filterText)!=tl.name.npos){
							ImGui::Checkbox(tl.name.c_str(), (bool*)&selectedTl[c]);
						}
						++c;
					}
					ImGui::EndChild();
					if(ImGui::Button("Set cluster bundles")){
						if(dl.clusterBundles) delete dl.clusterBundles;
						VkUtil::Context vkContext{g_PcPlotWidth, g_PcPlotHeight, g_PhysicalDevice, g_Device, g_DescriptorPool, g_PcPlotCommandPool, g_Queue};
						std::vector<std::pair<std::string, std::pair<float, float>>> attributes(pcAttributes.size());
						std::vector<std::pair<uint32_t, bool>> attributeOrder(pcAttributes.size());
						for(int i = 0; i < pcAttributes.size(); ++i) {
							attributes[i] = {pcAttributes[i].name, {pcAttributes[i].min, pcAttributes[i].max}};
							attributeOrder[i] = {pcAttrOrd[i], pcAttributeEnabled[pcAttrOrd[i]]};
						}
						std::vector<TemplateList*> templateLists;
						int c = 0;
						for(auto i = parent->drawLists.begin(); i != parent->drawLists.end(); ++i, ++c){
							if(selectedTl[c]){
								templateLists.push_back(&(*i));
							}
						}

						dl.clusterBundles = new ClusterBundles(vkContext, g_PcPlotRenderPass_noClear, g_PcPlotFramebuffer_noClear, dl.name, &parent->data, attributes, attributeOrder, &dl.color.x, templateLists);
						dl.renderClusterBundles = true;
						pcPlotRender = true;
						ImGui::CloseCurrentPopup();
					}
					ImGui::SameLine();
					if(ImGui::Button("Cancel")){
						ImGui::CloseCurrentPopup();
					};
					ImGui::EndPopup();
				}
				static int currentActiveLines = 0;
				if(openDrawlistInfos){
					ImGui::OpenPopup("DrawListStatistics");
					std::vector<uint8_t> activeIndices(std::find_if(g_PcPlotDataSets.begin(), g_PcPlotDataSets.end(), [&](DataSet& ds){return ds.name == dl.parentDataSet;})->data.size());
					VkUtil::downloadData(g_Device, dl.dlMem, dl.activeIndicesBufferOffset, activeIndices.size(), activeIndices.data());
					currentActiveLines = 0;
					for(uint8_t a: activeIndices) if(a) ++currentActiveLines;
				}
				if(ImGui::BeginPopupModal("DrawListStatistics")){
					ImGui::Text("Parent dataset: %s", dl.parentDataSet.c_str());
					ImGui::Text("Parent Templatelist: %s", dl.parentTemplateList->name.c_str());
					ImGui::Text("Max amount of lines: %d", int(dl.indices.size()));
					ImGui::Text("Current active Lines: %d", currentActiveLines);
					if(ImGui::Button("Close")) ImGui::CloseCurrentPopup();
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
					pcPlotSelectedDrawList.clear();
					changeList = &dl;
					destroy = true;
					pcPlotRender = true;
				}
				ImGui::NextColumn();

				int misc_flags = ImGuiColorEditFlags_AlphaBar;
				if (ImGui::ColorEdit4((std::string("Color##") + dl.name).c_str(), (float*)&dl.color, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | misc_flags)) {
					pcPlotRender = true;
				}
				ImGui::NextColumn();

				if (ImGui::Checkbox((std::string("##dh") + dl.name).c_str(), &dl.showHistogramm) && pcSettings.drawHistogramm) {
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
				ImGui::Text("Contains %8d points.", static_cast<int>(drawListComparator.aInd.size()));
				ImGui::Text("The intersection has %d points.", static_cast<int>(drawListComparator.aAndb.size()));
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
				ImGui::Text("Union has %8d points", static_cast<int>(drawListComparator.aOrB.size()));
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
				ImGui::Text("Difference has %8d points", static_cast<int>(drawListComparator.aMinusB.size()));
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
				ImGui::Text("Contains %8d points.", static_cast<int>(drawListComparator.bInd.size()));
				ImGui::Text("The intersection has %d points.", static_cast<int>(drawListComparator.aAndb.size()));
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
				ImGui::Text("Union has %8d points", static_cast<int>(drawListComparator.aOrB.size()));
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
				ImGui::Text("Difference has %8d points", static_cast<int>(drawListComparator.bMinusA.size()));
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
				pcSettings.updateBrushTemplates = true;
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

		if (animationItemsDisabled) {
			ImGui::PopItemFlag();
			animationItemsDisabled = false;
		}

		ImGui::End();

		//bubble window ----------------------------------------------------------------------------------
		int bubbleWindowSize = 0;
		if (bubbleWindowSettings.enabled) {
			ImGui::Begin("Bubble window", &bubbleWindowSettings.enabled, ImGuiWindowFlags_MenuBar);

			bubbleWindowSize = ImGui::GetWindowSize().y;

			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("Save Settings")) {
					addSaveSettingsMenu<BubbleWindowSettings>(&bubbleWindowSettings, "BubbleSettings", "bubblesettings");
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Coupling")) {
					ImGui::MenuItem("Couple to Parallel Coordinates", "", &bubbleWindowSettings.coupleToBrushing);
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Navigation")) {
					ImGui::SliderFloat("fly speed", &bubblePlotter->flySpeed, 0.01, 10);
					ImGui::SliderFloat("fast fly multiplier", &bubblePlotter->fastFlyMultiplier, 1, 10);
					ImGui::SliderFloat("rotation speed", &bubblePlotter->rotationSpeed, 0.01, 5);
					ImGui::SliderFloat("fov speed", &bubblePlotter->fovSpeed, 1, 100);
					if (ImGui::SliderFloat("far clip", &bubblePlotter->farClip, 10, 10000)) bubblePlotter->render();
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
					if (pcAttributes.size()) {
						ImGui::PushItemWidth(ImGui::CalcItemWidth() / 3);
						if (ImGui::BeginCombo("X", pcAttributes[bubblePlotter->posIndices.x].name.c_str())) {
							for (int i = 0; i < pcAttributes.size(); ++i) {
								if (ImGui::MenuItem(pcAttributes[i].name.c_str())) {
									bubblePlotter->attributeActivations[bubbleWindowSettings.posIndices.x] = true;
									bubblePlotter->attributeActivations[i] = false;
									bubblePlotter->posIndices.x = i;
									bubblePlotter->boundingRectMin.x = pcAttributes[i].min;
									bubblePlotter->boundingRectMax.x = pcAttributes[i].max;
									bubblePlotter->updateRenderOrder();
									bubblePlotter->render();
								}
							}
							ImGui::EndCombo();
						}
						ImGui::SameLine();
						if (ImGui::BeginCombo("Y", pcAttributes[bubblePlotter->posIndices.y].name.c_str())) {
							for (int i = 0; i < pcAttributes.size(); ++i) {
								if (ImGui::MenuItem(pcAttributes[i].name.c_str())) {
									bubblePlotter->attributeActivations[bubbleWindowSettings.posIndices.y] = true;
									bubblePlotter->attributeActivations[i] = false;
									bubblePlotter->posIndices.y = i;
									bubblePlotter->boundingRectMin.y = pcAttributes[i].min;
									bubblePlotter->boundingRectMax.y = pcAttributes[i].max;
									bubblePlotter->updateRenderOrder();
									bubblePlotter->render();
								}
							}
							ImGui::EndCombo();
						}
						ImGui::SameLine();
						if (ImGui::BeginCombo("Z", pcAttributes[bubblePlotter->posIndices.z].name.c_str())) {
							for (int i = 0; i < pcAttributes.size(); ++i) {
								if (ImGui::MenuItem(pcAttributes[i].name.c_str())) {
									bubblePlotter->attributeActivations[bubbleWindowSettings.posIndices.z] = true;
									bubblePlotter->attributeActivations[i] = false;
									bubblePlotter->posIndices.z = i;
									bubblePlotter->boundingRectMin.z = pcAttributes[i].min;
									bubblePlotter->boundingRectMax.z = pcAttributes[i].max;
									bubblePlotter->updateRenderOrder();
									bubblePlotter->render();
								}
							}
							ImGui::EndCombo();
						}
						ImGui::PopItemWidth();
					}
					//if (ImGui::DragInt3("Position indices", (int*)&bubblePlotter->posIndices.x, .05f, 0, pcAttributes.size())) {
					//	bubblePlotter->render();
					//}
					ImGui::EndMenu();
				}
				addExportMenu();
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
				//if (i == bubblePlotter->posIndices.x || i == bubblePlotter->posIndices.y || i == bubblePlotter->posIndices.z)
				//	continue;
				if (ImGui::Checkbox((bubblePlotter->attributeNames[i] + "##cb").c_str(), &bubblePlotter->attributeActivations[i])) {		//redistribute the remaining variales over the free layer space
					float count = 0;
					for (int j = 0; j < bubblePlotter->attributeNames.size(); ++j) {
						if (bubblePlotter->attributeActivations[j])//&& j != bubblePlotter->posIndices.x && j != bubblePlotter->posIndices.y && j != bubblePlotter->posIndices.z)
							count += 1;
					}
					if(count != 1.0f)
						count = 1 / (count - 1); //converting count to the percentage step
					float curP = 0;
					for (int j = 0; j < bubblePlotter->attributeNames.size(); ++j) {
						if (!bubblePlotter->attributeActivations[j]) {
							continue;
						}
						bubblePlotter->attributeTopOffsets[j] = curP;
						curP += count;
					}
					bubblePlotter->render();
				}
				ImGui::NextColumn();
				static char const* scales[] = { "Normal","Squareroot","Logarithmic" };
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
					bubblePlotter->setBubbleData(dl->indices, attributeNames, attributeMinMax, parent->data, dl->buffer, dl->activeIndicesBufferView, attributeNames.size(), parent->data.size());
				}
				ImGui::EndDragDropTarget();
			}

			ImGui::End();
		}
			
		//end of bubble window ---------------------------------------------------------------------------

		//begin of iso surface window --------------------------------------------------------------------
		if (isoSurfSettings.enabled) {
			ImGui::Begin("Isosurface Renderer",&isoSurfSettings.enabled,ImGuiWindowFlags_MenuBar);
			int dlbExport = -1;
			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("Save Settings")) {
					addSaveSettingsMenu<IsoSettings>(&isoSurfSettings, "IsoSettings", "isosettings");
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Settings")) {
					ImGui::Checkbox("Couple to brush", &isoSurfSettings.coupleIsoSurfaceRenderer);
					
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

				addExportMenu();
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
			if (ImGui::BeginCombo("Brush(if now brush is selected, active indices are used for iso Surface)", (selectedGlobalBrush == -1) ? choose : globalBrushes[selectedGlobalBrush].nam.c_str())) {
				if (ImGui::Selectable(choose, selectedGlobalBrush == -1)) selectedGlobalBrush = -1;
				for (int i = 0; i < globalBrushes.size(); ++i) {
					if (ImGui::Selectable(globalBrushes[i].nam.c_str(), selectedGlobalBrush == i)) selectedGlobalBrush = i;
				}
				ImGui::EndCombo();
			}
			ImGui::PopItemWidth();

			ImGui::PushItemWidth(100);
			//setting the position variables
			if (pcAttributes.size()) {
				if (ImGui::BeginCombo("##xdim", pcAttributes[isoSurfSettings.posIndices.x].name.c_str())) {
					for (int i = 0; i < pcAttributes.size(); ++i) {
						if (ImGui::MenuItem(pcAttributes[i].name.c_str())) {
							isoSurfSettings.posIndices.x = i;
						}
					}
					ImGui::EndCombo();
				}
				ImGui::SameLine();
				if (ImGui::BeginCombo("##ydim", pcAttributes[isoSurfSettings.posIndices.y].name.c_str())) {
					for (int i = 0; i < pcAttributes.size(); ++i) {
						if (ImGui::MenuItem(pcAttributes[i].name.c_str())) {
							isoSurfSettings.posIndices.y = i;
						}
					}
					ImGui::EndCombo();
				}
				ImGui::SameLine();
				if (ImGui::BeginCombo("Position indices (Order: lat, alt, lon)##zdim", pcAttributes[isoSurfSettings.posIndices.z].name.c_str())) {
					for (int i = 0; i < pcAttributes.size(); ++i) {
						if (ImGui::MenuItem(pcAttributes[i].name.c_str())) {
							isoSurfSettings.posIndices.z = i;
						}
					}
					ImGui::EndCombo();
				}
			}
			else {
				ImGui::Text("Placeholder for position indices settings (Settings appear when Attributes are available)");
			}
			//ImGui::DragInt3("Position indices (Order: lat, alt, lon)", (int*)&posIndices.x, 0.00000001f, 0, pcAttributes.size());
			ImGui::PopItemWidth();

			static bool showError = false;
			static bool positionError = false;
			if (ImGui::Button("Add new iso surface")) {
				if (selectedDrawlist == -1 || isoSurfSettings.posIndices.x== isoSurfSettings.posIndices.y || isoSurfSettings.posIndices.y== isoSurfSettings.posIndices.z || isoSurfSettings.posIndices.x== isoSurfSettings.posIndices.z) {
					showError = true;
				}
				else {
					DrawList* dl = &*std::next(g_PcPlotDrawLists.begin(), selectedDrawlist);
					DataSet* ds = nullptr;
					for (DataSet& d : g_PcPlotDataSets) {
						if (dl->parentDataSet == d.name) {
							ds = &d;
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
							if (isoSurfaceRenderer->drawlistBrushes[i].brush.size() && globalBrushes[selectedGlobalBrush].id == isoSurfaceRenderer->drawlistBrushes[i].brush) {
								index = i;
								break;
							}
							else if (!isoSurfaceRenderer->drawlistBrushes[i].brush.size()) {
								index = i;
								break;
							}
						}
					}
					auto& xDim = getDimensionValues(*ds, isoSurfSettings.posIndices.x), yDim = getDimensionValues(*ds, isoSurfSettings.posIndices.y), zDim = getDimensionValues(*ds, isoSurfSettings.posIndices.z);
					uint32_t w = xDim.second.size();
					uint32_t h = yDim.second.size();
					uint32_t d = zDim.second.size();
					bool regularGrid[3]{ xDim.first, yDim.first, zDim.first };
					if (index == -1) {
						isoSurfaceRenderer->drawlistBrushes.push_back({ dl->name,(selectedGlobalBrush == -1) ? "" : globalBrushes[selectedGlobalBrush].id,{ 1,0,0,1 }, {w, h, d} });
					}
					if (selectedGlobalBrush == -1) {
						std::vector<std::pair<float, float>> posBounds(3);
						for (int i = 0; i < 3; ++i) {
							posBounds[i].first = pcAttributes[isoSurfSettings.posIndices[i]].min;
							posBounds[i].second = pcAttributes[isoSurfSettings.posIndices[i]].max;
						}
						isoSurfaceRenderer->update3dBinaryVolume(xDim.second, yDim.second, zDim.second, &isoSurfSettings.posIndices.x, posBounds, pcAttributes.size(), ds->data.size(), dl->buffer, dl->activeIndicesBufferView, dl->indices.size(), dl->indicesBuffer, regularGrid, index);
					}
					else {
						isoSurfaceRenderer->update3dBinaryVolume(xDim.second, yDim.second, zDim.second, pcAttributes.size(), brushIndices, minMax, isoSurfSettings.posIndices, dl->buffer, ds->data.size() * pcAttributes.size() * sizeof(float), dl->indicesBuffer, dl->indices.size(), miMa, index);
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
				ImGui::Text("%s", db.drawlist.c_str());
				ImGui::NextColumn();
				ImGui::Text("%s", db.brush.c_str());
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
			ImGui::End();
		}
		//end of iso surface window -----------------------------------------------------------------------

#ifdef RENDER3D
		//begin view 3d window      ----------------------------------------------------------------------
		if (view3dSettings.enabled) {
			ImGui::Begin("3dview", &view3dSettings.enabled, ImGuiWindowFlags_MenuBar);
			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("Save Settings")) {
					addSaveSettingsMenu<View3dSettings>(&view3dSettings, "View3dSettings", "view3dsettings");
					ImGui::EndMenu();
				}
				//if (ImGui::BeginMenu("Settings")) {
				//	ImGui::Checkbox("Couple to brush", &isoSurfSettings.coupleIsoSurfaceRenderer);
				//
				//	ImGui::EndMenu();
				//}
				if (ImGui::BeginMenu("Rendering")) {
					static float boxSize[3]{ 1.5f,1.f,1.5f };
					if (ImGui::DragFloat3("Box dimensions", boxSize, .001f)) {
						view3d->resizeBox(boxSize[0], boxSize[1], boxSize[2]);
						view3d->render();
					}
					//if (ImGui::Checkbox("Activate shading", &view3d->shade)) {
					//	view3d->render();
					//}
					if (ImGui::SliderFloat("Ray march step size", &view3d->stepSize, 0.0005f, .05f, "%.5f")) {
						view3d->render();
					}
					//if (ImGui::SliderFloat("Step size for normal calc", &view3d->shadingStep, .1f, 10)) {
					//	view3d->render();
					//}
					//if (ImGui::SliderFloat("Wireframe width", &view3d->gridLineWidth, 0, .1f)) {
					//	view3d->render();
					//}
					if (ImGui::DragFloat3("Ligt direction", &isoSurfaceRenderer->lightDir.x)) {
						view3d->render();
					}
					//if (ImGui::ColorEdit4("Image background", isoSurfaceRenderer->imageBackground.color.float32, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar)) {
					//	view3d->imageBackGroundUpdated();
					//	view3d->render();
					//}
					ImGui::EndMenu();
				}
				ImGui::EndMenuBar();
			}

			ImGui::Image((ImTextureID)view3d->getImageDescriptorSet(), ImVec2(800, 800));

			//if (ImGui::IsWindowHovered() && ImGui::GetIO().MouseReleased[0]);
			//	view3d->resize(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowHeight());

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
				view3d->updateCameraPos(nav, io.DeltaTime);
				view3d->render();
				err = vkQueueWaitIdle(g_Queue);
				check_vk_result(err);
				ImGui::ResetMouseDragDelta();
			}

			ImGui::PushItemWidth(100);
			//setting the position variables
			if (pcAttributes.size()) {
				if (ImGui::BeginCombo("##xdim", pcAttributes[view3dSettings.posIndices[0]].name.c_str())) {
					for (int i = 0; i < pcAttributes.size(); ++i) {
						if (ImGui::MenuItem(pcAttributes[i].name.c_str())) {
							view3dSettings.posIndices[0] = i;
						}
					}
					ImGui::EndCombo();
				}
				ImGui::SameLine();
				if (ImGui::BeginCombo("##ydim", pcAttributes[view3dSettings.posIndices[1]].name.c_str())) {
					for (int i = 0; i < pcAttributes.size(); ++i) {
						if (ImGui::MenuItem(pcAttributes[i].name.c_str())) {
							view3dSettings.posIndices[1] = i;
						}
					}
					ImGui::EndCombo();
				}
				ImGui::SameLine();
				if (ImGui::BeginCombo("Position indices (Order: lat, alt, lon)##zdim", pcAttributes[view3dSettings.posIndices[2]].name.c_str())) {
					for (int i = 0; i < pcAttributes.size(); ++i) {
						if (ImGui::MenuItem(pcAttributes[i].name.c_str())) {
							view3dSettings.posIndices[2] = i;
						}
					}
					ImGui::EndCombo();
				}
			}
			else {
				ImGui::Text("Placeholder for position indices settings (Settings appear when Attributes are available)");
			}
			ImGui::PopItemWidth();

			static int densityAttribute = -1;
			if (ImGui::BeginCombo("Select Density Attribute", (densityAttribute == -1) ? "Select" : pcAttributes[densityAttribute].name.c_str())) {
				for (int i = 0; i < pcAttributes.size(); ++i) {
					if (ImGui::MenuItem(pcAttributes[i].name.c_str())) {
						densityAttribute = i;
					}
				}
				ImGui::EndCombo();
			}
			static int drawList = -1;
			if (ImGui::BeginCombo("Select Density Drawlist", (drawList == -1) ? "Select" : std::next(g_PcPlotDrawLists.begin(), drawList)->name.c_str())) {
				for (int i = 0; i < g_PcPlotDrawLists.size(); ++i) {
					if (ImGui::MenuItem(std::next(g_PcPlotDrawLists.begin(), i)->name.c_str())) {
						drawList = i;
					}
				}
				ImGui::EndCombo();
			}

			if (ImGui::Button("Set density")) {
				if (densityAttribute == -1 || drawList == -1) {
					std::cout << "Density attribute and draw list has to be selected";
				}
				else {
					uploadDrawListTo3dView(*std::next(g_PcPlotDrawLists.begin(), drawList), densityAttribute);
				}
			}
			ImGui::Text("Transfer function (Click to edit):");
			if (ImGui::ImageButton((ImTextureID)transferFunctionEditor->getTransferDescriptorSet(), { 300, 25 },{0,0},{1,1},0)) {
				transferFunctionEditor->setNextEditorPos(ImGui::GetMousePos(), { 0, 1 });
				transferFunctionEditor->show();
			}
			if(view3d->histogramBins.size()){
				ImGui::PlotHistogram("Denisty Histogram", &intArrayGetter, view3d->histogramBins.data(), view3d->histogramBins.size(), 0, 0, FLT_MAX, FLT_MAX, ImVec2(300,50));
			}
			ImGui::End();
		}
		//end view 3d window        ----------------------------------------------------------------------
#endif

		//brush iso surface window -----------------------------------------------------------------------
		if (brushIsoSurfSettings.enabled) {
			ImGui::Begin("Brush Isosurface Renderer", &brushIsoSurfSettings.enabled, ImGuiWindowFlags_MenuBar);
			int dlbExport = -1;
			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("Save Settings")) {
					addSaveSettingsMenu<IsoSettings>(&brushIsoSurfSettings, "BrushIsoSettings", "brushisosettings");
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Settings")) {
					ImGui::Checkbox("Couple to brush", &brushIsoSurfSettings.coupleBrushIsoSurfaceRenderer);

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
				addExportMenu();

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
			//ImGui::DragInt3("Position indices(order: lat, alt, lon)", (int*)posIndices, 1, 0, pcAttributes.size());

			ImGui::PushItemWidth(100);
			//setting the position variables
			if (pcAttributes.size()) {
				if (ImGui::BeginCombo("##xdim", pcAttributes[brushIsoSurfSettings.posIndices[0]].name.c_str())) {
					for (int i = 0; i < pcAttributes.size(); ++i) {
						if (ImGui::MenuItem(pcAttributes[i].name.c_str())) {
							brushIsoSurfSettings.posIndices[0] = i;
						}
					}
					ImGui::EndCombo();
				}
				ImGui::SameLine();
				if (ImGui::BeginCombo("##ydim", pcAttributes[brushIsoSurfSettings.posIndices[1]].name.c_str())) {
					for (int i = 0; i < pcAttributes.size(); ++i) {
						if (ImGui::MenuItem(pcAttributes[i].name.c_str())) {
							brushIsoSurfSettings.posIndices[1] = i;
						}
					}
					ImGui::EndCombo();
				}
				ImGui::SameLine();
				if (ImGui::BeginCombo("Position indices (Order: lat, alt, lon)##zdim", pcAttributes[brushIsoSurfSettings.posIndices[2]].name.c_str())) {
					for (int i = 0; i < pcAttributes.size(); ++i) {
						if (ImGui::MenuItem(pcAttributes[i].name.c_str())) {
							brushIsoSurfSettings.posIndices[2] = i;
						}
					}
					ImGui::EndCombo();
				}
			}
			else {
				ImGui::Text("Placeholder for position indices settings (Settings appear when Attributes are available)");
			}
			ImGui::PopItemWidth();

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
					brushIsoSurfaceRenderer->updateBrush(brush->id, minMax);
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
					std::pair<bool,std::vector<float>> xDim = getDimensionValues(*ds, brushIsoSurfSettings.posIndices.x), yDim = getDimensionValues(*ds, brushIsoSurfSettings.posIndices.y), zDim = getDimensionValues(*ds, brushIsoSurfSettings.posIndices.z);
					bool regularGrid[3]{ xDim.first, yDim.first, zDim.first };
					uint32_t w = xDim.second.size();
					uint32_t h = yDim.second.size();
					uint32_t d = zDim.second.size();
					std::vector<uint32_t> densityInds(pcAttributes.size());
					for (int i = 0; i < pcAttributes.size(); ++i) densityInds[i] = i;
					std::vector<std::pair<float, float>> bounds;// { {pcAttributes[posIndices[0]].min, pcAttributes[posIndices[0]].max}, { pcAttributes[posIndices[1]].min,pcAttributes[posIndices[1]].max }, { pcAttributes[posIndices[2]].min,pcAttributes[posIndices[2]].max } };
					for (int i = 0; i < pcAttributes.size(); ++i) {
						bounds.emplace_back(pcAttributes[i].min, pcAttributes[i].max);
					}
					brushIsoSurfaceRenderer->update3dBinaryVolume(xDim.second, yDim.second, zDim.second, pcAttributes.size(), densityInds, &brushIsoSurfSettings.posIndices.x, bounds, dl->buffer, ds->data.size(), dl->indicesBuffer, dl->indices.size(),regularGrid);
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
		if (violinPlotAttributeSettings.violinYScale == ViolinYScaleGlobalBrush || violinPlotAttributeSettings.violinYScale == ViolinYScaleBrushes) {
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
		if (violinPlotAttributeSettings.enabled) {
			ImGui::Begin("Violin attribute window", &violinPlotAttributeSettings.enabled, ImGuiWindowFlags_MenuBar);
			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("Save Settings")) {
					addSaveSettingsMenu<ViolinSettings>(&violinPlotAttributeSettings, "ViolinAttribute", "violinattribute");
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Settings")) {
					ImGui::Checkbox("Couple to Brushing", &violinPlotAttributeSettings.coupleViolinPlots);
					ImGui::Checkbox("Show Attribute min/max", &violinPlotAttributeSettings.showViolinPlotsMinMax);
					ImGui::SliderInt("Violin plots height", &violinPlotAttributeSettings.violinPlotHeight, 1, 4000);
					ImGui::SliderInt("Violin plots x spacing", &violinPlotAttributeSettings.violinPlotXSpacing, 0, 40);
					ImGui::SliderFloat("Violin plots line thickness", &violinPlotAttributeSettings.violinPlotThickness, 0, 10);
					ImGui::ColorEdit4("Violin plots background", &violinPlotAttributeSettings.violinBackgroundColor.x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
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

								if (violinPlotAttributeSettings.renderOrderAttConsider && ((drawL == 0) || (!violinPlotAttributeSettings.renderOrderBasedOnFirstAtt)))
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
						updateAllViolinPlotMaxValues(violinPlotAttributeSettings.renderOrderBasedOnFirstDL);
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

								if (violinPlotAttributeSettings.renderOrderAttConsider && ((drawL == 0) || (!violinPlotAttributeSettings.renderOrderBasedOnFirstAtt)))
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
					if (ImGui::SliderFloat("Smoothing kernel stdDev", &stdDev, 0, 25)) {
						histogramManager->setSmoothingKernelSize(stdDev);
						updateAllViolinPlotMaxValues(violinPlotAttributeSettings.renderOrderBasedOnFirstAtt);
						for(auto& plot: violinAttributePlots) updateSummedBins(plot);
					}
					static char const* violinYs[] = { "Standard","Local brush","Global brush","All brushes" };
					if (ImGui::BeginCombo("Y Scale", violinYs[violinPlotAttributeSettings.violinYScale])) {
                        ImGui::SetTooltip("This only affects to which range the bins are fitted. The y min and max are the PCPlot axis borders.");
						for (int v = 0; v < 4; ++v) {
							if (ImGui::MenuItem(violinYs[v])) {
								violinPlotAttributeSettings.violinYScale =(ViolinYScale) v;
							}
						}
						ImGui::EndCombo();
					}
					ImGui::Columns(3);
					ImGui::Checkbox("Overlay lines", &violinPlotAttributeSettings.violinPlotOverlayLines);
					ImGui::NextColumn();
					ImGui::Checkbox("Base render order on first attribute", &violinPlotAttributeSettings.renderOrderBasedOnFirstAtt);
					ImGui::NextColumn();
					ImGui::Checkbox("Optimize render order", &violinPlotAttributeSettings.renderOrderAttConsider);
					ImGui::Separator();

					ImGui::Checkbox("Optimize non-stop", &violinPlotAttributeSettings.renderOrderAttConsiderNonStop);
					
					//ImGui::EndMenu();

					ImGui::EndMenu();
				}
				addExportMenu();
				ImGui::EndMenuBar();
			}

			const static int plusWidth = 100;
            for (unsigned int i = 0; i < violinAttributePlots.size(); ++i) {
				ImGui::BeginChild(std::to_string(i).c_str(), ImVec2(-1, violinPlotAttributeSettings.violinPlotHeight), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
				//ImGui::BeginChild(std::to_string(i).c_str(), ImVec2(0, 0), true, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar);
				ImGui::PushItemWidth(150);
				//drawing uniform settings
				ImGui::Text("Uniform Settings");
				ImGui::SameLine(200);
				static char const* plotPositions[] = { "Left","Right","Middle" };
				static int uniformPlotPosition = 0;
				if (ImGui::BeginCombo("Position##uniform", plotPositions[uniformPlotPosition])) {
					for (int k = 0; k < 3; ++k) {
						if (ImGui::MenuItem(plotPositions[k])) {
							uniformPlotPosition = k;
							for (int j = 0; j < violinAttributePlots[i].drawLists.size(); ++j) {
								violinAttributePlots[i].violinPlacements[j] = (ViolinPlacement)k;
							}
						}
					}
					ImGui::EndCombo();
				}
				ImGui::SameLine(480);
				static char const* violinScales[] = { "Self","Local","Global","Global Attribute" };
				static int uniformPlotScale = 3;
				if (ImGui::BeginCombo("Scale##uniform", violinScales[uniformPlotScale])) {
					for (int k = 0; k < 4; ++k) {
						if (ImGui::MenuItem(violinScales[k])) {
							uniformPlotScale = k;
							for (int j = 0; j < violinAttributePlots[i].drawLists.size(); ++j) {
								violinAttributePlots[i].violinScalesX[j] = (ViolinScale)k;
							}
						}
					}
					ImGui::EndCombo();
				}
				ImGui::SameLine(730);
				static ImVec4 uniformLineColor = { 0,0,0,1 };
				if (ImGui::ColorEdit4("Line Col##uniform", &uniformLineColor.x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar)) {
					for (int j = 0; j < violinAttributePlots[i].drawLists.size(); ++j) {
						violinAttributePlots[i].drawListLineColors[j] = uniformLineColor;
					}
				}
				ImGui::SameLine(900);
				static ImVec4 uniformFillColor = { 0,0,0,.1 };
				if (ImGui::ColorEdit4("Fill Col##uniform", &uniformFillColor.x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar)) {
					for (int j = 0; j < violinAttributePlots[i].drawLists.size(); ++j) {
						violinAttributePlots[i].drawListFillColors[j] = uniformFillColor;
					}
				}
				//listing all histograms available
				if (ImGui::CollapsingHeader("Added Drawlists")) {
					for (int j = 0; j < violinAttributePlots[i].drawLists.size(); ++j) {
						if (ImGui::Checkbox(violinAttributePlots[i].drawLists[j].name.c_str(), &violinAttributePlots[i].drawLists[j].activated)) {
							updateAllViolinPlotMaxValues(violinPlotAttributeSettings.renderOrderBasedOnFirstAtt);
							updateSummedBins(violinAttributePlots[i]);
						}

						ImGui::SameLine(200);
						if (ImGui::BeginCombo(("Position##" + std::to_string(j)).c_str(), plotPositions[violinAttributePlots[i].violinPlacements[j]])) {
							for (int k = 0; k < 3; ++k) {
								if (ImGui::MenuItem(plotPositions[k], nullptr)) {
									violinAttributePlots[i].violinPlacements[j] = (ViolinPlacement)k;
								}
							}
							ImGui::EndCombo();
						}

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
						//ImGui::SameLine(1200);
						//if (ImGui::Checkbox(("log##" + std::to_string(j)).c_str(), &histogramManager->logScale[j])) {
						//	histogramManager->updateSmoothedValues();
						//	updateAllViolinPlotMaxValues(renderOrderBasedOnFirstAtt);
						//	updateSummedBins(violinAttributePlots[i]);
						//};
					}
				}
				static char choose[] = "Choose drawlist";
				if (ImGui::BeginCombo("Add drawlistdata", choose)) {
					for (auto k = g_PcPlotDrawLists.begin(); k != g_PcPlotDrawLists.end(); ++k) {
						if (ImGui::MenuItem(k->name.c_str(), "", false)) {
							violinAttributePlotAddDrawList(violinAttributePlots[i], *k, i);
						}
					}
					ImGui::EndCombo();
				}

                // Draw everything to load Colorbrewer Colorpalettes
                //if (violinAttributePlots[i].attributeNames.size() > 0){
                //    includeColorbrewerToViolinPlot((violinAttributePlots[i].colorPaletteManager),
                //                                   &(violinAttributePlots[i].drawListLineColors),
                //                                   &(violinAttributePlots[i].drawListFillColors));
                //}

				int amtOfAttributes = 0;
				for (int j = 0; j < violinAttributePlots[i].maxValues.size(); ++j) {
                    ImGui::Checkbox(pcAttributes[j].name.c_str(), violinAttributePlots[i].activeAttributes + j);
					ImGui::SameLine(200);
					if (ImGui::Checkbox(("Log##" + pcAttributes[j].name).c_str(), &histogramManager->logScale[j])) {
						histogramManager->updateSmoothedValues();
						updateAllViolinPlotMaxValues(violinPlotAttributeSettings.renderOrderBasedOnFirstDL);
					}

					if (violinAttributePlots[i].activeAttributes[j]) ++amtOfAttributes;
				}



				int previousNrOfColumns = ImGui::GetColumnsCount();
				ImGui::Separator();
				ImGui::Columns(5);


				if ((ImGui::Button("Optimize sides <right/left>")) || (violinPlotAttributeSettings.violinPlotAttrReplaceNonStop)) {
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
				ImGui::Checkbox("", &violinPlotAttributeSettings.violinPlotAttrInsertCustomColors);
				ImGui::SameLine(50);
				if (ImGui::BeginCombo("##appcolattr", "Apply colors of Dark2YellowSplit")) {
					std::vector<std::string> *availablePalettes = 
						violinAttributePlots[i].colorPaletteManager->colorPalette->getQualPaletteNames();

					std::vector<const char*>  vc = convertStringVecToConstChar(availablePalettes);
				
					if (vc.size() > 0) {
						//static char* violinYs[] = { "Standard","Local brush","Global brush","All brushes" };
						if (ImGui::BeginCombo("Line Palette", vc[violinPlotAttributeSettings.autoColorAssingLine])) {
							for (int v = 0; v < vc.size(); ++v) {
								if (ImGui::MenuItem(vc[v])) {
									violinAttributePlots[i].colorPaletteManager->chosenAutoColorPaletteLine =
										(*availablePalettes)[v];
									violinPlotAttributeSettings.autoColorAssingLine = v;
								}
							}
							ImGui::EndCombo();
						}
						if (ImGui::BeginCombo("Fill Palette", vc[violinPlotAttributeSettings.autoColorAssingFill])) {
							for (int v = 0; v < vc.size(); ++v) {
								if (ImGui::MenuItem(vc[v])) {
									violinAttributePlots[i].colorPaletteManager->chosenAutoColorPaletteFill =
										(*availablePalettes)[v];
									violinPlotAttributeSettings.autoColorAssingFill = v;
								}
							}
							ImGui::EndCombo();
                        }
                    }
					ImGui::EndCombo();
				}
				

				ImGui::NextColumn();
				ImGui::Checkbox("Re-place constantly", &violinPlotAttributeSettings.violinPlotAttrReplaceNonStop);

				ImGui::NextColumn();
				ImGui::Checkbox("Consider blending order", &violinPlotAttributeSettings.violinPlotAttrConsiderBlendingOrder);
				ImGui::NextColumn();
				if(ImGui::Checkbox("Reverse color pallette", &violinPlotAttributeSettings.violinPlotAttrReverseColorPallette))
				{
					violinAttributePlots[i].colorPaletteManager->setReverseColorOrder(violinPlotAttributeSettings.violinPlotDLReverseColorPallette);
				}

				if (ImGui::Button("Fix order and colors"))
				{
					violinPlotAttributeSettings.violinPlotAttrReplaceNonStop = false;
					violinAttributePlots[i].colorPaletteManager->useColorPalette = false;
					violinPlotAttributeSettings.renderOrderAttConsiderNonStop = false;
					violinPlotAttributeSettings.renderOrderAttConsider = false;
					//violinDrawlistPlots[i].colorPaletteManager->useColorPalette = false;				
				}

				includeColorbrewerToViolinPlot(violinAttributePlots[i].colorPaletteManager, &violinAttributePlots[i].drawListLineColors, &violinAttributePlots[i].drawListFillColors);


				ImGui::Columns(previousNrOfColumns);

				const char* plotCombinations[2] = { "stacked", "sum" };
				if(ImGui::BeginCombo("Violin plot combination", plotCombinations[violinPlotAttributeSettings.violinPlotAttrStacking])) {
					for (int comp = 0; comp < 2; ++comp) {
						if (ImGui::MenuItem(plotCombinations[comp])) violinPlotAttributeSettings.violinPlotAttrStacking = comp;
					}
					ImGui::EndCombo();
				}

				//labels for the plots
				ImGui::Separator();
				int c = 0;
				int c1 = 0;
				float xGap = (ImGui::GetWindowContentRegionWidth() - (amtOfAttributes - 1) * violinPlotAttributeSettings.violinPlotXSpacing) / amtOfAttributes + violinPlotAttributeSettings.violinPlotXSpacing;
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
						int p[] = { c,int(i) };		//holding the index in the pcAttriOrd array and the value of it
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

				// axis min max values
				if (violinPlotAttributeSettings.showViolinPlotsMinMax) {
					int c = 0;
					int c1 = 0;
					float xGap = (ImGui::GetWindowContentRegionWidth() - (amtOfAttributes - 1) * violinPlotAttributeSettings.violinPlotXSpacing) / amtOfAttributes + violinPlotAttributeSettings.violinPlotXSpacing;
					for (uint32_t j : violinAttributePlots[i].attributeOrder) {
						if (!violinAttributePlots[i].activeAttributes[j]) {
							c++;
							continue;
						}

						if (c1 != 0) {
							ImGui::SameLine(c1 * xGap + 10);
						}
						ImGui::Text("%s", (std::to_string(pcAttributes[j].max)).c_str());

						c++;
						c1++;
					}
				}

				// Drawing the violin plots
				ImVec2 leftUpperCorner = ImGui::GetCursorScreenPos();
				ImVec2 origLeftUpper = leftUpperCorner;
				ImVec2 size((ImGui::GetWindowContentRegionWidth() - (amtOfAttributes - 1) * violinPlotAttributeSettings.violinPlotXSpacing) / amtOfAttributes, ImGui::GetWindowContentRegionMax().y - leftUpperCorner.y + ImGui::GetWindowPos().y - (violinPlotAttributeSettings.showViolinPlotsMinMax ? ImGui::GetTextLineHeightWithSpacing(): 0));
				//ImVec2 size((ImGui::GetWindowContentRegionWidth() - (amtOfAttributes - 1) * violinPlotXSpacing) / amtOfAttributes, violinPlotHeight);
				ViolinDrawState drawState = (violinPlotAttributeSettings.violinPlotOverlayLines) ? ViolinDrawStateArea : ViolinDrawStateAll;
				bool done = false;
				while (!done) {
					leftUpperCorner = origLeftUpper;
					for (int j : violinAttributePlots[i].attributeOrder) {		//Drawing the plots per Attribute
						if (!violinAttributePlots[i].activeAttributes[j]) continue;
						if (drawState == ViolinDrawStateAll || drawState == ViolinDrawStateArea) ImGui::RenderFrame(leftUpperCorner, leftUpperCorner + size, ImGui::GetColorU32(violinPlotAttributeSettings.violinBackgroundColor), true, ImGui::GetStyle().FrameRounding);
						ImGui::PushClipRect(leftUpperCorner, leftUpperCorner + size + ImVec2{ 1,1 }, false);
						if (violinPlotAttributeSettings.violinPlotAttrStacking == 0) {
							for (int k = 0; k < violinAttributePlots[i].drawLists.size(); ++k) {
								if (!violinAttributePlots[i].drawLists[k].activated) continue;
								HistogramManager::Histogram& hist = histogramManager->getHistogram(violinAttributePlots[i].drawLists[k].name);
								DrawList* dl = nullptr;
								if (true || violinPlotAttributeSettings.yScaleToCurrenMax) {
									for (DrawList& draw : g_PcPlotDrawLists) {
										if (draw.name == violinAttributePlots[i].drawLists[k].name) {
											dl = &draw;
										}
									}
								}
								//std::vector<std::pair<float, float>> localMinMax(pcAttributes.size(), { std::numeric_limits<float>().max(),std::numeric_limits<float>().min() });
								if (violinPlotAttributeSettings.violinYScale == ViolinYScaleLocalBrush || violinPlotAttributeSettings.violinYScale == ViolinYScaleBrushes) {
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
								switch (violinPlotAttributeSettings.violinYScale) {
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
									div = violinAttributePlots[i].maxValues[j];
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
											ImGui::GetWindowDrawList()->AddLine(ImVec2(leftUpperCorner.x + hist.bins[j][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[j].size() - 1) * histYLineDiff),
												ImVec2(leftUpperCorner.x + hist.bins[j][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[j].size() - 1) * histYLineDiff), ImColor(violinAttributePlots[i].drawListLineColors[k]), violinPlotAttributeSettings.violinPlotThickness);
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
											ImGui::GetWindowDrawList()->AddLine(ImVec2(leftUpperCorner.x + size.x - hist.bins[j][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[j].size() - 1) * histYLineDiff),
												ImVec2(leftUpperCorner.x + size.x - hist.bins[j][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[j].size() - 1) * histYLineDiff), ImColor(violinAttributePlots[i].drawListLineColors[k]), violinPlotAttributeSettings.violinPlotThickness);
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
											ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase - .5f * hist.bins[j][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[j].size() - 1) * histYLineDiff),
												ImVec2(xBase - .5f * hist.bins[j][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[j].size() - 1) * histYLineDiff), ImColor(violinAttributePlots[i].drawListLineColors[k]), violinPlotAttributeSettings.violinPlotThickness);
											//right Line
											ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase + .5f * hist.bins[j][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[j].size() - 1) * histYLineDiff),
												ImVec2(xBase + .5f * hist.bins[j][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[j].size() - 1) * histYLineDiff), ImColor(violinAttributePlots[i].drawListLineColors[k]), violinPlotAttributeSettings.violinPlotThickness);
										}
									}

									break;
								}

							}
						}
						else if (violinAttributePlots[i].drawLists.size()) {
							HistogramManager::Histogram& hist = histogramManager->getHistogram(violinAttributePlots[i].drawLists[0].name);
							auto& summedBins = violinAttributePlots[i].summedBins;
							float histYStart;
							float histYEnd;
							histYStart = 0;
							histYEnd = size.y;
							float histYFillStart = (histYStart < 0) ? 0 : histYStart;
							float histYFillEnd = (histYEnd > size.y) ? size.y : histYEnd;
							float histYLineStart = histYStart + leftUpperCorner.y;
							float histYLineEnd = histYEnd + leftUpperCorner.y;
							float histYLineDiff = histYLineEnd - histYLineStart;

							float div = 0;
							std::vector<float> scals({});
							div = violinAttributePlots[i].maxSummedValues[j];//violinAttributePlots[i].maxValues[k];

							switch (violinAttributePlots[i].violinPlacements[0]) {
							case ViolinLeft:
								//filling
								if (drawState == ViolinDrawStateArea || drawState == ViolinDrawStateAll) {
									hist.binsRendered[j].clear();
									hist.areaRendered[j] = 0;
									for (int p = histYFillStart; p < histYFillEnd; ++p) {
										ImVec2 a(leftUpperCorner.x, leftUpperCorner.y + p);
										float v = getBinVal(((1 - (p + .5f) + histYEnd) / (histYEnd - histYStart)), summedBins[j]);
										ImVec2 b(leftUpperCorner.x + v / div * size.x, leftUpperCorner.y + p + 1);
										hist.binsRendered[j].push_back(std::abs(b.x - a.x));
										hist.areaRendered[j] += std::abs(b.x - a.x);
										if (b.x - a.x >= 1)
											ImGui::GetWindowDrawList()->AddRectFilled(a, b, ImColor(violinAttributePlots[i].drawListFillColors[0]));
									}
								}
								//outline
								if (drawState == ViolinDrawStateLine || drawState == ViolinDrawStateAll) {
									for (int l = 1; l < summedBins[j].size(); ++l) {
										ImGui::GetWindowDrawList()->AddLine(ImVec2(leftUpperCorner.x + summedBins[j][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (summedBins[j].size() - 1) * histYLineDiff),
											ImVec2(leftUpperCorner.x + summedBins[j][l] / div * size.x, histYLineEnd - ((float)l) / (summedBins[j].size() - 1) * histYLineDiff), ImColor(violinAttributePlots[i].drawListLineColors[0]), violinPlotAttributeSettings.violinPlotThickness);
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
										float v = getBinVal(((1 - (p + .5f) + histYEnd) / (histYEnd - histYStart)), summedBins[j]);
										ImVec2 b(leftUpperCorner.x + size.x - v / div * size.x, leftUpperCorner.y + p + 1);
										hist.binsRendered[j].push_back(std::abs(b.x - a.x));
										hist.areaRendered[j] += std::abs(b.x - a.x);
										if (a.x - b.x >= 1)
											ImGui::GetWindowDrawList()->AddRectFilled(a, b, ImColor(violinAttributePlots[i].drawListFillColors[0]));
									}
								}
								//outline
								if (drawState == ViolinDrawStateLine || drawState == ViolinDrawStateAll) {
									for (int l = 1; l < summedBins[j].size(); ++l) {
										ImGui::GetWindowDrawList()->AddLine(ImVec2(leftUpperCorner.x + size.x - summedBins[j][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (summedBins[j].size() - 1) * histYLineDiff),
											ImVec2(leftUpperCorner.x + size.x - summedBins[j][l] / div * size.x, histYLineEnd - ((float)l) / (summedBins[j].size() - 1) * histYLineDiff), ImColor(violinAttributePlots[i].drawListLineColors[0]), violinPlotAttributeSettings.violinPlotThickness);
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
										float v = getBinVal(((1 - (p + .5f) + histYEnd) / (histYEnd - histYStart)), summedBins[j]);
										ImVec2 a(xBase - .5f * v / div * size.x, leftUpperCorner.y + p);
										ImVec2 b(xBase + .5f * v / div * size.x, leftUpperCorner.y + p + 1);
										hist.binsRendered[j].push_back(std::abs(b.x - a.x));
										hist.areaRendered[j] += std::abs(b.x - a.x);
										if (b.x - a.x >= 1)
											ImGui::GetWindowDrawList()->AddRectFilled(a, b, ImColor(violinAttributePlots[i].drawListFillColors[0]));
									}
								}
								if (drawState == ViolinDrawStateLine || drawState == ViolinDrawStateAll) {
									for (int l = 1; l < summedBins[j].size(); ++l) {
										//left Line
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase - .5f * summedBins[j][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (summedBins[j].size() - 1) * histYLineDiff),
											ImVec2(xBase - .5f * summedBins[j][l] / div * size.x, histYLineEnd - ((float)l) / (summedBins[j].size() - 1) * histYLineDiff), ImColor(violinAttributePlots[i].drawListLineColors[0]), violinPlotAttributeSettings.violinPlotThickness);
										//right Line
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase + .5f * summedBins[j][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (summedBins[j].size() - 1) * histYLineDiff),
											ImVec2(xBase + .5f * summedBins[j][l] / div * size.x, histYLineEnd - ((float)l) / (summedBins[j].size() - 1) * histYLineDiff), ImColor(violinAttributePlots[i].drawListLineColors[0]), violinPlotAttributeSettings.violinPlotThickness);
									}
								}
							}
						}
						//optimizeViolinSidesAndAssignCustColors();

						ImGui::PopClipRect();
						leftUpperCorner.x += size.x + violinPlotAttributeSettings.violinPlotXSpacing;
					}

					if (drawState == ViolinDrawStateAll || drawState == ViolinDrawStateLine) done = true;
					if (drawState == ViolinDrawStateArea) drawState = ViolinDrawStateLine;
				}
				ImGui::PopItemWidth();
				
				//drawing min texts
				if (violinPlotAttributeSettings.showViolinPlotsMinMax) {
					ImGui::SetCursorPosY(leftUpperCorner.y + size.y);
					int c = 0;
					int c1 = 0;
					float xGap = (ImGui::GetWindowContentRegionWidth() - (amtOfAttributes - 1) * violinPlotAttributeSettings.violinPlotXSpacing) / amtOfAttributes + violinPlotAttributeSettings.violinPlotXSpacing;
					for (uint32_t j : violinAttributePlots[i].attributeOrder) {
						if (!violinAttributePlots[i].activeAttributes[j]) {
							c++;
							continue;
						}

						if (c1 != 0) {
							ImGui::SameLine(c1 * xGap + 10);
						}
						ImGui::Text("%s", (std::to_string(pcAttributes[j].min)).c_str());

						c++;
						c1++;
					}
				}
				//else
				//	ImGui::SetCursorPosY(leftUpperCorner.y + size.y);
				ImGui::EndChild();
				//drag and drop drawlists onto this plot child to add it to this violin plot
				if (ImGui::BeginDragDropTarget()) {
					if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Drawlist")) {
						for (int drawIndex : pcPlotSelectedDrawList) {
							auto dl = g_PcPlotDrawLists.begin();
							for (int iter = 0; iter < drawIndex; ++iter) ++dl;
							violinAttributePlotAddDrawList(violinAttributePlots[i], *dl, i);
						}
					}
					ImGui::EndDragDropTarget();
				}
			}

			//adding new Plots
			ImGui::SetCursorPosX(ImGui::GetWindowWidth() / 2 - plusWidth / 2);
			if (ImGui::Button("+", ImVec2(plusWidth, 0))) {
				//ViolinPlot *currVP = new ViolinPlot();
				violinAttributePlots.emplace_back();// *currVP);
				violinAttributePlots.back().activeAttributes = nullptr;
				//operator delete(currVP);
				//currVP = nullptr;
			}

			ImGui::End(); 
		}

		//begin of violin plots drawlist major --------------------------------------------------------------------------
		if (violinPlotDrawlistSettings.enabled) {
			ImGui::Begin("Violin drawlist window", &violinPlotDrawlistSettings.enabled, ImGuiWindowFlags_MenuBar);
			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("Save Settings")) {
					addSaveSettingsMenu<ViolinSettings>(&violinPlotDrawlistSettings, "ViolinDrawlist", "violidrawlist");
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Settings")) {
					ImGui::Checkbox("Couple to Brushing", &violinPlotDrawlistSettings.coupleViolinPlots);
					ImGui::Checkbox("Send to iso renderer on select", &violinPlotDrawlistSettings.violinPlotDLSendToIso);
					ImGui::SliderInt("Violin plots height", &violinPlotDrawlistSettings.violinPlotHeight, 1, 4000);
					ImGui::SliderInt("Violin plots x spacing", &violinPlotDrawlistSettings.violinPlotXSpacing, 0, 40);
					ImGui::SliderFloat("Violin plots line thickness", &violinPlotDrawlistSettings.violinPlotThickness, 0, 10);
					ImGui::ColorEdit4("Violin plots background", &violinPlotDrawlistSettings.violinBackgroundColor.x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
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
								if (violinPlotDrawlistSettings.renderOrderDLConsider && ((drawL == 0) || (!violinPlotDrawlistSettings.renderOrderBasedOnFirstDL)))
								{
									//if (!renderOrderDLReverse) {
										drawListPlot.attributeOrder[drawL] = sortHistogram(hist, drawListPlot, violinPlotDrawlistSettings.renderOrderDLConsider, violinPlotDrawlistSettings.renderOrderDLReverse);

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
						updateAllViolinPlotMaxValues(violinPlotDrawlistSettings.renderOrderBasedOnFirstDL);
                        for (unsigned int cpdlI; cpdlI < violinDrawlistPlots.size(); ++cpdlI){updateHistogramComparisonDL(cpdlI);}
					}
					static float stdDev = 1.5;
					if (ImGui::SliderFloat("Smoothing kernel stdDev", &stdDev, 0, 25)) {
						histogramManager->setSmoothingKernelSize(stdDev);
						updateAllViolinPlotMaxValues(violinPlotDrawlistSettings.renderOrderBasedOnFirstDL);
                        for (unsigned int cpdlI; cpdlI < violinDrawlistPlots.size(); ++cpdlI){updateHistogramComparisonDL(cpdlI);}
					}
					static char const* violinYs[] = { "Standard","Local brush","Global brush","All brushes" };

					ImGui::Columns(2);
					if (ImGui::BeginCombo("Y Scale", violinYs[violinPlotDrawlistSettings.violinYScale])) {
						for (int v = 0; v < 4; ++v) {
							if (ImGui::MenuItem(violinYs[v])) {
								violinPlotDrawlistSettings.violinYScale = (ViolinYScale)v;
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
					ImGui::Checkbox("Overlay lines", &violinPlotDrawlistSettings.violinPlotOverlayLines);
					ImGui::NextColumn();
					ImGui::Checkbox("Base render order on first DL", &violinPlotDrawlistSettings.renderOrderBasedOnFirstDL);
					ImGui::NextColumn();
					ImGui::Checkbox("Optimize render order", &violinPlotDrawlistSettings.renderOrderDLConsider);
					ImGui::Columns(2);
					if (ImGui::Checkbox("Reverse render order", &violinPlotDrawlistSettings.renderOrderDLReverse)) {
						for (unsigned int i = 0; i < violinDrawlistPlots.size(); ++i) {
							for (int jj = 0; jj < violinDrawlistPlots[i].drawLists.size(); ++jj) {
								HistogramManager::Histogram& hist = histogramManager->getHistogram(violinDrawlistPlots[i].drawLists[jj]);
								(violinPlotDrawlistSettings.renderOrderDLConsider && ((jj == 0) || (!violinPlotDrawlistSettings.renderOrderBasedOnFirstDL))) ? violinDrawlistPlots[i].attributeOrder[jj] = sortHistogram(hist, violinDrawlistPlots[i], violinPlotDrawlistSettings.renderOrderDLConsider, violinPlotDrawlistSettings.renderOrderDLReverse) : violinDrawlistPlots[i].attributeOrder[jj] = violinDrawlistPlots[i].attributeOrder[0];
							}
						}
					}
					ImGui::NextColumn();
					ImGui::Checkbox("Optimize non-stop", &violinPlotDrawlistSettings.renderOrderDLConsiderNonStop);

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

					static char const* plotPositions[] = { "Left","Right","Middle","Middle|Left","Middle|Right","Left|Half","Right|Half" };
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
					static char const* violinScales[] = { "Self","Local","Global","Global Attribute" };
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
					if (ImGui::Checkbox("ChangeLogScale", &violinPlotDrawlistSettings.logScaleDLGlobal)) {
						for (unsigned int i = 0; i < violinDrawlistPlots.size(); ++i) {
							for (int j = 0; j < violinDrawlistPlots[i].attributeFillColors.size(); ++j) {
								//if (histogramManager->logScale[j]) {
								(histogramManager->logScale[j]) = violinPlotDrawlistSettings.logScaleDLGlobal;
								//}
							}
							histogramManager->updateSmoothedValues();
							updateAllViolinPlotMaxValues(violinPlotDrawlistSettings.renderOrderBasedOnFirstDL);
							for (int jj = 0; jj < violinDrawlistPlots[i].drawLists.size(); ++jj) {
								HistogramManager::Histogram& hist = histogramManager->getHistogram(violinDrawlistPlots[i].drawLists[jj]);
								(violinPlotDrawlistSettings.renderOrderDLConsider && ((jj == 0) || (!violinPlotDrawlistSettings.renderOrderBasedOnFirstDL))) ? violinDrawlistPlots[i].attributeOrder[jj] = sortHistogram(hist, violinDrawlistPlots[i], violinPlotDrawlistSettings.renderOrderDLConsider, violinPlotDrawlistSettings.renderOrderDLReverse) : violinDrawlistPlots[i].attributeOrder[jj] = violinDrawlistPlots[i].attributeOrder[0];
							}
							
						}
                        for (unsigned int cpdlI; cpdlI < violinDrawlistPlots.size(); ++cpdlI){updateHistogramComparisonDL(cpdlI);}
					}

					ImGui::EndMenu();

					
				}
				addExportMenu();
				ImGui::EndMenuBar();
			}

			const static int plusWidth = 100;
			for (unsigned int i = 0; i < violinDrawlistPlots.size(); ++i) {
				float absHeight = violinPlotDrawlistSettings.violinPlotHeight;
				static bool settingsOpen = false;
				if(settingsOpen) absHeight += (violinDrawlistPlots[i].attributeNames.size() + 2) * (ImGui::GetTextLineHeightWithSpacing());
				ImGui::BeginChild(std::to_string(i).c_str(), ImVec2(-1, absHeight), true);
				ImGui::PushItemWidth(150);
				if (ImGui::CollapsingHeader("Attribute settings", &settingsOpen)) {
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
					//general settings
					static bool all_active = true;
					ImGui::PushStyleColor(0, { 1,1,0,1 });
					if (ImGui::Checkbox("General Settings", &all_active)) {
						for (int j = 0; j < violinDrawlistPlots[i].attributeNames.size(); ++j)
							violinDrawlistPlots[i].activeAttributes[j] = all_active;
						updateHistogramComparisonDL(i);
					}
					ImGui::PopStyleColor();
					ImGui::NextColumn();
					static int general_plot_pos = 0;
					static char const* plotPositions[] = { "Left","Right","Middle","Middle|Left","Middle|Right","Left|Half","Right|Half" };
					if (ImGui::BeginCombo("##generalpos", plotPositions[general_plot_pos])) {
						for (int k = 0; k < 7; ++k) {
							if (ImGui::MenuItem(plotPositions[k])) {
								general_plot_pos = k;
								for (int j = 0; j < violinDrawlistPlots[i].attributeNames.size(); ++j) {
									violinDrawlistPlots[i].attributePlacements[j] = (ViolinPlacement)general_plot_pos;
								}
							}
						}
						ImGui::EndCombo();
					}
					ImGui::NextColumn();
					static char const* violinScales[] = { "Self","Local","Global","Global Attribute" };
					static int general_plot_scale = 0;
					if (ImGui::BeginCombo("##generalscale", violinScales[general_plot_scale])) {
						for (int k = 0; k < 4; ++k) {
							if (ImGui::MenuItem(violinScales[k])) {
								general_plot_scale = k;
								for (int j = 0; j < violinDrawlistPlots[i].attributeNames.size(); ++j) {
									violinDrawlistPlots[i].violinScalesX[j] = (ViolinScale)general_plot_scale;
								}
								updateHistogramComparisonDL(i);
							}
						}
						ImGui::EndCombo();
					}
					ImGui::NextColumn();
					static float general_plot_multiplier = 1;
					if (ImGui::SliderFloat("##generalmultiplier", &general_plot_multiplier, 0, 1)) {
						for (int j = 0; j < violinDrawlistPlots[i].attributeNames.size(); ++j) {
							violinDrawlistPlots[i].attributeScalings[j] = general_plot_multiplier;
							for (int jj = 0; jj < violinDrawlistPlots[i].drawLists.size(); ++jj) {
								std::vector<std::pair<uint32_t, float>> area;
								HistogramManager::Histogram& hist = histogramManager->getHistogram(violinDrawlistPlots[i].drawLists[jj]);
								(violinPlotDrawlistSettings.renderOrderDLConsider && ((jj == 0) || (!violinPlotDrawlistSettings.renderOrderBasedOnFirstDL))) ? violinDrawlistPlots[i].attributeOrder[jj] = sortHistogram(hist, violinDrawlistPlots[i], violinPlotDrawlistSettings.renderOrderDLConsider, violinPlotDrawlistSettings.renderOrderDLReverse) : violinDrawlistPlots[i].attributeOrder[jj] = violinDrawlistPlots[i].attributeOrder[0];
							}
						}
						updateHistogramComparisonDL(i);
					}
					ImGui::NextColumn();
					static bool general_log = false;
					if (ImGui::Checkbox("##generallog", &general_log)) {
						for (int j = 0; j < violinDrawlistPlots[i].attributeNames.size(); ++j) {
							histogramManager->logScale[j] = general_log;
						}
						histogramManager->updateSmoothedValues();
						updateAllViolinPlotMaxValues(violinPlotDrawlistSettings.renderOrderBasedOnFirstDL);
						for (int jj = 0; jj < violinDrawlistPlots[i].drawLists.size(); ++jj) {
							HistogramManager::Histogram& hist = histogramManager->getHistogram(violinDrawlistPlots[i].drawLists[jj]);
							(violinPlotDrawlistSettings.renderOrderDLConsider && ((jj == 0) || (!violinPlotDrawlistSettings.renderOrderBasedOnFirstDL))) ? violinDrawlistPlots[i].attributeOrder[jj] = sortHistogram(hist, violinDrawlistPlots[i], violinPlotDrawlistSettings.renderOrderDLConsider, violinPlotDrawlistSettings.renderOrderDLReverse) : violinDrawlistPlots[i].attributeOrder[jj] = violinDrawlistPlots[i].attributeOrder[0];
						}
						updateHistogramComparisonDL(i);
					}
					ImGui::NextColumn();
					static ImVec4 general_col = { 0,0,0,1 };
					if (ImGui::ColorEdit4("##general_linecol", &general_col.x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar)) {
						for (int j = 0; j < violinDrawlistPlots[i].attributeNames.size(); ++j) {
							violinDrawlistPlots[i].attributeLineColors[j] = general_col;
						}
					}
					ImGui::NextColumn();
					static ImVec4 general_col_fill = { 0,0,0,.1 };
					if (ImGui::ColorEdit4("##general_fillcol", &general_col_fill.x, ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar)) {
						for (int j = 0; j < violinDrawlistPlots[i].attributeNames.size(); ++j) {
							violinDrawlistPlots[i].attributeFillColors[j] = general_col_fill;
						}
					}
					ImGui::NextColumn();
					ImGui::Separator();
					//settings for the attributes
					for (unsigned int j = 0; j < violinDrawlistPlots[i].attributeNames.size(); ++j) {
						if (ImGui::Checkbox(violinDrawlistPlots[i].attributeNames[j].c_str(), &violinDrawlistPlots[i].activeAttributes[j]))
						{
							updateHistogramComparisonDL(i);
						}
						ImGui::NextColumn();
						if (ImGui::BeginCombo(("##Position" + std::to_string(j)).c_str(), plotPositions[violinDrawlistPlots[i].attributePlacements[j]])) {
							for (int k = 0; k < 7; ++k) {
								if (ImGui::MenuItem(plotPositions[k], nullptr)) {
									violinDrawlistPlots[i].attributePlacements[j] = (ViolinPlacement)k;
								}
							}
							ImGui::EndCombo();
						}

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
								(violinPlotDrawlistSettings.renderOrderDLConsider && ((jj == 0) || (!violinPlotDrawlistSettings.renderOrderBasedOnFirstDL))) ? violinDrawlistPlots[i].attributeOrder[jj] = sortHistogram(hist, violinDrawlistPlots[i], violinPlotDrawlistSettings.renderOrderDLConsider, violinPlotDrawlistSettings.renderOrderDLReverse) : violinDrawlistPlots[i].attributeOrder[jj] = violinDrawlistPlots[i].attributeOrder[0];
							}
							updateHistogramComparisonDL(i);
						}
						ImGui::NextColumn();
						if (ImGui::Checkbox(("##log" + std::to_string(j)).c_str(), &histogramManager->logScale[j])) {
							histogramManager->updateSmoothedValues();
							updateAllViolinPlotMaxValues(violinPlotDrawlistSettings.renderOrderBasedOnFirstDL);
							for (int jj = 0; jj < violinDrawlistPlots[i].drawLists.size(); ++jj) {
								HistogramManager::Histogram& hist = histogramManager->getHistogram(violinDrawlistPlots[i].drawLists[jj]);
								(violinPlotDrawlistSettings.renderOrderDLConsider && ((jj == 0) || (!violinPlotDrawlistSettings.renderOrderBasedOnFirstDL))) ? violinDrawlistPlots[i].attributeOrder[jj] = sortHistogram(hist, violinDrawlistPlots[i], violinPlotDrawlistSettings.renderOrderDLConsider, violinPlotDrawlistSettings.renderOrderDLReverse) : violinDrawlistPlots[i].attributeOrder[jj] = violinDrawlistPlots[i].attributeOrder[0];
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
				}
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

				if ((ImGui::Button("Optimize sides <right/left>")) || (violinPlotDrawlistSettings.violinPlotDLReplaceNonStop)) {
					if (violinDrawlistPlots[i].drawLists.size() != 0) {
						// Only compute the order for the first histogram in the list (the first one in the matrix. Is that the same?)
						violinAdaptSidesAutoObj.vdlp = &(violinDrawlistPlots[i]);
						violinAdaptSidesAutoObj.optimizeSidesNowDL = true;
					}
				}
				ImGui::NextColumn();
				ImGui::Checkbox("", &violinPlotDrawlistSettings.violinPlotDLInsertCustomColors);
				ImGui::SameLine(50);
				if (ImGui::BeginCombo("##appcoldraw" ,"Apply colors of Dark2YellowSplit")) {
					std::vector<std::string> *availablePalettes =
						violinDrawlistPlots[i].colorPaletteManager->colorPalette->getQualPaletteNames();

					std::vector<const char*>  vc = convertStringVecToConstChar(availablePalettes);

					if (vc.size() > 0) {
						//static char* violinYs[] = { "Standard","Local brush","Global brush","All brushes" };
						if (ImGui::BeginCombo("Line Palette", vc[violinPlotDrawlistSettings.autoColorAssingLine])) {
							for (int v = 0; v < vc.size(); ++v) {
								if (ImGui::MenuItem(vc[v])) {
									violinDrawlistPlots[i].colorPaletteManager->chosenAutoColorPaletteLine =
										(*availablePalettes)[v];
									violinPlotDrawlistSettings.autoColorAssingLine = v;
								}
							}
							ImGui::EndCombo();
						}
						if (ImGui::BeginCombo("Fill Palette", vc[violinPlotDrawlistSettings.autoColorAssingFill])) {
							for (int v = 0; v < vc.size(); ++v) {
								if (ImGui::MenuItem(vc[v])) {
									violinDrawlistPlots[i].colorPaletteManager->chosenAutoColorPaletteFill =
										(*availablePalettes)[v];
									violinPlotDrawlistSettings.autoColorAssingFill = v;
								}
							}
							ImGui::EndCombo();
						}
					}
					ImGui::EndCombo();
				}


				ImGui::NextColumn();
				ImGui::Checkbox("Re-place constantly", &violinPlotDrawlistSettings.violinPlotDLReplaceNonStop);
				ImGui::NextColumn();
				ImGui::Checkbox("Consider blending order", &violinPlotDrawlistSettings.violinPlotDLConsiderBlendingOrder);
				ImGui::NextColumn();
				if (ImGui::Checkbox("Reverse color pallette", &violinPlotDrawlistSettings.violinPlotDLReverseColorPallette))
				{
					violinDrawlistPlots[i].colorPaletteManager->setReverseColorOrder(violinPlotDrawlistSettings.violinPlotDLReverseColorPallette);
				}
                ImGui::Columns(2);
				if (ImGui::Button("Fix order and colors"))
				{
					violinPlotDrawlistSettings.violinPlotDLReplaceNonStop = false;
					violinDrawlistPlots[i].colorPaletteManager->useColorPalette = false;
					violinPlotDrawlistSettings.renderOrderDLConsiderNonStop = false;
					violinPlotDrawlistSettings.renderOrderDLConsider = false;
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
                    if (j % 10 != 0)ImGui::SameLine();	//only 10 buttons per line
                    // String of the draggable button to drag in a dl into one position of the violin matrix
                    ImGui::Button(violinDrawlistPlots[i].drawLists[j].c_str());
					if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
						int p[] = { -1,j };		//holding the index in the pcAttriOrd array and the value of it
						ImGui::SetDragDropPayload("ViolinDrawlist", p, sizeof(p));
                        // Name shown during drag&drop event
                        ImGui::Text("%s", violinDrawlistPlots[i].drawLists[j].c_str());
						ImGui::EndDragDropSource();
					}

					if (ImGui::IsItemHovered()) {
						ImGui::BeginTooltip();
						ImGui::Text("right click to order plot matrix with distance to this drawlist.");
						ImGui::EndTooltip();
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
				ImVec2 size((ImGui::GetWindowContentRegionWidth() - (violinDrawlistPlots[i].matrixSize.second - 1) * violinPlotDrawlistSettings.violinPlotXSpacing) / violinDrawlistPlots[i].matrixSize.second, (ImGui::GetWindowContentRegionMax().y - leftUpperCorner.y + ImGui::GetWindowPos().y - violinDrawlistPlots[i].matrixSize.first * ImGui::GetFrameHeightWithSpacing()) / (float)violinDrawlistPlots[i].matrixSize.first);
				ViolinDrawState drawState = (violinPlotDrawlistSettings.violinPlotOverlayLines) ? ViolinDrawStateArea : ViolinDrawStateAll;
				bool done = false;
				while (!done) {
					leftUpperCorner = leftUpperCornerStart;
					for (int x = 0; x < violinDrawlistPlots[i].matrixSize.first; ++x) {	//Drawing the plots per matrix entry
						for (int y = 0; y < violinDrawlistPlots[i].matrixSize.second; ++y) {
							int j = violinDrawlistPlots[i].drawListOrder[x * violinDrawlistPlots[i].matrixSize.second + y];

							ImVec2 framePos = leftUpperCorner;
							framePos.y += ImGui::GetFrameHeightWithSpacing();
							if(drawState == ViolinDrawStateAll || drawState == ViolinDrawStateArea) ImGui::RenderFrame(framePos, framePos + size, ImGui::GetColorU32(violinPlotDrawlistSettings.violinBackgroundColor), true, ImGui::GetStyle().FrameRounding);
							ImGui::SetCursorScreenPos(framePos);
							if (size.x > 0 && size.y > 0) {	//safety check. ImGui crahes when button size is 0
								if (io.KeyCtrl) {
									ImGui::PushStyleColor(ImGuiCol_Button, { 0,0,0,0 });
									if (ImGui::Button(("##invBut" + std::to_string(x * violinDrawlistPlots[i].matrixSize.second + y)).c_str(), size) && j!= 0xffffffff) {
										if (violinDrawlistPlots[i].selectedDrawlists.find(j) == violinDrawlistPlots[i].selectedDrawlists.end()) {
											violinDrawlistPlots[i].selectedDrawlists.insert(j);
											if (isoSurfSettings.enabled && violinPlotDrawlistSettings.violinPlotDLSendToIso) {
												std::vector<std::pair<float, float>> posBounds(3);
												for (int i = 0; i < 3; ++i) {
													posBounds[i].first = pcAttributes[isoSurfSettings.posIndices[i]].min;
													posBounds[i].second = pcAttributes[isoSurfSettings.posIndices[i]].max;
												}
												DrawList* dl;
												for (auto& draw : g_PcPlotDrawLists) {
													if (violinDrawlistPlots[i].drawLists[j] == draw.name) {
														dl = &draw;
														break;
													}
												}
												DataSet* ds;
												for (auto& d : g_PcPlotDataSets) {
													if (d.name == dl->parentDataSet) {
														ds = &d;
													}
												}
												auto& xDim = getDimensionValues(*ds, isoSurfSettings.posIndices.x), yDim = getDimensionValues(*ds, isoSurfSettings.posIndices.y), zDim = getDimensionValues(*ds, isoSurfSettings.posIndices.z);
												uint32_t w = xDim.second.size();
												uint32_t h = yDim.second.size();
												uint32_t d = zDim.second.size();;
												bool regularDim[3]{ xDim.first, yDim.first, zDim.first };
												int index = -1;
												for (int in = 0; in < isoSurfaceRenderer->drawlistBrushes.size(); ++in) {
													if (isoSurfaceRenderer->drawlistBrushes[in].drawlist == dl->name && isoSurfaceRenderer->drawlistBrushes[in].brush == "") {
														index = in;
														break;
													}
												}
												if (index == -1) {
													glm::vec4 isoColor;
													(isoSurfaceRenderer->drawlistBrushes.size() == 0) ? isoColor = { 0,1,0, 0.627 } : isoColor = { 1,0,1,0.627 };
													isoSurfaceRenderer->drawlistBrushes.push_back({ dl->name, "",isoColor, {w, h, d} });
												}
												isoSurfaceRenderer->update3dBinaryVolume(xDim.second, yDim.second, zDim.second, &isoSurfSettings.posIndices.x, posBounds, pcAttributes.size(), ds->data.size(), dl->buffer, dl->activeIndicesBufferView, dl->indices.size(), dl->indicesBuffer, regularDim, index);
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
									for (int drawIndex : pcPlotSelectedDrawList) {
										auto dl = g_PcPlotDrawLists.begin();
										for (int iter = 0; iter < drawIndex; ++iter) ++dl;
										violinDrawListPlotAddDrawList(violinDrawlistPlots[i], *dl, i);
									}
									for (int selectIndex = 0; selectIndex < pcPlotSelectedDrawList.size(); ++selectIndex) {
										if (x * violinDrawlistPlots[i].matrixSize.second + y + selectIndex >= violinDrawlistPlots[i].drawListOrder.size()) break;
										violinDrawlistPlots[i].drawListOrder[x * violinDrawlistPlots[i].matrixSize.second + y + selectIndex] = violinDrawlistPlots[i].drawLists.size() + selectIndex - pcPlotSelectedDrawList.size();
									}
									
								}
								ImGui::EndDragDropTarget();
							}

							// if the current violin plot is selected draw a rect around it
							if (violinDrawlistPlots[i].selectedDrawlists.find(j) != violinDrawlistPlots[i].selectedDrawlists.end()) {
								ImGui::GetWindowDrawList()->AddRect(framePos, framePos + size, IM_COL32(255,200,0,255), ImGui::GetStyle().FrameRounding,ImDrawCornerFlags_All,5);
							}
							if (j == 0xffffffff) {
								leftUpperCorner.x += size.x + violinPlotDrawlistSettings.violinPlotXSpacing;
								continue;
							}
							ImVec2 textPos = framePos;
							textPos.y -= ImGui::GetTextLineHeight();
                            if (violinPlotDLIdxInListForHistComparison[i] != -1){textPos.y -= 1.1*ImGui::GetTextLineHeight();}
							ImGui::SetCursorScreenPos(textPos);

                            // Here, the text above each MPVP is written.


							ImGui::Text("%s", violinDrawlistPlots[i].drawLists[j].c_str());
                            if (violinPlotDLIdxInListForHistComparison[i] != -1){
                                ImVec2 textPosCurr = textPos;
                                textPosCurr.y += 1.1*ImGui::GetTextLineHeight();
                                ImGui::SetCursorScreenPos(textPosCurr);
                                ImGui::Text("%s", std::to_string(violinDrawlistPlots[i].histDistToRepresentative[j]).c_str());
                            }

							ImGui::PushClipRect(framePos, framePos + size, false);
							HistogramManager::Histogram& hist = histogramManager->getHistogram(violinDrawlistPlots[i].drawLists[j]);
							DrawList* dl = nullptr;
							if (violinPlotDrawlistSettings.violinYScale == ViolinYScaleLocalBrush || violinPlotDrawlistSettings.violinYScale == ViolinYScaleBrushes) {
								for (DrawList& draw : g_PcPlotDrawLists) {
									if (draw.name == violinDrawlistPlots[i].drawLists[j]) {
										dl = &draw;
									}
								}
							}
							//std::vector<std::pair<float, float>> localMinMax = std::vector<std::pair<float,float>>(pcAttributes.size(), { std::numeric_limits<float>().max(),std::numeric_limits<float>().min() });
							if (violinPlotDrawlistSettings.violinYScale == ViolinYScaleLocalBrush || violinPlotDrawlistSettings.violinYScale == ViolinYScaleBrushes) {
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
							bool mouseHover = framePos.x <= ImGui::GetMousePos().x && framePos.x + size.x >= ImGui::GetMousePos().x && framePos.y <= ImGui::GetMousePos().y && framePos.y + size.y >= ImGui::GetMousePos().y;
							float lineMultiplier = 1;
							for (int k : violinDrawlistPlots[i].attributeOrder[j]) {
								if (!violinDrawlistPlots[i].activeAttributes[k]) continue;

								float histYStart;
								float histYEnd;
								switch (violinPlotDrawlistSettings.violinYScale) {
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
										//hover check
										if (mouseHover) {
											float p = ((1 - ImGui::GetMousePos().y + framePos.y + histYEnd) / (histYEnd - histYStart));
											float v = getBinVal(p, hist.bins[k]);
											float lineX = framePos.x + v / div * size.x;
											float dist = std::abs(lineX - ImGui::GetMousePos().x);
											if (dist < LINEDISTANCE) {
												lineMultiplier = LINEMULTIPLIER;
												mouseHover = false;
												float data = p * pcAttributes[k].max + (1 - p) * pcAttributes[k].min;
												ImGui::BeginTooltip();
												ImGui::Text("%s {%f[val], %f[amt]}", pcAttributes[k].name.c_str(), data, v);
												ImGui::EndTooltip();
											}
										}
										ImGui::GetWindowDrawList()->AddLine(ImVec2(framePos.x, histYLineEnd),
											ImVec2(framePos.x + hist.bins[k][0] / div * size.x, histYLineEnd), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness);
										for (int l = 1; l < hist.bins[k].size(); ++l) {
											ImGui::GetWindowDrawList()->AddLine(ImVec2(framePos.x + hist.bins[k][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[k].size() - 1) * histYLineDiff),
												ImVec2(framePos.x + hist.bins[k][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[k].size() - 1) * histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness * lineMultiplier);
										}
										ImGui::GetWindowDrawList()->AddLine(ImVec2(framePos.x, histYLineEnd - histYLineDiff),
											ImVec2(framePos.x + hist.bins[k][hist.bins[k].size() - 1] / div * size.x, histYLineEnd - histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness);
										lineMultiplier = 1;
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
										if (mouseHover) {
											float p = ((1 - ImGui::GetMousePos().y + framePos.y + histYEnd) / (histYEnd - histYStart));
											float v = getBinVal(p, hist.bins[k]);
											float lineX = framePos.x + size.x - v / div * size.x;
											float dist = std::abs(lineX - ImGui::GetMousePos().x);
											if (dist < LINEDISTANCE) {
												lineMultiplier = LINEMULTIPLIER;
												mouseHover = false;
												float data = p * pcAttributes[k].max + (1 - p) * pcAttributes[k].min;
												ImGui::BeginTooltip();
												ImGui::Text("%s {%f[val], %f[amt]}", pcAttributes[k].name.c_str(), data, v);
												ImGui::EndTooltip();
											}
										}
										ImGui::GetWindowDrawList()->AddLine(ImVec2(framePos.x + size.x, histYLineEnd),
											ImVec2(framePos.x + size.x - hist.bins[k][0] / div * size.x, histYLineEnd), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness);
										for (int l = 1; l < hist.bins[k].size(); ++l) {
											ImGui::GetWindowDrawList()->AddLine(ImVec2(framePos.x + size.x - hist.bins[k][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[k].size() - 1) * histYLineDiff),
												ImVec2(framePos.x + size.x - hist.bins[k][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[k].size() - 1) * histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness * lineMultiplier);
										}
										ImGui::GetWindowDrawList()->AddLine(ImVec2(framePos.x + size.x, histYLineEnd - histYLineDiff),
											ImVec2(framePos.x + size.x - hist.bins[k][hist.bins[k].size() - 1] / div * size.x, histYLineEnd - histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness);
										lineMultiplier = 1;
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
										if (mouseHover) {
											float p = ((1 - ImGui::GetMousePos().y + framePos.y + histYEnd) / (histYEnd - histYStart));
											float v = getBinVal(p, hist.bins[k]);
											float lineX = xBase + .5f * v / div * size.x;
											float lineX2 = xBase - .5f * v / div * size.x;
											float dist = std::min(std::abs(lineX - ImGui::GetMousePos().x), std::abs(lineX2 - ImGui::GetMousePos().x));
											if (dist < LINEDISTANCE) {
												lineMultiplier = LINEMULTIPLIER;
												mouseHover = false;
												float data = p * pcAttributes[k].max + (1 - p) * pcAttributes[k].min;
												ImGui::BeginTooltip();
												ImGui::Text("%s {%f[val], %f[amt]}", pcAttributes[k].name.c_str(), data, v);
												ImGui::EndTooltip();
											}
										}
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase + hist.bins[k][0] / div * size.x / 2, histYLineEnd),
											ImVec2(xBase - hist.bins[k][0] / div * size.x / 2, histYLineEnd), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness);
										for (int l = 1; l < hist.bins[k].size(); ++l) {
											//left Line
											ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase - .5f * hist.bins[k][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[k].size() - 1) * histYLineDiff),
												ImVec2(xBase - .5f * hist.bins[k][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[k].size() - 1) * histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness * lineMultiplier);
											//right Line
											ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase + .5f * hist.bins[k][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[k].size() - 1) * histYLineDiff),
												ImVec2(xBase + .5f * hist.bins[k][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[k].size() - 1) * histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness * lineMultiplier);
										}
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase + hist.bins[k][hist.bins[k].size() - 1] / div * size.x / 2, histYLineEnd - histYLineDiff),
											ImVec2(xBase - hist.bins[k][hist.bins[k].size() - 1] / div * size.x / 2, histYLineEnd - histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness);
										lineMultiplier = 1;
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
										if (mouseHover) {
											float p = ((1 - ImGui::GetMousePos().y + framePos.y + histYEnd) / (histYEnd - histYStart));
											float v = getBinVal(p, hist.bins[k]);
											float lineX = xBase - .5f * v / div * size.x;
											float dist = std::abs(lineX - ImGui::GetMousePos().x);
											if (dist < LINEDISTANCE) {
												lineMultiplier = LINEMULTIPLIER;
												mouseHover = false;
												float data = p * pcAttributes[k].max + (1 - p) * pcAttributes[k].min;
												ImGui::BeginTooltip();
												ImGui::Text("%s {%f[val], %f[amt]}", pcAttributes[k].name.c_str(), data, v);
												ImGui::EndTooltip();
											}
										}
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase, histYLineEnd),
											ImVec2(xBase - hist.bins[k][0] / div * size.x / 2, histYLineEnd), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness);
										for (int l = 1; l < hist.bins[k].size(); ++l) {
											//left Line
											ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase - .5f * hist.bins[k][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[k].size() - 1) * histYLineDiff),
												ImVec2(xBase - .5f * hist.bins[k][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[k].size() - 1) * histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness * lineMultiplier);
										}
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase, histYLineEnd - histYLineDiff),
											ImVec2(xBase - hist.bins[k][hist.bins[k].size() - 1] / div * size.x / 2, histYLineEnd - histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness);
										//right Line
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase, framePos.y), ImVec2(xBase, framePos.y + size.y), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness);
										lineMultiplier = 1;
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
										if (mouseHover) {
											float p = ((1 - ImGui::GetMousePos().y + framePos.y + histYEnd) / (histYEnd - histYStart));
											float v = getBinVal(p, hist.bins[k]);
											float lineX = xBase + .5f * v / div * size.x;
											float dist = std::abs(lineX - ImGui::GetMousePos().x);
											if (dist < LINEDISTANCE) {
												lineMultiplier = LINEMULTIPLIER;
												mouseHover = false;
												float data = p * pcAttributes[k].max + (1 - p) * pcAttributes[k].min;
												ImGui::BeginTooltip();
												ImGui::Text("%s {%f[val], %f[amt]}", pcAttributes[k].name.c_str(), data, v);
												ImGui::EndTooltip();
											}
										}
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase + hist.bins[k][0] / div * size.x / 2, histYLineEnd),
											ImVec2(xBase, histYLineEnd), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness);
										for (int l = 1; l < hist.bins[k].size(); ++l) {
											//right Line
											ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase + .5f * hist.bins[k][l - 1] / div * size.x, histYLineEnd - (l - 1.0f) / (hist.bins[k].size() - 1) * histYLineDiff),
												ImVec2(xBase + .5f * hist.bins[k][l] / div * size.x, histYLineEnd - ((float)l) / (hist.bins[k].size() - 1) * histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness * lineMultiplier);
										}
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase + hist.bins[k][hist.bins[k].size() - 1] / div * size.x / 2, histYLineEnd - histYLineDiff),
											ImVec2(xBase, histYLineEnd - histYLineDiff), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness);
										//left Line
										ImGui::GetWindowDrawList()->AddLine(ImVec2(xBase, framePos.y), ImVec2(xBase, framePos.y + size.y), ImColor(violinDrawlistPlots[i].attributeLineColors[k]), violinPlotDrawlistSettings.violinPlotThickness);
										lineMultiplier = 1;
									}
									break;
								}
								}
							}
							optimizeViolinSidesAndAssignCustColors();
							leftUpperCorner.x += size.x + violinPlotDrawlistSettings.violinPlotXSpacing;
							ImGui::PopClipRect();
						}
						leftUpperCorner.x = leftUpperCornerStart.x;
						leftUpperCorner.y += size.y + ImGui::GetFrameHeightWithSpacing();
					}
					
					if (drawState == ViolinDrawStateAll || drawState == ViolinDrawStateLine) done = true;
					if (drawState == ViolinDrawStateArea) drawState = ViolinDrawStateLine;
				}
				ImGui::PopItemWidth();
				ImGui::EndChild() ;
				//drag and drop drawlists onto this plot child to add it to this violin plot
				if (ImGui::BeginDragDropTarget()) {
					if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Drawlist")) {
						for (int drawIndex : pcPlotSelectedDrawList) {
							auto dl = g_PcPlotDrawLists.begin();
							for (int iter = 0; iter < drawIndex; ++iter) ++dl;
							violinDrawListPlotAddDrawList(violinDrawlistPlots[i], *dl, i);
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

		transferFunctionEditor->draw();

		clusteringWorkbench->draw();
		if(clusteringWorkbench->requestPcPlotUpdate){
			pcPlotRender = true;
			updateDrawListIndexBuffer(*clusteringWorkbench->updateDl);
			updateWorkbenchRenderings(*clusteringWorkbench->updateDl);
			clusteringWorkbench->requestPcPlotUpdate = false;
		}

		scatterplotWorkbench->draw();
		// checking for lasso selections update
		if(scatterplotWorkbench->updatedDrawlists.size()){
			for(auto& dlName: scatterplotWorkbench->updatedDrawlists){
				auto curDl = std::find_if(g_PcPlotDrawLists.begin(), g_PcPlotDrawLists.end(), [&](DrawList& dl){return dl.name == dlName;});
				if(curDl != g_PcPlotDrawLists.end())
					updateActiveIndices(*curDl);
			}
			scatterplotWorkbench->updatedDrawlists.clear();
			pcPlotRender = true;		//updating the pcPlot to show new brushed lines
		}

		correlationMatrixWorkbench->draw({&g_PcPlotDrawLists, &pcPlotSelectedDrawList});
		if(correlationMatrixWorkbench->requestUpdate){
			correlationMatrixWorkbench->updateCorrelationScores(g_PcPlotDrawLists);
		}

		compressionWorkbench->draw();

		//checking data from hierarch importer
		for(auto& dl: g_PcPlotDrawLists){
			if(dl.hierarchyImportManager && dl.hierarchyImportManager->newDataLoaded){
				auto ds = std::find_if(g_PcPlotDataSets.begin(), g_PcPlotDataSets.end(), [&](DataSet& ds){return ds.name == dl.parentDataSet;});
				ds->data = dl.hierarchyImportManager->retrieveNewData();
				//todo upload new data, set index list ...
				fillVertexBuffer(ds->buffer, ds->data);
				dl.indices.resize(ds->data.size());
				std::iota(dl.indices.begin(), dl.indices.end(), 0);
				updateActiveIndices(dl);
				pcPlotRender = true;
			}
		}

		pcSettings.rescaleTableColumns = false;

		// Rendering
		ImGui::Render();

		// Image export
		if (g_ExportCountDown >= 0) {
			if (g_ExportCountDown == 0) {
				ImDrawData* exportDrawData = ImGui::GetCurrentContext()->Viewports[g_ExportViewportNumber]->DrawData;
				ImVec2 old_scale = exportDrawData->FramebufferScale;
				exportDrawData->FramebufferScale = { (float)g_ExportImageWidth / exportDrawData->DisplaySize.x, (float)g_ExportImageHeight / exportDrawData->DisplaySize.y };
				VkClearValue clear;
				memcpy(clear.color.float32, &clear_color, 4 * sizeof(float));
				check_vk_result(vkQueueWaitIdle(g_Queue)); // the previous frame draw has to be done
				FrameRenderExport(&g_ExportWindowFrame, exportDrawData, clear);
				//creating the image and copying the frame data to the image
				cimg_library::CImg<unsigned char> res(g_ExportImageWidth, g_ExportImageHeight, 1, 4);
				check_vk_result(vkQueueWaitIdle(g_Queue));
				exportDrawData->FramebufferScale = old_scale;
				unsigned char* img = new unsigned char[g_ExportImageWidth * g_ExportImageHeight * 4];
				VkUtil::downloadImageData(g_Device, g_PhysicalDevice, g_ExportWindowFrame.CommandPool, g_Queue, g_ExportWindowFrame.Backbuffer, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_ExportImageWidth, g_ExportImageHeight, 1, img, g_ExportImageWidth* g_ExportImageHeight * 4 * sizeof(unsigned char));
				//transforming the downloaded image to the correct image coordinates
				for (int x = 0; x < g_ExportImageWidth; ++x) {
					for (int y = 0; y < g_ExportImageHeight; ++y) {
						for (int c = 0; c < 4; ++c) {
							int cc = c;
							if (c != 3) cc = std::abs(c - 2);
							int oldIndex = x * g_ExportImageHeight * 4 + y * 4 + c;
							int newIndex = (cc) * g_ExportImageHeight * g_ExportImageWidth + x * g_ExportImageHeight + y;
							assert(oldIndex < g_ExportImageWidth* g_ExportImageHeight * 4);
							assert(newIndex < g_ExportImageWidth* g_ExportImageHeight * 4);
							
							if(c == 3) res.data()[newIndex] = 255;
							else res.data()[newIndex] = img[oldIndex];
						}
					}
				}
				delete[] img;
				
				res.save_png(g_ExportPath);
			}
			--g_ExportCountDown;
		}

		ImDrawData* main_draw_data = ImGui::GetDrawData();
		const bool main_is_minimized = (main_draw_data->DisplaySize.x <= 0.0f || main_draw_data->DisplaySize.y <= 0.0f);
		memcpy(&wd->ClearValue.color.float32[0], &clear_color, 4 * sizeof(float));
		if (!main_is_minimized)
			FrameRender(wd, main_draw_data);

		// Update and Render additional Platform Windows
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
		}

		// Present Main Platform Window
		if (!main_is_minimized)
			FramePresent(wd, window);
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
	delete transferFunctionEditor;
	for (ViolinPlot& vp : violinAttributePlots) {
		if (vp.activeAttributes) delete[] vp.activeAttributes;
	}
	for (ViolinDrawlistPlot& vp : violinDrawlistPlots) {
		if (vp.attributeNames.size()) delete[] vp.activeAttributes;
	}

	if(compressionWorkbench){
		compressionWorkbench->stopThreads();
	}


	err = vkDeviceWaitIdle(g_Device);
	check_vk_result(err);

	{//section to cleanup pcPlot
		vkDestroySampler(g_Device, g_PcPlotSampler, nullptr);
		cleanupPcPlotCommandPool();
		cleanupPcPlotFramebuffer();
		cleanupPcPlotDataSets();
		cleanupPcPlotPipeline();
		cleanupPcPlotRenderPass();
		cleanupPcPlotImageView();
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
		delete drawListColorPalette;
		delete gpuBrusher;
		delete histogramManager;
		delete scatterplotWorkbench;
		correlationMatrixWorkbench.reset();
		//pcRenderer.reset();
		clusteringWorkbench.reset();

		for (GlobalBrush& gb : globalBrushes) {
			if (gb.kdTree) delete gb.kdTree;
		}
	}

	{//other cleanups
		cleanupExportWindow();
	}

	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplSDL2_Shutdown();
	ImGui::DestroyContext();

	CleanupVulkanWindow();
	CleanupVulkan();

	SDL_DestroyWindow(window);
    SDL_Quit();

	return 0;
}
