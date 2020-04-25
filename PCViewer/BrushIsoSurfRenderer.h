/*
IsoSurfRenderer is a class to support iso surface rendering of 3d multivariate grid data.
The iso surfaces are set by brushes.
*/
#pragma once
#ifndef BrushIsoSurfRenderer_H
#define	BrushIsoSurfRenderer_H

#include "VkUtil.h"
#include "PCUtil.h"
#include <vulkan/vulkan.h>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/euler_angles.hpp"
#include "SpacialData.h"
#include "CameraNav.hpp"
#include <limits.h>
#include <string.h>
#include <map>
#include <vector>
#include <array>
#include <numeric>
#include <fstream>

#define VERTICALPANSPEED .01f
#define HORIZONTALPANSPEED .01f
#define ZOOMSPEED .03f

//#define AMTOF3DTEXTURES 7			//amount of textures used for the density values. The total amount of density attributes is AMTOF3DTEXTURES * 4 (4 density channels per image) NOTE: found out that it is easily possible to write an array of textures to a binding
#define MAXAMTOF3DTEXTURES 30
#define MAXAMTOFBRUSHES 30			//there is a max amount of brushes, as for every brush a integer in the gpu register has to be created
#define LOCALSIZE 256
#define LOCALSIZE3D 8				//patch size for 3d image in each dimension

#define IDX3D(x,y,z,width,height) ((x)+((y)*width)+((z)*width*height))

class BrushIsoSurfRenderer {
public:
	enum BrushIsoSurfRendererError {
		IsoSurfRendererError_Success,
		IsoSurfRendererError_GridDimensionMissmatch
	};

	struct DrawlistBrush {
		std::string drawlist;
		std::string brush;
		glm::vec4 brushSurfaceColor;
		uint32_t gridDimensions[3];			//width, height, depth
	};

	// height :		height of the rendered image
	// width :		width of the rendered image
	// device, pyhsicalDevice ... : valid vulkan instanes of the respective type
	BrushIsoSurfRenderer(uint32_t height, uint32_t width, VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool);
	~BrushIsoSurfRenderer();

	void resize(uint32_t width, uint32_t height);
	void resizeBox(float width, float height, float depth);
	bool update3dBinaryVolume(uint32_t width, uint32_t height, uint32_t depth, uint32_t amtOfAttributes, const std::vector<uint32_t>& densityAttributes, uint32_t positionIndices[3], std::vector<std::pair<float,float>>& posMinMax, VkBuffer data, uint32_t amtOfData, VkBuffer indices, uint32_t amtOfIndices, bool regularGrid);
	void getPosIndices(int index, uint32_t* ind);
	void updateCameraPos(CamNav::NavigationInput input, float deltaT);
	void setCameraPos(glm::vec3& newCameraPos, float **newRotation);
	void getCameraPos(glm::vec3& cameraPosReturn, float **rotationReturn);
	bool updateBrush(std::string& name, std::vector<std::vector<std::pair<float, float>>> minMax);			//this method only updates a brush.
	bool deleteBrush(std::string& name);
	void render();
	void setImageDescriptorSet(VkDescriptorSet descriptor);
	VkDescriptorSet getImageDescriptorSet();
	VkSampler getImageSampler();
	VkImageView getImageView();
	void exportBinaryCsv(std::string path, uint32_t binaryIndex);
	void setBinarySmoothing(float stdDiv);
	void imageBackGroundUpdated();

	std::string activeDrawlist;
	bool shade;
	float stepSize;
	float flySpeed;
	float fastFlyMultiplier;
	float rotationSpeed;
	float isoValue = .5f;
	glm::vec3 lightDir;
	VkClearValue imageBackground;
	std::vector<std::array<float, 4>> firstBrushColors;													//the first brush has for each attribute one color
	std::map<std::string, std::array<float, 4>> brushColors;											//each brush has its own colors

	// camera variables for the GUI are stored here
	glm::vec3 directIsoRendererCameraPositionGLM{};
	float directIsoRendererCameraPosition[3]{};
    float cameraRotationGUI[2]{};

private:
	struct UniformBuffer {
		glm::vec3 camPos;				//cameraPosition in model space
		alignas(16) glm::vec3 faces;	//face positions for intersection tests
		alignas(16) glm::vec3 lightDir;
		alignas(16) glm::mat4 mvp;		//modelViewProjection Matrix
	};

	struct ComputeInfos {
		uint32_t amtOfAttributes;		//amount of attributes in the dataset
		uint32_t amtOfBrushAttributes;	//amount of attributes for which the density maps should be created.
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
		uint32_t padding;				//if used with active ind padding is used to indicate if regular grid should be used
		//int array containing attribute infos:
		//index attr 1, amtOfBrushes 1, offset brush1
		//index attr 2, amtOfBrushes 2, offset brush2
		//...
	};

	struct BinaryComputeInfos {
		uint32_t amtOfAxis;			//This is also the amount of density attributes
		uint32_t maxX;
		uint32_t maxY;
		uint32_t maxZ;
		//float[] brushes structure (a stands for axis):
		//offset a1, offset a2, ..., offset an, a1, a2, ..., an
		//axis structure:
		//amtOfBrushes, b1, b2, ..., bn
		//brush structure:
		//minMax1, minMax2, ..., minMaxN
	};

	struct BrushInfos {		//Note, currently a maximum of 30 brushes is available. For more shader + define in this header have to be changed
		uint32_t amtOfAxis;
		uint32_t shade;
		float stepSize;
		float isoValue;
		uint32_t amtOfBrushes;
		//float[] colors for the brushes:
		//color brush0[4*float], color brush1[4*float], ... , color brush n[4*float]

		//[[Depricated]]float[] brushes structure (a stands for axis):
		//offset a1, offset a2, ..., offset an, a1, a2, ..., an
		//axis structure:
		//amtOfBrushes, offset b1, offset b2, ..., offset bn, b1, b2, ..., bn
		//brush structure:
		//bIndex, amtOfMinMax, color(vec4), minMax1, minMax2, ..., minMaxN
	};

	struct Brush {		//this corresponds to the brush structure above
		uint32_t bIndex;
		std::vector<std::pair<float, float>> minMax;
	};

	struct SmoothUBO {
		uint32_t index;			//position index to be smoothed(0->x, 1->y, 2->z)
		float stdDev;			//standard deviation
		uint32_t padding[2];
	};
	
	//shaderpaths
	static char vertPath[];
	static char fragPath[];
	static char computePath[];

	//general information about the 3d view
	uint32_t imageHeight;
	uint32_t imageWidth;

	//information about the rendered 3dBox
	float boxHeight = .5f;
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
	std::vector<VkSampler> image3dSampler;
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
	VkBuffer			brushBuffer;
	VkDeviceMemory		brushMemory;
	uint32_t			brushByteSize;

	//vulkan resources for the compute pipeline
	VkPipeline			computePipeline;
	VkPipelineLayout	computePipelineLayout;
	VkDescriptorSetLayout computeDescriptorSetLayout;

	std::vector<glm::uvec3> posIndices;
	uint32_t image3dExtent[3]{ 0,0,0 };

	//camera variables
	glm::vec3 cameraPos;		//camera position
	glm::vec2 cameraRot;


	//variables for the brushes
	std::map<std::string, std::vector<std::vector<std::pair<float, float>>>> brushes;		//each brush has a vector of minMax values. Each entry in the vector corresponds to an attribute
	std::vector<float*> attributeColors;													//if only one brush is active every attribute can be assigned a different color
	std::vector<uint32_t> activeDensities;													//vector containing the needed 3d density pictures to reduce amt of density pictures bound to the pipeline

	float smoothStdDiv = 1;

	const uint32_t densityMipLevels = 5;

	uint32_t uboAlignment;

	//private methods
	void smoothImage(int index);

	//methods to instatiate vulkan resources
	void createPrepareImageCommandBuffer();
	void createImageResources();
	void createBuffer();
	void createPipeline();
	void createDescriptorSets();
	void updateDescriptorSet();
	void updateBrushBuffer();
	void updateCommandBuffer();
};
#endif 