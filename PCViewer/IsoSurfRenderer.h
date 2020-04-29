/*
IsoSurfRenderer is a class to support iso surface rendering of 3d multivariate grid data.
The iso surfaces are set by brushes.
*/
#pragma once
#ifndef IsoSurfRenderer_H
#define	IsoSurfRenderer_H

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

class IsoSurfRenderer {
public:
	enum IsoSurfRendererError {
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
	IsoSurfRenderer(uint32_t height, uint32_t width, VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool);
	~IsoSurfRenderer();

	void resize(uint32_t width, uint32_t height);
	void resizeBox(float width, float height, float depth);
	bool update3dBinaryVolume(uint32_t width, uint32_t height, uint32_t depth, uint32_t amtOfAttributes, const std::vector<uint32_t>& densityAttributes,std::vector<std::pair<float, float>>& densityAttributesMinMax, glm::uvec3& positionIndices, std::vector<float*>& data, std::vector<uint32_t>& indices, std::vector<std::vector<std::pair<float, float>>>& brush, int index);
	bool update3dBinaryVolume(uint32_t width, uint32_t height, uint32_t depth, uint32_t amtOfAttributes, const std::vector<uint32_t>& brushAttributes, const std::vector<std::pair<float, float>>& densityAttributesMinMax, glm::uvec3& positionIndices, VkBuffer data, uint32_t dataByteSize, VkBuffer indices, uint32_t amtOfIndices, std::vector<std::vector<std::pair<float, float>>>& brush, int index);
	IsoSurfRendererError update3dBinaryVolume(uint32_t width, uint32_t height, uint32_t depth, uint32_t posIndices[3], std::vector<std::pair<float, float>>& posBounds, uint32_t amtOfAttributes, uint32_t dataSize, VkBuffer data, VkBufferView activeIndices, uint32_t indicesSize, VkBuffer indices, bool regularGrid, int index);
	void deleteBinaryVolume(uint32_t ind);
	void getPosIndices(int index, uint32_t* ind);
	void updateCameraPos(CamNav::NavigationInput input, float deltaT);
	void setCameraPos(glm::vec3& cameraPosReturn, float** newRotation);
	void getCameraPos(glm::vec3& cameraPosReturn, float** rotationReturn);
	void addBrush(std::string& name, std::vector<std::vector<std::pair<float, float>>> minMax);				//minMax has to be a vector containing for each attribute an array of minMax values
	bool updateBrush(std::string& name, std::vector<std::vector<std::pair<float, float>>> minMax);			//this method only updates a already added brush. Returns true if the brush was updated, else false
	bool deleteBrush(std::string& name);
	void render();
	void setImageDescriptorSet(VkDescriptorSet descriptor);
	VkDescriptorSet getImageDescriptorSet();
	VkSampler getImageSampler();
	VkImageView getImageView();
	void exportBinaryCsv(std::string path, uint32_t binaryIndex);
	void setBinarySmoothing(float stdDiv, bool keepOnes);
	void imageBackGroundUpdated();

	std::vector<DrawlistBrush> drawlistBrushes;
	bool shade;
	float stepSize;
	float shadingStep = 2;
	float flySpeed;
	float fastFlyMultiplier;
	float rotationSpeed;
	float isoValue = .1f;
	float gridLineWidth = .005f;
	glm::vec3 lightDir;
	VkClearValue imageBackground;

	// camera variables for the GUI are stored here
	glm::vec3 cameraPositionGLMGUI{};
	float cameraPositionGUI[3]{};
	float cameraRotationGUI[2]{};

private:
	struct UniformBuffer {
		glm::vec4 camPos;				//cameraPosition in model space
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
		float shadingStep;
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
	static char activeIndComputePath[];
	static char binaryComputePath[];
	static char binarySmoothPath[];
	static char binaryCopyOnesPath[];

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

	VkSampler					binaryImageSampler;
	std::vector<VkImage>		binaryImage;
	std::vector<VkImage>		binarySmooth;
	std::vector<VkImageView>	binaryImageView;
	std::vector<VkImageView>	binarySmoothView;
	std::vector<VkDeviceMemory>	binaryImageMemory;
	//vulkan resources for the compute pipeline
	VkPipeline			computePipeline;
	VkPipelineLayout	computePipelineLayout;
	VkDescriptorSetLayout computeDescriptorSetLayout;
	//vulkan resources for the compute pipeline used for filling the binary texture with active indices
	VkPipeline			activeIndComputePipeline;
	VkPipelineLayout	activeIndComputePipelineLayout;
	VkDescriptorSetLayout activeIndComputeDescriptorSetLayout;
	//vulkan resources for the binary image filling compute
	VkPipeline			binaryComputePipeline;
	VkPipelineLayout	binaryComputePipelineLayout;
	VkDescriptorSetLayout binaryComputeDescriptorSetLayout;
	//vulkan resources for the smoothing pipeline
	VkPipeline			binarySmoothPipeline;
	VkPipelineLayout	binarySmoothPipelineLayout;
	VkDescriptorSetLayout binarySmoothDescriptorSetLayout;
	//vulkan resources for copying one entrys into the smoothed binary images
	VkPipeline			binaryCopyOnesPipeline;
	VkPipelineLayout	binaryCopyOnesPipelineLayout;
	VkDescriptorSetLayout binaryCopyOnesDescriptorSetLayout;

	std::vector<glm::uvec3> posIndices;

	//camera variables
	glm::vec3 cameraPos;		//camera position
	glm::vec2 cameraRot;
	//variables for the brushes
	std::map<std::string, std::vector<std::vector<std::pair<float, float>>>> brushes;		//each brush has a vector of minMax values. Each entry in the vector corresponds to an attribute
	std::map<std::string, float*> brushColors;												//each brush has its own colors
	std::vector<float*> attributeColors;													//if only one brush is active every attribute can be assigned a different color

	float smoothStdDiv = 1;
	bool keepOnes = false;

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