#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <list>
#include "Data.hpp"
#include "LineBundles.hpp"
#include "ClusterBundles.hpp"
#include "TemplateList.hpp"
#include "Attribute.hpp"
#include "compression/HierarchyImportManager.hpp"
#include <optional>

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

struct Buffer {
	VkBuffer buffer;
	VkBuffer uboBuffer;
	VkDeviceMemory memory;
	VkDescriptorSet descriptorSet;

	bool operator==(const Buffer& other) {
		return this->buffer == other.buffer && this->memory == other.memory;
	}
};

enum class DataType{
	Continuous,
	ContinuousDlf,
	Hierarchichal
};

struct DataSet {
	std::string name;
	Buffer buffer;
	Data data;
	std::list<TemplateList> drawLists;
	int reducedDataSetSize;			//size of the reduced dataset(when clustering was applied). This is set to data.size() on creation.
	DataType dataType;

	bool operator==(const DataSet& other) const {
		return this->name == other.name;
	}
};

struct Brush {
	int id;
	std::pair<float, float> minMax;
};

enum class InheritanceFlags{
	dlf = 1,
	hierarchical = 1 << 1
};

// struct holding the information for a drawable instance of a TemplateList
//
// The id of the Drawlist is its name!
//
// The inheritedFlags field contains important information inherited from the dataset and template list this drawlistis created from
// Such inheritance bits can be:
//	- Hierarchical: Instead of creating buffers which are sized to hold the data information, the buffers have a size to be able to hold as much lines as set in maxHierarchyLines
//
struct DrawList {
	std::string name;
	std::string parentDataSet;
	TemplateList* parentTemplateList;
	const Data* data;
	const std::vector<Attribute>* attributes;
	InheritanceFlags inheritanceFlags;
	Vec4 color;
	Vec4 prefColor;
	bool show;
	bool showHistogramm;
	std::vector<float> brushedRatioToParent;     	// Stores the ratio of points of this data set and points going through the same 1D brushes of the parent.
	bool immuneToGlobalBrushes;
	VkBuffer buffer;								// vulkan data buffer
	VkDescriptorSet dataDescriptorSet;				//is relesed when dataset is removed
	VkBuffer indexBuffer;							//indexbuffer for line rendering!!!
	uint32_t indexBufferOffset;
	VkBuffer ubo;
	//VkBuffer histogramIndBuffer;
	//uint32_t histIndexBufferOffset;
	std::vector<VkBuffer> histogramUbos;
	VkBuffer medianBuffer;
	VkDescriptorSet medianBufferSet;				//has to be created/released in drawlist creation
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
	VkDeviceMemory indexBufferMemory;
	VkDescriptorSet uboDescSet;
	std::vector<uint32_t> indices;
	//std::vector<uint32_t> activeInd;
	std::vector<std::vector<Brush>> brushes;		//the pair contains first min and then max for the brush
	LineBundles* lineBundles;
	ClusterBundles* clusterBundles;
	bool renderBundles, renderClusterBundles;
	uint32_t activeLinesAmt;						//contains the amount of lines after brushing has been applied
	std::optional<HierarchyImportManager> hierarchImportManager;	//optional import manger for hierarchy files
};

struct DrawlistDragDropInfo{
	std::list<DrawList>* drawlists;
	std::vector<int>* selected;
};

struct UniformBufferObject {
	float alpha;
	uint32_t amtOfVerts;
	uint32_t amtOfAttributes;
	float padding;
	Vec4 color;
	std::vector<Vec4> vertTransformations;
	//Vec4 VertexTransormations[];			//is now a variable length array at the end of the UBO
	uint32_t size(){
		return sizeof(UniformBufferObject) - sizeof(vertTransformations) + sizeof(vertTransformations[0]) * vertTransformations.size();
	}
};

struct VertexBufferCreateInfo{
	DataType dataType;
	uint32_t maxLines;
	uint32_t additionalAttributeStorage;
};