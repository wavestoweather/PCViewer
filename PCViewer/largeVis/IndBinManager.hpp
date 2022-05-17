#pragma once
#include <string_view>
#include <string>
#include <vector>
#include "../LassoBrush.hpp"
#include "../Attribute.hpp"
#include "../Structures.hpp"
#include "../Brushing.hpp"
#include "../robin_hood_map/robin_hood.h"
#include "../VkUtil.h"
#include "../compression/Constants.hpp"
#include "RenderLineCounter.hpp"
#include "Renderer.hpp"
#include "LineCounter.hpp"
#include <atomic>
#include <future>
#include <roaring.hh>

// loads the roaring bitmaps from memory and makes them easily accesisble
// provides a general interface to be used in the main application for calculating the 2d bin counts as well as rendering them as pc plots
// the API consists of only 4 elements:
//  1. notifyBrushUpdate:       A method to notify brush updates
//  2. notifyAttributeUpdate:   A method to notify attribute updates such as reordering, subselecting etc.
//  3. render:                  A method to render the current active ordered attributes to the pc plot according to current 2d bin counts
//  4. checkRenderRequest:      A method which indicates that updates have been processed which require a rerendering -> call render() to set to false
class IndBinManager{
private:
    struct UVecHash{
        std::size_t operator()(std::vector<uint32_t> const& vec) const{
            std::size_t seed = vec.size();
            for(const auto& i : vec){
                seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

public:
    using RangeBrush = brushing::RangeBrush;

    enum class CountingMethod{
        CpuGeneric,
        CpuRoaring,
        GpuDrawPairwise,
        GpuDrawMultiViewport,
        GpuComputePairwise,
        GpuComputeFull,
        HybridRoaringGpuDraw
    };

    struct CreateInfo{
        std::string_view hierarchyFolder;
        uint32_t maxDrawLines = 1e6;
    };

    // maxDrawLines describes the max lines inbetween two attributes
    IndBinManager(const CreateInfo& info);
    IndBinManager(const IndBinManager&) = delete;   // no copy constructor
    IndBinManager& operator=(const IndBinManager&) = delete;    // no copy assignment
    // destructor triggers vulkan resource destruction
    ~IndBinManager();

    // updates the 2d bin counts and indicates a readily done brush update by setting teh requestRender bool to true
    void notifyBrushUpdate(const std::vector<RangeBrush>& rangeBrushes, const Polygons& lassoBrushes);
    // updates the attribute ordering of the attributes. Might trigger recalculation of 
    void notifyAttributeUpdate(const std::vector<int>& attributeOrdering, const std::vector<Attribute>& attributes, bool* atttributeActivations);
    // renders the current 2d bin counts to the pc plot
    void render();
    // checks if pc plot should be updated
    bool checkRenderRequest()const {return _requestRender;}

    // bool which indicates render update should be done (by calling render())
    std::vector<Attribute> attributes;
    robin_hood::unordered_map<std::vector<uint32_t>, std::vector<roaring::Roaring>, UVecHash> ndBuckets;    // contains all bin indices available (might also be multidimensional if 2d bin indexes are available)
    CountingMethod countingMethod{CountingMethod::HybridRoaringGpuDraw};    // variable to set the different counting techniques
private:
    // struct for holding all information for a counting image such as the vulkan resources, brushing infos...
    struct CountResource{
        VkBuffer countBuffer{};
        VkDeviceMemory countMemory{};
        size_t brushingId;      // used to identify brushing state (when brushing state is not on the current state recalculation of indices is required)

        VkDevice _vkDevice{};     // device handle needed for destruction
        ~CountResource(){
            if(countBuffer)
                vkDestroyBuffer(_vkDevice, countBuffer, nullptr);
            if(countMemory)
                vkFreeMemory(_vkDevice, countMemory, nullptr);
        }
        CountResource(const CountResource&) = delete;           // type is not copyable, only movable
        CountResource& operator=(const CountResource&) = delete;
    };

    std::atomic<bool> _requestRender{false};
    bool _hierarchyValid{true};
    uint32_t _baseLevel{0};
    uint32_t _hierarchyLevels;
    std::vector<uint32_t> _clusterLevelSizes{};

    std::string _hierarchyFolder;
    std::vector<uint32_t> _dimensionSizes;
    std::vector<std::vector<uint32_t>> _attributeDimensions;
    std::vector<std::vector<compression::IndexCenterFileData>> _attributeCenters; // for each level for all attributes a singel list with the centers exists 
    std::vector<std::vector<uint32_t>> _attributeIndices;

    std::thread _dataLoadThread;
    std::atomic<bool> _loadThreadActive{false};

    robin_hood::unordered_map<std::vector<uint32_t>, CountResource, UVecHash> _countResources; // map saving for each attribute combination the 2d bin counts. It saves additional information and keeps after order changes to avoid unnesecary recomputation
    
    // gpu pipeline handles which have to be noticed for deconstruction
    RenderLineCounter* _renderLineCounter{};
    LineCounter* _lineCounter{};
    compression::Renderer* _renderer{};

    uint32_t _managerByteSize{};                    // used to keep track of memory consumption to dynamically release intersection lists in "intersectionIndices"
};