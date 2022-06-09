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
#include "../compression/HirarchyCreation.hpp"
#include "RenderLineCounter.hpp"
#include "Renderer.hpp"
#include "LineCounter.hpp"
#include "ComputeBrusher.hpp"
#include <atomic>
#include <future>
#include <roaring64map.hh>
#include "../half/half.hpp"

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

    // note: The counting methods also include methods for min/max finding (needed for priority rendering)
    enum class CountingMethod{
        CpuGeneric,
        CpuMinGeneric,
        CpuGenericSingleField,
        CpuRoaring,
        GpuDrawPairwise,
        GpuDrawPairwiseTiled,
        GpuDrawMultiViewport,
        GpuComputePairwise,
        GpuComputeFull,
        HybridRoaringGpuDraw,
        Max
    };

    const char* countingMethodNames[10] = {
        "CpuGeneric",
        "CpuMinGeneric",
        "CpuGenericSingleField",
        "CpuRoaring",
        "GpuDrawPairwise",
        "GpuDrawPairwiseTiled",
        "GpuDrawMultiViewport",
        "GpuComputePairwise",
        "GpuComputeFull",
        "HybridRoaringGpuDraw"
    };

    struct CreateInfo{
        std::string_view hierarchyFolder;
        uint32_t maxDrawLines = 1e6;
        // needed for the backing rendering pipelines
        VkUtil::Context context;
        VkRenderPass renderPass;
        VkFramebuffer framebuffer;
    };

    struct CompressedColumnData{    // atm not compressed, future feature
        std::vector<half> cpuData;
        VkBuffer gpuData;
        VkDeviceMemory gpuMemory;
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
    // updates the priority distance for all data points according to an axis value set
    void notifyPriorityCenterUpdate(uint32_t attributeIndex, float attributeValue);
    // forces recounting of the bins with the current countingMethod
    void forceCountUpdate();
    // renders the current 2d bin counts to the pc plot (is done on main thread, as the main framebuffer is locked)
    void render(VkCommandBuffer commands, VkBuffer attributeInfos, bool clear = false);
    // checks if an update is enqueued
    void checkUpdateQueue(){
        if(_currentBrushState != _countBrushState && !_countUpdateThreadActive){
            notifyBrushUpdate(_currentBrushState.rangeBrushes, _currentBrushState.lassoBrushes);
        }
    };

    // bool which indicates render update should be done (by calling render())
    std::vector<Attribute> attributes;
    robin_hood::unordered_map<std::vector<uint32_t>, std::vector<roaring::Roaring64Map>, UVecHash> ndBuckets;    // contains all bin indices available (might also be multidimensional if 2d bin indexes are available)
    std::vector<CompressedColumnData> columnData;
    CountingMethod countingMethod{CountingMethod::GpuDrawPairwise};    // variable to set the different counting techniques
    uint32_t columnBins{1 << 10};
    uint32_t cpuLineCountingAmtOfThreads{12};
    std::atomic<bool> requestRender{false};
    std::vector<uint8_t> indexActivation{};         // bitset for all indices to safe index activation
    std::vector<half> priorityDistances;             // vector containing the priority distances for each data point
private:
    // struct for holding all information for a counting image such as the vulkan resources, brushing infos...
    struct CountResource{
        size_t binAmt{1 << 20}; // needed to check if resolution got updated and the buffer has to be recreated
        VkBuffer countBuffer{};
        VkDeviceMemory countMemory{};
        size_t brushingId{std::numeric_limits<size_t>::max()};      // used to identify brushing state (when brushing state is not on the current state recalculation of indices is required)

        VkDevice _vkDevice{};     // device handle needed for destruction
        CountResource() = default;
        ~CountResource(){
            if(countBuffer)
                vkDestroyBuffer(_vkDevice, countBuffer, nullptr);
            if(countMemory)
                vkFreeMemory(_vkDevice, countMemory, nullptr);
        }
        CountResource(const CountResource&) = delete;           // type is not copyable, only movable
        CountResource& operator=(const CountResource&) = delete;
    };

    struct BrushState{
        std::vector<RangeBrush> rangeBrushes;
        Polygons lassoBrushes;
        size_t id;
        bool operator==(const BrushState& other){
            return id == other.id;
        }
        bool operator!=(const BrushState& other){
            return id != other.id;
        }
    };

    const uint32_t _compressedBlockSize{1 << 20};
    VkUtil::Context _vkContext;

    bool _hierarchyValid{true};
    uint32_t _baseLevel{0};
    uint32_t _hierarchyLevels;
    std::vector<uint32_t> _clusterLevelSizes{};

    uint32_t binsMaxCenterAmt;

    std::string _hierarchyFolder;
    std::vector<uint32_t> _dimensionSizes;
    std::vector<std::vector<uint32_t>> _attributeDimensions;
    std::vector<std::vector<compression::IndexCenterFileData>> _attributeCenters; // for each level for all attributes a singel list with the centers exists 

    //std::thread _dataLoadThread;
    //std::atomic<bool> _loadThreadActive{false};
    std::thread _countUpdateThread;
    std::atomic<bool> _countUpdateThreadActive{false};
    bool _pendingCountUpdate{false};    // if a brush update or attribute update comes in while an update was still enqueud
    BrushState _currentBrushState;
    BrushState _countBrushState;    // used to check if it diverges from the current brush state. Happens when a brush update was entered while a count is still active

    std::vector<int> _attributeOrdering;
    bool* _attributeActivations;

    robin_hood::unordered_map<std::vector<uint32_t>, CountResource, UVecHash> _countResources; // map saving for each attribute combination the 2d bin counts. It saves additional information and keeps after order changes to avoid unnesecary recomputation
    VkBuffer _indexActivation{};
    VkDeviceMemory _indexActivationMemory{};

    // gpu pipeline handles which have to be noticed for deconstruction
    RenderLineCounter* _renderLineCounter{};
    LineCounter* _lineCounter{};
    compression::Renderer* _renderer{};
    ComputeBrusher* _computeBrusher{};

    size_t _curBrushingId{};                        // brush id to check brush status and need for count update
    uint32_t _managerByteSize{};                    // used to keep track of memory consumption to dynamically release intersection lists in "intersectionIndices"

    uint32_t _indexActivationState{};               // holds the brush state for which the index activation was last updated
    uint32_t _gpuIndexActivationState{};

    
    void updateCounts();
};