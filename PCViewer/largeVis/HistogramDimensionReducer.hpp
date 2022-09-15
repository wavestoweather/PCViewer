#pragma once
#include "../VkUtil.h"
#include "TimingInfo.hpp"
#include "PriorityInfo.hpp"
#include "../Brushing.hpp"
#include <map>
#include <atomic>

/*
                                HistogramDimensionReducer
    This class provides a singleton via acquireReference() which always has to be released again via reference->release()
    This is needed as internally it is being tracked how often the pipline is used and the pipeline is destroyed when no longer needed
    If Not all references are released, the pipeline will not be destructed before exiting the application and a vulkan warning will be shown

    The function reduceHistogram is completely stateless. It provides only the functionality of adding the commands required to 
    reduce the histograms to a PREALLOCATED command buffer without submittin anything to a queue. This has the benefit of easier
    implementation here, as well as no complicated waiting behaviour on destruction. Last but not least no multithreading behavioiur has
    to be implemented.

    The main reason reduceHistogram() does not internally dispatch its work is, that the amount of work needed for reduction is very small
    and has no big runtime. Thus dispatching at a later time is seen as sufficient

*/
class HistogramDimensionReducer{
    struct PushConstants{
        uint32_t        histogramWidth;
        uint32_t        xReduce;       // if 1 reduction accross x, else reduction accross y
        VkDeviceAddress srcHistogram;
        VkDeviceAddress dstHistogram;
    };

    VkUtil::Context         _vkContext{};

    const std::string_view  _shaderPath{"shaders/largeVisDimReduce.comp.spv"};

    VkUtil::PipelineInfo    _pipeline{};
    std::atomic_uint32_t    _refCounter{};

    HistogramDimensionReducer(const VkUtil::Context& context);
    ~HistogramDimensionReducer();

    static HistogramDimensionReducer* _singleton;

public:
    struct  reduceInfo{
        VkCommandBuffer commands;
        uint32_t        histogramWidth;
        uint32_t        xReduce;       // if 1 reduction accross x, else reduction accross y
        VkDeviceAddress srcHistogram;
        VkDeviceAddress dstHistogram;
    };

    // no copy or move and no standard creation
    HistogramDimensionReducer() = delete;
    HistogramDimensionReducer(const HistogramDimensionReducer&) = delete;
    HistogramDimensionReducer& operator=(const HistogramDimensionReducer&) = delete;
    HistogramDimensionReducer(HistogramDimensionReducer&&) = delete;
    HistogramDimensionReducer& operator=(HistogramDimensionReducer&&) = delete;

    static HistogramDimensionReducer* acquireReference(const VkUtil::Context& context){if(!_singleton) _singleton = new HistogramDimensionReducer(context); ++_singleton->_refCounter; return _singleton;}

    void release() {--_refCounter; if(_refCounter == 0){delete _singleton; _singleton = nullptr;}}
    // non realtime (only records the commands into reduceInfo::commands)
    void reduceHistogram(reduceInfo& info);
};