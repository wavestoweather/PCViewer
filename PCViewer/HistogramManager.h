#pragma once
#include <vector>
#include <vulkan/vulkan.h>
#include <string>
#include <map>
#include "VkUtil.h"
#include "PCUtil.h"

#define SHADERPATH "shader/histComp.spv"
#define LOCALSIZE 256


class HistogramManager {
public:
    struct Histogram {
        float maxGlobalCount;                            //maximung value accross all attributes
        float maxActiveCount = 0;
        std::vector<float> area;                        //area when scaling equals 1
        std::vector<float> areaShown;                    //area when scaling equals 1, but scaled to visible bins
        std::vector<float> areaRendered;                //area of rendered violin
        std::vector<float> maxCount;                    //maximum histogramm value for each attribute
        std::vector<std::pair<float, float>> ranges;    //the value ranges for each attribute
        std::vector<std::vector<float>> originalBins;    //histogramm values for each attribute before smoothing was applied
        std::vector<std::vector<float>> bins;            //histogramm values for each attribute
        std::vector<std::vector<float>> binsRendered;    //histogramm values for each attribute in final x-coordinate differences
        std::vector<unsigned int> side;                    //stores on which side each attribute is rendered.
        std::vector<unsigned int> attributeColorOrderIdx;//stores indices in the order in which colors have to be assigned to them. In combination with side, the right colors can be chosen from the colorpalette "Dark2ExtendedReorder", such that the colors are on the right side. 
    };


    HistogramManager(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue queue, VkDescriptorPool descriptorPool, uint32_t binsAmount);
    ~HistogramManager();

    void computeHistogramm(const std::string& name, std::vector<std::pair<float,float>>& minMax, VkBuffer data, uint32_t amtOfData, VkBuffer indices, uint32_t amtOfIndices, VkBufferView indicesActivations);
    Histogram& getHistogram(const std::string& name);
    bool containsHistogram(const std::string& name);

    /** Uses the chi-squared distance measure to compare all single attribute histograms
    and sums over all attributes. Mode = 0 -> originalBins, Mode = 1 -> binsRendered.
    */
    float computeHistogramDistance(std::string& nameRep, std::string& name, bool **active, int mode = 0);

    void setNumberOfBins(uint32_t n);
    //setting stdDev to a negative number leads to automatic choose of kernel size
    void setSmoothingKernelSize(float stdDev);
    void updateSmoothedValues();

    void determineSideHist(Histogram& hist, bool **active = nullptr, bool considerBlendingOrder = false);

    bool ignoreZeroValues;
    bool ignoreZeroBins;
    bool* logScale;

    bool adaptMinMaxToBrush;
    float stdDev;

private:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkCommandPool commandPool;
    VkQueue queue;
    VkDescriptorPool descriptorPool;

    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout descriptorSetLayout;
    // 3 ubo buffers for
    // informations:
    // numOfBins, numOfAttributes, amtOfIndices, ignoreZeroValues, float[min,max]
    // bins:
    // array for all bins
    VkBuffer uboBuffers[2];
    uint32_t uboOffsets[2];
    VkDeviceMemory uboMemory;

    uint32_t numOfBins;
    std::map<std::string, Histogram> histograms;

    void updateSmoothedValues(Histogram& hist);

    
};
