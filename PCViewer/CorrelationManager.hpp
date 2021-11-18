#pragma once
#include "VkUtil.h"
#include <map>
#include <vector>
#include <string>
#include "Structures.hpp"

class CorrelationManager{
public:
    enum class CorrelationMetric{
        Pearson,
        SpearmanRank,
        KendallRank
    };
    struct AttributeCorrelation{
        int baseAttribute;
        CorrelationMetric metric{};
        std::vector<float> correlationScores{};   //all axes are places consecutively in the vector(contains correlation to itself)
    };
    struct DrawlistCorrelations{
        std::string drawlist{};
        std::map<int, AttributeCorrelation> attributeCorrelations{};
    };

    CorrelationManager(const VkUtil::Context& context);
    ~CorrelationManager();

    void calculateCorrelation(const DrawList& dl, CorrelationMetric metric = CorrelationMetric::Pearson, int baseAttribute = -1, bool useGpu = false);

    std::map<std::string, DrawlistCorrelations> correlations{};
private:
    VkUtil::Context _vkContext{};
    std::string _pearsonShader = "shaders/corrPearson.comp.spv";
    std::string _spearmanShader = "shaders/corrSpearman.comp.spv";
    std::string _kendallShader = "shaders/corrKendall.comp.spv";
    std::string _meanShader = "shaders/corrMean.comp.spv";
    VkUtil::PipelineInfo _pearsonPipeline{}, _spearmanPipeline{}, _kendallPipeline{}, _meanPipeline{};
    void _execCorrelationCPU(const DrawList& dl, CorrelationMetric metric, int baseAttribute);
    void _execCorrelationGPU(const DrawList& dl, CorrelationMetric metric, int baseAttribute);
};