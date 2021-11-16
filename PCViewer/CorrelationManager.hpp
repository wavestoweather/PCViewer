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

    void calculateCorrelation(const DrawList& dl, CorrelationMetric metric = CorrelationMetric::Pearson, int baseAttribute = -1);

    std::map<std::string, DrawlistCorrelations> correlations{};
private:
    VkUtil::Context vkContext{};
};