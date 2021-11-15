#define NOSTATICS
#include "CorrelationManager.hpp"
#undef NOSTATICS
#include <cmath>
#include <algorithm>
#include <numeric>

CorrelationManager::CorrelationManager(const VkUtil::Context& context): vkContext(context)
{

}

void CorrelationManager::calculateCorrelation(const DrawList& dl, CorrelationMetric metric, int baseAttribute) 
{
    if(baseAttribute < 0) return;
    int amtOfAttributes = dl.data->columns.size();
    DrawlistCorrelations& dlCorrelations = correlations[dl.name];
    dlCorrelations.drawlist = dl.name;
    AttributeCorrelation& curAttCorrelation = dlCorrelations.attributeCorrelations[baseAttribute];
    curAttCorrelation.metric = metric;
    curAttCorrelation.baseAttribute = baseAttribute;
    curAttCorrelation.correlationScores.resize(amtOfAttributes);
    std::vector<double> means(amtOfAttributes, 0);
    std::vector<double> nominator(amtOfAttributes, 0), denom1(amtOfAttributes, 0), denom2(amtOfAttributes, 0);
    switch(metric){
    case CorrelationMetric::Pearson:
    {
        for(int a = 0; a < amtOfAttributes; ++a){
            for(size_t i = 0; i < dl.data->columns[a].size(); ++i){
                means[a] += (dl.data->columns[a][i] - means[a]) / (i + 1);
            }
        }
        for(size_t i = 0; i < dl.data->size(); ++i){
            for(int a = 0; a < amtOfAttributes; ++a){
                double aDiff = (*dl.data)(i, a) - means[a];
                double bDiff = (*dl.data)(i, baseAttribute) - means[baseAttribute];
                nominator[a] += aDiff * bDiff;
                denom1[a] += aDiff * aDiff;
                denom2[a] += bDiff * bDiff;
            }
        }
        for(int i = 0; i < amtOfAttributes; ++i){
            curAttCorrelation.correlationScores[i] = nominator[i] / std::sqrt(denom1[i] * denom2[i]);
        }
        break;
    }
    case CorrelationMetric::SpearmanRank:
    {
        std::vector<std::vector<size_t>> sortedIdx(amtOfAttributes);
        // sorts all attributes of data and puts a vector of indices into sortedIdx for each attribute
        // Index vector describes the position of a data point 
        for(int i =0; i < amtOfAttributes; ++i){
            auto& idx = sortedIdx[i];
            idx.resize(dl.data->columns[i].size());
            std::iota(idx.begin(), idx.end(), 0);
            std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b){return dl.data->columns[i][a] < dl.data->columns[i][b];});
            std::vector<size_t> cpy(idx.size());
            for(size_t i = 0; i < idx.size(); ++i) cpy[idx[i]] = i;
            idx = cpy;
        }
        for(int a = 0; a < amtOfAttributes; ++a) means[a] = (sortedIdx[a].size() - 1) / 2.0;
        for(size_t i = 0; i < dl.data->size(); ++i){
            for(int a = 0; a < amtOfAttributes; ++a){
                size_t aIdx = dl.data->columnIndex(i, a);
                size_t bIdx = dl.data->columnIndex(i, baseAttribute);
                double aDiff = sortedIdx[a][aIdx] - means[a];
                double bDiff = sortedIdx[baseAttribute][bIdx] - means[baseAttribute];
                nominator[a] += aDiff * bDiff;
                denom1[a] += aDiff * aDiff;
                denom2[a] += bDiff * bDiff;
            }
        }
        for(int i = 0; i < amtOfAttributes; ++i){
            curAttCorrelation.correlationScores[i] = nominator[i] / std::sqrt(denom1[i] * denom2[i]);
        }
        break;
    }
    case CorrelationMetric::KendallRank:
    {
        for(size_t i = 1; i < dl.data->size(); ++i){
            for(size_t j = 0; j < i; ++j){
                for(int a = 0; a < amtOfAttributes; ++a){
                double aDiff = (*dl.data)(i, a) - (*dl.data)(j, a);
                aDiff = std::signbit(aDiff) * -2.0 + 1;
                double bDiff = (*dl.data)(i, baseAttribute) - (*dl.data)(j, baseAttribute);
                bDiff = std::signbit(bDiff) * -2.0 + 1;
                nominator[a] += aDiff * bDiff;
                }
            }
        }
        double bin = dl.data->size() * (dl.data->size() - 1) / 2;
        for(int i = 0; i < amtOfAttributes; ++i){
            curAttCorrelation.correlationScores[i] = nominator[i] / bin;
        }
        break;
    }
    };
}
