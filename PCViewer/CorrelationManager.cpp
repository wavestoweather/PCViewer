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
    std::vector<uint8_t> activations(dl.data->size());
    VkUtil::downloadData(vkContext.device, dl.dlMem, dl.activeIndicesBufferOffset, activations.size(), activations.data());
    std::vector<double> means(amtOfAttributes, 0);
    std::vector<double> nominator(amtOfAttributes, 0), denom1(amtOfAttributes, 0), denom2(amtOfAttributes, 0);
    switch(metric){
    case CorrelationMetric::Pearson:
    {
        for(int a = 0; a < amtOfAttributes; ++a){
            int c = 0;
            for(size_t i = 0; i < dl.indices.size(); ++i){
                if(!activations[dl.indices[i]]) continue;
                means[a] += ((*dl.data)(dl.indices[i], a) - means[a]) / (++c);
            }
        }
        for(size_t i = 0; i < dl.indices.size(); ++i){
            for(int a = 0; a < amtOfAttributes; ++a){
                if(!activations[dl.indices[i]]) continue;
                size_t index = dl.indices[i];
                double aDiff = (*dl.data)(index, a) - means[a];
                double bDiff = (*dl.data)(index, baseAttribute) - means[baseAttribute];
                nominator[a] += aDiff * bDiff;
                denom1[a] += aDiff * aDiff;
                denom2[a] += bDiff * bDiff;
            }
        }
        for(int i = 0; i < amtOfAttributes; ++i){
            curAttCorrelation.correlationScores[i] = nominator[i] / std::sqrt((denom1[i] + 1e-5) * (denom2[i] + 1e-5));
            assert(curAttCorrelation.correlationScores[i] >= -1 && curAttCorrelation.correlationScores[i] <= 1);
        }
        break;
    }
    case CorrelationMetric::SpearmanRank:
    {
        std::vector<std::vector<size_t>> sortedIdx(amtOfAttributes);
        // sorts all attributes of data and puts a vector of indices into sortedIdx for each attribute
        // Index vector describes the position of a data point 
        std::vector<size_t> reducedInd;
        for(uint32_t i: dl.indices) if(activations[i]) reducedInd.push_back(i);
        for(int i =0; i < amtOfAttributes; ++i){
            auto& idx = sortedIdx[i];
            idx.resize(reducedInd.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b){return (*dl.data)(reducedInd[a], i) < (*dl.data)(reducedInd[b], i);});
            std::vector<size_t> cpy(idx.size());
            for(size_t i = 0; i < idx.size(); ++i) cpy[idx[i]] = i;
            idx = cpy;
        }
        for(int a = 0; a < amtOfAttributes; ++a) means[a] = (sortedIdx[a].size() - 1) / 2.0;
        for(size_t i = 0; i < reducedInd.size(); ++i){
            for(int a = 0; a < amtOfAttributes; ++a){
                size_t aIdx = i;
                size_t bIdx = i;
                double aDiff = sortedIdx[a][aIdx] - means[a];
                double bDiff = sortedIdx[baseAttribute][bIdx] - means[baseAttribute];
                nominator[a] += aDiff * bDiff;
                denom1[a] += aDiff * aDiff;
                denom2[a] += bDiff * bDiff;
            }
        }
        for(int i = 0; i < amtOfAttributes; ++i){
            curAttCorrelation.correlationScores[i] = nominator[i] / std::sqrt(denom1[i] * denom2[i]);
            assert(curAttCorrelation.correlationScores[i] >= -1 && curAttCorrelation.correlationScores[i] <= 1);
        }
        break;
    }
    case CorrelationMetric::KendallRank:
    {
        double bin = activations[dl.indices[0]];
        for(size_t i = 1; i < dl.indices.size(); ++i){
            bin += activations[dl.indices[i]];
            for(size_t j = 0; j < i; ++j){
                size_t indexA = dl.indices[i], indexB = dl.indices[j];
                if(!activations[indexA] || !activations[indexB]) continue;
                for(int a = 0; a < amtOfAttributes; ++a){
                    double aDiff = (*dl.data)(indexA, a) - (*dl.data)(indexB, a);
                    aDiff = std::signbit(aDiff) * -2.0 + 1;
                    double bDiff = (*dl.data)(indexA, baseAttribute) - (*dl.data)(indexB, baseAttribute);
                    bDiff = std::signbit(bDiff) * -2.0 + 1;
                    nominator[a] += aDiff * bDiff;
                }
            }
        }
        bin = bin * (bin - 1) / 2;
        for(int i = 0; i < amtOfAttributes; ++i){
            assert(nominator[i] / bin >= -1 && nominator[i] / bin <= 1);
            curAttCorrelation.correlationScores[i] = nominator[i] / bin;
        }
        break;
    }
    };
}
