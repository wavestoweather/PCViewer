#pragma once
#include "../half/half.hpp"
#include <vector>

namespace compression{

void testCounting();
std::vector<uint32_t> lineCounterPair(const std::vector<half>& aVals, const std::vector<half>& bVals, uint32_t aBins, uint32_t bBins, const std::vector<uint8_t>& activation, uint32_t amtOfThreads = 12);
std::vector<uint32_t> lineCounterPairSingleField(const std::vector<half>& aVals, const std::vector<half>& bVals, uint32_t aBins, uint32_t bBins, const std::vector<uint8_t>& activation, uint32_t amtOfThreads = 12);
std::vector<half> lineMinPair(const std::vector<half>& aVals, const std::vector<half>& bVals, uint32_t aBins, uint32_t bBins, const std::vector<uint8_t>& activation, const std::vector<half>& distances, uint32_t amtOfThreads = 12);
}