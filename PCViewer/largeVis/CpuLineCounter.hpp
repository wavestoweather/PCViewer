#pragma once
#include "../half/half.hpp"

namespace compression{

void testCounting();
std::vector<uint32_t> lineCounterPair(const std::vector<half>& aVals, const std::vector<half>& bVals, uint32_t aBins, uint32_t bBins, uint32_t amtOfThreads = 12);

}