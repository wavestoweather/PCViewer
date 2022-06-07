#pragma once
#include <roaring64map.hh>
#include <vector>

namespace compression{
    void testRoaringCounting();
    void testRoaringRealWorld();
    std::vector<uint32_t> lineCounterRoaring(uint32_t maxBins, const std::vector<roaring::Roaring64Map>& aIndices, const std::vector<uint32_t>& aIndexBins, const std::vector<roaring::Roaring64Map>& bIndices, const std::vector<uint32_t>& bIndexBins, uint32_t aBins, uint32_t bBins, uint32_t amtOfThreads = 12);
}