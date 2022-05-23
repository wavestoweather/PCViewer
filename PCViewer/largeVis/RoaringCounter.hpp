#pragma once
#include <roaring64map.hh>

namespace compression{
    void testRoaringCounting();
    void testRoaringRealWorld();
    std::vector<uint32_t> lineCounterRoaring(const std::vector<roaring::Roaring64Map>& aIndices, const std::vector<roaring::Roaring64Map>& bIndices, uint32_t aBins, uint32_t bBins, uint32_t amtOfThreads = 12);
}