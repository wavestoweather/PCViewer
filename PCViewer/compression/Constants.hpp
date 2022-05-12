#pragma once
#include <string>
#include <vector>

namespace compression{
    namespace constants{
        const std::string clusterCountName = "ClusterCount";
        const std::vector<std::string> reservedAttributeNames{clusterCountName};
    }
    enum class StoreFlags{
        RawColumnData = 1,          // indicates data values are available in column files with raw floats in them
        HalfColumnData = 1 << 1,    // same as RawColumnData with halfs instead of floats
        CuComColumnData = 1 << 2,   // indicates column files data compressed with cuda compress
        RawAttributeBins = 1 << 3,  // indicates attribute bins stored as indexlists
        RoaringAttributeBins = 1 << 4, // indicates attribute bins stored with roaring bitmpas
        Roaring2dBins = 1 << 5      // indicates 2d attribute bins tored with roaring bitmaps
    };
    constexpr StoreFlags operator|(StoreFlags a, StoreFlags b){return static_cast<StoreFlags>(static_cast<int>(a) | static_cast<int>(b));};

    constexpr StoreFlags flags = StoreFlags::RawColumnData | StoreFlags::HalfColumnData;
}