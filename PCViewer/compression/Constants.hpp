#pragma once
#include <string>
#include <vector>
#include <fstream>

namespace compression{
    namespace constants{
        const std::string clusterCountName = "ClusterCount";
        const std::vector<std::string> reservedAttributeNames{clusterCountName};
    }
    enum class DataStorageBits{
        None = 0,
        RawColumnData = 1,          // indicates data values are available in column files with raw floats in them
        HalfColumnData = 1 << 1,    // same as RawColumnData with halfs instead of floats
        CuComColumnData = 1 << 2,   // indicates column files data compressed with cuda compress
        RawAttributeBins = 1 << 3,  // indicates attribute bins stored as indexlists
        RoaringAttributeBins = 1 << 4, // indicates attribute bins stored with roaring bitmpas
        Roaring2dBins = 1 << 5      // indicates 2d attribute bins tored with roaring bitmaps
    };
    constexpr DataStorageBits operator|(DataStorageBits a, DataStorageBits b){return static_cast<DataStorageBits>(static_cast<int>(a) | static_cast<int>(b));};
    constexpr DataStorageBits operator&(DataStorageBits a, DataStorageBits b){return static_cast<DataStorageBits>(static_cast<int>(a) & static_cast<int>(b));};
    constexpr DataStorageBits& operator|=(DataStorageBits& a, DataStorageBits b){a = a | b; return a;};
    constexpr DataStorageBits& operator&=(DataStorageBits& a, DataStorageBits b){a = a & b; return a;};
    constexpr inline bool DataStorageBitSet(DataStorageBits a, DataStorageBits bit){return (a & bit) != DataStorageBits::None;};
    static std::ifstream& operator>>(std::ifstream& in, DataStorageBits& bits){
        uint32_t b;
        in >> b;
        bits = static_cast<DataStorageBits>(b);
        return in;
    }
    static std::ofstream& operator<<(std::ofstream& stream, DataStorageBits bits){
        stream << static_cast<uint32_t>(bits);
        return stream;
    }
}