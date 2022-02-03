#pragma once
#include "DataLoader.hpp"
#include "../Data.hpp"
#include <string_view>

namespace compression{
    void createHirarchy(const std::string_view& outputFolder, DataLoader* loader, float lvl0eps, int levels, int lvlMultiplier, int maxMemoryMB, int amtOfThreads, float quantizationStep);
    void createTempHirarchy(const std::string_view& outputFolder, DataLoader* loader, float lvl0eps, int levels, int lvlMultiplier, int maxMemoryMB, int amtOfThreads);
    void compressTempHirarchy(const std::string_view& outputFolder, int amtOfThreads, float quantizationStep);
    void loadAndDecompress(const std::string_view& file);
    void loadHirarchy(const std::string_view& outputFolder, Data& data);
};