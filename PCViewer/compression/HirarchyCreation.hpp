#pragma once
#include "DataLoader.hpp"
#include <string_view>

namespace compression{
    void createHirarchy(const std::string_view& outputFolder, DataLoader* loader, float lvl0eps, int levels, int lvlMultiplier, int maxMemoryMB, int amtOfThreads);
    void createTempHirarchy(const std::string_view& outputFolder, DataLoader* loader, float lvl0eps, int levels, int lvlMultiplier, int maxMemoryMB, int amtOfThreads);
    void compressTempHirarchy(const std::string_view& outputFolder, int amtOfThreads);
    void loadHirarchy(const std::string_view& outputFolder);
};