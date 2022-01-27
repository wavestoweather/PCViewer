#pragma once
#include "DataLoader.hpp"
#include <string_view>

namespace compression{
    void createHirarchy(const std::string_view& outputFolder, DataLoader& loader, float lvl0eps, int levels, int lvlMultiplier);
    void createTempHirarchy(const std::string_view& outputFolder, DataLoader& loader, float lvl0eps, int levels, int lvlMultiplier);
    void compressTempHirarchy(const std::string_view& outputFolder);
    void loadHirarchy(const std::string_view& outputFolder);
};