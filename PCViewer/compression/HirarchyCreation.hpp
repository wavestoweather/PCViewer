#pragma once
#include "DataLoader.hpp"

namespace compression{
    void createHirarchy(DataLoader& loader, float lvl0eps, int levels, int lvlMultiplier);
    void compressHirarchyFiles();
    void loadHirarchy();
};