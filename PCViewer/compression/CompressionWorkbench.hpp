#pragma once

#include "../imgui/imgui.h"
#include "DataLoader.hpp"
#include <memory>

class CompressionWorkbench{
public:
    CompressionWorkbench();

    void draw();

    bool active{};
private:
    std::string _inputFiles{};
    std::vector<std::string> _includedFiles{}, _excludedFiles{};
    std::shared_ptr<DataLoader> _loader{};

    std::vector<Attribute> _attributes{};
    size_t _dataSize{};
};