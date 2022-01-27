#pragma once

#include "../imgui/imgui.h"
#include "DataLoader.hpp"
#include <memory>
#include <future>

class CompressionWorkbench{
public:
    CompressionWorkbench(){};

    void draw();

    bool active{};
private:
    std::string _inputFiles{};
    std::vector<std::string> _includedFiles{}, _excludedFiles{};
    std::shared_ptr<DataLoader> _loader{};

    std::vector<Attribute> _attributes{};
    size_t _dataSize{};

    std::future<void> _analysisFuture;

    std::string _outputFolder{};
    float _epsStart{.1f};
    uint32_t _linesPerLvl{static_cast<uint32_t>(1e6)};
    uint32_t _levels{3};
};