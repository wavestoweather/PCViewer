#pragma once
#include "../WindowBases.hpp"

class DeriveWorkbench: public Workbench, public DatasetDependency{
public:
    void show() override;
    void addDataset(std::string_view datasetId) override;
    void removeDataset(std::string_view datasetId) override;
};