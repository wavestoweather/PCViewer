#pragma once
#include <string_view>

class Workbench{
public:
    // attributes --------------------------------------
    // attribuete to indicate if the workbench should be shown
    bool active{false};

    // methods -----------------------------------------
    // method to show the imgui window
    virtual void show() = 0;
};

class DrawlistDependency{
public:
    virtual void addDrawlist(std::string_view drawlistId) = 0;
    virtual void updateDrawlist(std::string_view drawlistId) = 0;
    virtual void removeDrawlist(std::string_view drawlistId) = 0;
};

class DatasetDependency{
    public:
    virtual void addDataset(std::string_view datasetId) = 0;
    virtual void removeDataset(std::string_view datasetId) = 0;
};
