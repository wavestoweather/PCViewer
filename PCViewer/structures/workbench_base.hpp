#pragma once
#include <string_view>
#include <vector>
#include <atomic>


namespace structures{
class workbench{
public:
    // attributes --------------------------------------
    // attribuete to indicate if the workbench should be shown
    bool active{false};

    // methods -----------------------------------------
    // method to show the imgui window
    virtual void show() = 0;
};

class drawlist_dependency{
public:
    virtual void addDrawlist(std::string_view drawlistId) = 0;
    virtual void signalDrawlistUpdate(const std::vector<std::string_view>& drawlistIds) = 0;
    virtual void removeDrawlist(std::string_view drawlistId) = 0;
};

class dataset_dependency{
    public:
    virtual void addDataset(std::string_view datasetId) = 0;
    virtual void signalDatasetUpdate(const std::vector<std::string_view>& datasetIds) = 0;
    virtual void removeDataset(std::string_view datasetId) = 0;
};
}