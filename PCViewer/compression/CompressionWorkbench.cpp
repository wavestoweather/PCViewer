#include "CompressionWorkbench.hpp"
#include "../imgui/imgui_stdlib.h"
#include "NetCdfLoader.hpp"
#include "../PCUtil.h"
#include "HirarchyCreation.hpp"
#include <thread>
#include <chrono>
#include <algorithm>

void analyse(std::shared_ptr<DataLoader> loader, size_t* dataSize, std::vector<Attribute>* attributes){
    loader->dataAnalysis(*dataSize, *attributes);
}

void CompressionWorkbench::draw() 
{
    if(!active) return;

    if(ImGui::Begin("CompresssionWorkbench", &active)){
        ImGui::Text("Open data files:");
        ImGui::InputText("Src Directory/File", &_inputFiles);
        ImGui::BeginChild("includes", {ImGui::GetWindowWidth() / 2.2f, 200});
        if(ImGui::CollapsingHeader("Include Formats"))
        {
            for(int i = 0; i < _includedFiles.size(); ++i){
                ImGui::InputText(("##inc" + std::to_string(i)).c_str(), &_includedFiles[i]);
            }
            if(_includedFiles.size() && ImGui::Button("Remove last Format")){
                _includedFiles.pop_back();
            }
            if(ImGui::Button("Add Include Format")){
                _includedFiles.push_back({});
            }
        }
        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("excludes", {ImGui::GetWindowWidth() / 2.2f, 200});
        if(ImGui::CollapsingHeader("Exclude Formats"))
        {
            for(int i = 0; i < _excludedFiles.size(); ++i){
                ImGui::InputText(("##exc" + std::to_string(i)).c_str(), &_excludedFiles[i]);
            }
            if(_excludedFiles.size() && ImGui::Button("Remove last Format")){
                _excludedFiles.pop_back();
            }
            if(ImGui::Button("Add Exclude Format")){
                _excludedFiles.push_back({});
            }
        }
        ImGui::EndChild();
        if(ImGui::Button("Create data loader")){
            std::vector<std::string_view> includeView(_includedFiles.begin(), _includedFiles.end());
            std::vector<std::string_view> excludeView(_excludedFiles.begin(), _excludedFiles.end());
            try{
                _loader = std::make_shared<NetCdfLoader>(_inputFiles, includeView, excludeView);
            }
            catch(std::runtime_error e){
                std::cout << e.what() << std::endl;
            }
        }
        if(_loader){
            ImGui::Text("Loader contains %i files", _loader->getFileAmt());
            if(ImGui::CollapsingHeader("Data dimension settings")){
                for(auto& e: _loader->queryAttributes){
                    if(e.dimensionSize == 0) 
                        ImGui::Checkbox(e.name.c_str(), &e.active);
                }
            }
        }
        if(_loader && ImGui::Button("Analyze")){
            _analysisFuture = std::async(analyse, _loader, &_dataSize, &_attributes);
            //_loader->dataAnalysis(_dataSize, _attributes);
        }
        ImGui::Text("Analyzed data size: %d", _dataSize);
        if(_loader)
            ImGui::Text("Loader progress: %.2f%%", _loader->progress() * 100);
        
        ImGui::Separator();
        ImGui::InputText("Output path", &_outputFolder);
        if(ImGui::InputFloat("Epsilon start", &_epsStart, .01f, .1f)) _epsStart = std::clamp(_epsStart, 1e-6f, 1.0f);
        if(ImGui::InputInt("Lines per level", reinterpret_cast<int*>(&_linesPerLvl))) _linesPerLvl = std::clamp<uint32_t>(_linesPerLvl, 1, 1e7);
        if(ImGui::InputInt("Levels", reinterpret_cast<int*>(&_levels))) _levels = std::clamp<uint32_t>(_levels, 1, 20);
        if(ImGui::InputInt("Max ram usage in MBytes", reinterpret_cast<int*>(&_maxWorkingMemory))) _maxWorkingMemory = std::max(10u, _maxWorkingMemory);
        if(ImGui::InputInt("Amt of threads", reinterpret_cast<int*>(&_amtOfThreads))) _amtOfThreads = std::max(1u, _amtOfThreads);
        using namespace std::chrono_literals;
        if(_analysisFuture.valid() && _analysisFuture.wait_for(0s) == std::future_status::ready && ImGui::Button("Build Hierarchy")){
            _buildHierarchyFuture = std::async(compression::createHirarchy, std::string_view(_outputFolder), _loader.get(), _epsStart, _levels, _linesPerLvl, _maxWorkingMemory, _amtOfThreads, _quantizationStep);
        }
        if(_buildHierarchyFuture.valid()){
            ImGui::Text("Hierarchy creation at %.2f%%", _loader->progress() * 100);
        }
        if(ImGui::Button("Compress")){
            compression::compressTempHirarchy(_outputFolder, _amtOfThreads, _quantizationStep);
        }
    }
    ImGui::End();
}

void CompressionWorkbench::stopThreads() 
{
    //canelling loading to also stop hirarchy building
    using namespace std::chrono_literals;
    if(_analysisFuture.valid() && _analysisFuture.wait_for(0s) == std::future_status::ready)
        _loader->reset();
}

CompressionWorkbench::~CompressionWorkbench() 
{

}