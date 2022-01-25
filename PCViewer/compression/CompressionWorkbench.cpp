#include "CompressionWorkbench.hpp"
#include "../imgui/imgui_stdlib.h"
#include "NetCdfLoader.hpp"
#include "../PCUtil.h"

void CompressionWorkbench::draw() 
{
    if(!active) return;

    if(ImGui::Begin("CompresssionWorkbench")){
        ImGui::Text("Open data files:");
        ImGui::InputText("Src Directory/File", &_inputFiles);
        if(ImGui::CollapsingHeader("Include Formats"));
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
        ImGui::SameLine();
        if(ImGui::CollapsingHeader("Exclude Formats"));
        {
            for(int i = 0; i < _excludedFiles.size(); ++i){
                ImGui::InputText(("##exc" + std::to_string(i)).c_str(), &_excludedFiles[i]);
            }
            if(_includedFiles.size() && ImGui::Button("Remove last Format")){
                _excludedFiles.pop_back();
            }
            if(ImGui::Button("Add Exclude Format")){
                _excludedFiles.push_back({});
            }
        }
        if(ImGui::Button("Create data loader")){
            _loader = std::make_shared<NetCdfLoader>(_inputFiles, _includedFiles, _excludedFiles);
        }
        if(_loader){
            ImGui::Text("Hier könnte jetzt eine subselektion der dimensionen ausgeführt werden...");
        }
        if(_loader && ImGui::Button("Analyze")){
            _loader->dataAnalysis(_dataSize, _attributes);
        }
        ImGui::Text(("Analyzed data size: " + std::to_string(_dataSize)).c_str());
        for(auto& a: _attributes){
            ImGui::Text(a.name.c_str());
        }
        ImGui::Text(("Loader Fortschritt: " + std::to_string(_loader->progress())).c_str());
    }
    ImGui::End();
}
