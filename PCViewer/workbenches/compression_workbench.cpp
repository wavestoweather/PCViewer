#include "compression_workbench.hpp"
#include <logger.hpp>
#include <imgui.h>
#include <imgui_stdlib.h>

namespace workbenches
{
compression_workbench::compression_workbench(std::string_view id):
    workbench(id)
{
}

compression_workbench::~compression_workbench(){
    // cancelling the worker threads if they are doing work
    if(_analysis_thread.joinable()){
        if(logger.logging_level >= logging::level::l_4);
            logger << logging::warning_prefix << " ~compression_workbench() Interrupting analysis thread" << logging::endl;
        _analysis_cancel = true;
        _analysis_thread.join();
    }
    if(_compression_thread.joinable()){
        if(logger.logging_level >= logging::level::l_4);
            logger << logging::warning_prefix << " ~compression_workbench() Interrupting compression thread. This might lead to corrupt intermediate file." << logging::endl;
        _compression_cancel = true;
        _compression_thread.join();
    }
}

void compression_workbench::show() 
{
    if(!active)
        return;
    
    ImGui::Begin(id.data(), &active);
    ImGui::Text("Open data files:");
    ImGui::InputText("Src Directory/File", &_input_files);
    ImGui::BeginChild("includes", {ImGui::GetWindowWidth() / 2.2f, 200});
    if(ImGui::CollapsingHeader("Include Formats"))
    {
        for(int i = 0; i < _included_files.size(); ++i){
            ImGui::InputText(("##inc" + std::to_string(i)).c_str(), &_included_files[i]);
        }
        if(_included_files.size() && ImGui::Button("Remove last Format")){
            _included_files.pop_back();
        }
        if(ImGui::Button("Add Include Format")){
            _included_files.push_back({});
        }
    }
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("excludes", {ImGui::GetWindowWidth() / 2.2f, 200});
    if(ImGui::CollapsingHeader("Exclude Formats"))
    {
        for(int i = 0; i < _excluded_files.size(); ++i){
            ImGui::InputText(("##exc" + std::to_string(i)).c_str(), &_excluded_files[i]);
        }
        if(_excluded_files.size() && ImGui::Button("Remove last Format")){
            _excluded_files.pop_back();
        }
        if(ImGui::Button("Add Exclude Format")){
            _excluded_files.push_back({});
        }
    }
    ImGui::EndChild();

    ImGui::End();
}
}