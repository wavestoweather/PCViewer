#pragma once
#include <imgui.h>
#include <workbench_base.hpp>
#include <logger.hpp>

namespace util{
namespace main_window{
inline void menu_bar(){
    if(ImGui::BeginMenuBar()){
        if(ImGui::BeginMenu("File")){
            ImGui::MenuItem("Nothing here yet");
            ImGui::EndMenu();
        }
        if(ImGui::BeginMenu("Edit")){
            if(ImGui::BeginCombo("Logging level", logging::level_names[logger.logging_level].data())){
                for(auto level: structures::enum_iteration<logging::level>{}){
                    if(ImGui::MenuItem(logging::level_names[level].data()))
                        logger.logging_level = level;
                }
                ImGui::EndCombo();
            }
            ImGui::EndMenu();
        }
        if(ImGui::BeginMenu("View")){
            // making all workbenches selectable
            for(const auto& wb: globals::workbenches){
                if(wb.get() == globals::primary_workbench)
                    continue;
                ImGui::MenuItem(wb->id.c_str(), "", &wb->active);
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }
}
}
}