#include "parallel_coordinates_workbench.hpp"

namespace workbenches{

void parallel_coordinates_workbench::show(){
    if(!active)
        return;
    ImGui::Begin(id.c_str(), &active);

    // -------------------------------------------------------------------------------
    // Plot region including labels and min max values
    // -------------------------------------------------------------------------------

    // -------------------------------------------------------------------------------
    // settings region
    // -------------------------------------------------------------------------------

    ImGui::BeginHorizontal("Settings");

        // general settings
        ImGui::BeginVertical("GeneralSettings");
        ImGui::BeginChild("testwarpp",{200, 0});
        if(ImGui::TreeNode("General settings")){
            ImGui::Text("Well here we go brudi");
            ImGui::TreePop();
        }
        ImGui::EndChild();
        ImGui::EndVertical();

        // drawlist settings
        ImGui::BeginVertical("DrawlistSettings");
        if(ImGui::BeginTable("Drawlist settings", 7, ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_SizingFixedFit)){
            ImGui::TableSetupScrollFreeze(0, 1);    // make top row always visible
            ImGui::TableSetupColumn("Drawlist", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Up");
            ImGui::TableSetupColumn("Down");
            ImGui::TableSetupColumn("Delete");
            ImGui::TableSetupColumn("Active");
            ImGui::TableSetupColumn("Color");
            ImGui::TableSetupColumn("Median");
            
            // top row
            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
            ImGui::TableNextColumn();
            ImGui::TableHeader("Drawlist");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Up");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Down");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Delete");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Active");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Color");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Median");

            ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("dl.drawlist_id.data()");
                ImGui::TableNextColumn();
                ImGui::ArrowButton("##testu", ImGuiDir_Up);
                ImGui::TableNextColumn();
                ImGui::ArrowButton("##testd", ImGuiDir_Down);
                ImGui::TableNextColumn();
                ImGui::Button("X##testx");
                ImGui::TableNextColumn();
                bool t;
                ImGui::Checkbox("##testact", &t);
            
            for(auto& dl: drawlist_infos){
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text(dl.drawlist_id.data());
                ImGui::NextColumn();
                ImGui::ArrowButton("##testu", ImGuiDir_Up);
                ImGui::NextColumn();
                ImGui::ArrowButton("##testd", ImGuiDir_Down);
                ImGui::NextColumn();
                ImGui::Button("X##testx");
                ImGui::NextColumn();
                ImGui::Checkbox("##testact", &dl.appearance->ref_no_track().show);
            }

            ImGui::EndTable();
        }
        ImGui::EndVertical();

    ImGui::EndHorizontal();

    ImGui::End();
}

}