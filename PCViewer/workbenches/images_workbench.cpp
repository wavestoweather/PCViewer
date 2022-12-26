#include "images_workbench.hpp"
#include <imgui.h>
#include <descriptor_set_storage.hpp>

namespace workbenches
{
void images_workbench::show() 
{
    if(!active)
        return;
    
    ImGui::Begin(id.data(), &active);

    // list with all images
    if(ImGui::BeginTable("Images", 2, ImGuiTableFlags_Resizable)){
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Image", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Delete");

        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableNextColumn();
        ImGui::TableHeader("Image");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Delete");

        for(const auto& [id, image]: globals::descriptor_sets){
            if(!image->flags.drawable_image)
                continue;
            
            ImGui::PushID(id.data());
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::ImageButton((ImTextureID)image->descriptor_set, {float(image->image_data->image_size.width) / image->image_data->image_size.height * _settings.image_height, _settings.image_height});
            ImGui::SameLine();
            ImGui::Text("%s", id.data());
            if(ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)){
                const std::string_view id_view = id;
                ImGui::SetDragDropPayload("image", &id_view, sizeof(id_view));
                ImGui::Text("Drop image onto plot to set as background (only for scatterplots)");
                ImGui::EndDragDropSource();
            }
            ImGui::TableNextColumn();
            if(ImGui::Button("X"))
                _popup_image_id = id;
            
            ImGui::PopID();
        }

        ImGui::EndTable();
    }

    ImGui::End();
}
}