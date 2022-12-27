#include "images_workbench.hpp"
#include <imgui.h>
#include <descriptor_set_storage.hpp>
#include <vk_util.hpp>
#include <imgui_util.hpp>

namespace workbenches
{
void images_workbench::show() 
{
    const static std::string_view delete_menu_id("image delete menu");

    if(!active)
        return;

    if(_popup_image_id.size())
        ImGui::OpenPopup(delete_menu_id.data());
    
    if(ImGui::BeginPopupModal(delete_menu_id.data(), {}, ImGuiWindowFlags_AlwaysAutoResize)){
        ImGui::Text("Do you want to delete %s?", _popup_image_id.data());
        if(ImGui::Button("Delete")){
            auto res = vkDeviceWaitIdle(globals::vk_context.device); util::check_vk_result(res);
            auto& image = globals::descriptor_sets[_popup_image_id];
            util::vk::destroy_image(image->image_data->image);
            util::vk::destroy_image_view(image->image_data->image_view);
            util::imgui::free_image_descriptor_set((ImTextureID)image->descriptor_set);
            globals::descriptor_sets.erase(_popup_image_id);
            _popup_image_id = {};
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if(ImGui::Button("Cancel")){
            _popup_image_id = {};
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    
    ImGui::Begin(id.data(), &active);

    // list with all images
    if(ImGui::BeginTable("Images", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)){
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Image");
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Delete");

        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableNextColumn();
        ImGui::TableHeader("Image");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Name");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Delete");

        for(const auto& [id, image]: globals::descriptor_sets){
            if(!image->flags.drawable_image)
                continue;
            
            ImGui::PushID(id.data());
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::ImageButton((ImTextureID)image->descriptor_set, {float(image->image_data->image_size.width) / image->image_data->image_size.height * _settings.image_height, _settings.image_height});
            if(ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)){
                const std::string_view id_view = id;
                ImGui::SetDragDropPayload("image", &id_view, sizeof(id_view));
                ImGui::Text("Drop image onto plot to set as background (only for scatterplots)");
                ImGui::EndDragDropSource();
            }
            ImGui::TableNextColumn();
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + _settings.image_height / 2 - ImGui::GetTextLineHeightWithSpacing() / 2);
            ImGui::Text("%s", id.data());
            ImGui::TableNextColumn();
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + _settings.image_height / 2 - ImGui::GetTextLineHeightWithSpacing() / 2);
            if(ImGui::Button("X"))
                _popup_image_id = id;
            
            ImGui::PopID();
        }

        ImGui::EndTable();
    }

    if(ImGui::TreeNodeEx("Settings", ImGuiTreeNodeFlags_Framed)){
        ImGui::DragFloat("Image preview height", &_settings.image_height, 1, 20, FLT_MAX);
        ImGui::TreePop();
    }

    ImGui::End();
}
}