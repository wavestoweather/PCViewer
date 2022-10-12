#include "data_workbench.hpp"
#include <imgui.h>
#include <datasets.hpp>
#include <drawlists.hpp>
#include <imgui_stdlib.h>
#include <dataset_util.hpp>
#include <open_filepaths.hpp>
#include <../imgui_file_dialog/ImGuiFileDialog.h>
#include <drawlist_util.hpp>

namespace workbenches
{
void data_workbench::show()
{
    const static std::string_view popup_tl_to_brush{"Templatelist to brush"};
    const static std::string_view popup_tl_to_dltl{"Templatelist to drawlist/templatelist"};
    const static std::string_view popup_add_tl{"Add templatelist"};
    const static std::string_view popup_split_ds{"Split dataset"};
    const static std::string_view popup_delete_ds{"Delete dataset"};
    const static std::string_view popup_add_empty_ds{"Add empty dataset"};
    const static std::string_view popup_delete_dl{"Delete drawlist"};

    bool popup_open_tl_to_brush{false};
    bool popup_open_tl_to_dltl{false};
    bool popup_open_add_tl{false};
    bool popup_open_split_ds{false};
    bool popup_open_delete_ds{false};
    bool popup_open_add_empty_ds{false};
    bool popup_open_delete_dl{false};

    ImGui::Begin(id.c_str());
    // 3 column layout with the following layout
    //  |   c1      |    c2     |    c3     |
    //  | datasets  | drawlists | global brushes|
    if(ImGui::BeginTable("data_workbench_cols", 3, ImGuiTableFlags_Resizable)){
        ImGui::TableSetupColumn("Datasets");
        ImGui::TableSetupColumn("Drawlists");
        ImGui::TableSetupColumn("Global brushes");
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableNextColumn();
        ImGui::TableHeader("Datasets");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Drawlists");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Global brushes");

        ImGui::TableNextRow();

        // c1
        ImGui::TableNextColumn();

        bool open = ImGui::InputText("Directory Path", &_open_filename, ImGuiInputTextFlags_EnterReturnsTrue);
		if (ImGui::IsItemHovered()) {
			ImGui::BeginTooltip();
			ImGui::Text("Enter either a file including filepath,\nOr a folder (division with /) and all datasets in the folder will be loaded\nOr drag and drop files to load onto application.");
			ImGui::EndTooltip();
		}

		ImGui::SameLine();

		//Opening a new Dataset into the Viewer
		if (ImGui::Button("Open") || open) {
			if(_open_filename.empty()){
				// opening the file dialogue
				ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".*,.nc,.csv", ".", 0);
			}
            else{
                globals::paths_to_open.push_back(_open_filename);
            }
        }
        ImGui::Separator();
        for(const auto& [id, dataset]: globals::datasets.read()){
            if(ImGui::TreeNode(id.data())){
                for(const auto& tl: dataset.read().templatelists){
                    if(ImGui::MenuItem(tl->name.c_str())){
                        _popup_tl_id = tl->name;
                        _popup_ds_id = id;
                        _tl_convert_data.trim = {0, tl->data_size};
                        popup_open_tl_to_dltl = true;
                    }
                    if(ImGui::Button("Add templatelist")){
                        _popup_ds_id = id;
                        popup_open_add_tl = true;
                    }
                    if(ImGui::Button("Split dataset")){
                        _popup_ds_id = id;
                        ImGui::OpenPopup(popup_split_ds.data());
                    }
                    ImGui::PushStyleColor(ImGuiCol_Button, (ImGuiCol)IM_COL32(220, 20, 0, 230));
                    if(ImGui::Button("Delete")){
                        _popup_ds_id = id;
                        ImGui::OpenPopup(popup_delete_ds.data());
                    }
                    ImGui::PopStyleColor();
                }
                ImGui::TreePop();
            }
        }
        if(ImGui::Button("Add empty dataset")){
            ImGui::OpenPopup(popup_add_empty_ds.data());
        }

        // c2
        ImGui::TableNextColumn();
        if(ImGui::BeginTable("Drawlists", 6, ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_SizingFixedFit)){
            ImGui::TableSetupScrollFreeze(0, 1);    // make top row always visible
            ImGui::TableSetupColumn("Drawlist", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Delete");
            ImGui::TableSetupColumn("Active");
            ImGui::TableSetupColumn("Color");
            ImGui::TableSetupColumn("Median");
            ImGui::TableSetupColumn("Median color");
            
            // top row
            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
            ImGui::TableNextColumn();
            ImGui::TableHeader("Drawlist");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Delete");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Active");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Color");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Median");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Median color");
            
            for(const auto& [id, dl]: globals::drawlists.read()){
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                bool selected = util::memory_view(globals::selected_drawlists).contains(id);
                if(ImGui::Selectable(id.data(), selected)){
                    // updating drawlist selection
                    if(selected && ImGui::GetIO().KeyCtrl)
                        globals::selected_drawlists.erase(std::find(globals::selected_drawlists.begin(), globals::selected_drawlists.end(), id));
                    else if(selected)
                        globals::selected_drawlists.clear();
                    else if(ImGui::GetIO().KeyShift){
                        uint32_t start_index = util::drawlist::drawlist_index(globals::selected_drawlists.back());
                        uint32_t end_index = ImGui::TableGetRowIndex() - 1;
                        if(start_index > end_index)
                            std::swap(start_index, end_index);
                        uint32_t cur_ind{};
                        for(const auto& [id_, dl_]: globals::drawlists.read()){
                            if(cur_ind > end_index)
                                break;
                            if(cur_ind > start_index)
                                globals::selected_drawlists.push_back(id_);
                            cur_ind++;
                        }
                    }
                    else if(ImGui::GetIO().KeyCtrl)
                        globals::selected_drawlists.push_back(id);
                    else
                        globals::selected_drawlists = {id};

                    // updating the brush selection
                    globals::brush_edit_data.clear();
                    if(globals::selected_drawlists.size()){
                        globals::brush_edit_data.brush_type = structures::brush_edit_data::brush_type::local;
                        globals::brush_edit_data.local_brush_id = id;
                    }
                }
                ImGui::TableNextColumn();
                if(ImGui::Button(("X##" + std::string(id)).c_str())){
                    _popup_ds_id = id;
                    ImGui::OpenPopup(popup_delete_dl.data());
                }
                auto& appearance_no_track = globals::drawlists.ref_no_track()[id].ref_no_track().appearance_drawlist.ref_no_track();
                ImGui::TableNextColumn();
                if(ImGui::Checkbox(("##dlactive" + std::string(id)).c_str(), &appearance_no_track.show))
                    globals::drawlists()[id]().appearance_drawlist().show;  // used to mark the changes which have to be propagated
                ImGui::TableNextColumn();
                int color_edit_flags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_AlphaBar;
                if(ImGui::ColorEdit4(("##dlcolor" + std::string(id)).c_str(), &appearance_no_track.color.x, color_edit_flags))
                    globals::drawlists()[id]().appearance_drawlist().color;
                ImGui::TableNextColumn();
                auto& median_no_track = globals::drawlists.ref_no_track()[id].ref_no_track().appearance_median.ref_no_track();
                ImGui::SetNextItemWidth(100);
                if(ImGui::BeginCombo(("##medtyp" + std::string(id)).c_str(), structures::median_type_names[dl.read().median_typ.read()].data())){
                    for(auto median_type: structures::median_iteration{}){
                        if(ImGui::MenuItem(structures::median_type_names[median_type].data()))
                            globals::drawlists()[id]().median_typ = median_type;
                    }
                    ImGui::EndCombo();
                }
                ImGui::TableNextColumn();
                if(ImGui::ColorEdit4(("##dlmedc" + std::string(id)).c_str(), &median_no_track.color.x, color_edit_flags))
                    globals::drawlists()[id]().appearance_median().color;
            }

            ImGui::EndTable();
        }

        // c3
        ImGui::TableNextColumn();


        ImGui::EndTable();
    }

    ImGui::End();

    // popups -------------------------------------------------------
    if(popup_open_tl_to_dltl)
        ImGui::OpenPopup(popup_tl_to_dltl.data());
    if(ImGui::BeginPopupModal(popup_tl_to_dltl.data())){
        const auto& tl = *globals::datasets.read().at(_popup_ds_id).read().templatelist_index.at(_popup_tl_id);
		if(ImGui::BeginTabBar("Destination")){
			if(ImGui::BeginTabItem("Drawlist")){
				_tl_convert_data.dst = structures::templatelist_convert_data::destination::drawlist;
				ImGui::Text("%s", (std::string("Creating a DRAWLIST list from ") + tl.name).c_str());
				ImGui::EndTabItem();
			}
			if(ImGui::BeginTabItem("TemplateList")){
				_tl_convert_data.dst = structures::templatelist_convert_data::destination::templatelist;
				ImGui::Text("%s", (std::string("Creating a TEMPLATELIST from ") + tl.name).c_str());
				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
		}
        ImGui::InputText("Output name", &_tl_convert_data.dst_name);
        if(ImGui::CollapsingHeader("Subsample/Trim")){
            ImGui::Checkbox("Random subsampling (If enabled subsampling rate is transformed into probaility)", &_tl_convert_data.random_subsampling);
            if(ImGui::InputInt("Subsampling Rate", &_tl_convert_data.subsampling)) _tl_convert_data.subsampling = std::max(_tl_convert_data.subsampling, 1);
            if(ImGui::InputScalarN("Trim indcies", ImGuiDataType_U64, _tl_convert_data.trim.data(), 2)){
				_tl_convert_data.trim.min = std::clamp<size_t>(_tl_convert_data.trim.min, 0u, _tl_convert_data.trim.max - 1);
				_tl_convert_data.trim.max = std::clamp<size_t>(_tl_convert_data.trim.max, _tl_convert_data.trim.min + 1, size_t(tl.data_size));
			}
        }

        if(ImGui::Button("Create") || ImGui::IsKeyPressed(ImGuiKey_Enter)){
            ImGui::CloseCurrentPopup();
            util::dataset::convert_templatelist(_tl_convert_data);
        }
        ImGui::SameLine();
        if(ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape))
            ImGui::CloseCurrentPopup();

        ImGui::EndPopup();
    }

    if(ImGui::BeginPopupModal(popup_tl_to_brush.data())){
        // TODO: implement
        ImGui::EndPopup();
    }

    if(popup_open_add_tl)
        ImGui::OpenPopup(popup_add_tl.data());
    if(ImGui::BeginPopupModal(popup_add_tl.data())){
        // TODO: implemnet
        ImGui::EndPopup();
    }

    if(ImGui::BeginPopupModal(popup_split_ds.data())){
        // TODO: implemnet
        ImGui::EndPopup();
    }

    if(ImGui::BeginPopupModal(popup_delete_ds.data())){
        // TODO: implemnet
        ImGui::EndPopup();
    }

    if(ImGui::BeginPopupModal(popup_add_empty_ds.data())){
        // TODO: implemnet
        ImGui::EndPopup();
    }

    if(ImGui::BeginPopupModal(popup_delete_dl.data())){
        // TODO: implemnet
        ImGui::EndPopup();
    }
}
}