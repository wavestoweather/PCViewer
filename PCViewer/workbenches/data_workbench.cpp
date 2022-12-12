#include "data_workbench.hpp"
#include <imgui.h>
#include <datasets.hpp>
#include <drawlists.hpp>
#include <imgui_stdlib.h>
#include <imgui_internal.h>
#include <dataset_util.hpp>
#include <open_filepaths.hpp>
#include <../imgui_file_dialog/ImGuiFileDialog.h>
#include <drawlist_util.hpp>
#include <regex>

namespace workbenches
{
void data_workbench::show()
{
    const static std::string_view popup_tl_to_brush{"Templatelist to brush"};
    const static std::string_view popup_tl_to_dltl{"Templatelist to drawlist/templatelist"};
    const static std::string_view popup_add_tl{"Add templatelist"};
    const static std::string_view popup_delete_ds{"Delete dataset"};
    const static std::string_view popup_add_empty_ds{"Add empty dataset"};

    bool popup_open_tl_to_brush{false};
    bool popup_open_tl_to_dltl{false};
    bool popup_open_add_tl{false};
    bool popup_open_split_ds{false};
    bool popup_open_delete_ds{false};
    bool popup_open_add_empty_ds{false};

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
                        _tl_convert_data.ds_id = id;
                        _tl_convert_data.tl_id = tl->name;
                        _tl_convert_data.trim = {0, tl->data_size};
                        popup_open_tl_to_dltl = true;
                    }
                    if(ImGui::Button("Add templatelist")){
                        _popup_ds_id = id;
                        popup_open_add_tl = true;
                    }
                    ImGui::PushStyleColor(ImGuiCol_Button, (ImGuiCol)IM_COL32(220, 20, 0, 230));
                    if(ImGui::Button("Delete")){
                        _popup_ds_id = id;
                        popup_open_delete_ds = true;
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
        if(ImGui::BeginTable("Drawlists", 6, ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)){
            ImGui::TableSetupScrollFreeze(0, 1);    // make top row always visible
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Active");
            ImGui::TableSetupColumn("Color");
            ImGui::TableSetupColumn("Median");
            ImGui::TableSetupColumn("Median color");
            ImGui::TableSetupColumn("Delete");
            
            // filter row
            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
            ImGui::TableNextColumn();
            if(_regex_error)
                ImGui::PushStyleColor(ImGuiCol_FrameBg, {1, 0, 0, .5});
            bool reselect_all = ImGui::InputText("Filter", &_table_filter);
            if(_regex_error)
                ImGui::PopStyleColor();
            std::regex table_regex;
            try{
                table_regex = std::regex(_table_filter);
                _regex_error = false;
            }
            catch(std::exception e){
                _regex_error = true;
                reselect_all = false;
            };
            ImGui::TableNextColumn();
            if(ImGui::Button("Deselect all")){
                globals::brush_edit_data.clear();
                globals::selected_drawlists.clear();
            }
            ImGui::TableNextColumn();
            if(ImGui::Button("Select all") || reselect_all){
                globals::brush_edit_data.clear();
                globals::selected_drawlists.clear();
                for(const auto& [id, dl]: globals::drawlists.read())
                    if(std::regex_search(id.begin(), id.end(), table_regex))
                        globals::selected_drawlists.push_back(id);
            }
            ImGui::TableNextColumn();
            if(ImGui::DragFloat("Uniform alpha", &_uniform_alpha, _uniform_alpha / 100, 1e-20, 1, "%.1g")){
                if(globals::selected_drawlists.size()){
                    for(const auto& dl: globals::selected_drawlists)
                        globals::drawlists()[dl]().appearance_drawlist().color.w = _uniform_alpha;
                }
                else{
                    for(auto& [dl_id, dl]: globals::drawlists())
                        dl().appearance_drawlist().color.w = _uniform_alpha;
                }
            }

            // top row
            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
            ImGui::TableNextColumn();
            ImGui::TableHeader("Name");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Active  ");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Color   ");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Delete          ");
            
            for(const auto& [id, dl]: globals::drawlists.read()){
                if(!std::regex_search(id.begin(), id.end(), table_regex))
                    continue;
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                bool selected = util::memory_view(globals::selected_drawlists).contains(id);
                if(ImGui::Selectable(id.data(), selected, ImGuiSelectableFlags_NoPadWithHalfSpacing)){
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
                if(ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)){
                    const std::string_view id_view = dl.read().id;
                    ImGui::SetDragDropPayload("drawlists", &id_view, sizeof(id_view));
                    ImGui::Text("Drop datasets on workbench to add");
                    ImGui::EndDragDropSource();
                }
                auto& appearance_no_track = globals::drawlists.ref_no_track()[id].ref_no_track().appearance_drawlist.ref_no_track();
                ImGui::TableNextColumn();
                if(ImGui::Checkbox(("##dlactive" + std::string(id)).c_str(), &appearance_no_track.show)){
                    if(globals::selected_drawlists.size()){
                        for(const auto& selected_dl: globals::selected_drawlists)
                            globals::drawlists()[selected_dl]().appearance_drawlist().show = appearance_no_track.show;
                    }
                    globals::drawlists()[id]().appearance_drawlist().show;  // used to mark the changes which have to be propagated
                }
                ImGui::TableNextColumn();
                int color_edit_flags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_AlphaBar;
                if(ImGui::ColorEdit4(("##dlcolor" + std::string(id)).c_str(), &appearance_no_track.color.x, color_edit_flags)){
                    if(globals::selected_drawlists.size()){
                        for(const auto& selected_dl: globals::selected_drawlists)
                            globals::drawlists()[selected_dl]().appearance_drawlist().color = appearance_no_track.color;
                    }
                    globals::drawlists()[id]().appearance_drawlist().color;
                }
                ImGui::TableNextColumn();
                if(ImGui::Button(("X##" + std::string(id)).c_str())){
                    globals::drawlists_to_delete.insert(id);
                }
            }

            ImGui::EndTable();
        }

        // c3
        ImGui::TableNextColumn();

        if(ImGui::BeginTable("global_brush_table", 3, ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)){
            ImGui::TableSetupScrollFreeze(0, 1);    // make top row always visible
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Active");
            ImGui::TableSetupColumn("Delete");
            
            // top row
            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
            ImGui::TableNextColumn();
            ImGui::TableHeader("Name");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Active");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Delete");
            ImGui::TableNextColumn();

            int brush_delete = -1;
            for(int i: util::size_range(globals::global_brushes.read())){
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                const auto& gb = globals::global_brushes.read()[i].read();
                bool selected = globals::brush_edit_data.global_brush_id == gb.id;
                if(ImGui::Selectable((gb.name + "##gbs" + std::to_string(gb.id)).c_str(), selected, ImGuiSelectableFlags_NoPadWithHalfSpacing)){
                    if(selected)
                        globals::brush_edit_data.clear();
                    else{
                        globals::selected_drawlists.clear();
                        globals::brush_edit_data.brush_type = structures::brush_edit_data::brush_type::global;
                        globals::brush_edit_data.global_brush_id = gb.id;
                    }
                }
                ImGui::TableNextColumn();
                if(ImGui::Checkbox(("##gba_" + std::to_string(gb.id)).c_str(), &globals::global_brushes.ref_no_track()[i].ref_no_track().active))
                    globals::global_brushes()[i]();
                ImGui::TableNextColumn();
                if(ImGui::Button(("X##gbd" + std::to_string(gb.id)).c_str()))
                    brush_delete = i;
            }
            if(brush_delete >= 0){
                globals::global_brushes().erase(globals::global_brushes().begin() + brush_delete);
                globals::brush_edit_data.clear();
            }

            ImGui::EndTable();
        }
        if(ImGui::Button("+ Add global brush")){
            structures::tracked_brush new_brush{};
            auto cur_id = globals::cur_global_brush_id++;
            new_brush.ref_no_track().id = cur_id;
            new_brush.ref_no_track().name = "Global brush " + std::to_string(cur_id);
            //globals::global_brushes.ref_no_track().emplace_back(structures::range_brush{}, structures::lasso_brush{}, structures::brush_id(cur_id), std::to_string(cur_id), true);
            globals::global_brushes.ref_no_track().push_back(new_brush);
            // selecting the last brush
            globals::selected_drawlists.clear();
            globals::brush_edit_data.brush_type = structures::brush_edit_data::brush_type::global;
            globals::brush_edit_data.global_brush_id =  cur_id;
        }

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
        ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    if(popup_open_delete_ds)
        ImGui::OpenPopup(popup_delete_ds.data());
    if(ImGui::BeginPopupModal(popup_delete_ds.data())){
        ImGui::Text("Do you really want to delete dataset %s?", _popup_ds_id.data());
        if(ImGui::Button("Cancel"))
            ImGui::CloseCurrentPopup();
        ImGui::SameLine();
        if(ImGui::Button("Confirm")){
            globals::datasets_to_delete.insert(_popup_ds_id);
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if(ImGui::BeginPopupModal(popup_add_empty_ds.data())){
        // TODO: implemnet
        ImGui::EndPopup();
    }
}
}