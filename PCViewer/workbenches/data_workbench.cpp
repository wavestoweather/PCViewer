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
#include <drawlist_colors_workbench.hpp>

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
        for(const auto& [ds_id, dataset]: globals::datasets.read()){
            if(ImGui::TreeNode(ds_id.data())){
                for(const auto& tl: dataset.read().templatelists){
                    if(ImGui::MenuItem(tl->name.c_str())){
                        _popup_tl_id = tl->name;
                        _popup_ds_id = ds_id;
                        _tl_convert_data.ds_id = ds_id;
                        _tl_convert_data.tl_id = tl->name;
                        _tl_convert_data.trim = {0, tl->data_size};
                        popup_open_tl_to_dltl = true;
                    }
                }
                if(ImGui::Button("Add templatelist")){
                    _popup_ds_id = ds_id;
                    popup_open_add_tl = true;
                }
                ImGui::PushStyleColor(ImGuiCol_Button, (ImGuiCol)IM_COL32(220, 20, 0, 230));
                if(ImGui::Button("Delete")){
                    _popup_ds_id = ds_id;
                    popup_open_delete_ds = true;
                }
                ImGui::PopStyleColor();
                ImGui::TreePop();
            }
        }
        if(ImGui::Button("Add empty dataset")){
            ImGui::OpenPopup(popup_add_empty_ds.data());
        }

        // c2
        ImGui::TableNextColumn();

        // filter row
        if(_regex_error)
            ImGui::PushStyleColor(ImGuiCol_FrameBg, {1, 0, 0, .5});
        ImGui::PushItemWidth(150);
        bool reselect_all = ImGui::InputText("Filter", &_table_filter) && _table_filter.size();
        ImGui::PopItemWidth();
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
        ImGui::SameLine();
        if(ImGui::Button("Deselect all")){
            globals::brush_edit_data.clear();
            globals::selected_drawlists.clear();
        }
        ImGui::SameLine();
        if(ImGui::Button("Select all") || reselect_all){
            globals::brush_edit_data.clear();
            globals::selected_drawlists.clear();
            for(const auto& [dl_id, dl]: globals::drawlists.read())
                if(std::regex_search(dl_id.begin(), dl_id.end(), table_regex))
                    globals::selected_drawlists.push_back(dl_id);
        }
        ImGui::SameLine();
        ImGui::PushItemWidth(150);
        _uniform_alpha = 0;
        int alpha_count = 0;
        for(const auto& [dl_id, dl]: globals::drawlists.read()){
            if(std::regex_search(dl_id.begin(), dl_id.end(), table_regex)){
                _uniform_alpha += dl.read().appearance_drawlist.read().color.w;
                ++alpha_count;
            }
        }
        if(alpha_count) _uniform_alpha /= alpha_count;
        if(ImGui::DragFloat("Uniform alpha", &_uniform_alpha, _uniform_alpha / 200, std::max(1e-20, _uniform_alpha * (180./200)), std::min(1., _uniform_alpha * (220./200)), "%.3g")){
            if(globals::selected_drawlists.size()){
                for(const auto& dl: globals::selected_drawlists)
                    globals::drawlists()[dl]().appearance_drawlist().color.w = _uniform_alpha;
            }
            else{
                for(auto& [dl_id, dl]: globals::drawlists())
                    dl().appearance_drawlist().color.w = _uniform_alpha;
            }
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if(ImGui::Button("Drawlist colors"))
            util::memory_view(globals::workbenches).find([](const globals::unique_workbench& wb){return wb->id == globals::drawlist_color_wb_id;})->active = true;
        

        if(ImGui::BeginTable("Drawlists", 6, ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)){
            ImGui::TableSetupScrollFreeze(0, 1);    // make top row always visible
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Active");
            ImGui::TableSetupColumn("Color");
            ImGui::TableSetupColumn("Median");
            ImGui::TableSetupColumn("Median color");
            ImGui::TableSetupColumn("Delete");

            // top row
            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
            ImGui::TableNextColumn();
            ImGui::TableHeader("Name");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Active");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Color");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Delete");
            
            ImGui::PushID(id.data());
            for(const auto& [dl_id, dl]: globals::drawlists.read()){
                if(!std::regex_search(dl_id.begin(), dl_id.end(), table_regex))
                    continue;
                ImGui::PushID(dl_id.data());
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                bool selected = util::memory_view(globals::selected_drawlists).contains(dl_id);
                if(ImGui::Selectable(dl_id.data(), selected, ImGuiSelectableFlags_NoPadWithHalfSpacing, {0, ImGui::GetTextLineHeightWithSpacing()})){
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
                        globals::selected_drawlists.push_back(dl_id);
                    else
                        globals::selected_drawlists = {dl_id};

                    // updating the brush selection
                    globals::brush_edit_data.clear();
                    if(globals::selected_drawlists.size()){
                        globals::brush_edit_data.brush_type = structures::brush_edit_data::brush_type::local;
                        globals::brush_edit_data.local_brush_id = dl_id;
                    }
                }
                if(ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)){
                    const std::string_view id_view = dl.read().id;
                    ImGui::SetDragDropPayload("drawlists", &id_view, sizeof(id_view));
                    ImGui::Text("Drop datasets on workbench to add");
                    ImGui::EndDragDropSource();
                }
                auto& appearance_no_track = globals::drawlists.ref_no_track()[dl_id].ref_no_track().appearance_drawlist.ref_no_track();
                ImGui::TableNextColumn();
                if(ImGui::Checkbox("##dlactive", &appearance_no_track.show)){
                    if(globals::selected_drawlists | util::contains(dl_id)){
                        for(const auto& selected_dl: globals::selected_drawlists)
                            globals::drawlists()[selected_dl]().appearance_drawlist().show = appearance_no_track.show;
                    }
                    globals::drawlists()[dl_id]().appearance_drawlist().show;  // used to mark the changes which have to be propagated
                }
                ImGui::TableNextColumn();
                int color_edit_flags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_AlphaBar;
                if(ImGui::ColorEdit4("##dlcolor", &appearance_no_track.color.x, color_edit_flags)){
                    if(globals::selected_drawlists | util::contains(dl_id)){
                        for(const auto& selected_dl: globals::selected_drawlists)
                            globals::drawlists()[selected_dl]().appearance_drawlist().color = appearance_no_track.color;
                    }
                    globals::drawlists()[dl_id]().appearance_drawlist().color;
                }
                ImGui::TableNextColumn();
                if(ImGui::Button("X##")){
                    globals::drawlists_to_delete.insert(dl_id);
                }
                ImGui::PopID();
            }
            ImGui::PopID();

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
                if(ImGui::Selectable((gb.name + "##gbs" + std::to_string(gb.id)).c_str(), selected, ImGuiSelectableFlags_NoPadWithHalfSpacing, {0, ImGui::GetTextLineHeightWithSpacing()})){
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
    using uniform_split = structures::templatelist_split_data::uniform_value_split;
    using value_split = structures::templatelist_split_data::value_split;
    using quantile_split = structures::templatelist_split_data::quantile_split;
    using automatic_split = structures::templatelist_split_data::automatic_split;
    if(popup_open_tl_to_dltl)
        ImGui::OpenPopup(popup_tl_to_dltl.data());
    if(ImGui::BeginPopupModal(popup_tl_to_dltl.data(), {}, ImGuiWindowFlags_AlwaysAutoResize)){
        const auto& ds = globals::datasets.read().at(_popup_ds_id).read();
        const auto& tl = *ds.templatelist_index.at(_popup_tl_id);
        if(ImGui::BeginTabBar("Destination")){
            if(ImGui::BeginTabItem("Drawlist")){
                _tl_convert_data.dst = structures::templatelist_convert_data::destination::drawlist;
                _tl_split_data.create_drawlists = true;
                ImGui::Text("Creating a DRAWLIST list from %s", tl.name.c_str());
                ImGui::EndTabItem();
            }
            if(ImGui::BeginTabItem("TemplateList")){
                _tl_convert_data.dst = structures::templatelist_convert_data::destination::templatelist;
                _tl_split_data.create_drawlists = false;
                ImGui::Text("Creating a TEMPLATELIST from %s", tl.name.c_str());
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        bool split{true};
        _tl_split_data.attribute = std::min(_tl_split_data.attribute, int(ds.attributes.size()) - 1);
        if(ImGui::BeginTabBar("Trim/Subsample/Split")){
            if(ImGui::BeginTabItem("Subsample/Trim")){
                split = false;
                ImGui::Checkbox("Random subsampling (If enabled subsampling rate is transformed into probaility)", &_tl_convert_data.random_subsampling);
                if(ImGui::InputInt("Subsampling Rate", &_tl_convert_data.subsampling)) _tl_convert_data.subsampling = std::max(_tl_convert_data.subsampling, 1);
                if(ImGui::InputScalarN("Trim indcies", ImGuiDataType_U64, _tl_convert_data.trim.data(), 2)){
                    _tl_convert_data.trim.min = std::clamp<size_t>(_tl_convert_data.trim.min, 0u, _tl_convert_data.trim.max - 1);
                    _tl_convert_data.trim.max = std::clamp<size_t>(_tl_convert_data.trim.max, _tl_convert_data.trim.min + 1, size_t(tl.data_size));
                }
                ImGui::EndTabItem();
            }
            if(ImGui::BeginTabItem("Uniform Value Split")){
                if(ImGui::BeginCombo("Split axis", ds.attributes[_tl_split_data.attribute].display_name.c_str())){
                    for(int att: util::size_range(ds.attributes)){
                        if(ImGui::MenuItem(ds.attributes[att].display_name.c_str())) _tl_split_data.attribute = att;
                    }
                    ImGui::EndCombo();
                }
                if(!std::holds_alternative<uniform_split>(_tl_split_data.additional_info))
                    _tl_split_data.additional_info = uniform_split{};
                ImGui::InputInt("Amount of split groups", &std::get<uniform_split>(_tl_split_data.additional_info).split_count);
                ImGui::EndTabItem();
            }
            if(ImGui::BeginTabItem("Value Split")){
                if(!std::holds_alternative<value_split>(_tl_split_data.additional_info))
                    _tl_split_data.additional_info = value_split{{ds.attributes[_tl_split_data.attribute].bounds.read().min, ds.attributes[_tl_split_data.attribute].bounds.read().max}};
                if(ImGui::BeginCombo("Split axis", ds.attributes[_tl_split_data.attribute].display_name.c_str())){
                    for(int att: util::size_range(ds.attributes)){
                        if(ImGui::MenuItem(ds.attributes[att].display_name.c_str())){
                            _tl_split_data.attribute = att;
                            std::get<value_split>(_tl_split_data.additional_info).values = {ds.attributes[_tl_split_data.attribute].bounds.read().min, ds.attributes[_tl_split_data.attribute].bounds.read().max};
                        }
                    }
                    ImGui::EndCombo();
                }

                auto& values = std::get<value_split>(_tl_split_data.additional_info).values;
                int delete_item{-1}, add_item{-1};
                ImGui::Text("Split values:");
                for(int i: util::size_range(values)){
                    float min = ds.attributes[_tl_split_data.attribute].bounds.read().min, max = ds.attributes[_tl_split_data.attribute].bounds.read().max, speed = .01f;
                    if(i == 0) speed = 0.0000000001;
                    else if(i == values.size() - 1) speed = 0.000000001;
                    else {min = values[i - 1], max = values[i + 1]; speed = (max - min) / 500;}
                    ImGui::DragFloat(("##quantile" + std::to_string(i)).c_str(), values.data() + i, speed, min, max);
                    if(i != 0 && i != values.size()-1){
                        ImGui::SameLine();
                        if(ImGui::Button(("X##deleteQuant" + std::to_string(i)).c_str())) delete_item = i;
                    }
                    if(i < values.size() - 1){
                        static float buttonHeight = 10;
                        static float space = 5;
                        float prevCursorPosY = ImGui::GetCursorPosY();
                        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetTextLineHeightWithSpacing() / 2.0f + space);
                        if(ImGui::Button(("##addButton" + std::to_string(i)).c_str(), ImVec2(250,buttonHeight))){
                            add_item = i;
                        }
                        ImGui::SetCursorPosY(prevCursorPosY + space);
                    }
                }
                if(add_item >= 0) values.insert(values.begin() + add_item + 1, (values[add_item] + values[add_item + 1]) / 2.0f);
                if(delete_item >= 0) values.erase(values.begin() + delete_item);

                if(ImGui::Button("Unify value differences")){
                    for(int i: util::i_range(1, values.size() - 1))
                        values[i] = i * (values.back() - values.front()) / (values.size() - 1);
                }

                ImGui::EndTabItem();
            }
            if(ImGui::BeginTabItem("Quantile Split")){
                if(!std::holds_alternative<quantile_split>(_tl_split_data.additional_info))
                    _tl_split_data.additional_info = quantile_split{{.0f, 1.f}};
                if(ImGui::BeginCombo("Split axis", ds.attributes[_tl_split_data.attribute].display_name.c_str())){
                    for(int att: util::size_range(ds.attributes)){
                        if(ImGui::MenuItem(ds.attributes[att].display_name.c_str())){
                            _tl_split_data.attribute = att;
                            std::get<quantile_split>(_tl_split_data.additional_info).quantiles = {.0f,1.f};
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::Text("Split quantlies:");
                auto& quantiles = std::get<quantile_split>(_tl_split_data.additional_info).quantiles;
                int delete_item{-1}, add_item{-1};
                for(int i: util::size_range(quantiles)){
                    float min = 0, max = 1, speed = .01f;
                    if(i == 0) speed = 0.0000000001;
                    else if(i == quantiles.size() - 1) speed = 0.000000001;
                    else {min = quantiles[i - 1], max = quantiles[i + 1];}
                    ImGui::DragFloat(("##quantile" + std::to_string(i)).c_str(), quantiles.data() + i, speed, min, max);
                    if(i != 0 && i != quantiles.size()-1){
                        ImGui::SameLine();
                        if(ImGui::Button(("X##deleteQuant" + std::to_string(i)).c_str())) delete_item = i;
                    }
                    if(i < quantiles.size() - 1){
                        static float buttonHeight = 10;
                        static float space = 5;
                        float prevCursorPosY = ImGui::GetCursorPosY();
                        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetTextLineHeightWithSpacing() / 2.0f + space);
                        if(ImGui::Button(("##addButton" + std::to_string(i)).c_str(), ImVec2(250,buttonHeight))){
                            add_item = i;
                        }
                        ImGui::SetCursorPosY(prevCursorPosY + space);
                    }
                }
                if(add_item >= 0) quantiles.insert(quantiles.begin() + add_item + 1, (quantiles[add_item] + quantiles[add_item + 1]) / 2.0f);
                if(delete_item >= 0) quantiles.erase(quantiles.begin() + delete_item);

                if(ImGui::Button("Unify quantiles")){
                    for(int i: util::i_range(1, quantiles.size() - 1))
                        quantiles[i] = i / double(quantiles.size() - 1);
                }

                ImGui::EndTabItem();
            }
            if(ImGui::BeginTabItem("Automatic Split")){
                if(!std::holds_alternative<automatic_split>(_tl_split_data.additional_info))
                    _tl_split_data.additional_info = automatic_split{};
                ImGui::Text("Only select variables with discrete values.\n If too much values wrt data size are found, no split will be performed.");
                if(ImGui::BeginCombo("Split axis", ds.attributes[_tl_split_data.attribute].display_name.c_str())){
                    for(int att: util::size_range(ds.attributes)){
                        if(ImGui::MenuItem(ds.attributes[att].display_name.c_str())){
                            _tl_split_data.attribute = att;
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        
        ImGui::InputText("Output name", split ? &_tl_split_data.dst_name_format: &_tl_convert_data.dst_name);
        if(ImGui::IsItemHovered()){
            ImGui::BeginTooltip();
            ImGui::Text("for split enter a format string to distinguish resulting drawlists, eg. test_%%d");
            ImGui::EndTooltip();
        }

        if(ImGui::Button("Create") || ImGui::IsKeyPressed(ImGuiKey_Enter)){
            ImGui::CloseCurrentPopup();
            _tl_convert_data.ds_id = ds.id;
            _tl_convert_data.tl_id = tl.name;
            _tl_split_data.ds_id = ds.id;
            _tl_split_data.tl_id = tl.name;
            if(split)
                util::dataset::split_templatelist(_tl_split_data);
            else
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