#include "violin_drawlist_workbench.hpp"
#include <imgui.h>
#include <violin_util.hpp>
#include <histogram_registry_util.hpp>
#include <stager.hpp>

namespace workbenches{
void violin_drawlist_workbench::_update_attribute_histograms(){
    // check for still active histogram update
    for(const auto& [dl_id, dl]: session_state.drawlists){
        if(_registered_histograms.contains(dl.drawlist_id) && _registered_histograms[dl.drawlist_id].size()){
            auto access = dl.drawlist_read().histogram_registry.const_access();
            if(!access->dataset_update_done)    
                return;
        }
    }

    // blending the histogram values according to the settings ---------------------------------------------
    const auto active_drawlist_attributes = get_active_drawlist_attributes();
    std::tie(_global_max, _per_attribute_max, _drawlist_attribute_histograms) = util::violins::update_histograms(active_drawlist_attributes, settings.read().smoothing_std_dev, settings.read().histogram_bin_count, session_state.attributes.size(), settings.read().ignore_zero_bins, session_state.attribute_log);

    settings.changed = false;
    for(auto& [dl, registrators]: _registered_histograms){
        // locking the registry before signaling the reigstrators
        auto registry_lock = globals::drawlists.read().at(dl).read().histogram_registry.const_access();
        for(auto& registrator: registrators)
            registrator.signal_registry_used();
    }
}

void violin_drawlist_workbench::_update_registered_histograms(){
    auto active_drawlist_attributes = get_active_drawlist_attributes();
    auto active_attribute_indices = get_active_indices();
    for(const auto& [dl_id, dl]: session_state.drawlists){

        std::vector<bool> registrator_needed(_registered_histograms[dl.drawlist_id].size(), false);
        for(auto a: active_attribute_indices){
            auto registrator_id = util::histogram_registry::get_id_string(a, settings.read().histogram_bin_count, false, false);
            int registrator_index{-1};
            for(int j: util::size_range(_registered_histograms[dl.drawlist_id])){
                if(_registered_histograms[dl.drawlist_id][j].registry_id == registrator_id){
                    registrator_index = j;
                    break;
                }
            }
            if(registrator_index >= 0)
                registrator_needed[registrator_index] = true;
            else{
                // adding the new histogram
                auto& drawlist = dl.drawlist_write();
                _registered_histograms[dl.drawlist_id].emplace_back(*drawlist.histogram_registry.access(), a, settings.read().histogram_bin_count, false, false, true);
                registrator_needed.push_back(true);
            }
        }
        // removing unused registrators
        // locking registry
        auto registry_lock = dl.drawlist_read().histogram_registry.const_access();
        for(int i: util::rev_size_range(_registered_histograms[dl.drawlist_id])){
            if(!registrator_needed[i])
                _registered_histograms[dl.drawlist_id].erase(_registered_histograms[dl.drawlist_id].begin() + i);
        }
        // printing out the registrators
        if(logger.logging_level >= logging::level::l_5){
            logger << logging::info_prefix << " violin_drawlist_workbench (" << active_attribute_indices.size() << " attributes, " << registry_lock->registry.size() <<" registrators, " << registry_lock->name_to_registry_key.size() << " name to registry entries), registered histograms: ";
            for(const auto& [key, val]: registry_lock->registry)
                logger << val.hist_id << " ";
            logger << logging::endl;
        }
    }
}

violin_drawlist_workbench::violin_drawlist_workbench(std::string_view id): workbench(id){
}

void violin_drawlist_workbench::show(){
    const std::string_view drag_type_matrix{"dl_matrix_element"};

    if(!active)
        return;

    // checking updates

    ImGui::Begin(id.data(), &active);
    ImGui::PushID(id.data());

    // violin plots ---------------------------------------------------------------------------------------
    const float max_width = ImGui::GetWindowContentRegionWidth();
    const ImVec2 base_pos = ImGui::GetCursorScreenPos();
    const float plot_width_padded = max_width / settings.read().matrix_dimensions[1];
    const float plot_width = plot_width_padded - settings.read().plot_padding;
    for(auto&& [dl_id, i]: util::enumerate(session_state.matrix_elements)){
        ImGui::PushID(dl_id.data());
        
        const int x = i / settings.read().matrix_dimensions[1];
        const int y = i % settings.read().matrix_dimensions[1];

        const ImVec2 plot_min{base_pos.x + y * plot_width_padded, base_pos.y + x * (settings.read().plot_height + settings.read().plot_padding)};
        const ImVec2 plot_max{plot_min.x + plot_width, plot_min.y + settings.read().plot_height};
        
        // drawing the background
        ImGui::GetWindowDrawList()->AddRectFilled(plot_min, plot_max, ImGui::ColorConvertFloat4ToU32(settings.read().plot_background), 4);
        
        // drag drop stuff
        ImGui::SetCursorScreenPos(plot_min);
        ImGui::BeginDisabled(dl_id.empty());
        ImGui::InvisibleButton("invis", {plot_max.x - plot_min.x, plot_max.y - plot_min.y});
        ImGui::EndDisabled();
        if(ImGui::BeginDragDropTarget()){
            if(auto payload = ImGui::AcceptDragDropPayload(drag_type_matrix.data()))
                dl_id = *reinterpret_cast<std::string_view*>(payload->Data);

            ImGui::EndDragDropTarget();
        }
        if(ImGui::BeginDragDropSource()){
            ImGui::SetDragDropPayload(drag_type_matrix.data(), &dl_id, sizeof(dl_id));
            ImGui::Text("%s", dl_id.data());
            ImGui::EndDragDropSource();
        }

        if(dl_id.empty() || !session_state.drawlists[dl_id].appearance->read().show)
            continue;
        if(x >= settings.read().matrix_dimensions[0])
            break;

        // infill drawing
        for(const auto& [att, active]: util::rev_iter(session_state.attribute_order_infos)){
            if(!active)
                continue;
            
            const auto& violin_app = session_state.attribute_violin_appearances[att];
            assert(_drawlist_attribute_histograms.contains({dl_id, att}));
            const auto& histogram = _drawlist_attribute_histograms[{dl_id, att}];
            // TODO adjust max val
            const float hist_normalization_fac = histogram.max_val;
            util::violins::imgui_violin_infill(plot_min, plot_max, histogram.smoothed_values, hist_normalization_fac, violin_app);
        }
        // border drawing
        for(const auto& [att, active]: util::rev_iter(session_state.attribute_order_infos)){
            if(!active)
                continue;
            
            const auto& violin_app = session_state.attribute_violin_appearances[att];
            assert(_drawlist_attribute_histograms.contains({dl_id, att}));
            const auto& histogram = _drawlist_attribute_histograms[{dl_id, att}];
            // TODO adjust max val
            const float hist_normalization_fac = histogram.max_val;
            float line_thickness = settings.read().line_thickness;
            const bool hovered_plot = _hovered_dl_attribute == std::tuple{dl_id, att};
            if(hovered_plot)
                line_thickness *= 2;
            float hover_val = util::violins::imgui_violin_border(plot_min, plot_max, histogram.smoothed_values, hist_normalization_fac, violin_app, line_thickness);
            if(isnanf(hover_val)){
                if(hovered_plot)
                    _hovered_dl_attribute = {};
                continue;
            }
            _hovered_dl_attribute = {dl_id, att};
        }
        ImGui::PopID();
    }

    // drawlists and settings -----------------------------------------------------------------------------
    if(ImGui::BeginTable("settings", 3, ImGuiTableFlags_Resizable)){
        ImGui::TableSetupColumn("General Settings");
        ImGui::TableSetupColumn("Attribute Settings");
        ImGui::TableSetupColumn("Drawlist Settings");

        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableNextColumn();
        ImGui::TableHeader("Settings");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Attributes");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Drawlists");

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        if(ImGui::DragInt2("Matrix dimensions", settings.read().matrix_dimensions.data(), 1, 1, 100))
            session_state.matrix_elements.resize(settings.read().matrix_dimensions[0] * settings.read().matrix_dimensions[1], {});
        ImGui::ColorEdit4("##cols", &settings.read().plot_background.x, ImGuiColorEditFlags_NoInputs);
        ImGui::DragFloat("Line hover distance", &settings.read().line_hover_dist, 1, 1, 200);
        ImGui::DragFloat("Line alpha", &settings.read().line_alpha, .05, 0, 1);
        ImGui::DragFloat("Area alpha", &settings.read().area_alpha, .05, 0, 1);
        if(ImGui::InputInt("Bin count", &settings.ref_no_track().histogram_bin_count, 1, 100, ImGuiInputTextFlags_EnterReturnsTrue))
            settings();
        if(ImGui::DragFloat("Smoothing std dev", &settings.ref_no_track().smoothing_std_dev, 1, -1, 500))
            settings();
        if(ImGui::Checkbox("Ignore 0 bins", &settings.ref_no_track().ignore_zero_bins))
            settings();
        ImGui::DragFloat("Plot height", &settings.read().plot_height, 1, 5, 50000);
        ImGui::DragFloat("Plot padding", &settings.read().plot_padding, 1, 0, 100);
        ImGui::Checkbox("Auto-repositon attributes", &settings.read().reposition_attributes_on_update);

        ImGui::TableNextColumn();
        if(ImGui::BeginTable("attributes", 7, ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)){
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Up");
            ImGui::TableSetupColumn("Down");
            ImGui::TableSetupColumn("Show");
            ImGui::TableSetupColumn("Color");
            ImGui::TableSetupColumn("Position");
            ImGui::TableSetupColumn("Log");

            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
            ImGui::TableNextColumn();
            ImGui::TableHeader("Name");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Up");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Down");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Show");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Color");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Position");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Log");

            int up_index{-1}, down_index{-1};
            for(auto&& [att, i]: util::enumerate(session_state.attribute_order_infos)){
                const auto& attribute = session_state.attributes[att.attribut_index];
                auto& att_app = session_state.attribute_violin_appearances[att.attribut_index];
                ImGui::PushID(attribute.id.c_str());

                ImGui::TableNextRow();
                ImGui::NextColumn();
                ImGui::Text("%s", attribute.display_name.c_str());
                ImGui::NextColumn();
                ImGui::BeginDisabled(i == 0);
                if(ImGui::ArrowButton("##up", ImGuiDir_Up))
                    up_index = i;
                ImGui::EndDisabled();
                ImGui::TableNextColumn();
                ImGui::BeginDisabled(i == session_state.attribute_order_infos.size() - 1);
                if(ImGui::ArrowButton("##do", ImGuiDir_Down))
                    down_index = i;
                ImGui::EndDisabled();
                ImGui::TableNextColumn();
                ImGui::Checkbox("##en", &att.active);
                ImGui::TableNextColumn();
                ImGui::ColorEdit4("##col", &session_state.attribute_violin_appearances[att.attribut_index].color.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
                ImGui::TableNextColumn();
                if(ImGui::BeginCombo("##pos", structures::violins::violin_positions.at({att_app.base_pos, att_app.dir, att_app.span_full}).data())){
                    for(const auto& [pos, name]: structures::violins::violin_positions)
                        if(ImGui::MenuItem(name.data()))
                            std::tie(att_app.base_pos, att_app.dir, att_app.span_full) = pos;
                    ImGui::EndCombo();
                }
                ImGui::TableNextColumn();
                if(ImGui::Checkbox("##log", (bool*)&session_state.attribute_log[att.attribut_index]))
                    _update_attribute_histograms();

                ImGui::PopID();
            }
            if(up_index >= 0)
                std::swap(session_state.attribute_order_infos[up_index], session_state.attribute_order_infos[up_index - 1]);
            if(down_index >= 0)
                std::swap(session_state.attribute_order_infos[down_index], session_state.attribute_order_infos[down_index + 1]);
        }

        if(ImGui::BeginTable("drawlists", 2, ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)){
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Show");
            ImGui::TableSetupColumn("Remove");

            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
            ImGui::TableNextColumn();
            ImGui::TableHeader("Name");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Show");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Remove");

            std::string_view delete_dl{};
            for(auto&& [dl_id, dl]: session_state.drawlists){
                ImGui::PushID(dl_id.data());

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Selectable(dl_id.data());
                if(ImGui::BeginDragDropSource()){
                    ImGui::SetDragDropPayload(drag_type_matrix.data(), &dl_id, sizeof(dl_id));
                    ImGui::Text("%s", dl_id.data());
                    ImGui::EndDragDropSource();
                }
                ImGui::TableNextColumn();
                ImGui::Checkbox("##en", &dl.appearance->ref_no_track().show);
                ImGui::TableNextColumn();
                if(ImGui::Button("X"))
                    delete_dl = dl_id;

                ImGui::PopID();
            }
            if(delete_dl.size())
                remove_drawlists(delete_dl);

            ImGui::EndTable();
        }

        ImGui::EndTable();
    }

    ImGui::PopID();
    ImGui::End();
}

void violin_drawlist_workbench::signal_dataset_update(const util::memory_view<std::string_view>& dataset_ids, update_flags flags, const structures::gpu_sync_info& sync_info){
    bool any_affected{};
    for(const auto& [dl_id, dl]: session_state.drawlists){
        if(dataset_ids.contains(dl.drawlist_read().parent_dataset)){
            any_affected = true;
            break;
        }
    }
    if(!any_affected)
        return;
    
    // checking intersection of the new attributes
    std::vector<std::string> new_attributes;
    for(const auto& [dl, pos]: util::pos_iter(session_state.drawlists)){
        int intersection_index{-1};
        for(int i: util::size_range(dl.second.dataset_read().attributes)){
            if(pos == util::iterator_pos::first && i >= new_attributes.size())
                new_attributes.emplace_back(dl.second.dataset_read().attributes[i].id);
            
            if(pos != util::iterator_pos::first && (i < new_attributes.size() || dl.second.drawlist_read().dataset_read().attributes[i].id != new_attributes[i])){
                intersection_index = i;
                break;
            }
        }
        // removing attributes wich are not in the intersectoin set
        if(intersection_index > 0)
            new_attributes.erase(new_attributes.begin() + intersection_index, new_attributes.end());
    }

    if(logger.logging_level >= logging::level::l_5)
        logger << logging::info_prefix << " violin_drawlist_workbench::signal_dataset_update() New attributes will be: " << util::memory_view(new_attributes) << logging::endl;

    session_state.attributes.clear();
    for(int i: util::size_range(new_attributes)){
        session_state.attributes.emplace_back(structures::attribute{new_attributes[i], session_state.drawlists.begin()->second.dataset_read().attributes[i].display_name});
        for(const auto& [dl_id, dl]: session_state.drawlists){
            const auto& ds_bounds = dl.dataset_read().attributes[i].bounds.read();
            const auto& dl_bounds = session_state.attributes.back().bounds.read();
            if(ds_bounds.min < dl_bounds.min)
                session_state.attributes.back().bounds().min = ds_bounds.min;
            if(ds_bounds.max > dl_bounds.max)
                session_state.attributes.back().bounds().max = ds_bounds.max;
        }
    }

    // deleting all removed attributes in sorting order
    for(int i: util::rev_size_range(session_state.attribute_order_infos)){
        if(session_state.attribute_order_infos[i].attribut_index >= session_state.attributes.size())
            session_state.attribute_order_infos.erase(session_state.attribute_order_infos.begin() + i);
    }
    // adding new attribute references
    for(int i: util::i_range(session_state.attributes.size() - session_state.attribute_order_infos.size())){
        uint32_t cur_index = session_state.attribute_order_infos.size();
        session_state.attribute_order_infos.emplace_back(structures::attribute_order_info{cur_index});
    }

    _update_registered_histograms();
}

void violin_drawlist_workbench::add_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info){
    if(session_state.drawlists.empty()){
        session_state.attributes = globals::drawlists.read().at(drawlist_ids.front()).read().dataset_read().attributes;
        const auto attribute_count = session_state.attributes.size();
        session_state.attribute_log.resize(attribute_count, false);
        session_state.attribute_order_infos.clear();
        for(uint32_t i: util::size_range(session_state.attributes))
            session_state.attribute_order_infos.emplace_back(structures::attribute_order_info{i, true});
        session_state.attribute_violin_appearances.resize(attribute_count);
    }
    
    for(const auto& dl_id: drawlist_ids){
        // checking for already contained drawlists and attribute consistency
        if(session_state.drawlists.count(dl_id))
            continue;

        auto&       dl = globals::drawlists()[dl_id]();
        const auto& ds = dl.dataset_read();

        int merge_index{};
        for(int var: util::size_range(session_state.attributes)){
            if(session_state.attributes[var].id != ds.attributes[var].id)
                break;
            merge_index = var;
        }

        if(merge_index < session_state.attributes.size()){
            session_state.attributes.resize(merge_index);
            session_state.attribute_log.resize(merge_index);
            session_state.attribute_order_infos.clear();
            for(uint32_t i: util::size_range(session_state.attributes))
                session_state.attribute_order_infos.emplace_back(structures::attribute_order_info{i, true});
            session_state.attribute_violin_appearances.resize(merge_index);
        }

        // combining min max values
        for(auto&& [attribute, i]: util::enumerate(session_state.attributes)){
            if(attribute.bounds.read().min > ds.attributes[i].bounds.read().min)
                attribute.bounds().min = ds.attributes[i].bounds.read().min;
            if(attribute.bounds.read().max < ds.attributes[i].bounds.read().max)
                attribute.bounds().max = ds.attributes[i].bounds.read().max;
        }

        session_state.drawlists[dl_id] = structures::violins::drawlist_info{dl_id, true, dl.appearance_drawlist};
        for(auto& me: session_state.matrix_elements){
            if(me.empty()){
                me = dl_id;
                break;
            }
        }

        _update_registered_histograms();
    }
}

void violin_drawlist_workbench::remove_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info){
    for(const auto delete_dl: drawlist_ids){
        if(session_state.drawlists.count(delete_dl) == 0)
            continue;
        for(auto& dl: session_state.matrix_elements)
            if(dl == delete_dl)
                dl = {};
        // freeing registry
        auto registry_lock = globals::drawlists.read().at(delete_dl).read().histogram_registry.const_access();
        _registered_histograms.erase(delete_dl);
        session_state.drawlists.erase(delete_dl);
    }
}

void violin_drawlist_workbench::signal_drawlist_update(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info){
    _update_attribute_histograms();
}

std::vector<violin_drawlist_workbench::drawlist_attribute> violin_drawlist_workbench::get_active_drawlist_attributes() const{
    std::vector<drawlist_attribute> ret;
    for(const auto& i: session_state.attribute_order_infos){
        if(!i.active)
            continue;
        for(const auto& [dl_info, dl]: session_state.drawlists){
            if(!dl.appearance->read().show)
                continue;
            ret.emplace_back(drawlist_attribute{dl.drawlist_id, i.attribut_index});
        }
    }
    return ret;
}

std::vector<uint32_t> violin_drawlist_workbench::get_active_indices() const{
    std::vector<uint32_t> ret;
    for(const auto& i: session_state.attribute_order_infos)
        if(i.active)
            ret.emplace_back(i.attribut_index);
    return ret;
}
}