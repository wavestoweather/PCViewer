#include "violin_drawlist_workbench.hpp"
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_util.hpp>
#include <violin_util.hpp>
#include <histogram_registry_util.hpp>
#include <stager.hpp>
#include <cmath>
#include <flat_set.hpp>

namespace workbenches{
void violin_drawlist_workbench::_update_attribute_histograms(){
    // check for still active histogram update
    for(const auto& [dl_id, dl]: session_state.read().drawlists){
        if(_registered_histograms.contains(dl.drawlist_id) && _registered_histograms[dl.drawlist_id].size()){
            auto access = dl.drawlist_read().histogram_registry.const_access();
            if(!access->dataset_update_done)    
                return;
        }
    }

    // blending the histogram values according to the settings ---------------------------------------------
    const auto active_drawlist_attributes = get_active_ordered_drawlist_attributes();
    const auto attribute_bounds = get_attribute_min_max();
    std::tie(_global_max, _per_attribute_max, _drawlist_attribute_histograms) = util::violins::update_histograms(active_drawlist_attributes, attribute_bounds, settings.read().smoothing_std_dev, settings.read().histogram_bin_count, static_cast<int>(attribute_bounds.size()), settings.read().ignore_zero_bins, session_state.read().attribute_log);

    _update_attribute_positioning();

    settings.changed = false;
    session_state.changed = false;
    for(auto& [dl, registrators]: _registered_histograms){
        // locking the registry before signaling the reigstrators
        auto registry_lock = globals::drawlists.read().at(dl).read().histogram_registry.const_access();
        for(auto& registrator: registrators)
            registrator.signal_registry_used();
    }
}

void violin_drawlist_workbench::_update_registered_histograms(){
    auto const active_drawlist_attributes = get_active_ordered_drawlist_attributes();
    auto const active_attributes = get_active_ordered_attributes();
    for(const auto& [dl_id, dl]: session_state.read().drawlists){
        const auto active_attribute_indices = util::data::active_attribute_refs_to_indices(active_attributes, dl.dataset_read().attributes);

        std::vector<bool> registrator_needed(_registered_histograms[dl.drawlist_id].size(), false);
        for(auto&& [att, i]: util::enumerate(active_attributes)){
            auto registrator_id = util::histogram_registry::get_id_string(active_attribute_indices[i], settings.read().histogram_bin_count, att.get().bounds->read(), false, false);
            int registrator_index{-1};
            for(size_t j: util::size_range(_registered_histograms[dl.drawlist_id])){
                if(_registered_histograms[dl.drawlist_id][j].registry_id == registrator_id){
                    registrator_index = static_cast<int>(j);
                    break;
                }
            }
            if(registrator_index >= 0)
                registrator_needed[registrator_index] = true;
            else{
                // adding the new histogram
                auto& drawlist = dl.drawlist_write();
                _registered_histograms[dl.drawlist_id].emplace_back(drawlist.histogram_registry.access()->scoped_registrator(active_attribute_indices[i], settings.read().histogram_bin_count, att.get().bounds->read(), false, false, true));
                registrator_needed.emplace_back(true);
            }
        }
        // removing unused registrators
        // locking registry
        auto registry_lock = dl.drawlist_read().histogram_registry.const_access();
        for(size_t i: util::rev_size_range(_registered_histograms[dl.drawlist_id])){
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
    for(auto& att: session_state.ref_no_track().attribute_order_infos)
        if(!att.linked_with_attribute)
            att.bounds->changed = false;
}

void violin_drawlist_workbench::_update_attribute_positioning(bool update_direct){
    if(!settings.read().reposition_attributes_on_update && !update_direct)
        return;
    structures::violins::histogram d;
    std::map<std::string_view, std::vector<std::reference_wrapper<structures::violins::histogram>>> per_attribute_histograms;
    for(const auto& [dl_id, dl]: session_state.read().drawlists){
        if(!dl.appearance->read().show)
            continue;
        for(const auto& att: session_state.read().attribute_order_infos){
            if(_drawlist_attribute_histograms.contains({dl_id, att.attribute_id}))
                per_attribute_histograms[att.attribute_id].emplace_back(_drawlist_attribute_histograms.at({dl_id, att.attribute_id}));
            else
                per_attribute_histograms[att.attribute_id].emplace_back(d);
        }
    }
    if(per_attribute_histograms.empty())
        return;

    std::vector<std::string_view> attributes;
    for(const auto& i: session_state.read().attribute_order_infos)
        if(i.active->read())
            attributes.emplace_back(i.attribute_id);
    auto [appearances, order] = util::violins::get_violin_pos_order(per_attribute_histograms, attributes);
    for(const auto& [att, app]: appearances)
        session_state().attribute_violin_appearances[att] = app;
    std::map<std::string_view, int> attribute_pos;
    for(auto&& [att, i]: util::enumerate(order))
        attribute_pos[att] = static_cast<int>(i + 1);
    std::sort(session_state().attribute_order_infos.begin(), session_state().attribute_order_infos.end(), [&attribute_pos](auto& a, auto&b){return attribute_pos[a.attribute_id] < attribute_pos[b.attribute_id];});
}

void violin_drawlist_workbench::_update_attribute_order_infos(){
    // calculating the intersection of all drawlist attributes
    structures::flat_set<std::string_view> new_attributes;
    for(const auto& [dl, first]: util::first_iter(session_state.read().drawlists)){
        structures::flat_set<std::string_view> n;
        for(const auto& att: dl.second.dataset_read().attributes)
            n |= att.id;
        if(first)
            new_attributes = std::move(n);
        else
            new_attributes &= n;
    }

    if(logger.logging_flags.additional_info)
        logger << logging::info_prefix << " violin_drawlist_workbench::_update_attribute_order_infos() New attributes will be: " << util::memory_view(new_attributes.data(), new_attributes.size()) << logging::endl;
    
    structures::flat_set<std::string_view> old_attributes;
    for(const auto& att_info: session_state.read().attribute_order_infos)
        old_attributes |= att_info.attribute_id;
    auto attributes_to_add = new_attributes / old_attributes;

    // deleting all unused attributes in reverse order to avoid length problems
    for(size_t i: util::rev_size_range(session_state.read().attribute_order_infos)){
        std::string_view cur_att = session_state.read().attribute_order_infos[i].attribute_id;
        if(!new_attributes.contains(cur_att)){
            if(!session_state.read().attribute_order_infos[i].linked_with_attribute)
                bool todo = true;
            session_state().attribute_log.erase(cur_att);
            session_state().attribute_violin_appearances.erase(cur_att);
            session_state().attribute_order_infos.erase(session_state().attribute_order_infos.begin() + i);
        }
    }
    // adding new attribute references
    for(std::string_view att: attributes_to_add){
        auto& attribute = globals::attributes.ref_no_track()[att].ref_no_track();
        session_state().attribute_order_infos.emplace_back(structures::attribute_info{att, true, attribute.active, attribute.bounds, attribute.color});
        session_state().attribute_log[att] = {};
        session_state().attribute_violin_appearances[att] = {};
    }
}

violin_drawlist_workbench::violin_drawlist_workbench(std::string_view id): workbench(id){
    session_state.ref_no_track().matrix_elements.resize(settings.read().matrix_dimensions[0] * settings.read().matrix_dimensions[1]);
}

void violin_drawlist_workbench::show(){
    const std::string_view drag_type_matrix{"dl_matrix_element"};
    const std::string_view matrix_popup_id{"Matrix element popup"};

    bool open_matrix_popup{};

    if(!active){
        for(auto& [dl, regs]: _registered_histograms)
            for(auto& reg: regs)
                reg.signal_registry_used();
        return;
    }

    // checking updates
    bool request_register_update{};
    for(const auto& att: session_state.read().attribute_order_infos)
        request_register_update |= att.bounds->changed;

    if(request_register_update)
        _update_registered_histograms();

    if(session_state.changed || settings.changed)
        _update_attribute_histograms();

    ImGui::Begin(id.data(), &active);
    ImGui::PushID(id.data());

    // violin plots ---------------------------------------------------------------------------------------
    const float max_width = ImGui::GetWindowContentRegionWidth();
    const ImVec2 base_pos = ImGui::GetCursorScreenPos();
    const float plot_width_padded = max_width / settings.read().matrix_dimensions[1];
    const float plot_width = plot_width_padded - settings.read().plot_padding;
    const float plot_height_padded = settings.read().plot_height + settings.read().plot_padding + ImGui::GetTextLineHeightWithSpacing();
    for(auto&& [dl_id, i]: util::enumerate(session_state.read().matrix_elements)){
        util::imgui::scoped_id imgui_id(dl_id.size() ? dl_id.data(): "def");
        
        const int x = static_cast<int>(i / settings.read().matrix_dimensions[1]);
        const int y = static_cast<int>(i % settings.read().matrix_dimensions[1]);

        const ImVec2 violin_min{base_pos.x + y * plot_width_padded, base_pos.y + x * plot_height_padded};
        const ImVec2 plot_min{violin_min.x, violin_min.y + ImGui::GetTextLineHeightWithSpacing()};
        const ImVec2 plot_max{plot_min.x + plot_width, plot_min.y + settings.read().plot_height};
        
        // drawing the label and background
        ImGui::SetCursorScreenPos(violin_min);
        ImGui::Text("%s", dl_id.empty() ? "": dl_id.data());
        ImGui::GetWindowDrawList()->AddRectFilled(plot_min, plot_max, ImGui::ColorConvertFloat4ToU32(settings.read().plot_background), 4);
        
        // drag drop stuff
        ImGui::SetCursorScreenPos(plot_min);
        ImGui::BeginDisabled(dl_id.empty());
        ImGui::InvisibleButton("invis", {plot_max.x - plot_min.x, plot_max.y - plot_min.y});
        ImGui::EndDisabled();
        if(ImGui::BeginDragDropTarget()){
            if(auto payload = ImGui::AcceptDragDropPayload(drag_type_matrix.data()))
                session_state().matrix_elements[i] = *reinterpret_cast<std::string_view*>(payload->Data);

            ImGui::EndDragDropTarget();
        }
        if(ImGui::BeginDragDropSource()){
            ImGui::SetDragDropPayload(drag_type_matrix.data(), &dl_id, sizeof(dl_id));
            ImGui::Text("%s", dl_id.data());
            ImGui::EndDragDropSource();
        }
        if(ImGui::IsItemClicked(ImGuiMouseButton_Right)){
            _popup_matrix_element = static_cast<int>(i);
            open_matrix_popup = true;
        }

        if(dl_id.empty() || !session_state.read().drawlists.at(dl_id).appearance->read().show)
            continue;
        if(x >= settings.read().matrix_dimensions[0])
            break;

        // infill drawing
        for(const auto& att: util::rev_iter(session_state.read().attribute_order_infos)){
            if(!att.active->read() || !_drawlist_attribute_histograms.contains({dl_id, att.attribute_id}))
                continue;
            
            auto& violin_app = session_state.ref_no_track().attribute_violin_appearances[att.attribute_id];
            violin_app.color.x = att.color->read().x;
            violin_app.color.y = att.color->read().y;
            violin_app.color.z = att.color->read().z;
            violin_app.color.w = settings.read().area_alpha;
            const auto& histogram = _drawlist_attribute_histograms[{dl_id, att.attribute_id}];
            const float hist_normalization_fac = violin_app.scale == structures::violins::violin_scale_t::self ? histogram.max_val: violin_app.scale == structures::violins::violin_scale_t::per_attribute ? _per_attribute_max[att.attribute_id]: _global_max;
            util::violins::imgui_violin_infill(plot_min, plot_max, histogram.smoothed_values, hist_normalization_fac, violin_app);
        }
        // border drawing
        for(const auto& att: util::rev_iter(session_state.read().attribute_order_infos)){
            if(!att.active->read() || !_drawlist_attribute_histograms.contains({dl_id, att.attribute_id}))
                continue;
            
            auto& violin_app = session_state.ref_no_track().attribute_violin_appearances[att.attribute_id];
            violin_app.color.x = att.color->read().x;
            violin_app.color.y = att.color->read().y;
            violin_app.color.z = att.color->read().z;
            violin_app.color.w = settings.read().line_alpha;
            const auto& histogram = _drawlist_attribute_histograms[{dl_id, att.attribute_id}];
            const float hist_normalization_fac = violin_app.scale == structures::violins::violin_scale_t::self ? histogram.max_val: violin_app.scale == structures::violins::violin_scale_t::per_attribute ? _per_attribute_max[att.attribute_id]: _global_max;
            float line_thickness = settings.read().line_thickness;
            const bool hovered_plot = _hovered_dl_attribute == drawlist_attribute{dl_id, att.attribute_id};
            if(hovered_plot)
                line_thickness *= 2;
            float hover_val = util::violins::imgui_violin_border(plot_min, plot_max, histogram.smoothed_values, hist_normalization_fac, violin_app, line_thickness);
            if(isnan(hover_val)){
                if(hovered_plot)
                    _hovered_dl_attribute = {};
                continue;
            }
            _hovered_dl_attribute = {dl_id, att.attribute_id};
        }
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
            session_state().matrix_elements.resize(settings.read().matrix_dimensions[0] * settings.read().matrix_dimensions[1], {});
        ImGui::ColorEdit4("Plot background", &settings.read().plot_background.x, ImGuiColorEditFlags_NoInputs);
        ImGui::DragFloat("Line thickness", &settings.read().line_thickness, .1f, 1, 100);
        ImGui::DragFloat("Line hover distance", &settings.read().line_hover_dist, 1, 1, 200);
        ImGui::DragFloat("Line alpha", &settings.read().line_alpha, .05f, 0, 1);
        ImGui::DragFloat("Area alpha", &settings.read().area_alpha, .05f, 0, 1);
        if(ImGui::InputInt("Bin count", &settings.ref_no_track().histogram_bin_count, 1, 100, ImGuiInputTextFlags_EnterReturnsTrue))
            settings();
        if(ImGui::DragFloat("Smoothing std dev", &settings.ref_no_track().smoothing_std_dev, .1f, -1, 500))
            settings();
        if(ImGui::Checkbox("Ignore 0 bins", &settings.ref_no_track().ignore_zero_bins))
            settings();
        ImGui::DragFloat("Plot height", &settings.read().plot_height, 1, 5, 50000);
        ImGui::DragFloat("Plot padding", &settings.read().plot_padding, 1, 0, 100);
        ImGui::Separator();
        ImGui::Text("Attrribute coloring");
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x / 4);
        if(ImGui::BeginCombo("Palette type", settings.read().attribute_color_palette_type.c_str())){
            for(const auto& [name, palette]: brew_palette_types)
                if(ImGui::MenuItem(name.data()))
                    settings.read().attribute_color_palette_type = name;
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x / 4);
        if(ImGui::BeginCombo("Color Scheme", settings.read().attribute_color_palette.c_str())){
            for(const auto& palette: brew_palette_types.at(settings.read().attribute_color_palette_type))
                if(ImGui::MenuItem(palette.data()))
                    settings.read().attribute_color_palette = palette;
            ImGui::EndCombo();
        }
        ImGui::Checkbox("Auto-repositon attributes", &settings.read().reposition_attributes_on_update);
        if(ImGui::Button("Reposition/recolor attributes"))
            _update_attribute_positioning(true);

        ImGui::TableNextColumn();
        if(ImGui::BeginTable("attributes", 8, ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)){
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Up");
            ImGui::TableSetupColumn("Down");
            ImGui::TableSetupColumn("Show");
            ImGui::TableSetupColumn("Color");
            ImGui::TableSetupColumn("Position");
            ImGui::TableSetupColumn("Scale");
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
            ImGui::TableHeader("Scale");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Log");

            int up_index{-1}, down_index{-1};
            for(auto&& [att, i]: util::enumerate(session_state.read().attribute_order_infos)){
                auto& att_app = session_state.read().attribute_violin_appearances[att.attribute_id];
                util::imgui::scoped_id imgui_id(att.attribute_id.data());

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", att.attribute_read().display_name.c_str());
                ImGui::TableNextColumn();
                ImGui::BeginDisabled(i == 0);
                if(ImGui::ArrowButton("##up", ImGuiDir_Up))
                    up_index = static_cast<int>(i);
                ImGui::EndDisabled();
                ImGui::TableNextColumn();
                ImGui::BeginDisabled(i == session_state.read().attribute_order_infos.size() - 1);
                if(ImGui::ArrowButton("##do", ImGuiDir_Down))
                    down_index = static_cast<int>(i);
                ImGui::EndDisabled();
                ImGui::TableNextColumn();
                if(ImGui::Checkbox("##en", &session_state.ref_no_track().attribute_order_infos[i].active->ref_no_track())){
                    //_update_registered_histograms();
                    session_state().attribute_order_infos[i].active->write();
                }
                ImGui::TableNextColumn();
                ImGui::ColorEdit4("##col", &session_state.ref_no_track().attribute_order_infos[i].color->ref_no_track().x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
                ImGui::TableNextColumn();
                ImGui::SetNextItemWidth(70);
                if(ImGui::BeginCombo("##pos", structures::violins::violin_positions.at({att_app.base_pos, att_app.dir, att_app.span_full}).data())){
                    for(const auto& [pos, name]: structures::violins::violin_positions)
                        if(ImGui::MenuItem(name.data()))
                            std::tie(att_app.base_pos, att_app.dir, att_app.span_full) = pos;
                    ImGui::EndCombo();
                }
                ImGui::TableNextColumn();
                ImGui::SetNextItemWidth(70);
                if(ImGui::BeginCombo("##scale", structures::violins::violin_scale_names[att_app.scale].data())){
                    for(const auto scale: structures::enum_iteration<structures::violins::violin_scale_t>{})
                        if(ImGui::MenuItem(structures::violins::violin_scale_names[scale].data()))
                            att_app.scale = scale;
                    ImGui::EndCombo();
                }
                ImGui::TableNextColumn();
                if(ImGui::Checkbox("##log", &session_state.ref_no_track().attribute_log[att.attribute_id]))
                    session_state();
                    //_update_attribute_histograms();
            }
            if(up_index >= 0)
                std::swap(session_state.ref_no_track().attribute_order_infos[up_index], session_state.ref_no_track().attribute_order_infos[up_index - 1]);
            if(down_index >= 0)
                std::swap(session_state.ref_no_track().attribute_order_infos[down_index], session_state.ref_no_track().attribute_order_infos[down_index + 1]);
            ImGui::EndTable();
        }
        ImGui::TableNextColumn();
        if(ImGui::BeginTable("drawlists", 3, ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)){
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
            for(auto&& [dl_id, dl]: session_state.ref_no_track().drawlists){
                util::imgui::scoped_id imgui_id(dl_id.data());

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                bool selected = globals::selected_drawlists | util::contains(dl_id);
                if(ImGui::Selectable(dl_id.data(), selected, ImGuiSelectableFlags_NoPadWithHalfSpacing, {0, ImGui::GetTextLineHeightWithSpacing()})){
                    if(selected)
                        globals::selected_drawlists.clear();
                    else
                        globals::selected_drawlists.emplace_back(dl_id);
                }
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
            }
            if(delete_dl.size())
                remove_drawlists(delete_dl);

            ImGui::EndTable();
        }

        ImGui::EndTable();
    }

    if(open_matrix_popup)
        ImGui::OpenPopup(matrix_popup_id.data());
    if(ImGui::BeginPopup(matrix_popup_id.data())){
        if(ImGui::MenuItem("Remove Violins")){
            session_state.ref_no_track().matrix_elements[_popup_matrix_element] = {};
            _popup_matrix_element = -1;
        }
        ImGui::EndPopup();
    }

    ImGui::PopID();
    ImGui::End();
}

void violin_drawlist_workbench::signal_dataset_update(const util::memory_view<std::string_view>& dataset_ids, update_flags flags, const structures::gpu_sync_info& sync_info){
    bool any_affected{};
    for(const auto& [dl_id, dl]: session_state.read().drawlists){
        if(dataset_ids.contains(dl.drawlist_read().parent_dataset)){
            any_affected = true;
            break;
        }
    }
    if(!any_affected)
        return;
    
    _update_attribute_order_infos();

    //_update_registered_histograms();
}

void violin_drawlist_workbench::add_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info){
    for(const auto& dl_id: drawlist_ids){
        // checking for already contained drawlists and attribute consistency
        if(session_state.read().drawlists.count(dl_id))
            continue;

        auto&       dl = globals::drawlists()[dl_id]();
        session_state().drawlists[dl_id] = {dl_id, true, dl.appearance_drawlist};

        // adding the drawlist to the first free matrix field
        for(size_t i: util::size_range(session_state.read().matrix_elements)){
            if(session_state.read().matrix_elements[i].empty()){
                session_state().matrix_elements[i] = dl.id;
                break;
            }
        }
    }
    _update_attribute_order_infos();
    // copy colors and disconnect from global state
    for(auto& att: session_state().attribute_order_infos){
        att.linked_with_attribute = false;
        _local_storage.emplace_back(std::make_unique<structures::violins::local_storage>(structures::violins::local_storage{att.active->read(), att.bounds->read(), att.color->read()}));
        att.active = _local_storage.back()->active;
        att.bounds = _local_storage.back()->bounds;
        att.color = _local_storage.back()->color;
    }

    _update_registered_histograms();
}

void violin_drawlist_workbench::remove_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info){
    for(const auto delete_dl: drawlist_ids){
        if(session_state.read().drawlists.count(delete_dl) == 0)
            continue;
        for(auto& dl: session_state().matrix_elements)
            if(dl == delete_dl)
                dl = {};
        // freeing registry
        auto registry_lock = globals::drawlists.read().at(delete_dl).read().histogram_registry.const_access();
        _registered_histograms.erase(delete_dl);
        session_state().drawlists.erase(delete_dl);
    }
    _update_attribute_order_infos();
}

void violin_drawlist_workbench::signal_drawlist_update(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info){
    //_update_attribute_histograms();
    session_state();
}

std::vector<violin_drawlist_workbench::drawlist_attribute> violin_drawlist_workbench::get_active_ordered_drawlist_attributes() const{
    std::vector<drawlist_attribute> ret;
    for(const auto& i: session_state.read().attribute_order_infos){
        if(!i.active->read())
            continue;
        for(const auto& [dl_info, dl]: session_state.read().drawlists){
            if(!dl.appearance->read().show)
                continue;
            ret.emplace_back(drawlist_attribute{dl.drawlist_id, i.attribute_id});
        }
    }
    return ret;
}

std::vector<structures::const_attribute_info_ref> violin_drawlist_workbench::get_active_ordered_attributes() const{
    std::vector<structures::const_attribute_info_ref> ret;
    for(const auto& i: session_state.read().attribute_order_infos)
        if(i.active->read())
            ret.emplace_back(i);
    return ret;
}

std::map<std::string_view, structures::min_max<float>> violin_drawlist_workbench::get_attribute_min_max() const{
    std::map<std::string_view, structures::min_max<float>> bounds;
    for(const auto& a: session_state.read().attribute_order_infos)
        bounds[a.attribute_id] = a.bounds->read();
    return bounds;
}
}