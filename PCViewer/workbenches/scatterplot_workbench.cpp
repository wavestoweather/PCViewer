#include "scatterplot_workbench.hpp"
#include <imgui_util.hpp>
#include <vma_initializers.hpp>
#include <imgui_internal.h>
#include <imgui_stdlib.h>
#include <scatterplot_renderer.hpp>
#include <mutex>
#include <brushes.hpp>
#include <brush_util.hpp>
#include <util.hpp>
#include <descriptor_set_storage.hpp>
#include <flat_set.hpp>
#include <regex>

namespace workbenches
{
void scatterplot_workbench::_update_registered_histograms(){
    if(!all_registrators_updated(true)) // also waits for rendering to avoid updating before visualization has updated
        return;

    robin_hood::unordered_map<std::string_view, std::vector<bool>> drawlist_registator_needed;
    std::array<int, 2> bucket_sizes{static_cast<int>(settings.read().plot_width), static_cast<int>(settings.read().plot_width)};
    std::array<uint32_t, 2> indices;
    std::array<structures::min_max<float>, 2> bounds;
    std::vector<const_attribute_info_ref> attributes; for(const auto& att: attribute_order_infos.read()) attributes.emplace_back(att);
    for(const auto& [atts, dls]: plot_list.read()){
        const auto p = atts;
        size_t att_a_index = attributes | util::index_of_if<const_attribute_info_ref>([&p](auto& r){return r.get().attribute_id == p.a;});
        size_t att_b_index = attributes | util::index_of_if<const_attribute_info_ref>([&p](auto& r){return r.get().attribute_id == p.b;});
        assert(att_a_index != util::n_pos && att_b_index != util::n_pos);
        for(auto dl: dls){
            if(DRAWLIST_READ(dl).const_templatelist().data_size < settings.read().large_vis_threshold)
                continue;
            
            if(!drawlist_registator_needed.contains(dl))
                drawlist_registator_needed[dl].resize(_registered_histograms[dl].size());

            const auto& ds = DRAWLIST_READ(dl).dataset_read();
            const auto attribute_indices = util::data::active_attribute_refs_to_indices(attributes, ds.attributes); // if the dataset does not have the attribute its index will be util::memory_view<>::npos
            indices = {attribute_indices[att_a_index], attribute_indices[att_b_index]};
            bounds = {attributes[att_a_index].get().bounds->read(), attributes[att_b_index].get().bounds->read()};
            auto registrator_id = util::histogram_registry::get_id_string(indices, bucket_sizes, bounds, false, false);
            size_t registrator_index = util::memory_view(_registered_histograms[dl]).index_of([&registrator_id](const registered_histogram& h){return registrator_id == h.registry_id;});
            if(registrator_index != util::memory_view<>::n_pos)
                drawlist_registator_needed[dl][registrator_index] = true;
            else{
                // adding new histogram
                _registered_histograms[dl].emplace_back(DRAWLIST_WRITE(dl).histogram_registry.access()->scoped_registrator(indices, bucket_sizes, bounds, false, false, false));
                drawlist_registator_needed[dl].emplace_back(true);
            }
        }
    }

    // removing unused registrators --------------------------------------------------
    std::vector<std::string_view> removed_dls;
    for(auto& [dl, registrators]: _registered_histograms){
        if(!drawlist_registator_needed.contains(dl)){
            removed_dls.emplace_back(dl);
            continue;
        }
        // acquire registry lock and erase registrators
        auto lock = DRAWLIST_READ(dl).histogram_registry.const_access();
        for(size_t i: util::rev_size_range(registrators)){
            if(!drawlist_registator_needed[dl][i])
                registrators.erase(registrators.begin() + i);
        }
    }
    if(removed_dls.size()){
        for(const auto& dl: removed_dls){
            auto lock = DRAWLIST_READ(dl).histogram_registry.const_access();
            _registered_histograms.erase(dl);
        }
    }
    if(logger.logging_level >= logging::level::l_5){
        size_t registrator_sum{};
        for(const auto& [dl, registrators]: _registered_histograms){
            logger << logging::info_prefix << " scatterplot_workbench::_update_registered_histograms() For " << dl << " " << registrators.size() << " histograms are registered" << logging::endl;
            registrator_sum += registrators.size();
        }
        logger << logging::info_prefix << "scatterplot_workbench::_update_registered_histograms() Registrators update done, overall need " << registrator_sum << " registrators" << logging::endl;
    }

    // setting update singal flags
    for(const auto& dl: drawlist_infos.read())
        dl.drawlist_write().histogram_registry.access()->request_change_all();
    attribute_order_infos.changed = false;
    plot_list.changed = false;
    _request_registrators_update = false;
}

void scatterplot_workbench::_update_plot_images(){
    constexpr VkImageUsageFlags image_usage{VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};
    // waiting for the device to finish all command buffers using the image before statring to destroy/create plot images
    {
        for(auto& m: globals::vk_context.mutex_storage)
            m->lock();
        auto res = vkDeviceWaitIdle(globals::vk_context.device); util::check_vk_result(res);
        for(auto& m: globals::vk_context.mutex_storage)
            m->unlock();
    }
    
    robin_hood::unordered_set<att_pair_dls> used_attribute_pairs;
    std::vector<VkImageMemoryBarrier> image_barriers;
    // creating new pairs / recreating images if size changed
    auto destroy_image=[&](const att_pair_dls& p) {
        auto& plot_data = plot_datas[p];
        util::vk::destroy_image(plot_data.image);
        util::vk::destroy_image_view(plot_data.image_view);
        util::imgui::free_image_descriptor_set(plot_data.image_descriptor);
    };
    auto register_image = [&](const att_pair_dls& p) {
        if(!p)
            return;
        // destruction of old image if necessary
        if(plot_datas.contains(p) && (plot_datas[p].image_width != settings.read().plot_width || plot_datas[p].image_format != settings.read().plot_format))
            destroy_image(p);
        // creating new image if needed
        if(!plot_datas[p].image){
            auto& plot_data = plot_datas[p];
            auto image_info = util::vk::initializers::imageCreateInfo(settings.read().plot_format, {settings.read().plot_width, settings.read().plot_width, 1}, image_usage);
            auto mem_alloc = util::vma::initializers::allocationCreateInfo();
            std::tie(plot_data.image, plot_data.image_view) = util::vk::create_image_with_view(image_info, mem_alloc);
            plot_data.image_descriptor = util::imgui::create_image_descriptor_set(plot_data.image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            plot_data.image_format = settings.read().plot_format;

            // updating the image layout
            image_barriers.emplace_back(util::vk::initializers::imageMemoryBarrier(plot_data.image.image, VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, {}, {}, {}, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
        }
        used_attribute_pairs.insert(p);
    };

    for(const auto& p: plot_list.read())
        register_image(p);
    util::vk::convert_image_layouts_execute(image_barriers);

    // removing all unused images
    robin_hood::unordered_set<att_pair_dls> unused_attribute_pairs;
    for(const auto& [pair, image_data]: plot_datas) if(!used_attribute_pairs.contains(pair)) unused_attribute_pairs.insert(pair);
    for(const auto& pair: unused_attribute_pairs){
        destroy_image(pair);
        plot_datas.erase(pair);
    }
}

void scatterplot_workbench::_update_plot_list(){
    auto active_attributes = get_active_ordered_attributes();
    plot_list().clear();
    switch(settings.read().plot_type){
    case plot_type_t::matrix:{
        std::vector<std::string_view> dls; for(const auto& dl: drawlist_infos.read()) dls.emplace_back(dl.drawlist_id);
        for(size_t i: util::i_range(size_t(1), active_attributes.size())){
            for(size_t j: util::i_range(i))
                plot_list().emplace_back(att_pair_dls{active_attributes[i].get().attribute_id, active_attributes[j].get().attribute_id, dls});
        }
        break;
    }
    case plot_type_t::list:
        robin_hood::unordered_set<att_pair_dls> added_pairs;
        for(const auto& p: _matrix_scatterplots){
            if(p && !added_pairs.contains(p)){
                plot_list().emplace_back(p);
                added_pairs.insert(p);
            }
        }
        break;
    }
}

void scatterplot_workbench::_render_plot(){
    // check for still active histogram update
    if(!all_registrators_updated())
        return;

    _update_plot_images();  // all plot images are recreated before rendering is issued

    if(logger.logging_level >= logging::level::l_5)
        logger << logging::info_prefix << " scatterplot_workbench::_render_plot()" << logging::endl;
    pipelines::scatterplot_renderer::render_info render_info{*this};
    pipelines::scatterplot_renderer::instance().render(render_info);

    for(auto& dl_info: drawlist_infos())
        if(!dl_info.linked_with_drawlist)
            dl_info.clear_changes();
    drawlist_infos.changed = false;
    attribute_order_infos.changed = false;
    settings.changed = false;
    for(auto& [dl, registrators]: _registered_histograms){
        auto registry_lock = globals::drawlists.read().at(dl).read().histogram_registry.const_access();
        for(auto& registrator: registrators)
            registrator.signal_registry_used();
    }
}

void scatterplot_workbench::_update_attribute_order_infos(){
    // calculating the intersection of all drawlist attributes
    structures::flat_set<std::string_view> new_attributes;
    for(const auto& [dl, first]: util::first_iter(drawlist_infos.read())){
        structures::flat_set<std::string_view> n;
        for(const auto& att: dl.dataset_read().attributes)
            n |= att.id;
        if(first)
            new_attributes = std::move(n);
        else
            new_attributes &= n;
    }

    if(logger.logging_flags.additional_info)
        logger << logging::info_prefix << " violin_drawlist_workbench::_update_attribute_order_infos() New attributes will be: " << util::memory_view(new_attributes.data(), new_attributes.size()) << logging::endl;
    
    structures::flat_set<std::string_view> old_attributes;
    for(const auto& att_info: attribute_order_infos.read())
        old_attributes |= att_info.attribute_id;
    auto attributes_to_add = new_attributes / old_attributes;

    // deleting all unused attributes in reverse order to avoid length problems
    for(size_t i: util::rev_size_range(attribute_order_infos.read())){
        std::string_view cur_att = attribute_order_infos.read()[i].attribute_id;
        if(!new_attributes.contains(cur_att)){
            if(!attribute_order_infos.read()[i].linked_with_attribute)
                bool todo = true;
            attribute_order_infos().erase(attribute_order_infos.read().begin() + i);
        }
    }
    // adding new attribute references
    for(std::string_view att: attributes_to_add){
        auto& attribute = globals::attributes.ref_no_track()[att].ref_no_track();
        attribute_order_infos().emplace_back(structures::attribute_info{att, true, attribute.active, attribute.bounds, attribute.color});
    }
}

void scatterplot_workbench::_show_general_settings(){
    if(ImGui::BeginCombo("plot type(coming soon)", plot_type_names[settings.read().plot_type].data())){
        for(auto t: structures::enum_iteration<plot_type_t>()){
            if(ImGui::MenuItem(plot_type_names[t].data())){
                settings().plot_type = t;
                _update_plot_list();
            }
        }
        ImGui::EndCombo();
    }
    if(settings.read().plot_type == plot_type_t::list){
        if(ImGui::DragInt2("Scatterplot matrix size", settings.ref_no_track().plot_matrix.data(), .1f, 1, 100)){
            _matrix_scatterplots.resize(settings.read().plot_matrix[0] * settings.read().plot_matrix[1]);
            _update_plot_list();
        }
    }
    ImGui::ColorEdit4("Plot background", &settings.read().plot_background_color.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
    if(ImGui::InputScalar("Plot width", ImGuiDataType_U32, &settings.ref_no_track().plot_width, {}, {}, {}, ImGuiInputTextFlags_EnterReturnsTrue)){
        settings().plot_width = std::clamp(settings.read().plot_width, 50u, 10000u);
        _request_registrators_update |= true;
    }
    if(ImGui::DragFloat("Uniform radius", &settings.read().uniform_radius, 1, 1, 100))
        for(auto& dl: drawlist_infos())
            dl.scatter_appearance().radius = settings.read().uniform_radius;
    ImGui::InputDouble("plot padding", &settings.read().plot_padding);
}

scatterplot_workbench::scatterplot_workbench(std::string_view id):
    workbench(id)
{
    _matrix_scatterplots.resize(settings.read().plot_matrix[0] * settings.read().plot_matrix[1]);
}

void draw_lassos(scatterplot_workbench::attribute_pair p, util::memory_view<const uint32_t> p_indices, float plot_width, ImVec4 min_max, ImU32 color = util::brushes::get_brush_color(), float thickness = globals::brush_edit_data.brush_line_width, const ImVec2& base_pos = ImGui::GetCursorScreenPos()){
    if(globals::brush_edit_data.brush_type == structures::brush_edit_data::brush_type::none)
        return;
    
    bool swap{p_indices[0] > p_indices[1]};
    if(swap)
        std::swap(p.a, p.b);
    const auto& lassos = util::brushes::get_selected_lasso_brush_const();
    auto lasso = lassos | util::try_find_if<const structures::polygon>([&](const structures::polygon& e){return e.attr1 == p.a && e.attr2 == p.b;});
    if(lasso){
        if(lasso->get().borderPoints.empty()) return;
        ImVec2 const* last_p = &lasso->get().borderPoints.back();
        for(const ImVec2& cur_p: lasso->get().borderPoints){
            ImVec2 normalized_start{util::normalize_val_for_range(last_p->x, min_max[0], min_max[2]), util::normalize_val_for_range(last_p->y, min_max[1], min_max[3])};
            ImVec2 normalized_end{util::normalize_val_for_range(cur_p.x, min_max[0], min_max[2]), util::normalize_val_for_range(cur_p.y, min_max[1], min_max[3])};
            normalized_start.y = 1 - normalized_start.y;
            normalized_end.y = 1 - normalized_end.y;
            ImVec2 start{util::unnormalize_val_for_range(normalized_start.x, base_pos.x, base_pos.x + plot_width), util::unnormalize_val_for_range(normalized_start.y, base_pos.y, base_pos.y + plot_width)}; 
            ImVec2 end{util::unnormalize_val_for_range(normalized_end.x, base_pos.x, base_pos.x + plot_width), util::unnormalize_val_for_range(normalized_end.y, base_pos.y, base_pos.y + plot_width)}; 
            if(swap)
                std::swap(start, end);
            ImGui::GetWindowDrawList()->AddLine(start, end, color, thickness);
            last_p = &cur_p;
        }
    }
}
    
void scatterplot_workbench::show() 
{
    const std::string_view plot_menu_id{"plot_menu"};

    if(!active) 
        return;

    // checking for setting updates and updating the rendering if necessary
    _request_registrators_update |= attribute_order_infos.changed;// && std::any_of(attribute_order_infos.read().begin(), attribute_order_infos.read().end(), [](const auto& info){return info.active->changed || info.bounds->changed;});
    _request_registrators_update |= plot_list.changed;

    if(_request_registrators_update){
        _update_plot_list();
        _update_registered_histograms();
    }

    bool local_change{false};
    bool request_render{false};
    local_change |= drawlist_infos.changed;
    request_render |= local_change;
    request_render |= attribute_order_infos.changed;
    request_render |= settings.changed;
    if(globals::drawlists.changed){
        for(const auto& dl: drawlist_infos.read())
            request_render |= globals::drawlists.read().at(dl.drawlist_id).changed;
    }

    if(request_render)
        _render_plot();


    ImGui::Begin(id.data(), &active, ImGuiWindowFlags_HorizontalScrollbar);

    const auto active_attributes = get_active_ordered_attributes();
    robin_hood::unordered_map<std::string_view, const_attribute_info_ref> att_id_to_attribute;
    for(const auto& att: active_attributes)
        att_id_to_attribute.insert({att.get().attribute_id, att});
    if(_plot_x_vals.size() < active_attributes.size()) _plot_x_vals.resize(active_attributes.size());
    // plot views ------------------------------------------------------
    att_pair_dls    hovered_pair{};
    ImVec4          hovered_rect{};
    switch(settings.read().plot_type){
    case plot_type_t::matrix:{
        structures::flat_set<std::string_view> dls; for(const auto& dl: drawlist_infos.read()) dls |= dl.drawlist_id;
        // matrix should be displayed as a left lower triangular matrix
        for(size_t i: util::i_range(size_t(1), active_attributes.size())){
            ImVec2 text_pos = ImGui::GetCursorScreenPos(); text_pos.y += settings.read().plot_width / 2;
            util::imgui::AddTextVertical(active_attributes[i].get().attribute_read().display_name.c_str(), text_pos, .5f);
            ImGui::SetCursorScreenPos({ImGui::GetCursorScreenPos().x + ImGui::GetTextLineHeightWithSpacing(), ImGui::GetCursorScreenPos().y});
            for(size_t j: util::i_range(i)){
                att_pair_dls p = {active_attributes[i].get().attribute_id, active_attributes[j].get().attribute_id, dls};
                if(!plot_datas.contains(p))
                    continue;
                if(j != 0)  
                    ImGui::SameLine();
                auto c_pos = ImGui::GetCursorScreenPos();
                ImGui::GetWindowDrawList()->AddRectFilled(c_pos, {c_pos.x + settings.read().plot_width, c_pos.y + settings.read().plot_width}, ImColor(settings.read().plot_background_color));
                if(plot_additional_datas.contains(p) && globals::descriptor_sets.count(plot_additional_datas[p].background_image)){
                    ImGui::Image(globals::descriptor_sets[plot_additional_datas[p].background_image]->descriptor_set, {float(settings.read().plot_width), float(settings.read().plot_width)});
                    ImGui::SetCursorScreenPos(c_pos);
                }
                ImGui::Image(plot_datas[p].image_descriptor, {float(settings.read().plot_width), float(settings.read().plot_width)});
                if(ImGui::IsItemClicked(ImGuiMouseButton_Right)){
                    ImGui::OpenPopup(plot_menu_id.data());
                    _popup_attributes = p;
                }
                if(ImGui::IsItemHovered()){
                    hovered_pair = p;
                    hovered_rect = {c_pos.x, c_pos.x + settings.read().plot_width, c_pos.y, c_pos.y + settings.read().plot_width};
                }
                if(ImGui::BeginDragDropTarget()){
                    if(const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("image"))
                        plot_additional_datas[p].background_image = *reinterpret_cast<const std::string_view*>(payload->Data);
                    ImGui::EndDragDropTarget();
                }
                std::array<uint32_t, 2> indices{uint32_t(i), uint32_t(j)};
                draw_lassos(p.atts, indices, settings.read().plot_width, {active_attributes[j].get().bounds->read().min, active_attributes[i].get().bounds->read().min, active_attributes[j].get().bounds->read().max, active_attributes[i].get().bounds->read().max}, util::brushes::get_brush_color(), globals::brush_edit_data.brush_line_width, c_pos);
                _plot_x_vals[j] = c_pos.x;
            }
        }
        for(int i: util::i_range(int(active_attributes.size()) - 1)){
            if(i) ImGui::SameLine();
            float width = ImGui::CalcTextSize(active_attributes[i].get().attribute_read().display_name.c_str()).x;
            ImGui::SetCursorScreenPos({_plot_x_vals[i] + settings.read().plot_width / 2.f - width / 2.f, ImGui::GetCursorScreenPos().y});
            ImGui::Text("%s", active_attributes[i].get().attribute_read().display_name.c_str());
        }
        }
        break;
    case plot_type_t::list:
        for(const auto& [p, i]: util::enumerate(_matrix_scatterplots)){
            util::imgui::scoped_id list_id{int(i)};
            const bool first = i % settings.read().plot_matrix[1] == 0;
            if(!first) ImGui::SameLine();
            // background
            auto c_pos = ImGui::GetCursorScreenPos();
            ImGui::GetWindowDrawList()->AddRectFilled(c_pos, {c_pos.x + settings.read().plot_width, c_pos.y + settings.read().plot_width}, ImColor(settings.read().plot_background_color));
            if(p){
                if(plot_additional_datas.contains(p) && globals::descriptor_sets.count(plot_additional_datas[p].background_image)){
                    ImGui::Image(globals::descriptor_sets[plot_additional_datas[p].background_image]->descriptor_set, {float(settings.read().plot_width), float(settings.read().plot_width)});
                    ImGui::SetCursorScreenPos(c_pos);
                }
                ImGui::Image(plot_datas[p].image_descriptor, {float(settings.read().plot_width), float(settings.read().plot_width)});
                if(ImGui::IsItemClicked(ImGuiMouseButton_Right)){
                    ImGui::OpenPopup(plot_menu_id.data());
                    _popup_attributes = p;
                }
                if(ImGui::IsItemHovered()){
                    hovered_pair = p;
                    hovered_rect = {c_pos.x, c_pos.x + settings.read().plot_width, c_pos.y, c_pos.y + settings.read().plot_width};
                }
                if(ImGui::BeginDragDropTarget()){
                    if(const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("image"))
                        plot_additional_datas[p].background_image = *reinterpret_cast<const std::string_view*>(payload->Data);
                    ImGui::EndDragDropTarget();
                }
                //std::array<uint32_t, 2> indices{uint32_t(i), uint32_t(j)};
                //draw_lassos(p, indices, settings.read().plot_width, {active_attributes[j].get().bounds->read().min, active_attributes[i].get().bounds->read().min, active_attributes[j].get().bounds->read().max, active_attributes[i].get().bounds->read().max}, util::brushes::get_brush_color(), globals::brush_edit_data.brush_line_width, c_pos);
                //_plot_x_vals[j] = c_pos.x;
            }
            else{
                ImGui::InvisibleButton("plot_alt", {float(settings.read().plot_width), float(settings.read().plot_width)});
                if(ImGui::BeginDragDropTarget()){
                    if(auto payload = ImGui::AcceptDragDropPayload("att_combination")){

                    }
                    ImGui::EndDragDropTarget();
                }
            }
        }
        break;
    }
    // lasso brushes ----------------------------------------------------
    // cleearing old points if fresh button press
    if(hovered_pair && &att_id_to_attribute.at(hovered_pair.atts.a).get() > &att_id_to_attribute.at(hovered_pair.atts.b).get())
        std::swap(hovered_pair.atts.a, hovered_pair.atts.b);

    auto get_attr_pos = [&](const ImVec2& pos = ImGui::GetMousePos()){
        ImVec2 mouse_norm{util::normalize_val_for_range(pos.x, hovered_rect[0], hovered_rect[1]), util::normalize_val_for_range(pos.y, hovered_rect[2], hovered_rect[3])};
        mouse_norm.y = 1 - mouse_norm.y;
        ImVec2 attr_pos{util::unnormalize_val_for_range(mouse_norm.x, att_id_to_attribute.at(hovered_pair.atts.a).get().bounds->read().min, att_id_to_attribute.at(hovered_pair.atts.a).get().bounds->read().max), util::unnormalize_val_for_range(mouse_norm.y, att_id_to_attribute.at(hovered_pair.atts.b).get().bounds->read().min, att_id_to_attribute.at(hovered_pair.atts.b).get().bounds->read().max)};
        return attr_pos;
    };
    if(hovered_pair != attribute_pair{} && globals::brush_edit_data.brush_type != structures::brush_edit_data::brush_type::none)
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    if(ImGui::IsMouseClicked(ImGuiMouseButton_Left) && globals::brush_edit_data.brush_type != structures::brush_edit_data::brush_type::none && hovered_pair != attribute_pair{}){
        try{
            auto& polygon = util::memory_view(util::brushes::get_selected_lasso_brush()).find([&hovered_pair](const structures::polygon& e){return e.attr1 == hovered_pair.atts.a && e.attr2 == hovered_pair.atts.b;});
            polygon.borderPoints.clear();
        } catch(std::exception e){
            util::brushes::get_selected_lasso_brush().emplace_back(structures::polygon{hovered_pair.atts.a, hovered_pair.atts.b, {}});
        }
        _last_lasso_point = ImGui::GetMousePos();
        _started_lasso_attributes = hovered_pair;
    }
    if(ImGui::IsMouseDown(ImGuiMouseButton_Left) && globals::brush_edit_data.brush_type != structures::brush_edit_data::brush_type::none &&
        util::distance(_last_lasso_point, ImGui::GetMousePos()) > globals::brush_edit_data.drag_threshold &&
        _started_lasso_attributes && _started_lasso_attributes == hovered_pair){
        auto& polygon = util::memory_view(util::brushes::get_selected_lasso_brush()).find([&hovered_pair](const structures::polygon& e){return e.attr1 == hovered_pair.atts.a && e.attr2 == hovered_pair.atts.b;});
        if(polygon.borderPoints.empty())
            polygon.borderPoints.emplace_back(get_attr_pos(_last_lasso_point));
        polygon.borderPoints.emplace_back(get_attr_pos());
        _last_lasso_point = ImGui::GetMousePos();
    }

    // settings ---------------------------------------------------------
    std::string_view delete_drawlist{};
    if(ImGui::BeginTable("scatterplot_setting", 3, ImGuiTableFlags_Resizable)){
        // column setup
        ImGui::TableSetupScrollFreeze(0, 1);    // make top row always visible
        ImGui::TableSetupColumn("Settings");
        ImGui::TableSetupColumn("Attributes");
        ImGui::TableSetupColumn("Drawlists");
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableNextColumn();
        ImGui::TableHeader("Settings");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Attributes");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Drawlists");

        // settings
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        _show_general_settings();
        if(settings.read().plot_type == structures::scatterplot_wb::plot_type_t::list){
            ImGui::Separator();
            if(_regex_error)
                ImGui::PushStyleColor(ImGuiCol_FrameBg, {1.f, 0.f, 0.f, .5f});
            ImGui::InputText("Filter##scatterplot", &_att_pair_regex);
            if(_regex_error)
                ImGui::PopStyleColor();
            std::regex regex;
            try{ regex = std::regex(_att_pair_regex); _regex_error = false;}
            catch(std::exception) {_regex_error = true;}

            for(const auto& att1: active_attributes){
                for(const auto& att2: active_attributes){
                    if(att1.get().attribute_id == att2.get().attribute_id) continue;
                    std::string combination = std::string(att1.get().attribute_id) + "|" + std::string(att2.get().attribute_id);
                    if(!std::regex_search(combination.begin(), combination.end(), regex))
                        continue;
                    if(ImGui::Selectable(combination.c_str())){
                        for(auto& el: plot_list.ref_no_track()){

                        }
                    }
                    if(ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)){
                        ImGui::SetDragDropPayload("att_combination", combination.data(), combination.size());
                        ImGui::Text("Drop attribute combination on matrix spot to add");
                        ImGui::EndDragDropSource();
                    }
                }
            }
        }

        // attributes
        ImGui::TableNextColumn();
        if(ImGui::BeginTable("Attributes", 6, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg | ImGuiTableFlags_Hideable)){
            ImGui::TableSetupScrollFreeze(1, 0);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Up");
            ImGui::TableSetupColumn("Down");
            ImGui::TableSetupColumn("Active");
            ImGui::TableSetupColumn("Min");
            ImGui::TableSetupColumn("Max");

            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
            ImGui::TableNextColumn();
            ImGui::TableHeader("Name");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Up");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Down");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Active");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Min");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Max");

            int up_index{-1}, down_index{-1};
            for(auto&& [att_ref, i]: util::enumerate(attribute_order_infos.read())){
                auto& att_no_track = attribute_order_infos.ref_no_track()[i];
                util::imgui::scoped_id attribute_id(att_no_track.attribute_id.data());
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                bool selected = globals::selected_attributes | util::contains(att_ref.attribute_id);
                if(ImGui::Selectable(att_ref.attribute_id.data(), selected, ImGuiSelectableFlags_NoPadWithHalfSpacing, {0, ImGui::GetTextLineHeightWithSpacing()})){

                }
                ImGui::TableNextColumn();
                ImGui::BeginDisabled(i == 0);
                if(ImGui::ArrowButton("##up", ImGuiDir_Up))
                    up_index = static_cast<int>(i);
                ImGui::EndDisabled();
                ImGui::TableNextColumn();
                ImGui::BeginDisabled(i == attribute_order_infos.read().size() - 1);
                if(ImGui::ArrowButton("##do", ImGuiDir_Down))
                    down_index = static_cast<int>(i);
                ImGui::EndDisabled();
                ImGui::TableNextColumn();
                if(ImGui::Checkbox("##a", &att_no_track.active->ref_no_track())){
                    //if(globals::selected_attributes.size() && selected)
                    //    for(std::string_view att: globals::selected_attributes)
                    attribute_order_infos();
                }
                ImGui::TableNextColumn();
                ImGui::SetNextItemWidth(70);
                if(ImGui::DragFloat("##mi", &att_no_track.bounds->ref_no_track().min, (att_no_track.bounds->read().max - att_no_track.bounds->read().min) / 200, 0.f, 0.f, "%.3g"))
                    attribute_order_infos()[i].bounds->write();
                ImGui::TableNextColumn();
                ImGui::SetNextItemWidth(70);
                if(ImGui::DragFloat("##ma", &att_no_track.bounds->ref_no_track().max, (att_no_track.bounds->read().max - att_no_track.bounds->read().min) / 200, 0.f, 0.f, "%.3g"))
                    attribute_order_infos()[i].bounds->write();
            }
            if(up_index >= 0)
                std::swap(attribute_order_infos()[up_index], attribute_order_infos()[up_index - 1]);
            if(down_index >= 0)
                std::swap(attribute_order_infos()[down_index], attribute_order_infos()[down_index + 1]);
            ImGui::EndTable();
        }

        // drawlists
        ImGui::TableNextColumn();
        if(ImGui::BeginTable("drawlists", 8, ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)){
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Up");
            ImGui::TableSetupColumn("Down");
            ImGui::TableSetupColumn("Delete");
            ImGui::TableSetupColumn("Active");
            ImGui::TableSetupColumn("Color");
            ImGui::TableSetupColumn("Splat form");
            ImGui::TableSetupColumn("Radius");

            // header labels
            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
            ImGui::TableNextColumn();
            ImGui::TableHeader("Name");
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
            ImGui::TableHeader("Splat form");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Radius");

            int up_index{-1}, down_index{-1};
            ImGui::PushID(id.data());  // used to distinguish all ui elements in different workbenches
            for(size_t dl_index: util::rev_size_range(drawlist_infos.read())){
                auto& dl = drawlist_infos.ref_no_track()[dl_index];
                ImGui::PushID(dl.drawlist_id.data());
                
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                DRAWLIST_SELECTABLE(dl.drawlist_id);
                ImGui::TableNextColumn();
                ImGui::BeginDisabled(dl_index == drawlist_infos.read().size() - 1);
                if(ImGui::ArrowButton("##us", ImGuiDir_Up))
                    up_index = dl_index;
                ImGui::EndDisabled();
                ImGui::TableNextColumn();
                ImGui::BeginDisabled(dl_index == 0);
                if(ImGui::ArrowButton("##ds", ImGuiDir_Down))
                    down_index = dl_index;
                ImGui::EndDisabled();
                ImGui::TableNextColumn();
                if(ImGui::Button("X##xs"))
                    delete_drawlist = dl.drawlist_id;
                ImGui::TableNextColumn();
                if(ImGui::Checkbox("##acts", &dl.appearance->ref_no_track().show))
                    dl.appearance->write();
                ImGui::TableNextColumn();
                if(ImGui::ColorEdit4("##cols", &dl.appearance->ref_no_track().color.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar))
                    dl.appearance->write();
                ImGui::TableNextColumn();
                if(ImGui::BeginCombo("##form", splat_form_names[dl.scatter_appearance.read().splat].data())){
                    for(auto e: structures::enum_iteration<splat_form_t>()){
                        if(ImGui::MenuItem(splat_form_names[e].data()))
                            drawlist_infos()[dl_index].scatter_appearance().splat = e;
                    }
                    ImGui::EndCombo();
                }
                ImGui::TableNextColumn();
                if(ImGui::DragFloat("##rad", &dl.scatter_appearance.ref_no_track().radius, 1, 1.f, 100.f))
                    drawlist_infos()[dl_index].scatter_appearance();
                ImGui::PopID();
            }
            ImGui::PopID();
            if(up_index >= 0 && up_index < drawlist_infos.read().size() - 1)
                std::swap(drawlist_infos()[up_index], drawlist_infos()[up_index + 1]);
            if(down_index > 0)
                std::swap(drawlist_infos()[down_index], drawlist_infos()[down_index - 1]);

            ImGui::EndTable();
        }

        ImGui::EndTable();
    }

    // popups
    if(ImGui::BeginPopup(plot_menu_id.data())){
        auto& att_a = (attribute_order_infos.ref_no_track() | util::try_find_if<attribute_order_info>([this](auto a){return a.attribute_id == _popup_attributes.atts.a;}))->get();
        auto& att_b = (attribute_order_infos.ref_no_track() | util::try_find_if<attribute_order_info>([this](auto a){return a.attribute_id == _popup_attributes.atts.b;}))->get();
        ImGui::PushItemWidth(100);
        _show_general_settings();
        ImGui::Separator();
        ImGui::BeginDisabled(!plot_additional_datas.contains(_popup_attributes));
        if(ImGui::MenuItem("Remove background image"))
            plot_additional_datas.erase(_popup_attributes);
        ImGui::EndDisabled();
        float diff = att_a.bounds->read().max - att_a.bounds->read().min;
        if(ImGui::DragFloat2(att_a.attribute_read().display_name.c_str(), att_a.bounds->ref_no_track().data(), diff * 1e-3f))
            attribute_order_infos(), att_a.bounds->write();
        diff = att_b.bounds->read().max - att_b.bounds->read().min;
        if(ImGui::DragFloat2(att_b.attribute_read().display_name.c_str(), att_b.bounds->ref_no_track().data(), diff * 1e-3f))
            attribute_order_infos(), att_b.bounds->write();
        if(ImGui::MenuItem(("Swap " + att_a.attribute_read().display_name + " bounds").c_str()))
            std::swap(att_a.bounds->write().min, att_a.bounds->write().max), attribute_order_infos();
        if(ImGui::MenuItem(("Swap " + att_b.attribute_read().display_name + " bounds").c_str()))
            std::swap(att_b.bounds->write().min, att_b.bounds->write().max), attribute_order_infos();
        ImGui::PopItemWidth();
        ImGui::EndPopup();
    }

    ImGui::End();

    // deleting local drawlist
    if(delete_drawlist.size()){
        remove_drawlists(delete_drawlist);
    }
}

void scatterplot_workbench::add_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info){
    for(const auto& dl_id: drawlist_ids){
        // check for already added drawlists
        bool exists = false;
        for(const auto& dl: drawlist_infos.read()){
            if(dl.drawlist_id == dl_id){
                exists = true;
                break;
            }
        }
        if(exists)
            continue;

        auto& dl = globals::drawlists.write().at(dl_id).write();
        drawlist_infos.write().emplace_back(drawlist_info{dl_id, true, dl.appearance_drawlist});
    }
    _update_attribute_order_infos();
    // copying global attribute settings and disconnecting them from global state
    for(auto& att: attribute_order_infos()){
        att.linked_with_attribute = false;
        _local_attribute_storage.emplace(att.attribute_id, std::make_unique<structures::scatterplot_wb::local_attribute_storage>(structures::scatterplot_wb::local_attribute_storage{att.active->read(), att.bounds->read()}));
        att.active = _local_attribute_storage[att.attribute_id]->active;
        att.bounds = _local_attribute_storage[att.attribute_id]->bounds;
    }

    // checking histogram (large vis/axis histograms) rendering or standard rendering
    //_update_registered_histograms();
}

void scatterplot_workbench::signal_dataset_update(const util::memory_view<std::string_view>& dataset_ids, update_flags flags, const structures::gpu_sync_info& sync_info){
    bool any_dl_affected{};
    for(const auto& dl: drawlist_infos.read()){
        if(dataset_ids.contains(dl.drawlist_read().parent_dataset)){
            any_dl_affected = true;
            break;
        }
    }
    if(!any_dl_affected)    
        return;
    
    attribute_order_infos();
    //_update_attribute_order_infos();
    //_update_plot_list();
    //_update_registered_histograms();
}

void scatterplot_workbench::remove_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info){
    for(size_t i: util::rev_size_range(drawlist_infos.read())){
        if(drawlist_ids.contains(drawlist_infos.read()[i].drawlist_id)){
            std::string_view dl = drawlist_infos.read()[i].drawlist_id;
            // locking registry
            auto registry_lock = globals::drawlists.read().at(dl).read().histogram_registry.const_access();
            _registered_histograms.erase(dl);
            drawlist_infos().erase(drawlist_infos().begin() + i);
        }
    }
}

void scatterplot_workbench::signal_drawlist_update(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info){
    bool request_render{};
    for(auto drawlist_id: drawlist_ids){
        if(globals::drawlists.read().at(drawlist_id).changed){
            request_render = true;
            break;
        }
    }
    if(request_render)
        drawlist_infos();  // has to be done delayed as current plot imageas might be still in use and might have to be recreated
}

std::vector<scatterplot_workbench::const_attribute_info_ref> scatterplot_workbench::get_active_ordered_attributes() const{
    std::vector<const_attribute_info_ref> attributes;
    for(const auto& i: attribute_order_infos.read()){
        if(i.active->read())
            attributes.emplace_back(i);
    }
    return attributes;
}

bool scatterplot_workbench::all_registrators_updated(bool rendered) const{
    for(const auto& dl: drawlist_infos.read()){
        if(_registered_histograms.contains(dl.drawlist_id) && _registered_histograms.at(dl.drawlist_id).size()){
            auto access = dl.drawlist_read().histogram_registry.const_access();
            if(!access->dataset_update_done)    
                return false;
            if(rendered && !access->registrators_done)
                return false;
        }
    }
    return true;
}

const scatterplot_workbench::attribute_order_info& scatterplot_workbench::get_attribute_order_info(std::string_view attribute) const{
    return (attribute_order_infos.read() | util::try_find_if<const attribute_order_info>([&attribute](auto a){return a.attribute_id == attribute;}))->get();
}
}