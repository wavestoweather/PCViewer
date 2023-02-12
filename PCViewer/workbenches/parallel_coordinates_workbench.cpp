#include "parallel_coordinates_workbench.hpp"
#include <vk_initializers.hpp>
#include <vma_initializers.hpp>
#include <vk_util.hpp>
#include <parallel_coordinates_renderer.hpp>
#include <imgui_util.hpp>
#include <imgui_stdlib.h>
#include <imgui_internal.h>
#include <util.hpp>
#include <brush_util.hpp>
#include <algorithm>
#include <splines.hpp>
#include <priority_globals.hpp>
#include <json_util.hpp>
#include <data_util.hpp>
#include <flat_set.hpp>

namespace workbenches{

parallel_coordinates_workbench::parallel_coordinates_workbench(const std::string_view id):
    workbench(id)
{
    plot_data.ref_no_track().height = static_cast<uint32_t>(ImGui::GetIO().DisplaySize.y * .5f);
    _update_plot_image();
}

void parallel_coordinates_workbench::_update_plot_image(){
    // waiting for the device to avoid destruction errors
    auto res = vkDeviceWaitIdle(globals::vk_context.device); util::check_vk_result(res);
    if(plot_data.ref_no_track().image)
        util::vk::destroy_image(plot_data.ref_no_track().image);
    if(plot_data.ref_no_track().image_view)
        util::vk::destroy_image_view(plot_data.ref_no_track().image_view);
    if(plot_data.ref_no_track().image_descriptor)
        util::imgui::free_image_descriptor_set(plot_data.ref_no_track().image_descriptor);
    auto image_info = util::vk::initializers::imageCreateInfo(plot_data.read().image_format, {plot_data.read().width, plot_data.read().height, 1}, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    auto alloc_info = util::vma::initializers::allocationCreateInfo();
    std::tie(plot_data.ref_no_track().image, plot_data.ref_no_track().image_view) = util::vk::create_image_with_view(image_info, alloc_info);
    plot_data.ref_no_track().image_descriptor = util::imgui::create_image_descriptor_set(plot_data.read().image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // updating the image layout
    auto image_barrier = util::vk::initializers::imageMemoryBarrier(plot_data.ref_no_track().image.image, VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, {}, {}, {}, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    util::vk::convert_image_layouts_execute(image_barrier);
}

void parallel_coordinates_workbench::_draw_setting_list(){
    ImGui::MenuItem("Enable axis lines", {}, &setting.read().enable_axis_lines);
    ImGui::MenuItem("Enable min max labels", {}, &setting.read().min_max_labes);
    ImGui::MenuItem("Enable axis tick lables", {}, &setting.read().axis_tick_label);
    ImGui::MenuItem("Enable category labels", {}, &setting.read().enable_category_labels);
    ImGui::InputText("Tick format", &setting.read().axis_tick_fmt);
    ImGui::InputInt("Axis tick count", &setting.read().axis_tick_count);
}

void parallel_coordinates_workbench::_swap_attributes(const attribute_order_info_t& from, const attribute_order_info_t& to){
    auto from_it = std::find(attribute_order_infos().begin(), attribute_order_infos().end(), from);
    auto to_it = std::find(attribute_order_infos().begin(), attribute_order_infos().end(), to);
    if(ImGui::IsKeyDown(ImGuiKey_ModCtrl)){
        auto from_ind = std::distance(attribute_order_infos().begin(), from_it);
        auto to_ind = std::distance(attribute_order_infos().begin(), to_it);
        auto lower = std::min(from_ind, to_ind);
        auto higher = std::max(from_ind, to_ind);
        int direction = from_ind < to_ind ? -1 : 1; // direction is the shuffle direction of all elements except from
        auto start_ind = from_ind - direction;
        for(;start_ind >= lower && start_ind <= higher; start_ind -= direction)
            attribute_order_infos()[start_ind + direction] = attribute_order_infos()[start_ind];
        attribute_order_infos()[to_ind] = from;
    }
    else
        std::swap(*from_it, *to_it); 

    _update_registered_histograms();
}

void parallel_coordinates_workbench::_update_registered_histograms(bool request_update){
    if(!all_registrators_updated(true))
        return;
    // updating registered histograms (iterating through indices pairs and checking for registered histogram)
    const auto active_attributes = get_active_ordered_attributes();
    for(const auto& dl: drawlist_infos.read()){
        // active indices can only be calculated for a dataset as the attribute ordering in meory has to be considered
        const auto& attributes = dl.dataset_read().attributes;
        const auto active_indices = util::data::active_attribute_refs_to_indices(active_attributes, attributes);
        // setting the flag for resorting priority rendering
        if(dl.priority_render.read()){
            dl.drawlist_write().delayed_ops.delayed_ops_done = false;
            dl.drawlist_write().delayed_ops.priority_sorting_done = false;
            dl.drawlist_write().delayed_ops.priority_rendering_sorting_started = false;
        }
        if(dl.drawlist_read().const_templatelist().data_size < setting.read().histogram_rendering_threshold){
            _registered_histograms.erase(dl.drawlist_id);
            continue;
        }

        // multidimensional histograms for parallel coordinates plotting
        std::vector<bool> registrator_needed(_registered_histograms[dl.drawlist_id].size(), false);
        for(size_t i: util::i_range(active_indices.size() - 1)){
            std::vector<uint32_t> indices;
            std::vector<int> bucket_sizes;
            std::vector<structures::min_max<float>> bounds;
            int height = plot_data.read().height;
            if(setting.read().render_splines){
                indices = {active_indices[std::max(static_cast<int>(i) - 1, 0)], active_indices[i], active_indices[i + 1], active_indices[std::min(static_cast<uint32_t>(i) + 2, static_cast<uint32_t>(active_indices.size()) - 1)]};
                bounds = {active_attributes[std::max(static_cast<int>(i) - 1, 0)].get().bounds->read(), active_attributes[i].get().bounds->read(), active_attributes[i + 1].get().bounds->read(), active_attributes[std::min(i + 2, active_attributes.size() - 1)].get().bounds->read()};
                bucket_sizes = {config::histogram_splines_hidden_res, height, height, config::histogram_splines_hidden_res};
            }
            else{
                indices = {active_indices[i], active_indices[i + 1]};
                bounds = {active_attributes[i].get().bounds->read(), active_attributes[i + 1].get().bounds->read()};
                bucket_sizes = {height, height};
            }
            bool max_needed = dl.priority_render.read();
            auto registrator_id = util::histogram_registry::get_id_string(indices, bucket_sizes, bounds, false, max_needed);
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
                _registered_histograms[dl.drawlist_id].emplace_back(drawlist.histogram_registry.access()->scoped_registrator(indices, bucket_sizes, bounds, false, max_needed, false));
                registrator_needed.push_back(true);
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
            logger << logging::info_prefix << " parallel_coordinates_workbench (" << active_indices.size() << " attributes, " << registry_lock->registry.size() <<" registrators, " << registry_lock->name_to_registry_key.size() << " name to registry entries), registered histograms: ";
            for(const auto& [key, val]: registry_lock->registry)
                logger << val.hist_id << " ";
            logger << logging::endl;
        }
    }
    // 1 dimensional histograms for axis axis histogram rendering
    for(const auto& dl: drawlist_infos.read()){
        if(setting.read().hist_type == histogram_type::none || !dl.appearance->read().show_histogram){
            if(_registered_axis_histograms.contains(dl.drawlist_id))
                _registered_axis_histograms.erase(dl.drawlist_id);
            continue;
        }

        const auto& attributes = dl.dataset_read().attributes;
        const auto active_indices = util::data::active_attribute_refs_to_indices(active_attributes, attributes);

        std::vector<bool> registrator_needed(_registered_axis_histograms[dl.drawlist_id].size(), false);
        for(uint32_t i: active_indices){
            int height = plot_data.read().height;
            auto registrator_id = util::histogram_registry::get_id_string(i, height, attributes[i].bounds.read(), false, false);
            int registrator_index{-1};
            for(size_t j: util::size_range(_registered_axis_histograms[dl.drawlist_id])){
                if(_registered_axis_histograms[dl.drawlist_id][j].registry_id == registrator_id){
                    registrator_id = static_cast<int>(j);
                    break;
                }
            }
            if(registrator_index >= 0)
                registrator_needed[registrator_index] = true;
            else{
                auto& drawlist = dl.drawlist_write();
                _registered_axis_histograms[dl.drawlist_id].emplace_back(drawlist.histogram_registry.access()->scoped_registrator(i, height, active_attributes[i].get().bounds->read(), false, false, false));
                registrator_needed.push_back(true);
            }
        }
        // removing unused registrators
        auto registry_lock = dl.drawlist_read().histogram_registry.const_access();
        for(size_t i: util::rev_size_range(_registered_axis_histograms[dl.drawlist_id])){
            if(!registrator_needed[i])
                _registered_axis_histograms[dl.drawlist_id].erase(_registered_axis_histograms[dl.drawlist_id].begin() + i);
        }
        if(logger.logging_level >= logging::level::l_5){
            logger << logging::info_prefix << " parallel_coordinates_workbench:: Updated axis histogram registry for drawlist " << dl.drawlist_id << " now requiring " << _registered_axis_histograms[dl.drawlist_id].size() << " histograms" << logging::endl;
        }
    }
    if(request_update){
        for(const auto& dl: drawlist_infos.read()){
            auto registry_lock = dl.drawlist_write().histogram_registry.access();
            registry_lock->request_change_all();
        }
    }
    for(auto& att: attribute_order_infos())
        att.clear_change();
    attribute_order_infos.changed = false;
    for(auto& dl: drawlist_infos.ref_no_track())
        dl.priority_render.changed = false;
}

void parallel_coordinates_workbench::_update_attribute_order_infos(){
    // getting the updated interesction of all dataset attributes
    structures::flat_set<std::string_view> new_attributes;
    for(const auto& [dl, first]: util::first_iter(drawlist_infos.read())){
        structures::flat_set<std::string_view> n;
        for(const auto& att: dl.dataset_read().attributes)
            n |= structures::flat_set<std::string_view>{{att.id}};
        if(first)
            new_attributes = std::move(n);
        else
            new_attributes &= n;
    }

    if(logger.logging_level >= logging::level::l_5)
        logger << logging::info_prefix << " parallel_coordinates_workbench::signal_dataset_update() New attributes will be: " << util::memory_view(new_attributes.data(), new_attributes.size()) << logging::endl;

    structures::flat_set<std::string_view> old_attributes;
    for(const auto& att_info: attribute_order_infos.read())
        old_attributes |= structures::flat_set<std::string_view>{{att_info.attribute_id}};
    auto attributes_to_add = new_attributes / old_attributes;
    
    // deleting all removed attributes in sorting order
    for(size_t i: util::rev_size_range(attribute_order_infos.read())){
        if(!new_attributes.contains(attribute_order_infos.read()[i].attribute_id)){
            if(!attribute_order_infos.read()[i].linked_with_attribute)
                bool todo = true;
            attribute_order_infos().erase(attribute_order_infos.read().begin() + i);
        }
    }
    // adding new attribute references
    if(attribute_order_infos.read().empty() && drawlist_infos.read().size()){
        for(auto& att: drawlist_infos.read()[0].dataset_read().attributes){
            auto& attribute = globals::attributes.ref_no_track()[att.id].ref_no_track();
            attribute_order_infos().emplace_back(attribute_order_info_t{attribute.id, true, attribute.active, attribute.bounds});
        }
    }
    else{
        for(std::string_view att: attributes_to_add){
            auto& attribute = globals::attributes.ref_no_track()[att].ref_no_track();
            attribute_order_infos().emplace_back(attribute_order_info_t{attribute.id, true, attribute.active, attribute.bounds});
        }
    }
}

void parallel_coordinates_workbench::show(){
    if(!active){
        for(auto& [dl, regs]: _registered_histograms)
            for(auto& reg: regs)
                reg.signal_registry_used();
        return;
    }
    const static std::string_view brush_menu_id{"brush menu"};
    const static std::string_view pc_menu_id{"parallel coordinates menu"};

    bool brush_menu_open{false};
    bool pc_menu_open{false};

    // checking for drawlist change (if drawlist has changed, render_plot() will be called later in the frame by update_drawlists)
    // checking for local change
    bool any_drawlist_change{false};
    bool local_change{false};
    bool request_render{false};
    bool request_registered_histograms_update{false};
    bool request_registered_histograms_update_var{false};
    for(const auto& dl_info: drawlist_infos.read()){
        const auto& dl = globals::drawlists.read().at(dl_info.drawlist_id);
        if(!dl.changed && !dl_info.linked_with_drawlist)
            continue;

        bool any_change = dl.read().any_change();       // propagating change to drawlist top
        if(any_change)
            globals::drawlists()[dl_info.drawlist_id]();
        any_drawlist_change |= dl.read().any_change();
    }
    local_change |= drawlist_infos.changed;
    request_render |= local_change;

    // checking for attributes change
    for(auto&& [att, i]: util::enumerate(attribute_order_infos.read()))
        if(att.any_change())
            attribute_order_infos();
    request_registered_histograms_update |= attribute_order_infos.changed;
    request_render |= attribute_order_infos.changed;
    request_render |= setting.changed;

    // checking for priority change
    for(const auto& dl: drawlist_infos.read())
        request_registered_histograms_update |= dl.priority_render.changed;

    request_registered_histograms_update |= setting.changed | _dl_added;
    _dl_added = false;

    // checking for changed image
    if(plot_data.changed){
        _update_plot_image();
        plot_data.changed = false;

        request_render |= true;

        // updating requested histograms
        request_registered_histograms_update = true;
    }

    // check for requested registered histogram update
    if(request_registered_histograms_update)
        _update_registered_histograms(request_registered_histograms_update_var);

    if(request_render)
        render_plot();

    ImGui::Begin(id.c_str(), &active);

    // -------------------------------------------------------------------------------
    // Plot region including labels and min max values
    // -------------------------------------------------------------------------------
    auto content_size = ImGui::GetWindowContentRegionMax();

    uint32_t labels_count{};
    for(const auto& att_ref: attribute_order_infos.read())
        if(att_ref.active->read())
            ++labels_count;

    constexpr float tick_width = 10;
    const float padding_side = 10;
    const ImVec2 button_size{70, 20};
    const float gap = static_cast<float>((content_size.x - 2 * padding_side) / (labels_count - 1));
    const float button_gap = static_cast<float>((content_size.x - 2 * padding_side - button_size.x) / (labels_count - 1));

    float cur_offset{};
    ImGui::Dummy({1, 1});
    // attribute labels
    for(const auto& att_ref: attribute_order_infos.read()){
        if(!att_ref.active->read())
            continue;
        
        ImGui::SameLine(cur_offset);
        
        const auto& global_attribute = att_ref.attribute_read();
        std::string name = global_attribute.display_name;
        float text_size = ImGui::CalcTextSize(name.c_str()).x;
        if(text_size > button_size.x){
            //add ellipsis at the end of the text
            bool too_long = true;
            std::string cur_substr = name.substr(0, name.size() - 4) + "...";
            while (too_long) {
                text_size = ImGui::CalcTextSize(cur_substr.c_str()).x;
                too_long = text_size > button_size.x;
                cur_substr = cur_substr.substr(0, cur_substr.size() - 4) + "...";
            }
            name = cur_substr + "##" + name;
        }
        ImGui::Button(name.size() ? name.c_str(): "##place", button_size);
        if (name != global_attribute.display_name && ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("%s", global_attribute.display_name.c_str());
            ImGui::Text("Drag and drop to switch axes, hold ctrl to shuffle");
            ImGui::EndTooltip();
        }
        if (name == global_attribute.display_name && ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("Drag and drop to switch axes, hold ctrl to shuffle");
            ImGui::EndTooltip();
        }
        if (ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {// TODO implement            //editAttributeName = i;
            //strcpy(newAttributeName, pcAttributes[i].originalName.c_str());
        }

        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
            ImGui::SetDragDropPayload("ATTRIBUTE", &att_ref, sizeof(att_ref));
            ImGui::Text("Swap %s", name.c_str());
            ImGui::EndDragDropSource();
        }
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ATTRIBUTE")) {
                const attribute_order_info_t other = *(const attribute_order_info_t*)payload->Data;

                _swap_attributes(other, att_ref);
            }
        }
        cur_offset += button_gap;
    }
    // attribute max values
    ImGui::Dummy({1, 1});
    cur_offset = 0;
    for(const auto& [att_ref, i]: util::enumerate(attribute_order_infos.read())){
        util::imgui::scoped_id att_id(att_ref.attribute_id.data());
        if(!att_ref.active->read())
            continue;
        
        ImGui::SetNextItemWidth(button_size.x);
        ImGui::SameLine(cur_offset);
        float diff = att_ref.bounds->read().max - att_ref.bounds->read().min;
        if(ImGui::DragFloat(setting.read().min_max_labes ? "max": "##ma", &attribute_order_infos.ref_no_track()[i].bounds->ref_no_track().max, diff * .001f, 0, 0, "%6.4g")){
            //_request_registered_histograms_update = true;
            //_request_registered_histograms_update_var = true;
            attribute_order_infos()[i].bounds->write();
        }
        if(ImGui::IsItemClicked(ImGuiMouseButton_Right)){
            // todo minmaxpopup
        }
        cur_offset += button_gap;
    }

    auto pic_pos = ImGui::GetCursorScreenPos();
    auto pic_size = ImVec2{content_size.x - 10, static_cast<float>(plot_data.read().height)};
    ImGui::Image(plot_data.read().image_descriptor, pic_size);

    if(setting.read().enable_axis_lines){
        auto col = ImGui::ColorConvertFloat4ToU32(setting.read().plot_background);
        float y1 = pic_pos.y;
        float y2 = y1 + pic_size.y - 1;
        ImVec4 inv_color{(1 - setting.read().plot_background.x), (1 - setting.read().plot_background.y), (1 - setting.read().plot_background.z), 1.f};
        auto line_col = IM_COL32(inv_color.x * 255, inv_color.y * 255, inv_color.z * 255, 255);
        float hist_width = setting.read().hist_type == histogram_type::none? .0f: float(setting.read().histogram_width) * pic_size.x / 2.f;
        for(int i: util::i_range(labels_count)){
            float x = pic_pos.x + (pic_size.x - 1 - hist_width) * i / (labels_count - 1) + hist_width / 2.f;
            ImGui::GetWindowDrawList()->AddLine({x - 1, y1}, {x - 1, y2}, col);
            ImGui::GetWindowDrawList()->AddLine({x + 1, y1}, {x + 1, y2}, col);
            ImGui::GetWindowDrawList()->AddLine({x, y1    }, {x, y2    }, line_col);
        }
        if(setting.read().enable_category_labels){
            int att_pos{};
            const float label_height = ImGui::GetTextLineHeightWithSpacing() + ImGui::GetStyle().WindowPadding.y * 2;
            for(const auto& att_ref: attribute_order_infos.read()){
                if(!att_ref.active->read())
                    continue;
                if(att_ref.attribute_read().categories.empty()){
                    ++att_pos;
                    continue;
                }
                float x = pic_pos.x + (pic_size.x - 1 - hist_width) * att_pos / (labels_count - 1) + hist_width / 2.f;
                const auto& attribute = att_ref.attribute_read();
                float min_val = attribute.bounds.read().min;
                float max_val = attribute.bounds.read().max;
                // going through the ordered categories and drawing a label if the distance to the previous label is large enough
                const double max_label_count = (attribute.categories.at(std::string(attribute.ordered_categories.back())) - attribute.categories.at(std::string(attribute.ordered_categories.front()))) / (max_val - min_val) * pic_size.y / label_height;
                const size_t label_step = std::max(size_t(1), static_cast<size_t>(std::ceil(attribute.ordered_categories.size() / std::abs(max_label_count))));
                size_t step = static_cast<size_t>(std::exp2(std::ceil(std::log2(label_step))));
                for(auto c: util::i_range(size_t(0), attribute.ordered_categories.size(), step)){
                    auto cat = attribute.ordered_categories[c];
                    float p = util::normalize_val_for_range(attribute.categories.at(std::string(cat)), min_val, max_val);
                    if(p < 0 || p > 1)
                        continue;
                    p = util::unnormalize_val_for_range(p, pic_pos.y + pic_size.y, pic_pos.y);
                    ImGuiWindowFlags flags = ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking;
                    ImGui::SetNextWindowPos({x, p - label_height / 2.f});
                    ImGui::Begin(cat.data(), {}, flags);
                    ImGui::Text("%s", cat.data());
                    ImGui::End();
                }
                ++att_pos;
            }
        }
        if(setting.read().axis_tick_label){
            ImGui::PushFontShadow(col);
            int att_pos{};
            for(const auto& att_ref: attribute_order_infos.read()){
                if(!att_ref.active->read())
                    continue;
                if(att_ref.attribute_read().categories.size()){
                    ++att_pos;
                    continue;
                }

                float x = pic_pos.x + (pic_size.x - 1 - hist_width) * att_pos / (labels_count - 1) + hist_width / 2.f;
                const auto& attribute = att_ref.attribute_read();
                float min_tick = attribute.bounds.read().min;
                float max_tick = attribute.bounds.read().max;
                double diff = max_tick - min_tick;
                double cell = diff / setting.read().axis_tick_count;
                double base = std::pow(10., std::floor(std::log10(cell)));
                double unit = base, U;
                constexpr double h = .5, h5 = .5 + 1.5 * h;
                if((U = 2 * base) - cell < h * (cell - unit)) { 
                    unit = U;
                    if((U = 5 * base) - cell < h5 * (cell - unit)){
                        unit = U;
                        if((U = 10 * base) - cell < h * (cell - unit))
                            unit = U;
                }}
                constexpr double rounding_eps = 1e-10;
                double ns = std::floor(min_tick / unit + rounding_eps) * unit; while(ns < min_tick) ns += unit;
                double nu = std::ceil(max_tick / unit - rounding_eps) * unit; while(nu > max_tick) nu -= unit;
                const auto cursor_pos = ImGui::GetCursorScreenPos();
                for(double tick_val = ns; tick_val <= nu; tick_val += unit){
                    float y = static_cast<float>((tick_val - attribute.bounds.read().max) / (attribute.bounds.read().min - attribute.bounds.read().max) * pic_size.y + pic_pos.y);
                    ImGui::GetWindowDrawList()->AddLine({x - tick_width / 2, y - 1}, {x + tick_width / 2, y - 1}, col);
                    ImGui::GetWindowDrawList()->AddLine({x - tick_width / 2, y + 1}, {x + tick_width / 2, y + 1}, col);
                    ImGui::GetWindowDrawList()->AddLine({x - tick_width / 2, y    }, {x + tick_width / 2, y    }, line_col);
                    if(x > pic_pos.x + (1. - (1. / labels_count)) * pic_size.x)
                        ImGui::SetCursorScreenPos({x - ImGui::GetFontSize() * 3, y - .5f * ImGui::GetTextLineHeight()});
                    else
                        ImGui::SetCursorScreenPos({x, y - .5f * ImGui::GetTextLineHeight()});
                    
                    ImGui::TextColored(inv_color, setting.read().axis_tick_fmt.c_str(), tick_val);
                }
                ImGui::SetCursorScreenPos(cursor_pos);
                ++att_pos;
            }
            ImGui::PopFontShadow();
        }
    }

    // attribute min values
    ImGui::Dummy({1, 1});
    cur_offset = 0;
    for(const auto& [att_ref, i]: util::enumerate(attribute_order_infos.read())){
        util::imgui::scoped_id att_id(att_ref.attribute_id.data());
        if(!att_ref.active->read())
            continue;
        
        ImGui::SetNextItemWidth(button_size.x);
        ImGui::SameLine(cur_offset);
        float diff = att_ref.bounds->read().max - att_ref.bounds->read().min;
        if(ImGui::DragFloat(setting.read().min_max_labes ? "min": "##mi", &attribute_order_infos.ref_no_track()[i].bounds->ref_no_track().min, diff * .001f, 0, 0, "%6.4g")){
            //_request_registered_histograms_update = true;
            //_request_registered_histograms_update_var = true;
            attribute_order_infos()[i].bounds->write();
        }
        if(ImGui::IsItemClicked(ImGuiMouseButton_Right)){
            // todo minmaxpopup
        }
        cur_offset += button_gap;
    }

    // brush windows
    std::map<std::string_view, uint32_t> place_of_attribute;
    if(globals::brush_edit_data.brush_type != structures::brush_edit_data::brush_type::none || _select_priority_center_single || _select_priority_center_all){
        uint32_t place = 0;
        for(const auto& att_ref: attribute_order_infos.read())
            if(att_ref.active->read())
                place_of_attribute[att_ref.attribute_id] = place++;
    }

    if(globals::brush_edit_data.brush_type != structures::brush_edit_data::brush_type::none){ 
        bool any_hover = false;
        robin_hood::unordered_set<structures::brush_id> brush_delete;
        ImVec2 mouse_pos = {ImGui::GetIO().MousePos.x - ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, static_cast<float>(setting.read().brush_drag_threshold)).x, ImGui::GetIO().MousePos.y - ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0).y};

        const structures::range_brush& selected_brush = util::brushes::get_selected_range_brush_const();
        float brush_gap = pic_size.x / (labels_count - 1);

        for(const auto& brush: selected_brush){
            if(place_of_attribute.count(brush.attr) == 0)
                continue;   // attribute not active

            const auto& att_ref = (attribute_order_infos.read() | util::try_find_if<const attribute_order_info_t>([&brush](auto& i){return i.attribute_id == brush.attr;}))->get();
            
            float x = brush_gap * place_of_attribute[brush.attr] + pic_pos.x - static_cast<float>(setting.read().brush_box_width / 2.);

            float y = util::normalize_val_for_range(brush.max, att_ref.bounds->read().max, att_ref.bounds->read().min) * pic_size.y + pic_pos.y;
            float height = (brush.max - brush.min) / (att_ref.bounds->read().max - att_ref.bounds->read().min) * pic_size.y;
            
            structures::brush_edit_data::brush_region hovered_region{structures::brush_edit_data::brush_region::COUNT};
            if(util::point_in_box(mouse_pos, {x, y}, {x + float(setting.read().brush_box_width), y + height}))
                hovered_region = structures::brush_edit_data::brush_region::body;
            else if(util::point_in_box(mouse_pos, {x, y - float(setting.read().brush_box_border_hover_width)}, {x + float(setting.read().brush_box_width), y}))
                hovered_region = structures::brush_edit_data::brush_region::top;
            else if(util::point_in_box(mouse_pos, {x, y + height}, {x + float(setting.read().brush_box_width), y + height + float(setting.read().brush_box_border_hover_width)}))
                hovered_region = structures::brush_edit_data::brush_region::bottom;
            if(!ImGui::IsWindowHovered())
                hovered_region = structures::brush_edit_data::brush_region::COUNT;  // resetting hover to avoid dragging when window is not focues
            bool brush_hovered = hovered_region != structures::brush_edit_data::brush_region::COUNT;
            
            ImU32 border_color;
            if(globals::brush_edit_data.selected_ranges.contains(brush.id))
                border_color = util::vec4_to_imu32(setting.read().brush_box_selected_color);
            else if(globals::brush_edit_data.brush_type == structures::brush_edit_data::brush_type::global)
                border_color = util::vec4_to_imu32(setting.read().brush_box_global_color);
            else if(globals::brush_edit_data.brush_type == structures::brush_edit_data::brush_type::local)
                border_color = util::vec4_to_imu32(setting.read().brush_box_local_color);

            ImGui::GetWindowDrawList()->AddRect({x, y}, {x + float(setting.read().brush_box_width), y + height}, border_color, 1, ImDrawCornerFlags_All, float(setting.read().brush_box_border_width));

            // set mouse cursor
            switch(hovered_region){
            case structures::brush_edit_data::brush_region::top:
            case structures::brush_edit_data::brush_region::bottom:
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
                break;
            case structures::brush_edit_data::brush_region::body:
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
                break;
            }

            // start dragging
            if(brush_hovered && ImGui::GetIO().MouseClicked[ImGuiMouseButton_Left]){
                if(!ImGui::GetIO().KeyCtrl)
                    globals::brush_edit_data.selected_ranges.clear();
                globals::brush_edit_data.selected_ranges.insert(brush.id);
                globals::brush_edit_data.hovered_region_on_click = hovered_region;
            }
            // dragging
            if(globals::brush_edit_data.selected_ranges.contains(brush.id) && ((ImGui::IsMouseDown(ImGuiMouseButton_Left) && ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, static_cast<float>(setting.read().brush_drag_threshold)).y) || ImGui::IsKeyPressed(ImGuiKey_DownArrow) || ImGui::IsKeyPressed(ImGuiKey_UpArrow))){
                float delta;
                if(ImGui::IsMouseDown(ImGuiMouseButton_Left))
                    delta = -ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, float(setting.read().brush_drag_threshold)).y / pic_size.y;
                else if(ImGui::IsKeyPressed(ImGuiKey_UpArrow))
                    delta = float(setting.read().brush_arrow_button_move);
                else if(ImGui::IsKeyPressed(ImGuiKey_DownArrow))
                    delta = -float(setting.read().brush_arrow_button_move);
                
                structures::range_brush& range_brush = util::brushes::get_selected_range_brush();
                structures::axis_range& range       = *std::find(range_brush.begin(), range_brush.end(), brush);
                switch(globals::brush_edit_data.hovered_region_on_click){
                case structures::brush_edit_data::brush_region::top:
                    range.max += delta * (att_ref.bounds->read().max - att_ref.bounds->read().min);
                    break;
                case structures::brush_edit_data::brush_region::bottom:
                    range.min += delta * (att_ref.bounds->read().max - att_ref.bounds->read().min);
                    break;
                case structures::brush_edit_data::brush_region::body:
                    range.max += delta * (att_ref.bounds->read().max - att_ref.bounds->read().min);
                    range.min += delta * (att_ref.bounds->read().max - att_ref.bounds->read().min);
                    break;
                }

                if(range.min > range.max){
                    std::swap(range.min, range.max);
                    if(globals::brush_edit_data.hovered_region_on_click == structures::brush_edit_data::brush_region::top)
                        globals::brush_edit_data.hovered_region_on_click = structures::brush_edit_data::brush_region::bottom;
                    else
                        globals::brush_edit_data.hovered_region_on_click = structures::brush_edit_data::brush_region::top;
                }
            }
            // brush right click menu
            if(ImGui::IsMouseClicked(ImGuiMouseButton_Right) && brush_hovered)
                brush_menu_open = true;
            
            if(ImGui::IsKeyPressed(ImGuiKey_Delete)){
                brush_delete = globals::brush_edit_data.selected_ranges;
            }

            if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && brush_hovered){
                brush_delete = {brush.id};
            }

            // TODO mu adoption

            // Tooltip for hovered brush or selected brush
            if (brush_hovered || globals::brush_edit_data.selected_ranges.contains(brush.id)) {
                float x_anchor = .5f;
                if (place_of_attribute[brush.attr] == 0) x_anchor = 0;
                if (place_of_attribute[brush.attr] == labels_count - 1) x_anchor = 1;

                ImGui::SetNextWindowPos({ x + float(setting.read().brush_box_width) / 2,y }, 0, { x_anchor,1 });
                ImGui::SetNextWindowBgAlpha(ImGui::GetStyle().Colors[ImGuiCol_PopupBg].w * 0.60f);
                ImGuiWindowFlags flags = ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking;
                ImGui::Begin(("Tooltip brush max##" + std::to_string(brush.id)).c_str(), NULL, flags);
                ImGui::Text("%f", brush.max);
                ImGui::End();

                ImGui::SetNextWindowPos({ x + float(setting.read().brush_box_width) / 2, y + height }, 0, { x_anchor,0 });
                ImGui::SetNextWindowBgAlpha(ImGui::GetStyle().Colors[ImGuiCol_PopupBg].w * 0.60f);
                ImGui::Begin(("Tooltip brush min##" + std::to_string(brush.id)).c_str(), NULL, flags);
                ImGui::Text("%f", brush.min);
                ImGui::End();
            }

            any_hover |= brush_hovered;
        }
        if(ImGui::IsWindowFocused())
            ImGui::ResetMouseDragDelta();   // has to be reset to avoid dragging open the box too far

        // brush creation
        bool new_brush{false};
        for(const auto& att_ref: attribute_order_infos.read()){
            if(!att_ref.active->read())
                continue;
            float x = static_cast<float>(brush_gap * place_of_attribute[att_ref.attribute_id] + pic_pos.x - setting.read().brush_box_width / 2);
            bool axis_hover = util::point_in_box(mouse_pos, {x, pic_pos.y}, {x + float(setting.read().brush_box_width), pic_pos.y + pic_size.y}) && ImGui::IsWindowHovered();
            if(!any_hover && axis_hover && globals::brush_edit_data.selected_ranges.empty()){
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

                if(ImGui::IsMouseClicked(ImGuiMouseButton_Left)){
                    float val = util::normalize_val_for_range(mouse_pos.y, pic_pos.y, pic_pos.y + pic_size.y) * (att_ref.bounds->read().min - att_ref.bounds->read().max) + att_ref.bounds->read().max;

                    structures::axis_range new_range{att_ref.attribute_id, globals::cur_brush_range_id++, val, val};
                    globals::brush_edit_data.selected_ranges.insert(new_range.id);
                    globals::brush_edit_data.hovered_region_on_click = structures::brush_edit_data::brush_region::top;
                    util::brushes::get_selected_range_brush().push_back(std::move(new_range));

                    new_brush = true;
                }
            }
        }

        // brush deletion
        if(brush_delete.size()){
            util::brushes::delete_brushes(brush_delete);
            brush_delete.clear();
        }

        // releasing edge
        if(!ImGui::IsPopupOpen(brush_menu_id.data()) && !any_hover && globals::brush_edit_data.selected_ranges.size() &&  (ImGui::IsMouseReleased(ImGuiMouseButton_Left) || (!new_brush && ImGui::IsMouseClicked(ImGuiMouseButton_Left))) && !ImGui::GetIO().KeyCtrl)
            globals::brush_edit_data.selected_ranges.clear();
    }
    
    // priority selection
    if(_select_priority_center_single || _select_priority_center_all){
        globals::brush_edit_data.clear();
        globals::selected_drawlists.clear();
        ImVec2 b{pic_pos.x + pic_size.x, pic_pos.y + pic_size.y};
        ImGui::GetWindowDrawList()->AddRect(pic_pos, b, IM_COL32(255, 255, 0, 255), 0, 15, 5);
        if(!util::point_in_box(ImGui::GetIO().MousePos, pic_pos, b) && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            _select_priority_center_all = _select_priority_center_single = false;

        for(const auto& att_ref: attribute_order_infos.read()){
            if(!att_ref.active->read())
                continue;
            float x = gap * place_of_attribute[att_ref.attribute_id] + pic_pos.x - static_cast<float>(setting.read().brush_box_width / 2.);
            bool axis_hover = util::point_in_box(ImGui::GetMousePos(), {x, pic_pos.y}, {x + float(setting.read().brush_box_width), b.y});
            if(axis_hover){
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                if(ImGui::IsMouseClicked(ImGuiMouseButton_Left)){
                    const auto& bounds = att_ref.bounds->read();
                    globals::priority_center_attribute_id = att_ref.attribute_id;
                    globals::priority_center_vealue = util::unnormalize_val_for_range(util::normalize_val_for_range(ImGui::GetMousePos().y, b.y, pic_pos.y), bounds.min, bounds.max);
                    globals::priority_center_distance = std::max(globals::priority_center_vealue - bounds.min, bounds.max - globals::priority_center_vealue);
                    logger << logging::info_prefix << " priority attribute: " <<att_ref.attribute_read().display_name << ", priority center: " << globals::priority_center_vealue << ", priority distance " << globals::priority_center_distance << logging::endl;
                    for(auto& dl: drawlist_infos.ref_no_track()){
                        if(_select_priority_center_single && !util::memory_view(globals::selected_drawlists).contains(dl.drawlist_id) && !globals::selected_drawlists.empty())
                            continue;
                        dl.drawlist_write().delayed_ops.priority_rendering_requested = true;
                        dl.drawlist_write().delayed_ops.priority_sorting_done = false;
                        dl.drawlist_write().delayed_ops.delayed_ops_done = false;
                        dl.appearance->write().color.w = .9f;
                        dl.priority_render = true;
                        if(_select_priority_center_single)
                            break;
                    }
                    _select_priority_center_all = _select_priority_center_single = false;
                    //_request_registered_histograms_update = true;
                    //_request_registered_histograms_update_var = true;
                }
            }
        }
        
    }

    // -------------------------------------------------------------------------------
    // settings region
    // -------------------------------------------------------------------------------
    std::string_view delete_drawlist{};

    if(ImGui::BeginTable("Settings", 3, ImGuiTableFlags_Hideable | ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings)){
        ImGui::TableSetupScrollFreeze(1, 0);
        ImGui::TableSetupColumn("General settings");
        ImGui::TableSetupColumn("Attributes");
        ImGui::TableSetupColumn("Drawlists");

        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableNextColumn();
        ImGui::TableHeader("General settings");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Attributes");
        ImGui::TableNextColumn();
        ImGui::TableHeader("Drawlists");
        

        // general settings
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        _draw_setting_list();

        // attribute settings
        ImGui::TableNextColumn();
        if(ImGui::BeginTable("Attributes", 4, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)){
            ImGui::TableSetupScrollFreeze(1, 0);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Up");
            ImGui::TableSetupColumn("Down");
            ImGui::TableSetupColumn("Active");

            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
            ImGui::TableNextColumn();
            ImGui::TableHeader("Name");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Up");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Down");
            ImGui::TableNextColumn();
            ImGui::TableHeader("Active");

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
            }
            if(up_index >= 0)
                std::swap(attribute_order_infos()[up_index], attribute_order_infos()[up_index - 1]);
            if(down_index >= 0)
                std::swap(attribute_order_infos()[down_index], attribute_order_infos()[down_index + 1]);
            ImGui::EndTable();
        }


        // drawlist settings
        ImGui::TableNextColumn();
        if(ImGui::BeginTable("Drawlist settings", 7, ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)){
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
            
            int up_index{-1}, down_index{-1};
            for(size_t dl_index: util::rev_size_range(drawlist_infos.read())){
                auto& dl = drawlist_infos.ref_no_track()[dl_index];
                std::string dl_string(dl.drawlist_id);
                const auto& drawlist = globals::drawlists.read().at(dl.drawlist_id);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                bool selected = util::memory_view(globals::selected_drawlists).contains(dl.drawlist_id);
                if(ImGui::Selectable((drawlist.read().name + "##pc_wb").c_str(), selected, ImGuiSelectableFlags_NoPadWithHalfSpacing, {0, ImGui::GetTextLineHeightWithSpacing()})){
                    globals::selected_drawlists.clear();
                    globals::brush_edit_data.clear();
                    if(!selected){
                        globals::selected_drawlists.push_back(drawlist.read().name);
                        globals::brush_edit_data.brush_type = structures::brush_edit_data::brush_type::local;
                        globals::brush_edit_data.local_brush_id = dl.drawlist_id;
                    }
                }
                ImGui::TableNextColumn();
                if(ImGui::ArrowButton(("##u" + dl_string).c_str(), ImGuiDir_Up))
                    up_index = static_cast<int>(dl_index);
                ImGui::TableNextColumn();
                if(ImGui::ArrowButton(("##d" + dl_string).c_str(), ImGuiDir_Down))
                    down_index = static_cast<int>(dl_index);
                ImGui::TableNextColumn();
                if(ImGui::Button(("X##x" + dl_string).c_str()))
                    delete_drawlist = dl.drawlist_id;
                ImGui::TableNextColumn();
                if(ImGui::Checkbox(("##act" + dl_string).c_str(), &dl.appearance->ref_no_track().show))
                    dl.appearance->write();
                ImGui::TableNextColumn();
                if(ImGui::ColorEdit4(("##col" + dl_string).c_str(), &dl.appearance->ref_no_track().color.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar))
                    dl.appearance->write();
                ImGui::TableNextColumn();
                ImGui::SetNextItemWidth(100);
                if(ImGui::BeginCombo(("##med" + dl_string).c_str(), structures::median_type_names[dl.median->read()].data())){
                    for(auto m: structures::median_iteration{}){
                        if(ImGui::MenuItem(structures::median_type_names[m].data()))
                            dl.median->write() = m;
                    }
                    ImGui::EndCombo();
                }
            }
            if(up_index >= 0 && up_index < drawlist_infos.read().size() - 1)
                std::swap(drawlist_infos()[up_index], drawlist_infos()[up_index + 1]);
            if(down_index > 0)
                std::swap(drawlist_infos()[down_index], drawlist_infos()[down_index - 1]);

            ImGui::EndTable();
        }
        ImGui::EndTable();
    }

    // popups ----------------------------------------------------------------
    pc_menu_open = !brush_menu_open && ImGui::IsMouseClicked(ImGuiMouseButton_Right) && ImGui::IsWindowHovered() && util::point_in_box(ImGui::GetMousePos(), pic_pos, {pic_pos.x + pic_size.x, pic_pos.y + pic_size.y});

    if(brush_menu_open)
        ImGui::OpenPopup(brush_menu_id.data());
    if(ImGui::BeginPopup(brush_menu_id.data())){
        if(ImGui::MenuItem("Delete##b", {}, false, bool(globals::brush_edit_data.selected_ranges.size()))){
            util::brushes::delete_brushes(globals::brush_edit_data.selected_ranges);
        }
        if(ImGui::MenuItem("Fit axis to bounds", {}, false, bool(globals::brush_edit_data.selected_ranges.size()))){
            const auto& selected_ranges = util::brushes::get_selected_range_brush_const();
            // getting the extremum values for each axis if existent
            std::map<std::string_view, std::optional<structures::min_max<float>>> axis_values;
            for(const auto& range: selected_ranges){
                if(axis_values[range.attr])
                    axis_values[range.attr] = std::min_max(*axis_values[range.attr], {range.min, range.max});
                else
                    axis_values[range.attr] = structures::min_max<float>{range.min, range.max};
            }
            for(size_t i: util::size_range(attribute_order_infos.read())){
                if(axis_values[attribute_order_infos.read()[i].attribute_id])
                    attribute_order_infos()[i].bounds->write() = *axis_values[attribute_order_infos.read()[i].attribute_id];
            }
        }
        ImGui::DragInt("Live brush treshold", &setting.read().live_brush_threshold);
        ImGui::EndPopup();
    }
    if(pc_menu_open)
        ImGui::OpenPopup(pc_menu_id.data());
    if(ImGui::BeginPopup(pc_menu_id.data())){
        if(ImGui::ColorEdit4("Plot background", &setting.ref_no_track().plot_background.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar))
            setting();
        if(ImGui::MenuItem("Render splines", {}, &setting.ref_no_track().render_splines)){
            //_request_registered_histograms_update = true;
            setting();
        }
        if(ImGui::BeginMenu("Axis histograms")){
            if(ImGui::BeginCombo("Histogram", histogram_type_names[setting.read().hist_type].data())){
                for(auto type: structures::enum_iteration<histogram_type>()){
                    if(ImGui::MenuItem(histogram_type_names[type].data())){
                        setting().hist_type = type;
                        //_request_registered_histograms_update = true;
                    }
                }
                ImGui::EndCombo();
            }
            if(ImGui::SliderDouble("Blur radius", &setting.ref_no_track().histogram_blur_width, .001, .3))
                setting();
            if(ImGui::SliderDouble("Histogram width", &setting.ref_no_track().histogram_width, .0, .1))
                setting();
            ImGui::EndMenu();
        }
        if(ImGui::BeginMenu("Priority rendering")){
            if(ImGui::MenuItem("Priority rendering off")){
                for(auto& dl: drawlist_infos()){
                    if(!dl.priority_render.read())
                        continue;
                    dl.priority_render = false;
                    dl.drawlist_write().delayed_ops.priority_rendering_requested = false;
                    dl.drawlist_write().delayed_ops.priority_sorting_done = true;
                }
                //_request_registered_histograms_update = true;
            }
            if(ImGui::MenuItem("Set priority center"))
                _select_priority_center_single = true;
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("Set priority center for the selected drawlist/top most drawlist");
            
            if(ImGui::MenuItem("Set priority center all"))
                _select_priority_center_all = true;
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("Set priority center for all drawlists");

            ImGui::EndMenu();
        }
        if(ImGui::BeginMenu("Plot visuals")){
            _draw_setting_list();
            ImGui::EndMenu();
        }
        if(ImGui::BeginMenu("Plot Size")){
            if(ImGui::InputInt2("width/height", reinterpret_cast<int*>(&plot_data.ref_no_track().width), ImGuiInputTextFlags_EnterReturnsTrue))
                plot_data();
            
            if(ImGui::BeginCombo("Sample per pixel", util::vk::sample_count_infos.at(plot_data.read().image_samples).name.data())){
                for(const auto& [bit, count_name]: util::vk::sample_count_infos)
                    if(ImGui::MenuItem(count_name.name.data()))
                        plot_data().image_samples = bit;
                ImGui::EndCombo();
            }

            static std::map<VkFormat, std::string_view> format_names{{VK_FORMAT_R8G8B8A8_UNORM, "8 Bit Unorm"}, {VK_FORMAT_R16G16B16A16_UNORM, "16 Bit Unorm"}, {VK_FORMAT_R16G16B16A16_SFLOAT, "16 Bit Float"}, {VK_FORMAT_R32G32B32A32_SFLOAT, "32 Bit Float"}};
            if(ImGui::BeginCombo("PCP Format", format_names[plot_data.read().image_format].data())){
                for(const auto& [bit, name]: format_names)
                    if(ImGui::MenuItem(name.data()))
                        plot_data().image_format = bit;
                ImGui::EndCombo();
            }

            ImGui::EndMenu();
        }
        ImGui::EndPopup();
    }

    ImGui::End();

    // deleting local drawlist
    if(delete_drawlist.size()){
        auto del = drawlist_infos().begin();
        while(del->drawlist_id != delete_drawlist)
            ++del;
        remove_drawlists(del->drawlist_id);
    }
}

void parallel_coordinates_workbench::render_plot()
{
    // if histogram rendering requested and not yet finished delay rendering
    if(!all_registrators_updated())// || attribute_order_infos.changed)
        return;

    if(logger.logging_level >= logging::level::l_5)
        logger << logging::info_prefix << " parallel_coordinates_workbench::render_plot()" << logging::endl;
    pipelines::parallel_coordinates_renderer::render_info render_info{*this};
    pipelines::parallel_coordinates_renderer::instance().render(render_info);
    for(auto& dl_info: drawlist_infos()){
        if(!dl_info.linked_with_drawlist)
            dl_info.clear_change();
    }
    drawlist_infos.changed = false;
    for(auto& att_ref: attribute_order_infos.ref_no_track()){
        if(att_ref.bounds->changed)
            att_ref.bounds->changed = false;
    }
    attribute_order_infos.changed = false;
    setting.changed = false;
    // signaling all registrators
    for(auto& [dl, registrators]: _registered_histograms){
        // locking registry to avoid mutithread clash
        auto registry_lock = globals::drawlists.read().at(dl).read().histogram_registry.const_access();
        for(auto& registrator: registrators)
            registrator.signal_registry_used();
    }
    for(auto& [dl, registrators]: _registered_axis_histograms){
        auto registry_lock = globals::drawlists.read().at(dl).read().histogram_registry.const_access();
        for(auto& registrator: registrators)
            registrator.signal_registry_used();
    }
}

std::vector<parallel_coordinates_workbench::const_attribute_info_ref> parallel_coordinates_workbench::get_active_ordered_attributes() const{
    std::vector<const_attribute_info_ref> attributes;
    for(const auto& i: attribute_order_infos.read()){
        if(i.active->read())
            attributes.emplace_back(i);
    }
    return attributes;
}
const parallel_coordinates_workbench::attribute_order_info_t& parallel_coordinates_workbench::get_attribute_order_info(std::string_view attribute) const{
    return (attribute_order_infos.read() | util::try_find_if<const attribute_order_info_t>([&attribute](auto a){return a.attribute_id == attribute;}))->get();
}

bool parallel_coordinates_workbench::all_registrators_updated(bool rendered) const{
    // if histogram rendering requested and not yet finished delay rendering
    for(const auto& dl: drawlist_infos.read()){
        if(_registered_histograms.contains(dl.drawlist_id) && _registered_histograms.at(dl.drawlist_id).size() ||
            _registered_axis_histograms.contains(dl.drawlist_id) && _registered_axis_histograms.at(dl.drawlist_id).size()){
            const auto access = dl.drawlist_read().histogram_registry.const_access();
            if(!access->dataset_update_done)
                return false;     // a registered histogram is being currently updated, no rendering possible
            if(rendered && !access->registrators_done)
                return false;
        }
        if(!dl.drawlist_read().delayed_ops.delayed_ops_done)
            return false;
    }
    return true;
}

void parallel_coordinates_workbench::signal_dataset_update(const util::memory_view<std::string_view>& dataset_ids, structures::dataset_dependency::update_flags flags, const structures::gpu_sync_info& sync_info){
    // checking attribute intersection of all drawlists if any is decendent of the datasets
    bool any_drawlist_affected{};
    for(const auto& dl: drawlist_infos.read()){
        if(dataset_ids.contains(dl.drawlist_read().parent_dataset)){
            any_drawlist_affected = true;
            break;
        }
    }
    if(!any_drawlist_affected)
        return;

    _update_attribute_order_infos();
    
    //_request_registered_histograms_update = true; should be obsolete
}

void parallel_coordinates_workbench::remove_datasets(const util::memory_view<std::string_view>& dataset_ids, const structures::gpu_sync_info& sync_info){
    // going through the drawlists and check if they have to be removed
    std::vector<std::string_view> drawlists_to_remove;
    for(size_t i: util::size_range(drawlist_infos.read())){
        if(dataset_ids.contains(drawlist_infos.read()[i].drawlist_read().parent_dataset))
            drawlists_to_remove.push_back(drawlist_infos.read()[i].drawlist_id);
    }
    remove_drawlists(drawlists_to_remove);
}

void parallel_coordinates_workbench::add_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info){
    for(auto drawlist_id: drawlist_ids){
        // checking if the drawlist is already added
        bool exists = false;
        for(const auto& dl: drawlist_infos.read()){
            if(dl.drawlist_id == drawlist_id){
                exists = true;
                break;
            }
        }
        if(exists)
            continue;

        auto& dl = globals::drawlists.write().at(drawlist_id).write();
        drawlist_infos.write().push_back(drawlist_info{drawlist_id, true, dl.appearance_drawlist, dl.median_typ});
        _dl_added = true;
    }
    _update_attribute_order_infos();
}

void parallel_coordinates_workbench::signal_drawlist_update(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info) {
    for(auto drawlist_id: drawlist_ids){
        if(globals::drawlists.read().at(drawlist_id).changed){
            drawlist_infos();
            break;
        }
    }
}

void parallel_coordinates_workbench::remove_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info){
    for(size_t i: util::rev_size_range(drawlist_infos.read())){
        if(drawlist_ids.contains(drawlist_infos.read()[i].drawlist_id)){
            std::string_view dl = drawlist_infos.read()[i].drawlist_id;
            // locking registry
            auto registry_lock = globals::drawlists.read().at(dl).read().histogram_registry.const_access();
            _registered_histograms.erase(dl);
            _registered_axis_histograms.erase(dl);
            drawlist_infos().erase(drawlist_infos().begin() + i);
        }
    }
    _update_attribute_order_infos();
}

// settings conversions
parallel_coordinates_workbench::settings_t::settings_t(const crude_json::value& json){
    auto& t = *this;
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT(json, t, enable_axis_lines);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT(json, t, min_max_labes);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT(json, t, axis_tick_label);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT(json, t, enable_category_labels);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT(json, t, axis_tick_fmt);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT_CAST(json, t, axis_tick_count, double);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT_CAST(json, t, render_batch_size, double);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT(json, t, brush_box_width);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT(json, t, brush_box_border_width);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT(json, t, brush_box_border_hover_width);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT_VEC4(json, t, brush_box_global_color);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT_VEC4(json, t, brush_box_local_color);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT_VEC4(json, t, brush_box_selected_color);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT(json, t, brush_arrow_button_move);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT(json, t, brush_drag_threshold);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT_CAST(json, t, live_brush_threshold, double);

    JSON_ASSIGN_JSON_FIELD_TO_STRUCT_CAST(json, t, hist_type, double);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT(json, t, histogram_blur_width);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT(json, t, histogram_width);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT_VEC4(json, t, plot_background);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT_CAST(json, t, histogram_rendering_threshold, double);
    JSON_ASSIGN_JSON_FIELD_TO_STRUCT(json, t, render_splines);
}
parallel_coordinates_workbench::settings_t::operator crude_json::value() const{
    auto& t = *this;
    crude_json::value json(crude_json::type_t::object);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON(json, t, enable_axis_lines);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON(json, t, min_max_labes);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON(json, t, axis_tick_label);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON(json, t, enable_category_labels);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON(json, t, axis_tick_fmt);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON_CAST(json, t, axis_tick_count, double);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON_CAST(json, t, render_batch_size, double);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON(json, t, brush_box_width);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON(json, t, brush_box_border_width);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON(json, t, brush_box_border_hover_width);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON_VEC4(json, t, brush_box_global_color);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON_VEC4(json, t, brush_box_local_color);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON_VEC4(json, t, brush_box_selected_color);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON(json, t, brush_arrow_button_move);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON(json, t, brush_drag_threshold);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON_CAST(json, t, live_brush_threshold, double);

    JSON_ASSIGN_STRUCT_FIELD_TO_JSON_CAST(json, t, hist_type, double);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON(json, t, histogram_blur_width);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON(json, t, histogram_width);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON_VEC4(json, t, plot_background);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON_CAST(json, t, histogram_rendering_threshold, double);
    JSON_ASSIGN_STRUCT_FIELD_TO_JSON(json, t, render_splines);
    return json;
}
bool parallel_coordinates_workbench::settings_t::operator==(const settings_t& o) const{
    COMP_EQ_OTHER(o, enable_axis_lines);
    COMP_EQ_OTHER(o, min_max_labes);
    COMP_EQ_OTHER(o, axis_tick_label);
    COMP_EQ_OTHER(o, enable_category_labels);
    COMP_EQ_OTHER(o, axis_tick_fmt);
    COMP_EQ_OTHER(o, axis_tick_count);
    COMP_EQ_OTHER(o, render_batch_size);
    COMP_EQ_OTHER(o, brush_box_width);
    COMP_EQ_OTHER(o, brush_box_border_width);
    COMP_EQ_OTHER(o, brush_box_border_hover_width);
    COMP_EQ_OTHER_VEC4(o, brush_box_global_color);
    COMP_EQ_OTHER_VEC4(o, brush_box_local_color);
    COMP_EQ_OTHER_VEC4(o, brush_box_selected_color);
    COMP_EQ_OTHER(o, brush_arrow_button_move);
    COMP_EQ_OTHER(o, brush_drag_threshold);
    COMP_EQ_OTHER(o, live_brush_threshold);

    COMP_EQ_OTHER(o, hist_type);
    COMP_EQ_OTHER(o, histogram_blur_width);
    COMP_EQ_OTHER(o, histogram_width);
    COMP_EQ_OTHER_VEC4(o, plot_background);
    COMP_EQ_OTHER(o, histogram_rendering_threshold);
    COMP_EQ_OTHER(o, render_splines);
    return true;
}

}