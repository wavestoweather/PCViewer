#include "parallel_coordinates_workbench.hpp"
#include <vk_initializers.hpp>
#include <vma_initializers.hpp>
#include <vk_util.hpp>
#include <parallel_coordinates_renderer.hpp>
#include <imgui_util.hpp>
#include <imgui_stdlib.h>
#include <util.hpp>
#include <brush_util.hpp>
#include <algorithm>

namespace workbenches{

parallel_coordinates_workbench::parallel_coordinates_workbench(const std::string_view id):
    workbench(id)
{
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
    ImGui::Checkbox("Enable axis lines", &setting.read().enable_axis_lines);
    ImGui::Checkbox("Enable min max labels", &setting.read().min_max_labes);
    ImGui::Checkbox("Enable axis tick lables", &setting.read().axis_tick_label);
    ImGui::InputText("Tick format", &setting.read().axis_tick_fmt);
    ImGui::InputInt("Axis tick count", &setting.read().axis_tick_count);
}

void parallel_coordinates_workbench::_swap_attributes(const attribute_order_info& from, const attribute_order_info& to){
    auto from_it = std::find(attributes_order_info().begin(), attributes_order_info().end(), from);
    auto to_it = std::find(attributes_order_info().begin(), attributes_order_info().end(), to);
    if(ImGui::IsKeyDown(ImGuiKey_ModCtrl)){
        int from_ind = std::distance(attributes_order_info().begin(), from_it);
        int to_ind = std::distance(attributes_order_info().begin(), to_it);
        int lower = std::min(from_ind, to_ind);
        int higher = std::max(from_ind, to_ind);
        int direction = from_ind < to_ind ? -1 : 1; // direction is the shuffle direction of all elements except from
        int start_ind = from_ind - direction;
        for(;start_ind >= lower && start_ind <= higher; start_ind -= direction)
            attributes_order_info()[start_ind + direction] = attributes_order_info()[start_ind];
        attributes_order_info()[to_ind] = from;
    }
    else
        std::swap(*from_it, *to_it); 

    _update_registered_histograms();
}

void parallel_coordinates_workbench::_update_registered_histograms(){
    // updating registered histograms (iterating through indices pairs and checking for registered histogram)
    auto active_indices = get_active_ordered_indices();
    for(const auto& dl: drawlist_infos.read()){
        if(!_registered_histograms.contains(dl.drawlist_id))
            continue;

        std::vector<bool> registrator_needed(_registered_histograms[dl.drawlist_id].size(), false);
        for(int i: util::i_range(active_indices.size() - 1)){
            util::memory_view<const uint32_t> indices(active_indices.data() + i, 2);
            std::vector<int> bucket_sizes(2, plot_data.read().height);
            auto registrator_id = util::histogram_registry::get_id_string(indices, bucket_sizes, false, false);
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
                auto& drawlist = globals::drawlists()[dl.drawlist_id]();
                _registered_histograms[dl.drawlist_id].emplace_back(*drawlist.histogram_registry.access(), indices, bucket_sizes, false, false, false);
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
            logger << logging::info_prefix << " parallel_coordinates_workbench (" << active_indices.size() << " attributes, " << registry_lock->registry.size() <<" registrators, " << registry_lock->name_to_registry_key.size() << " name to registry entries), registered histograms: ";
            for(const auto& [key, val]: registry_lock->registry)
                logger << val.hist_id << " ";
            logger << logging::endl;
        }
    }
}

void parallel_coordinates_workbench::show(){
    if(!active)
        return;
    const static std::string_view brush_menu_id{"brush menu"};
    const static std::string_view pc_menu_id{"parallel coordinates menu"};

    bool brush_menu_open{false};
    bool pc_menu_open{false};

    // checking for drawlist change (if drawlist has changed, render_plot() will be called later in the frame by update_drawlists)
    // checking for local change
    bool any_drawlist_change{false};
    bool local_change{false};
    bool request_render{false};
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
    request_render |= attributes.changed;
    request_render |= attributes_order_info.changed;
    request_render |= setting.changed;

    // checking for changed image
    if(plot_data.changed){
        _update_plot_image();
        plot_data.changed = false;

        request_render |= true;

        // updating requested histograms
        _update_registered_histograms();
    }

    request_render &= !any_drawlist_change;

    if(request_render)
        render_plot();

    ImGui::Begin(id.c_str(), &active);

    // -------------------------------------------------------------------------------
    // Plot region including labels and min max values
    // -------------------------------------------------------------------------------
    auto content_size = ImGui::GetWindowContentRegionMax();

    uint32_t labels_count{};
    for(const auto& att_ref: attributes_order_info.read())
        if(att_ref.active)
            ++labels_count;

    constexpr float tick_width = 10;
    const size_t padding_side = 10;
    const ImVec2 button_size{70, 20};
    const size_t gap = (content_size.x - 2 * padding_side) / (labels_count - 1);
    const size_t button_gap = (content_size.x - 2 * padding_side - button_size.x) / (labels_count - 1);

    size_t cur_offset{};
    ImGui::Dummy({1, 1});
    // attribute labels
    for(const auto& att_ref: attributes_order_info.read()){
        if(!att_ref.active)
            continue;
        
        ImGui::SameLine(cur_offset);
        
        std::string name = attributes.read()[att_ref.attribut_index].display_name;
        int text_size = ImGui::CalcTextSize(name.c_str()).x;
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
        ImGui::Button(name.c_str(), button_size);
        if (name != attributes.read()[att_ref.attribut_index].id && ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("%s", attributes.read()[att_ref.attribut_index].id.c_str());
            ImGui::Text("Drag and drop to switch axes, hold ctrl to shuffle");
            ImGui::EndTooltip();
        }
        if (name == attributes.read()[att_ref.attribut_index].id && ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("Drag and drop to switch axes, hold ctrl to shuffle");
            ImGui::EndTooltip();
        }
        if (ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {// TODO implement            //editAttributeName = i;
            //strcpy(newAttributeName, pcAttributes[i].originalName.c_str());
        }

        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
            const attribute_order_info p = att_ref;        //holding the attribute reference which should be switched
            ImGui::SetDragDropPayload("ATTRIBUTE", &p, sizeof(p));
            ImGui::Text("Swap %s", name.c_str());
            ImGui::EndDragDropSource();
        }
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ATTRIBUTE")) {
                const attribute_order_info other = *(const attribute_order_info*)payload->Data;

                _swap_attributes(other, att_ref);
            }
        }
        cur_offset += button_gap;
    }
    // attribute max values
    ImGui::Dummy({1, 1});
    cur_offset = 0;
    for(const auto& att_ref: attributes_order_info.read()){
        if(!att_ref.active)
            continue;
        
        std::string name = setting.read().min_max_labes ? "max##" + attributes.read()[att_ref.attribut_index].id : "##max" + attributes.read()[att_ref.attribut_index].id;
        ImGui::SetNextItemWidth(button_size.x);
        ImGui::SameLine(cur_offset);
        float diff = attributes.read()[att_ref.attribut_index].bounds.read().max - attributes.read()[att_ref.attribut_index].bounds.read().min;
        if(ImGui::DragFloat(name.c_str(), &attributes.ref_no_track()[att_ref.attribut_index].bounds.ref_no_track().max, diff * .001f, 0, 0, "%6.4g"))
            attributes()[att_ref.attribut_index].bounds().max;
        if(ImGui::IsItemClicked(ImGuiMouseButton_Right)){
            // todo minmaxpopup
        }
        cur_offset += button_gap;
    }

    auto pic_pos = ImGui::GetCursorScreenPos();
    auto pic_size = ImVec2{content_size.x - 10, content_size.y * .7f};
    ImGui::Image(plot_data.read().image_descriptor, pic_size);

    if(setting.read().enable_axis_lines){
        float y1 = pic_pos.y;
        float y2 = y1 + pic_size.y - 1;
        auto line_col = IM_COL32((1 - setting.read().plot_background.x) * 255, (1 - setting.read().plot_background.y) * 255, (1 - setting.read().plot_background.z) * 255, 255);
        for(int i: util::i_range(labels_count)){
            float x = pic_pos.x + (pic_size.x - 1) * i / (labels_count - 1);
            ImGui::GetWindowDrawList()->AddLine({x, y1}, {x, y2}, line_col);
        }
        int att_pos = 0;
        for(const auto& att_ref: attributes_order_info.read()){
            if(!att_ref.active)
                continue;

            float x = pic_pos.x + (pic_size.x - 1) * att_pos / (labels_count - 1);
            const auto& attribute = attributes.read()[att_ref.attribut_index];
            float min_tick = attribute.bounds.read().min + (attribute.bounds.read().max - attribute.bounds.read().min) / setting.read().axis_tick_count;
            float max_tick = attribute.bounds.read().max;
            float diff = max_tick - min_tick;
            int exp;
            double m = frexp(diff, &exp);
            m *= 2;
            exp -= 1;
            double t = pow(2, exp);
            int e = int(log10(t));
            double dec = pow(10, e);
            for(int i: util::i_range(setting.read().axis_tick_count)){
                double tick_val = min_tick + diff * i / (setting.read().axis_tick_count - 1);
                // rounding down to multiple of 10
                tick_val /= dec;
                tick_val = double(int(tick_val));
                tick_val *= dec;
                float y = (tick_val - attribute.bounds.read().max) / (attribute.bounds.read().min - attribute.bounds.read().max) * pic_size.y + pic_pos.y;
                ImGui::GetWindowDrawList()->AddLine({x - tick_width / 2, y}, {x + tick_width / 2, y}, line_col);
            }
            ++att_pos;
        }  
    }

    // attribute min values
    ImGui::Dummy({1, 1});
    cur_offset = 0;
    for(const auto& att_ref: attributes_order_info.read()){
        if(!att_ref.active)
            continue;
        
        std::string name = setting.read().min_max_labes ? "min##" + attributes.read()[att_ref.attribut_index].id : "##min" + attributes.read()[att_ref.attribut_index].id;
        ImGui::SetNextItemWidth(button_size.x);
        ImGui::SameLine(cur_offset);
        float diff = attributes.read()[att_ref.attribut_index].bounds.read().max - attributes.read()[att_ref.attribut_index].bounds.read().min;
        if(ImGui::DragFloat(name.c_str(), &attributes.ref_no_track()[att_ref.attribut_index].bounds.ref_no_track().min, diff * .001f, 0, 0, "%6.4g"))
            attributes()[att_ref.attribut_index].bounds().min;
        if(ImGui::IsItemClicked(ImGuiMouseButton_Right)){
            // todo minmaxpopup
        }
        cur_offset += button_gap;
    }

    // brush windows
    if(globals::brush_edit_data.brush_type != structures::brush_edit_data::brush_type::none){
        std::map<uint32_t, uint32_t> place_of_ind;
        uint32_t place = 0;
        for(const auto& attr_ref: attributes_order_info.read())
            if(attr_ref.active)
                place_of_ind[attr_ref.attribut_index] = place++;
            
        bool any_hover = false;
        robin_hood::unordered_set<structures::brush_id> brush_delete;
        ImVec2 mouse_pos = {ImGui::GetIO().MousePos.x - ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, setting.read().brush_drag_threshold).x, ImGui::GetIO().MousePos.y - ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0).y};

        const structures::range_brush& selected_brush = util::brushes::get_selected_range_brush_const();
        float brush_gap = pic_size.x / (labels_count - 1);

        for(const auto& brush: selected_brush){
            if(place_of_ind.count(brush.axis) == 0)
                continue;   // attribute not active
            
            float x = brush_gap * place_of_ind[brush.axis] + pic_pos.x - setting.read().brush_box_width / 2;

            float y = util::normalize_val_for_range(brush.max, attributes.read()[brush.axis].bounds.read().max, attributes.read()[brush.axis].bounds.read().min) * pic_size.y + pic_pos.y;
            float height = (brush.max - brush.min) / (attributes.read()[brush.axis].bounds.read().max - attributes.read()[brush.axis].bounds.read().min) * pic_size.y;
            
            structures::brush_edit_data::brush_region hovered_region{structures::brush_edit_data::brush_region::COUNT};
            if(util::point_in_box(mouse_pos, {x, y}, {x + setting.read().brush_box_width, y + height}))
                hovered_region = structures::brush_edit_data::brush_region::body;
            else if(util::point_in_box(mouse_pos, {x, y - setting.read().brush_box_border_hover_width}, {x + setting.read().brush_box_width, y}))
                hovered_region = structures::brush_edit_data::brush_region::top;
            else if(util::point_in_box(mouse_pos, {x, y + height}, {x + setting.read().brush_box_width, y + height + setting.read().brush_box_border_hover_width}))
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

            ImGui::GetWindowDrawList()->AddRect({x, y}, {x + setting.read().brush_box_width, y + height}, border_color, 1, ImDrawCornerFlags_All, setting.read().brush_box_border_width);

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
            if(globals::brush_edit_data.selected_ranges.contains(brush.id) && ((ImGui::IsMouseDown(ImGuiMouseButton_Left) && ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, setting.read().brush_drag_threshold).y) || ImGui::IsKeyPressed(ImGuiKey_DownArrow) || ImGui::IsKeyPressed(ImGuiKey_UpArrow))){
                float delta;
                if(ImGui::IsMouseDown(ImGuiMouseButton_Left))
                    delta = -ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, setting.read().brush_drag_threshold).y / pic_size.y;
                else if(ImGui::IsKeyPressed(ImGuiKey_UpArrow))
                    delta = setting.read().brush_arrow_button_move;
                else if(ImGui::IsKeyPressed(ImGuiKey_DownArrow))
                    delta = -setting.read().brush_arrow_button_move;
                
                structures::range_brush& range_brush = util::brushes::get_selected_range_brush();
                structures::axis_range& range       = *std::find(range_brush.begin(), range_brush.end(), brush);
                switch(globals::brush_edit_data.hovered_region_on_click){
                case structures::brush_edit_data::brush_region::top:
                    range.max += delta * (attributes.read()[brush.axis].bounds.read().max - attributes.read()[brush.axis].bounds.read().min);
                    break;
                case structures::brush_edit_data::brush_region::bottom:
                    range.min += delta * (attributes.read()[brush.axis].bounds.read().max - attributes.read()[brush.axis].bounds.read().min);
                    break;
                case structures::brush_edit_data::brush_region::body:
                    range.max += delta * (attributes.read()[brush.axis].bounds.read().max - attributes.read()[brush.axis].bounds.read().min);
                    range.min += delta * (attributes.read()[brush.axis].bounds.read().max - attributes.read()[brush.axis].bounds.read().min);
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
                if (place_of_ind[brush.axis] == 0) x_anchor = 0;
                if (place_of_ind[brush.axis] == labels_count - 1) x_anchor = 1;

                ImGui::SetNextWindowPos({ x + setting.read().brush_box_width / 2,y }, 0, { x_anchor,1 });
                ImGui::SetNextWindowBgAlpha(ImGui::GetStyle().Colors[ImGuiCol_PopupBg].w * 0.60f);
                ImGuiWindowFlags flags = ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking;
                ImGui::Begin(("Tooltip brush max##" + std::to_string(brush.id)).c_str(), NULL, flags);
                ImGui::Text("%f", brush.max);
                ImGui::End();

                ImGui::SetNextWindowPos({ x + setting.read().brush_box_width / 2, y + height }, 0, { x_anchor,0 });
                ImGui::SetNextWindowBgAlpha(ImGui::GetStyle().Colors[ImGuiCol_PopupBg].w * 0.60f);
                ImGui::Begin(("Tooltip brush min##" + std::to_string(brush.id)).c_str(), NULL, flags);
                ImGui::Text("%f", brush.min);
                ImGui::End();
            }

            any_hover |= brush_hovered;
        }
        ImGui::ResetMouseDragDelta();   // has to be reset to avoid draggin open the box too far

        // brush creation
        bool new_brush{false};
        for(const auto& attr_ref: attributes_order_info.read()){
            if(!attr_ref.active)
                continue;
            float x = brush_gap * place_of_ind[attr_ref.attribut_index] + pic_pos.x - setting.read().brush_box_width / 2;
            bool axis_hover = util::point_in_box(mouse_pos, {x, pic_pos.y}, {x + setting.read().brush_box_width, pic_pos.y + pic_size.y}) && ImGui::IsWindowHovered();
            if(!any_hover && axis_hover && globals::brush_edit_data.selected_ranges.empty()){
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

                if(ImGui::IsMouseClicked(ImGuiMouseButton_Left)){
                    float val = util::normalize_val_for_range(mouse_pos.y, pic_pos.y, pic_pos.y + pic_size.y) * (attributes.read()[attr_ref.attribut_index].bounds.read().min - attributes.read()[attr_ref.attribut_index].bounds.read().max) + attributes.read()[attr_ref.attribut_index].bounds.read().max;

                    structures::axis_range new_range{attr_ref.attribut_index, globals::cur_brush_range_id++, val, val};
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
        if(!any_hover && globals::brush_edit_data.selected_ranges.size() &&  (ImGui::IsMouseReleased(ImGuiMouseButton_Left) || (!new_brush && ImGui::IsMouseClicked(ImGuiMouseButton_Left))) && !ImGui::GetIO().KeyCtrl)
            globals::brush_edit_data.selected_ranges.clear();
    }

    pc_menu_open = !brush_menu_open && ImGui::IsMouseClicked(ImGuiMouseButton_Right) && ImGui::IsWindowHovered() && util::point_in_box(ImGui::GetMousePos(), pic_pos, {pic_pos.x + pic_size.x, pic_pos.y + pic_size.y});

    if(brush_menu_open)
        ImGui::OpenPopup(brush_menu_id.data());
    if(ImGui::BeginPopup(brush_menu_id.data())){
        if(ImGui::MenuItem("Delete", {}, false, bool(globals::brush_edit_data.selected_ranges.size()))){
            util::brushes::delete_brushes(globals::brush_edit_data.selected_ranges);
        }
        if(ImGui::MenuItem("Fit axis to boudns", {}, false, bool(globals::brush_edit_data.selected_ranges.size()))){
            const auto& selected_ranges = util::brushes::get_selected_range_brush_const();
            // getting the extremum values for each axis if existent
            std::vector<std::optional<structures::min_max<float>>> axis_values(attributes.read().size());
            for(const auto& range: selected_ranges){
                if(axis_values[range.axis])
                    axis_values[range.axis] = std::min_max(*axis_values[range.axis], {range.min, range.max});
                else
                    axis_values[range.axis] = structures::min_max<float>{range.min, range.max};
            }
            for(int i: util::size_range(attributes.read())){
                if(axis_values[i])
                    attributes()[i].bounds = *axis_values[i];
            }
        }
        ImGui::DragInt("Live brush treshold", &setting.read().live_brush_threshold);
        ImGui::EndPopup();
    }
    if(pc_menu_open)
        ImGui::OpenPopup(pc_menu_id.data());
    if(ImGui::BeginPopup(pc_menu_id.data())){
        if(ImGui::BeginCombo("Histogram", histogram_type_names[setting.read().hist_type].data())){
            for(auto type: structures::enum_iteration<histogram_type>()){
                if(ImGui::MenuItem(histogram_type_names[type].data()))
                    setting().hist_type = type;
            }
            ImGui::EndCombo();
        }
        if(ImGui::ColorEdit4("Plot background", &setting.ref_no_track().pc_background.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar))
            setting();
        if(ImGui::MenuItem("Render splines", {}, &setting.ref_no_track().render_splines))
            setting();
        if(ImGui::BeginMenu("Plot Size")){
            if(ImGui::InputInt2("width/height", reinterpret_cast<int*>(&plot_data.ref_no_track().width), ImGuiInputTextFlags_EnterReturnsTrue))
                plot_data();
            
            static std::map<VkSampleCountFlagBits, std::string_view> flag_names{{VK_SAMPLE_COUNT_1_BIT, "1Spp"}, {VK_SAMPLE_COUNT_1_BIT, "1Spp"}, {VK_SAMPLE_COUNT_2_BIT, "2Spp"}, {VK_SAMPLE_COUNT_4_BIT, "4Spp"}, {VK_SAMPLE_COUNT_8_BIT, "8Spp"}, {VK_SAMPLE_COUNT_16_BIT, "16Spp"}};
            if(ImGui::BeginCombo("Sample per pixel", flag_names[plot_data.read().image_samples].data())){
                for(const auto& [bit, name]: flag_names)
                    if(ImGui::MenuItem(name.data()))
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
    

    // -------------------------------------------------------------------------------
    // settings region
    // -------------------------------------------------------------------------------
    std::string_view delete_drawlist{};

    ImGui::BeginHorizontal("Settings");

        // general settings
        ImGui::BeginVertical("GeneralSettings");
        ImGui::BeginChild("testwarpp",{400, 0});
        // activating the attributes
        struct attr_ref_t{
            std::string_view name; bool* active;
            bool operator<(const attr_ref_t& o) const {return name < o.name;}
        };
        std::set<attr_ref_t> attribute_set;
        for(auto& att_ref: attributes_order_info.ref_no_track())
            attribute_set.insert(attr_ref_t{attributes.read()[att_ref.attribut_index].id, &att_ref.active});
        for(auto& a: attribute_set){
            if(ImGui::Checkbox((std::string(a.name) + "##activation").c_str(), a.active)){
                attributes_order_info();
                _update_registered_histograms();
            }
        }
        if(ImGui::TreeNode("General settings")){
            _draw_setting_list();
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
            
            for(auto& dl: drawlist_infos.ref_no_track()){
                std::string dl_string(dl.drawlist_id);
                const auto& drawlist = globals::drawlists.read().at(dl.drawlist_id);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                bool selected = util::memory_view(globals::selected_drawlists).contains(dl.drawlist_id);
                if(ImGui::Selectable((drawlist.read().name + "##pc_wb").c_str(), selected)){
                    globals::selected_drawlists.clear();
                    if(!selected)
                        globals::selected_drawlists.push_back(drawlist.read().name);
                }
                ImGui::TableNextColumn();
                if(ImGui::ArrowButton(("##u" + dl_string).c_str(), ImGuiDir_Up))
                    drawlist_infos.write();
                ImGui::TableNextColumn();
                if(ImGui::ArrowButton(("##d" + dl_string).c_str(), ImGuiDir_Down))
                    drawlist_infos.write();
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

            ImGui::EndTable();
        }
        ImGui::EndVertical();

    ImGui::EndHorizontal();

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
    for(const auto& dl: drawlist_infos.read()){
        if(_registered_histograms.contains(dl.drawlist_id) && _registered_histograms[dl.drawlist_id].size()){
            const auto access = globals::drawlists.read().at(dl.drawlist_id).read().histogram_registry.const_access();
            if(!access->dataset_update_done)
                return;     // a registered histogram is being currently updated, no rendering possible
        }
    }

    if(logger.logging_level >= logging::level::l_5)
        logger << logging::info_prefix << " parallel_coordinates_workbench::render_plot()" << logging::endl;
    pipelines::parallel_coordinates_renderer::render_info render_info{
        *this,  // workbench (is not changed, the renderer only reads information)
        {},     // wait_semaphores;
        {}      // signal_semaphores;
    };
    pipelines::parallel_coordinates_renderer::instance().render(render_info);
    for(auto& dl_info: drawlist_infos()){
        if(!dl_info.linked_with_drawlist)
            dl_info.clear_change();
    }
    drawlist_infos.changed = false;
    for(auto& attribute: attributes.ref_no_track()){
        if(attribute.bounds.changed)
            attribute.bounds.changed = false;
    }
    attributes.changed = false;
    attributes_order_info.changed = false;
    // signaling all registrators
    if(_registered_histograms.size()){
        for(auto& [dl, registrators]: _registered_histograms){
            // locking registry
            auto registry_lock = globals::drawlists.read().at(dl).read().histogram_registry.const_access();
            for(auto& registrator: registrators)
                registrator.signal_registry_used();
        }
    }
}

std::vector<uint32_t> parallel_coordinates_workbench::get_active_ordered_indices() const{
    std::vector<uint32_t> indices;
    for(const auto& i: attributes_order_info.read()){
        if(i.active)
            indices.push_back(i.attribut_index);
    }
    return indices;
}

void parallel_coordinates_workbench::remove_datasets(const util::memory_view<std::string_view>& dataset_ids, const structures::gpu_sync_info& sync_info){
    // going through the drawlists and check if they have to be removed
    std::vector<std::string_view> drawlists_to_remove;
    for(int i: util::size_range(drawlist_infos.read())){
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
        auto& ds = dl.dataset_read();
        if(drawlist_infos.read().empty()){
            // setting up the internal states
            attributes = ds.attributes;
            for(auto& attribute: attributes()){
                if(attribute.bounds.read().min == attribute.bounds.read().max){
                    float diff = attribute.bounds.read().max * .01f;
                    attribute.bounds().min -= diff;
                    attribute.bounds().max += diff;
                }
            }
            attributes_order_info().resize(attributes.read().size());
            for(int i: util::size_range(attributes_order_info.read()))
                attributes_order_info.write()[i].attribut_index = i;
        }
        // check attribute consistency
        for(int var: util::size_range(attributes.read()))
            if(attributes.read()[var].id != ds.attributes[var].id)
                throw std::runtime_error{"parallel_coordinates_workbench::addDrawlist() Inconsistent attributes for the new drawlist"};

        drawlist_infos.write().push_back(drawlist_info{drawlist_id, true, dl.appearance_drawlist, dl.median_typ});

        // checking histogram (large vis) rendering or standard rendering
        if(dl.const_templatelist().data_size > setting.read().histogram_rendering_threshold){
            std::vector<uint32_t> indices = get_active_ordered_indices();
            std::vector<int> bin_sizes(2, static_cast<int>(plot_data.read().height));
            for(int i: util::i_range(indices.size() - 1)){
                _registered_histograms[drawlist_id].emplace_back(*dl.histogram_registry.access(), util::memory_view<const uint32_t>(indices.data() + i, 2), util::memory_view<const int>(bin_sizes), false, false, false);
            }
        }
    }
}

void parallel_coordinates_workbench::signal_drawlist_update(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info) {
    bool request_render{false};
    for(auto drawlist_id: drawlist_ids){
        if(globals::drawlists.read().at(drawlist_id).changed){
            request_render = true;
            break;
        }
    }
    if(request_render)
        render_plot();
}

void parallel_coordinates_workbench::remove_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info){
    for(int i: util::rev_size_range(drawlist_infos.read())){
        if(drawlist_ids.contains(drawlist_infos.read()[i].drawlist_id)){
            std::string_view dl = drawlist_infos.read()[i].drawlist_id;
            // locking registry
            auto registry_lock = globals::drawlists.read().at(dl).read().histogram_registry.const_access();
            _registered_histograms.erase(dl);
            drawlist_infos().erase(drawlist_infos().begin() + i);
        }
    }
}

}