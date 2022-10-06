#include "parallel_coordinates_workbench.hpp"
#include <vk_initializers.hpp>
#include <vma_initializers.hpp>
#include <vk_util.hpp>
#include <parallel_coordinates_renderer.hpp>
#include <imgui_util.hpp>

namespace workbenches{

parallel_coordinates_workbench::parallel_coordinates_workbench(const std::string_view id):
    workbench(id)
{
    _update_plot_image();
}

void parallel_coordinates_workbench::_update_plot_image(){
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

void parallel_coordinates_workbench::show(){
    if(!active)
        return;
    ImGui::Begin(id.c_str(), &active);

    // -------------------------------------------------------------------------------
    // Plot region including labels and min max values
    // -------------------------------------------------------------------------------
    auto content_size = ImGui::GetWindowContentRegionMax();

    uint32_t labels_count{};
    for(const auto& att_ref: attributes_order_info.read())
        if(att_ref.active)
            ++labels_count;

    const size_t padding_side = 10;
    const ImVec2 button_size{70, 20};
    const size_t gap = (content_size.x - 2 * padding_side - button_size.x) / (labels_count - 1);

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
			name = cur_substr;
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
		if (ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {// TODO implement			//editAttributeName = i;
			//strcpy(newAttributeName, pcAttributes[i].originalName.c_str());
		}

        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
			const attribute_order_info* p[] = {&att_ref};		//holding the index in the pcAttriOrd array and the value of it
			ImGui::SetDragDropPayload("ATTRIBUTE", p, sizeof(p));
			ImGui::Text("Swap %s", name.c_str());
			ImGui::EndDragDropSource();
		}
		if (ImGui::BeginDragDropTarget()) {
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ATTRIBUTE")) {
				const attribute_order_info* other = (const attribute_order_info*)payload->Data;

				//reorder_attributes(c, other[0], io.KeyCtrl);
			}
		}
        cur_offset += gap;
    }
    // attribute max values
    ImGui::Dummy({1, 1});
    cur_offset = 0;
    for(const auto& att_ref: attributes_order_info.read()){
        if(!att_ref.active)
            continue;
        
        std::string name{"max##" + attributes.read()[att_ref.attribut_index].id};
        ImGui::SetNextItemWidth(button_size.x);
        ImGui::SameLine(cur_offset);
        float diff = attributes.read()[att_ref.attribut_index].bounds.read().max - attributes.read()[att_ref.attribut_index].bounds.read().min;
        if(ImGui::DragFloat(name.c_str(), &attributes.ref_no_track()[att_ref.attribut_index].bounds.ref_no_track().max, diff * .001f, 0, 0, "%6.4g"))
            attributes()[att_ref.attribut_index].bounds().max;
        if(ImGui::IsItemClicked(ImGuiMouseButton_Right)){
            // todo minmaxpopup
        }
        cur_offset += gap;
    }

    ImGui::Image(plot_data.read().image_descriptor, {content_size.x, content_size.y * .7f});

    // attribute min values
    ImGui::Dummy({1, 1});
    cur_offset = 0;
    for(const auto& att_ref: attributes_order_info.read()){
        if(!att_ref.active)
            continue;
        
        std::string name{"min##" + attributes.read()[att_ref.attribut_index].id};
        ImGui::SetNextItemWidth(button_size.x);
        ImGui::SameLine(cur_offset);
        float diff = attributes.read()[att_ref.attribut_index].bounds.read().max - attributes.read()[att_ref.attribut_index].bounds.read().min;
        if(ImGui::DragFloat(name.c_str(), &attributes.ref_no_track()[att_ref.attribut_index].bounds.ref_no_track().min, diff * .001f, 0, 0, "%6.4g"))
            attributes()[att_ref.attribut_index].bounds().min;
        if(ImGui::IsItemClicked(ImGuiMouseButton_Right)){
            // todo minmaxpopup
        }
        cur_offset += gap;
    }

    // -------------------------------------------------------------------------------
    // settings region
    // -------------------------------------------------------------------------------

    ImGui::BeginHorizontal("Settings");

        // general settings
        ImGui::BeginVertical("GeneralSettings");
        ImGui::BeginChild("testwarpp",{200, 0});
        if(ImGui::TreeNode("General settings")){
            ImGui::Text("Well here we go brudi");
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
                ImGui::Text(drawlist.read().name.c_str());
                ImGui::TableNextColumn();
                if(ImGui::ArrowButton(("##u" + dl_string).c_str(), ImGuiDir_Up))
                    drawlist_infos.write();
                ImGui::TableNextColumn();
                if(ImGui::ArrowButton(("##d" + dl_string).c_str(), ImGuiDir_Down))
                    drawlist_infos.write();
                ImGui::TableNextColumn();
                if(ImGui::Button(("X##x" + dl_string).c_str()))
                    drawlist_infos.write();
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

        if(!drawlist_infos.changed)
            continue;
        local_change |= dl_info.any_change();
    }
    request_render |= local_change;

    // checking for attributes change
    request_render |= attributes.changed;
    request_render |= attributes_order_info.changed;

    // checking for changed image
    if(plot_data.changed){
        std::cout << "plot_data changed, recreating..." << std::endl;
        if(plot_data.read().image)
            util::vk::destroy_image(plot_data().image);
        if(plot_data.read().image_view)
            util::vk::destroy_image_view(plot_data().image_view);

        auto image_info = util::vk::initializers::imageCreateInfo(plot_data.read().image_format, {plot_data.read().width, plot_data.read().height, 1}, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_TYPE_2D, 1, 1, plot_data.read().image_samples);
        auto alloc_info = util::vma::initializers::allocationCreateInfo();
        std::tie(plot_data.ref_no_track().image, plot_data.ref_no_track().image_view) = util::vk::create_image_with_view(image_info, alloc_info);
        plot_data.changed = false;

        request_render |= true;
    }

    request_render &= !any_drawlist_change;

    if(request_render)
        render_plot();
}

void parallel_coordinates_workbench::render_plot()
{
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
}

void parallel_coordinates_workbench::add_drawlists(const util::memory_view<std::string_view>& drawlist_ids){
    for(auto drawlist_id: drawlist_ids){
        auto& dl = globals::drawlists.write().at(drawlist_id).write();
        auto& ds = dl.dataset_read();
        if(drawlist_infos.read().empty()){
            // setting up the internal states
            attributes = ds.attributes;
            attributes_order_info().resize(attributes.read().size());
            for(int i: util::size_range(attributes_order_info.read()))
                attributes_order_info.write()[i].attribut_index = i;
        }
        // check attribute consistency
        for(int var: util::size_range(attributes.read()))
            if(attributes.read()[var].id != ds.attributes[var].id)
                throw std::runtime_error{"parallel_coordinates_workbench::addDrawlist() Inconsistent attributes for the new drawlist"};

        drawlist_infos.write().push_back(drawlist_info{drawlist_id, true, dl.appearance_drawlist, dl.median_typ});
    }
}

void parallel_coordinates_workbench::signal_drawlist_update(const util::memory_view<std::string_view>& drawlist_ids) {
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

}