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

    ImGui::Image(plot_data.read().image_descriptor, {content_size.x, content_size.y * .7f});

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
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text(dl.drawlist_id.data());
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

        render_plot();
    }
}

void parallel_coordinates_workbench::render_plot()
{
    pipelines::parallel_coordinates_renderer::render_info render_info{
        *this,  // workbench (is not changed, the renderer only reads information)
        {},     // wait_semaphores;
        {}      // signal_semaphores;
    };
    pipelines::parallel_coordinates_renderer::instance().render(render_info);
}

void parallel_coordinates_workbench::add_drawlist(std::string_view drawlist_id){
    auto& dl = globals::drawlists.write().at(drawlist_id).write();
    auto& ds = dl.dataset_read();
    if(drawlist_infos.read().empty()){
        // setting up the internal states
        attributes.write() = ds.attributes;
    }
    // check attribute consistency
    for(int var: util::size_range(attributes.read()))
        if(attributes.read()[var].id != ds.attributes[var].id)
            throw std::runtime_error{"parallel_coordinates_workbench::addDrawlist() Inconsistent attributes for the new drawlist"};
    
    drawlist_infos.write().push_back(drawlist_info{drawlist_id, true, dl.appearance_drawlist, dl.median_typ});
}

}