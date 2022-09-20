#include "parallel_coordinates_workbench.hpp"
#include <vk_initializers.hpp>
#include <vma_initializers.hpp>
#include <vk_util.hpp>
#include <parallel_coordinates_renderer.hpp>

namespace workbenches{

parallel_coordinates_workbench::parallel_coordinates_workbench(const std::string_view id):
    workbench(id)
{
    auto image_info = util::vk::initializers::imageCreateInfo(plot_data.read().image_format, {plot_data.read().width, plot_data.read().height, 1}, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    auto alloc_info = util::vma::initializers::allocationCreateInfo();
    std::tie(plot_data.ref_no_track().image, plot_data.ref_no_track().image_view) = util::vk::create_image_with_view(image_info, alloc_info);
}

void parallel_coordinates_workbench::show(){
    if(!active)
        return;
    ImGui::Begin(id.c_str(), &active);

    // -------------------------------------------------------------------------------
    // Plot region including labels and min max values
    // -------------------------------------------------------------------------------

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

            ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("dl.drawlist_id.data()");
                ImGui::TableNextColumn();
                ImGui::ArrowButton("##testu", ImGuiDir_Up);
                ImGui::TableNextColumn();
                ImGui::ArrowButton("##testd", ImGuiDir_Down);
                ImGui::TableNextColumn();
                ImGui::Button("X##testx");
                ImGui::TableNextColumn();
                bool t;
                ImGui::Checkbox("##testact", &t);
            
            for(auto& dl: drawlist_infos){
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text(dl.drawlist_id.data());
                ImGui::NextColumn();
                ImGui::ArrowButton("##testu", ImGuiDir_Up);
                ImGui::NextColumn();
                ImGui::ArrowButton("##testd", ImGuiDir_Down);
                ImGui::NextColumn();
                ImGui::Button("X##testx");
                ImGui::NextColumn();
                ImGui::Checkbox("##testact", &dl.appearance->ref_no_track().show);
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
        *this,  // workbench
        {},     // wait_semaphores;
        {}      // signal_semaphores;
    };
    pipelines::parallel_coordinates_renderer::instance().render(render_info);
}

}