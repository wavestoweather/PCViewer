#pragma once
#include <workbench_base.hpp>
#include <memory_view.hpp>
#include <imgui.h>
#include <parallel_coordinates_renderer.hpp>

namespace workbenches{

class parallel_coordinates_workbench: public structures::workbench, public structures::drawlist_dependency{
    using appearance_tracker = structures::change_tracker<structures::drawlist::appearance>;
    using drawlist_info = pipelines::parallel_coordinates_renderer::drawlist_info;

    // both are unique_ptrs to avoid issues with the memory_views when data elements are deleted in the vector
    std::vector<std::unique_ptr<appearance_tracker>>         _storage_appearance;
    std::vector<std::unique_ptr<structures::median_type>>    _storage_median_type;
public:
    std::vector<drawlist_info>                  drawlist_infos;     // the order here is the render order of the drawlists
    structures::alpha_mapping_type              alpha_mapping_typ;

    void show() override;

};

}