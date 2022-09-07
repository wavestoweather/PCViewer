#pragma once
#include <workbench_basse.hpp>
#include <array>

namespace workbenches{

class parallel_coordinates_workbench: public workbench, public drawlist_dependency{
    const std::string_view vertex_shader{""};
    const std::string_view geometry_shader{""};
    const std::string_view fragment_shader{""};

public:
    enum class render_type{
        polyline,
        spline, 
        COUNT
    };
    static std::array<std::string_view, static_cast<size_t>(render_type::COUNT){
        "polyline",
        "spline"
    };

    void show() override;

}

}