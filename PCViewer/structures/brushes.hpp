#pragma once
#include <inttypes.h>
#include <vector>
#include <imgui.h>
#include <change_tracker.hpp>
#include <buffer_info.hpp>
#include <robin_hood.h>

namespace structures{
typedef uint32_t range_id;
typedef uint32_t brush_id;
struct axis_range{
    uint32_t axis;
    range_id id;
    float min;
    float max;

    bool operator==(const axis_range& o) const {return id == o.id;}
};
using range_brush = std::vector<axis_range>;
using range_brushes = std::vector<range_brush>;
using range_brushes_map = robin_hood::unordered_map<brush_id, range_brush>;

struct polygon{
    int                 attr1, attr2;
    std::vector<ImVec2> borderPoints;
};
using lasso_brush = std::vector<polygon>;
using lasso_brushes = std::vector<lasso_brush>;
using lasso_brushes_map = robin_hood::unordered_map<brush_id, lasso_brush>;

struct brush{
    range_brush ranges{};
    lasso_brush lassos{};
    brush_id    id{};
    mutable std::string name{};   // mutable as it should not be tracked

    bool empty() const {return ranges.empty() && lassos.empty();}
};
using tracked_brush = change_tracker<brush>;
using tracked_brushes = change_tracker<std::vector<tracked_brush>>;

struct global_brushes: public tracked_brushes{
    buffer_info brushes_gpu{};
};

struct brush_edit_data{
    enum class brush_type{
        none,
        global,
        local,
        COUNT
    };
    enum class brush_region{
        top,
        bottom,
        body,
        COUNT
    };
    brush_type          brush_type{};
    brush_id            global_brush_id{}; 
    std::string_view    local_brush_id{};  // drawlist id
    brush_region        hovered_region_on_click{brush_region::COUNT};
    robin_hood::unordered_set<range_id>  selected_ranges{};

    void clear(){
        brush_type = brush_type::none;
        hovered_region_on_click = brush_region::COUNT;
        selected_ranges.clear();
    }
};
}

namespace globals{
extern structures::global_brushes global_brushes;
extern structures::brush_edit_data brush_edit_data;
extern std::atomic<structures::brush_id> cur_global_brush_id;
extern std::atomic<structures::range_id> cur_brush_range_id;
}
