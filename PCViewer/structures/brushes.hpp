#pragma once
#include <inttypes.h>
#include <vector>
#include <imgui.h>
#include <change_tracker.hpp>

namespace structures{
    struct axis_range{
        uint32_t axis;
        float min;
        float max;
    };
    using range_brush = std::vector<axis_range>;
    using range_brushes = std::vector<range_brush>;

    struct polygon{
        int attr1, attr2;
        std::vector<ImVec2> borderPoints;
    };
    using lasso_brush = std::vector<polygon>;
    using lasso_brushes = std::vector<lasso_brush>;

    struct brushes{
        range_brushes ranges;
        lasso_brushes lassos;
    };

    using tracked_brushes = change_tracker<brushes>;
}

namespace globals{
extern structures::tracked_brushes global_brushes;
}
