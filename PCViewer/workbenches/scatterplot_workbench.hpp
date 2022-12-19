#pragma once

#include <workbench_base.hpp>

namespace workbenches{
class scatterplot_workbench: public structures::workbench{

public:
    scatterplot_workbench(std::string_view id);

    void notify_drawlist_dataset_update() override;
    void show() override;
};
}