#pragma once
#include <workbench_base.hpp>

namespace workbenches{
class images_workbench: public structures::workbench{
    struct settings_t{
        float image_height{100.f};
    }               _settings{};

    std::string_view _popup_image_id{};

public:
    images_workbench(std::string_view id): workbench(id) {}

    void show() override;
};
}