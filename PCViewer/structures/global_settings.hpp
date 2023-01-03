#pragma once
#include <json_util.hpp>

namespace structures{
struct globals_settings_t{
    bool    drawlist_creation_assign_color{true};
    double  drawlist_creation_alpha_factor{.001}; // dl_alpha = 1/ (factor * tl.size)

    bool operator==(const globals_settings_t& o) const{
        COMP_EQ_OTHER(o, drawlist_creation_assign_color);
        COMP_EQ_OTHER(o, drawlist_creation_alpha_factor);
        return true;
    }
    globals_settings_t() = default;
    globals_settings_t(const crude_json::value& json){
        JSON_ASSIGN_JSON_FIELD_TO_THIS(json, drawlist_creation_assign_color);
        JSON_ASSIGN_JSON_FIELD_TO_THIS(json, drawlist_creation_alpha_factor);
    }
    operator crude_json::value() const{
        crude_json::value json(crude_json::type_t::object);
        JSON_ASSIGN_THIS_FIELD_TO_JSON(json, drawlist_creation_assign_color);
        JSON_ASSIGN_THIS_FIELD_TO_JSON(json, drawlist_creation_alpha_factor);
        return json;
    }
};
}

namespace globals{
extern structures::globals_settings_t settings;
}