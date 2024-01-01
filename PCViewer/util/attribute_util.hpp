#pragma once
#include <attributes.hpp>
#include <settings_manager.hpp>
#include <logger.hpp>

namespace util{
namespace attribute{
const std::string_view attribute_rename_setting_name{"attribute_renames"};
const std::string_view attribute_rename_setting_type{"attribute_renames"};
inline void load_attribute_renames(){
    auto rename_settings = globals::settings_manager.get_setting(attribute_rename_setting_name);
    // load all renames (paires of from to renames)
    bool success = true;
    auto rename_settings_array = rename_settings["from_tos"];
    if(rename_settings_array.is_array()){
        for(size_t i: util::size_range(rename_settings_array)){
            if(!rename_settings_array[i].is_object()){
                logger << logging::error_prefix << " util::attribute::load_attribute_renames() array element " << i + 1 << " of rename settings is not an object, reverting to default." << logging::endl;
                success = false;
                break;
            }
            try{
                globals::attribute_renames.emplace_back(std::pair<std::string, std::string>(rename_settings_array[i]["from"].get<std::string>(), rename_settings_array[i]["to"].get<std::string>()));
            }
            catch(std::exception){
                logger << logging::error_prefix << " util::attribute::load_attribute_renames() array element " << i + 1 << " of rename settings is not a valid renaming, reverting to default." << logging::endl;
                success = false;
                break;
            }
        }
    }
    else if(!rename_settings_array.is_null()){
        success = false;
        logger << logging::error_prefix << " util::attribute::load_attribute_renames() Attribute setting " << attribute_rename_setting_name << " is not an array, loading default renamings" << logging::endl;
    }
    else 
        success = false;
    if(!success)
        globals::attribute_renames = {{"longitude", "lon"}, {"latitude", "lat"}, {"level", "lev"}};
}

inline void store_attribute_renames(){
    crude_json::value renames(crude_json::type_t::array);
    for(const auto& [from, to]: globals::attribute_renames){
        if(from.size() && to.size()){
            renames.push_back(crude_json::object{{"from", from}, {"to", to}});
        }
    }
    globals::settings_manager.add_setting(crude_json::object{{"id", std::string(attribute_rename_setting_name)}, {"type", std::string(attribute_rename_setting_type)}, {"from_tos", renames}});
}
}
}