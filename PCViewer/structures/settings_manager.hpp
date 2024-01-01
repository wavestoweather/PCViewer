#pragma once
#include <vector>
#include <set>
#include <robin_hood.h>
#include <string>
#include <string_view>
#include <memory_view.hpp>
#include <json_util.hpp>

namespace util{
namespace settings{
inline crude_json::value setting_base(const std::string& id, const std::string& type){
    crude_json::value json;
    json["id"] = std::string(id);
    json["type"] = type;
    return json;
}
}
}

namespace structures{
class settings_manager {
public:
    using setting = crude_json::value;

    settings_manager();
    ~settings_manager();

    bool add_setting(const setting& s, bool autostore = true);
    bool delete_setting(std::string_view id);
    // if setting was not found returns an empty json (check with json.is_null())
    setting& get_setting(std::string_view id);
    const std::set<std::string>& get_settings_type(const std::string& type) const;

private:
    const std::string_view settings_file{"settings.json"};
    const std::string_view file_path{""};
    void store_settings(std::string_view filename);
    void load_settings(std::string_view filename);

    robin_hood::unordered_map<std::string, setting> settings;
    robin_hood::unordered_map<std::string, std::set<std::string>> settings_type;
};

}

namespace globals{
extern structures::settings_manager settings_manager;
}
