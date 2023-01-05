#pragma once
#include <c_file.hpp>
#include <filesystem>
#include <fstream>
#include <enum_names.hpp>
#include "../imgui_nodes/crude_json.h"

#define JSON_ASSIGN_STRUCT_FIELD_TO_JSON(json, struct, field) json[#field] = struct.field
#define JSON_ASSIGN_THIS_FIELD_TO_JSON(json, field) json[#field] = field
#define JSON_ASSIGN_STRUCT_FIELD_TO_JSON_CAST(json, struct, field, type_json) json[#field] = static_cast<type_json>(struct.field)
#define JSON_ASSIGN_STRUCT_FIELD_TO_JSON_VEC4(json, struct, field) json[#field][0] = static_cast<double>(struct.field.x); json[#field][1] = static_cast<double>(struct.field.y); json[#field][2] = static_cast<double>(struct.field.z); json[#field][3] = static_cast<double>(struct.field.w);
#define JSON_ASSIGN_STRUCT_FIELD_TO_JSON_ENUM_NAME(json, struct, field, enum_names) json[#field] = std::string(enum_names[struct.field]);
#define JSON_ASSIGN_JSON_FIELD_TO_STRUCT(json, struct, field) struct.field = json[#field].get<decltype(field)>()
#define JSON_ASSIGN_JSON_FIELD_TO_THIS(json, field) field = json[#field].get<decltype(field)>()
#define JSON_ASSIGN_JSON_FIELD_TO_STRUCT_CAST(json, struct, field, type_json) struct.field = static_cast<decltype(struct.field)>(json[#field].get<type_json>())
#define JSON_ASSIGN_JSON_FIELD_TO_STRUCT_VEC4(json, struct, field) struct.field.x = json[#field][0].get<double>(); struct.field.y = json[#field][1].get<double>(); struct.field.z = json[#field][2].get<double>(); struct.field.w = json[#field][3].get<double>();
#define JSON_ASSIGN_JSON_FIELD_TO_STRUCT_ENUM_NAME(json, struct, field, enum_names) struct.field = enum_names[json[#field].get<std::string>()];

#define COMP_EQ_OTHER(o, field) if(field != o.field) return false;
#define COMP_EQ_OTHER_VEC4(o, field) if(field.x != o.field.x || field.y != o.field.y || field.z != o.field.z || field.w != o.field.w) return false;

namespace util{
namespace json{
inline crude_json::value open_json(std::string_view filename){
    structures::c_file file(filename, "rb");
    if(!file)
        throw std::runtime_error{"util::json::open_json() File " + std::string(filename) + " could not be found."};
    std::string json_string; json_string.resize(std::filesystem::file_size(std::string(filename)));
    file.read<char>({json_string.data(), json_string.size()});
    return crude_json::value::parse(json_string);
}

// for indent < 0 dumps json as single string,
// for indent == 0 dumps json line by line without indent
// fur indent >0 dumps json line by line and indents inner contents by indent spaces
inline void save_json(std::string_view filename, const crude_json::value& json, int indent = -1){
    std::ofstream file(std::string(filename), std::ios::binary);
    file << json.dump(indent);
}

inline bool is_dimension_selection(const crude_json::value& val){
    return val.is_object() && val.contains("type") && val["type"].is_string() && val["type"].get<std::string>() == "dim_sel" && val.contains("selected_dim") && val["selected_dim"].is_string();
}

inline bool is_enumeration(const crude_json::value& val){
    return val.is_object() && val.contains("type") && val["type"].is_string() && val["type"].get<std::string>() == "enumeration" && val.contains("choices") && val["choices"].is_array() && val.contains("chosen") && val["chosen"].is_number();
}

template<typename T>
inline void add_enum(crude_json::value& val, const std::string& enum_name, const structures::enum_names<T> names, T default_val){
    val[enum_name]["type"] = "enumeration"; 
    for(auto e: structures::enum_iteration<T>{}) 
        val[enum_name]["choices"][int(e)] = std::string(names[e]); 
    val[enum_name]["chosen"] = double(default_val);
}

template<typename T>
inline T get_enum_val(const crude_json::value& val){
    if(!is_enumeration(val))
        throw std::runtime_error{"util::json::get_enum_val() json given is not an enum"};
    return static_cast<T>(val["chosen"].get<double>());
}
}
}