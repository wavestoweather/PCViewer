#pragma once
#include <dataset_convert_data.hpp>
#include <memory_view.hpp>
#include <attributes.hpp>
#include <datasets.hpp>
#include <map>
#include <open_filepaths.hpp>
#include <imgui.h>
#include <filesystem>
#include <file_util.hpp>

namespace util{
namespace dataset{
namespace open_internals{
struct load_information{
    uint32_t block_index_to_load{};
    uint32_t block_count_to_load{};
};
template<class T>
struct load_result{
    structures::data<T>                             data{};
    std::vector<structures::attribute>              attributes{};
    std::vector<std::optional<T>>                   fill_values{};
};

load_result<float> open_netcdf_float(std::string_view filename, memory_view<structures::query_attribute> query_attributes = {}, const load_information* partial_info = {});
load_result<half> open_netcdf_half(std::string_view filename, memory_view<structures::query_attribute> query_attributes = {}, const load_information* partial_info = {});
template<class T>
load_result<T> open_netcdf(std::string_view filename, memory_view<structures::query_attribute> query_attributes = {}, const load_information* partial_info = {}){
    if constexpr(std::is_same_v<T, float>)
        return open_netcdf_float(filename, query_attributes, partial_info);
    if constexpr(std::is_same_v<T, half>)
        return open_netcdf_half(filename, query_attributes, partial_info);
    return {};
}

load_result<float> open_csv_float(std::string_view filename, memory_view<structures::query_attribute> query_attributes = {}, const load_information* partial_info = {});
load_result<half> open_csv_half(std::string_view filename, memory_view<structures::query_attribute> query_attributes = {}, const load_information* partial_info = {});
template<class T>
load_result<T> open_csv(std::string_view filename, memory_view<structures::query_attribute> query_attributes = {}, const load_information* partial_info = {}){
    if constexpr(std::is_same_v<T, float>)
        return open_csv_float(filename, query_attributes, partial_info);
    if constexpr(std::is_same_v<T, half>)
        return open_csv_half(filename, query_attributes, partial_info);
    return {};
}
load_result<half> open_combined(std::string_view folder, memory_view<structures::query_attribute> query_attributes = {}, const load_information* partial_info = {});
load_result<uint32_t> open_combined_compressed(std::string_view folder, memory_view<structures::query_attribute> query_attributes = {}, const load_information* partial_info = {});

std::vector<structures::query_attribute> get_netcdf_qeuery_attributes(std::string_view file);
std::vector<structures::query_attribute> get_csv_query_attributes(std::string_view file);
std::vector<structures::query_attribute> get_combined_query_attributes(std::string_view folder);
}
enum class data_type_preference{
    half_precision,
    float_precision,
    COUNT
};

globals::dataset_t open_dataset(std::string_view filename, memory_view<structures::query_attribute> query_attributes = {}, data_type_preference data_type_pref = data_type_preference::float_precision);


void convert_dataset(const structures::dataset_convert_data& convert_data);

inline void fill_query_attributes(){
    // the attribute query is done for the first dataset to open
    for(std::string_view path: globals::paths_to_open){
        if(std::filesystem::exists(path)){
            try{
                auto [file, file_extension] = util::get_file_extension(path);
                if(file_extension.empty())
                    globals::attribute_query = open_internals::get_combined_query_attributes(path);
                else if(file_extension == ".nc")
                    globals::attribute_query = open_internals::get_netcdf_qeuery_attributes(path);
                else if(file_extension == ".csv")
                    globals::attribute_query = open_internals::get_csv_query_attributes(path);
                break;
            }
            catch(std::runtime_error e){
                std::cout << "[error] " << e.what() << std::endl;
            }
        }
    }
}

inline void check_datasets_to_open(){
    const std::string_view open_dataset_popup_name{"Open dataset(s)"};

    if(globals::paths_to_open.size()){
        if(globals::attribute_query.empty()){
            fill_query_attributes();
        }

        ImGui::OpenPopup(open_dataset_popup_name.data());
    }

    if(ImGui::BeginPopupModal(open_dataset_popup_name.data())){
        if(ImGui::CollapsingHeader("Variables/Dimensions settings")){
            // TODO complete
        }
        static std::vector<uint8_t> activations;
        if(activations.size() != globals::paths_to_open.size())
            activations = std::vector<uint8_t>(globals::paths_to_open.size(), true);
        for(int i: util::size_range(activations))
            ImGui::Checkbox(globals::paths_to_open[i].c_str(), reinterpret_cast<bool*>(activations.data()));

        if(ImGui::Button("Open") || ImGui::IsKeyPressed(ImGuiKey_Enter)){
            for(std::string_view path: globals::paths_to_open){
                try{
                    auto dataset = open_dataset(path, globals::attribute_query);
                    bool ok = false;
                }
                catch(std::runtime_error e){
                    std::cout << "[error] " << e.what() << std::endl;
                }
            }
            ImGui::CloseCurrentPopup();
            globals::paths_to_open.clear();
            globals::attribute_query.clear();
        }
        ImGui::SameLine();
        if(ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)){
            ImGui::CloseCurrentPopup();
            globals::paths_to_open.clear();
            globals::attribute_query.clear();
        }
        ImGui::EndPopup();
    }
}

};
};