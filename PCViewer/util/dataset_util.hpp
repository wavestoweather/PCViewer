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
#include <scale_offset.hpp>

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

template<class T>
load_result<T> open_netcdf(std::string_view filename, memory_view<structures::query_attribute> query_attributes = {}, const load_information* partial_info = {});

template<class T>
load_result<T> open_csv(std::string_view filename, memory_view<structures::query_attribute> query_attributes = {}, const load_information* partial_info = {});

load_result<half> open_combined(std::string_view folder, memory_view<structures::query_attribute> query_attributes = {}, const load_information* partial_info = {});
load_result<uint32_t> open_combined_compressed(std::string_view folder, memory_view<structures::query_attribute> query_attributes = {}, const load_information* partial_info = {});

std::vector<structures::query_attribute> get_netcdf_qeuery_attributes(std::string_view file);
std::vector<structures::query_attribute> get_csv_query_attributes(std::string_view file);
std::vector<structures::query_attribute> get_combined_query_attributes(std::string_view folder);
}
enum class data_type_preference{
    none,
    half_precision,
    float_precision,
    COUNT
};

globals::dataset_t open_dataset(std::string_view filename, memory_view<structures::query_attribute> query_attributes = {}, data_type_preference data_type_pref = data_type_preference::none);


void convert_templatelist(const structures::templatelist_convert_data& convert_data);

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

void check_datasets_to_open();

};
};