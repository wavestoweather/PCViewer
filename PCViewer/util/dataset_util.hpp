#pragma once
#include <dataset_convert_data.hpp>
#include <memory_view.hpp>
#include <attributes.hpp>
#include <datasets.hpp>
#include <map>

namespace util{
namespace dataset{
namespace open_internals{
struct load_information{
    uint32_t block_index_to_load{};
    uint32_t block_count_to_load{};
};
template<class T>
struct load_result{
    structures::data<T>                             data;
    std::vector<std::optional<T>>                   fill_values;
    std::map<uint32_t, std::vector<std::string>>    categories;
};

template<class T>
load_result<T> open_netcdf(std::string_view filename, memory_view<structures::query_attribute> query_attributes = {}, const load_information* partial_info = {});
template<class T>
load_result<T> open_csv(std::string_view filenmae, memory_view<structures::query_attribute> query_attributes = {}, const load_information* partial_info = {});
load_result<half> open_combined(std::string_view folder, memory_view<structures::query_attribute> query_attributes = {}, const load_information* partial_info = {});
load_result<uint32_t> open_combined_compressed(std::string_view folder, memory_view<structures::query_attribute> query_attributes = {}, const load_information* partial_info = {});
}

globals::dataset_t open_dataset(std::string_view filename, memory_view<structures::query_attribute> query_attributes = {});


void convert_dataset(const structures::dataset_convert_data& convert_data);

};
};