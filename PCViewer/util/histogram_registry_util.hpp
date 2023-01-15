#pragma once
#include <string>
#include <memory_view.hpp>
#include <string_view_util.hpp>
#include <min_max.hpp>
#include <charconv>
#include <fast_float.h>
#include <numeric>

namespace util{
namespace histogram_registry{

inline std::string get_id_string(util::memory_view<const uint32_t> attribute_indices, util::memory_view<const int> bin_sizes, util::memory_view<const structures::min_max<float>> attribute_bounds, bool is_min_hist, bool is_max_hist){
    std::vector<uint32_t> sorted(attribute_indices.size()); std::iota(sorted.begin(), sorted.end(), 0);
    std::sort(sorted.begin(), sorted.end(), [&](uint32_t l, uint32_t r){return attribute_indices[l] < attribute_indices[r];});
    std::string id;
    for(size_t i: util::size_range(attribute_indices)){
        id += std::to_string(attribute_indices[sorted[i]]);
        if(i < attribute_indices.size() - 1)
            id += '_';
    }
    id += '|';
    for(size_t i: util::size_range(bin_sizes)){
        id += std::to_string(bin_sizes[sorted[i]]);
        if(i < bin_sizes.size() - 1)
            id += '_';
    }
    id += '|';
    for(size_t i: util::size_range(attribute_bounds)){
        id += std::to_string(attribute_bounds[sorted[i]].min);
        id += ',';
        id += std::to_string(attribute_bounds[sorted[i]].max);
        if(i < attribute_bounds.size() - 1)
            id += '_';
    }
    if(is_min_hist)
        id += "|min";
    if(is_max_hist)
        id += "|max";
    return id;
}

inline std::tuple<std::vector<uint32_t>, std::vector<int>, std::vector<structures::min_max<float>>, bool, bool> get_indices_bins(std::string_view in){
    std::string_view indices;
    std::string_view bins;
    std::string_view min_max;
    getline(in, indices, '|');
    getline(in, bins, '|');
    getline(in, min_max, '|');
    std::vector<uint32_t> parsed_indices;
    std::vector<int> parsed_bins;
    std::vector<structures::min_max<float>> parsed_min_max;
    for(std::string_view cur; getline(indices, cur, '_');){
        parsed_indices.push_back({});
        std::from_chars(cur.data(), cur.data() + cur.size(), parsed_indices.back());
    }
    for(std::string_view cur; getline(bins, cur, '_');){
        parsed_bins.push_back({});
        std::from_chars(cur.data(), cur.data() + cur.size(), parsed_bins.back());
    }
    for(std::string_view cur; getline(min_max, cur, '_');){
        structures::min_max<float> mm;
        std::string_view c; getline(cur, c, ',');
        fast_float::from_chars(c.data(), c.data() + c.size(), mm.min);
        getline(cur, c, ',');
        fast_float::from_chars(c.data(), c.data() + c.size(), mm.max);
        parsed_min_max.emplace_back(mm);
    }
    bool min_hist{};
    bool max_hist{};
    for(std::string_view cur; getline(in, cur, '|');){
        if(cur == "min")
            min_hist = true;
        if(cur == "max")
            max_hist = true;
    }
    return {parsed_indices, parsed_bins, parsed_min_max, min_hist, max_hist};
}

inline bool id_contains_attributes(std::string_view id, util::memory_view<const uint32_t> attribute_indices){
    std::string_view indices;
    getline(id, indices, '|');
    int index_pos{};
    for(std::string_view cur; getline(indices, cur, '_');){
        uint32_t cur_num; std::from_chars(cur.data(), cur.data() + cur.size(), cur_num);
        if(cur_num != attribute_indices[index_pos++])
            return false;
    }
    return index_pos == attribute_indices.size();
}

void check_histogram_update();
}
}