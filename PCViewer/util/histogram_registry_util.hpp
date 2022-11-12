#pragma once
#include <string>
#include <memory_view.hpp>
#include <string_view_util.hpp>
#include <charconv>
#include <numeric>

namespace util{
namespace histogram_registry{

inline std::string get_id_string(util::memory_view<const uint32_t> attribute_indices, util::memory_view<const int> bin_sizes, bool is_min_hist, bool is_max_hist){
    std::vector<uint32_t> sorted(attribute_indices.size()); std::iota(sorted.begin(), sorted.end(), 0);
    std::sort(sorted.begin(), sorted.end(), [&](uint32_t l, uint32_t r){return attribute_indices[l] < attribute_indices[r];});
    std::string id;
    for(int i: util::size_range(attribute_indices)){
        id += std::to_string(attribute_indices[sorted[i]]);
        if(i < attribute_indices.size() - 1)
            id += '_';
    }
    id += '|';
    for(int i: util::size_range(bin_sizes)){
        id += std::to_string(bin_sizes[sorted[i]]);
        if(i < bin_sizes.size() - 1)
            id += '_';
    }
    if(is_min_hist)
        id += "|min";
    if(is_max_hist)
        id += "|max";
    return id;
}

inline std::tuple<std::vector<uint32_t>, std::vector<int>, bool, bool> get_indices_bins(std::string_view in){
    std::string_view indices;
    std::string_view bins;
    getline(in, indices, '|');
    getline(in, bins, '|');
    std::vector<uint32_t> parsed_indices;
    std::vector<int> parsed_bins;
    for(std::string_view cur; getline(indices, cur, '_');){
        parsed_indices.push_back({});
        std::from_chars(cur.data(), cur.data() + cur.size(), parsed_indices.back());
    }
    for(std::string_view cur; getline(bins, cur, '_');){
        parsed_bins.push_back({});
        std::from_chars(cur.data(), cur.data() + cur.size(), parsed_bins.back());
    }
    bool min_hist{};
    bool max_hist{};
    for(std::string_view cur; getline(in, cur, '|');){
        if(cur == "min")
            min_hist = true;
        if(cur == "max")
            max_hist = true;
    }
    return {parsed_indices, parsed_bins, min_hist, max_hist};
}

}
}