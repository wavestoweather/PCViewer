#pragma once
#include <vector>
#include <violin_structures.hpp>

namespace util{
namespace violins{
using histograms_t = robin_hood::unordered_map<structures::violins::drawlist_attribute, structures::violins::histogram>;
using drawlist_attribute = structures::violins::drawlist_attribute;
using attribute_histograms_t = std::tuple<float, std::vector<float>, histograms_t>;
using violin_position_order_t = std::tuple<std::vector<structures::violins::violin_appearance_t>, std::vector<structures::attribute_order_info>>;

inline attribute_histograms_t update_histograms(const std::vector<drawlist_attribute>& active_drawlist_attributes, float std_dev, int histogram_bin_count, int active_attributes_count, bool ignore_zero_bins, util::memory_view<const uint8_t> attribute_log){
    float               global_max{};
    std::vector<float>  per_attribute_max(active_attributes_count, .0f);
    histograms_t        histograms;
    double std_dev2 = 2 * std_dev * std_dev;
    int k_size = std_dev < 0 ? .2 * histogram_bin_count + 1: std::ceil(std_dev * 3);

    // integrating to 3 sigma std dev
    for(const auto& da: active_drawlist_attributes){
        const auto histogram_id = util::histogram_registry::get_id_string(da.attribute, histogram_bin_count, false, false);
        const auto histogram_access = da.drawlist_read().histogram_registry.const_access();
        const auto& original_histogram = histogram_access->cpu_histograms.at(histogram_id);
        auto& histogram = histograms[da];
        histogram.area = histogram.max_val = 0;
        for(int bin: util::size_range(original_histogram)){
            float divisor{}, divider{};

            for(int k: util::i_range(-k_size, k_size + 1)){
                if (bin + k >= ((ignore_zero_bins)? 1:0) && bin + k < original_histogram.size()) {
                    float factor = std::exp(-(k * k) / std_dev2);
                    divisor += original_histogram[bin + k] * factor;
                    divider += factor;
                }
            }

            histogram.smoothed_values[bin] = divisor / divider;
            if (attribute_log[da.attribute]) histogram.smoothed_values[bin] = log(histogram.smoothed_values[bin] + 1);
            histogram.area += histogram.smoothed_values[bin];
            histogram.max_val = std::max(histogram.max_val, histogram.smoothed_values[bin]);
        }
        per_attribute_max[da.attribute] = std::max(per_attribute_max[da.attribute], histogram.max_val);
        global_max = std::max(global_max, histogram.max_val);
    }
    return {global_max, per_attribute_max, histograms};
}

inline void imgui_violin_infill(const ImVec2& plot_min, const ImVec2& plot_max, util::memory_view<const float> histogram_values, float hist_normalization_fac, const structures::violins::violin_appearance_t& violin_app){
    const int pixel_footprint{std::max(static_cast<int>((plot_max.y - plot_min.y) / histogram_values.size()), 2)};   // pixel footprint to know how many values i have to blend from the histogram values to get the correct width for the current pixel
    // norm and base already encompass violin position and direction. The base x position is always base_x, while the top x position can be calculated by base_x + hist_val * norm;
    const float base_x = violin_app.base_pos == structures::violins::violin_base_pos_t::left ? plot_min.x : violin_app.base_pos == structures::violins::violin_base_pos_t::middle ? .5f * (plot_min.x + plot_max.x) : plot_max.x;
    const float norm = hist_normalization_fac * (violin_app.span_full ? 1.f : 2.f) * (violin_app.dir == structures::violins::violin_dir_t::right ? 1.f: -1.f);
    // for each pixel add rectangle
    const int max_iter = static_cast<int>(plot_max.y - plot_min.y);
    for(int i: util::i_range(max_iter)){
        const float center_hist = (i + .5f) / max_iter * histogram_values.size();
        const int start_bin = std::max(int(std::ceil(center_hist)) - pixel_footprint / 2, 0);
        const int end_bin = std::min(start_bin + pixel_footprint, int(histogram_values.size()) - 1);
        const int diff = end_bin - start_bin; assert(histogram_values.size() && diff > 0);
        float hist_val{};
        for(int h: util::i_range(start_bin, end_bin))
            hist_val += histogram_values[h] / diff;
        // transforming the histogram value to x offeset for the area
        hist_val *= norm;
        const ImVec2 min_box{base_x           , plot_min.y + i};
        const ImVec2 max_box{base_x + hist_val, plot_min.y + i + 1};
        ImGui::GetWindowDrawList()->AddRectFilled(min_box, max_box, ImGui::ColorConvertFloat4ToU32(violin_app.color));
    }
}

// returns a value between 0 and 1 if border is hovered, nan otherwise
inline float imgui_violin_border(const ImVec2& plot_min, const ImVec2& plot_max, util::memory_view<const float> histogram_values, float hist_normalization_fac, const structures::violins::violin_appearance_t& violin_app, bool is_hovered = false, float hover_dist = 5){
    return std::numeric_limits<float>::quiet_NaN();
}

inline violin_position_order_t get_violin_pos_order(){
    return {};
}
}
}