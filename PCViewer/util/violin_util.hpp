#pragma once
#include <vector>
#include <violin_structures.hpp>
#include <util.hpp>
#include <color_brewer_util.hpp>

namespace util{
namespace violins{
using histograms_t = robin_hood::unordered_map<structures::violins::drawlist_attribute, structures::violins::histogram>;
using drawlist_attribute = structures::violins::drawlist_attribute;
using attribute_histograms_t = std::tuple<float, std::vector<float>, histograms_t>;
using violin_position_order_t = std::tuple<std::vector<structures::violins::violin_appearance_t>, std::vector<structures::attribute_order_info>>;

inline attribute_histograms_t update_histograms(const std::vector<drawlist_attribute>& active_drawlist_attributes, util::memory_view<const structures::min_max<float>> attribute_min_max, float std_dev, int histogram_bin_count, int active_attributes_count, bool ignore_zero_bins, util::memory_view<const uint8_t> attribute_log){
    float               global_max{};
    std::vector<float>  per_attribute_max(active_attributes_count, .0f);
    histograms_t        histograms;
    double std_dev2 = 2 * std_dev * std_dev;
    int k_size = std_dev < 0 ? .2 * histogram_bin_count + 1: std::ceil(std_dev * 3);

    // integrating to 3 sigma std dev
    for(const auto& da: active_drawlist_attributes){
        const auto histogram_id = util::histogram_registry::get_id_string(da.attribute, histogram_bin_count, attribute_min_max[da.attribute], false, false);
        const auto histogram_access = da.drawlist_read().histogram_registry.const_access();
        const auto& original_histogram = histogram_access->cpu_histograms.at(histogram_id);
        auto& histogram = histograms[da];
        histogram.smoothed_values.resize(original_histogram.size());
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
    // norm and base already encompass violin position and direction. The base x position is always base_x, while the top x position can be calculated by base_x + hist_val * norm;
    const float base_x = (violin_app.base_pos == structures::violins::violin_base_pos_t::left ? plot_min.x : violin_app.base_pos == structures::violins::violin_base_pos_t::middle ? .5f * (plot_min.x + plot_max.x) : plot_max.x) + .5f;
    const float norm = hist_normalization_fac * (violin_app.span_full ? 1.f : 2.f) * (violin_app.dir == structures::violins::violin_dir_t::right ? 1.f: -1.f);
    const int max_hist = histogram_values.size() - 1;
    // for each pixel add rectangle
    const int max_iter = static_cast<int>(plot_max.y - plot_min.y);
    for(int i: util::i_range(max_iter)){
        const float center_hist = float(i) / max_iter * (histogram_values.size() - 1);
        const int base_hist = int(center_hist);
        const float a = center_hist - base_hist;
        float hist_val = (1.f - a) * histogram_values[max_hist - base_hist];
        if(base_hist < histogram_values.size() - 1)
            hist_val += a * histogram_values[max_hist - base_hist - 1];
        // transforming the histogram value to x offeset for the area
        hist_val = hist_val / norm * (plot_max.x - plot_min.x) + .5f;
        ImVec2 min_box{base_x           , plot_min.y + i};
        const ImVec2 max_box{base_x + hist_val, plot_min.y + i + 1};
        if(violin_app.dir == structures::violins::violin_dir_t::left_right) min_box.x -= hist_val; // add other side 
        ImGui::GetWindowDrawList()->AddRectFilled(min_box, max_box, ImGui::ColorConvertFloat4ToU32(violin_app.color));
    }
}

// returns a value between 0 and 1 if border is hovered, nan otherwise
inline float imgui_violin_border(const ImVec2& plot_min, const ImVec2& plot_max, util::memory_view<const float> histogram_values, float hist_normalization_fac, const structures::violins::violin_appearance_t& violin_app, float line_thickness, float hover_dist = 5){
    // draws lines on the left and the right of the voilin
    const float base_x = violin_app.base_pos == structures::violins::violin_base_pos_t::left ? plot_min.x : violin_app.base_pos == structures::violins::violin_base_pos_t::middle ? .5f * (plot_min.x + plot_max.x) : plot_max.x;
    const float norm = hist_normalization_fac * (violin_app.span_full ? 1.f : 2.f) * (violin_app.dir == structures::violins::violin_dir_t::right ? 1.f: -1.f);
    const bool double_sided = violin_app.dir == structures::violins::violin_dir_t::left_right;
    const ImU32 col = ImGui::ColorConvertFloat4ToU32(violin_app.color);
    const float plot_width = plot_max.x - plot_min.x;
    const int max_hist = histogram_values.size() - 1;

    // connection line
    float b_w = histogram_values[max_hist] / norm * plot_width;
    ImVec2 b_a{base_x, plot_min.y};
    ImVec2 b_b{base_x + b_w, plot_min.y};
    if(double_sided) b_a.x -= b_w; // add other side 
    ImGui::GetWindowDrawList()->AddLine(b_a, b_b, col, line_thickness);
    // inbetween lines
    for(int i: util::i_range(histogram_values.size() - 1)){
        const float start_r = i / float(histogram_values.size() - 1);
        const float end_r = (i + 1) / float(histogram_values.size() - 1);
        const float hist_val_start = histogram_values[max_hist - i] / norm * plot_width;
        const float hist_val_end = histogram_values[max_hist - i - 1] / norm * plot_width;
        ImVec2 a{base_x + hist_val_start, util::unnormalize_val_for_range(start_r, plot_min.y, plot_max.y)};
        ImVec2 b{base_x + hist_val_end, util::unnormalize_val_for_range(end_r, plot_min.y, plot_max.y)};
        ImGui::GetWindowDrawList()->AddLine(a, b, col, line_thickness);
        if(double_sided){
            a.x = base_x - hist_val_start;
            b.x = base_x - hist_val_end;
            ImGui::GetWindowDrawList()->AddLine(a, b, col, line_thickness);
        }
    }
    if(!double_sided)
        ImGui::GetWindowDrawList()->AddLine({base_x, plot_min.y}, {base_x, plot_max.y}, col, line_thickness);
    // end connecting line
    b_w = histogram_values[0] / norm * plot_width;
    b_a = {base_x, plot_max.y};
    b_b = {base_x + b_w, plot_max.y};
    if(double_sided) b_a.x -= b_w; // add other side 
    ImGui::GetWindowDrawList()->AddLine(b_a, b_b, col, line_thickness);
    return std::numeric_limits<float>::quiet_NaN();
}

inline violin_position_order_t get_violin_pos_order(const std::vector<std::vector<std::reference_wrapper<structures::violins::histogram>>> per_attribute_histograms, const std::vector<uint32_t>& active_attributes, const std::vector<structures::violins::violin_appearance_t>& attribute_position_scaling, std::string_view palette_name){
    const auto color_palette = (brew_palette_infos() | util::try_pick_if<palette_info_t>([&palette_name](const palette_info_t& t){return t.name == palette_name;})).value();
    const size_t attribute_count = per_attribute_histograms.size();

    std::vector<structures::violins::violin_appearance_t> violin_positions(attribute_count, structures::violins::violin_appearance_t{});

    // the overlap is calculated between all combinations of attributes
    std::vector<std::vector<float>> hist_overlaps(attribute_count, std::vector<float>(attribute_count, 0));
    for(uint32_t i: util::i_range(attribute_count)){
        if(!(active_attributes | util::contains(i))) continue;
        for(int j: util::i_range(i + 1, attribute_count)){
            if(!(active_attributes | util::contains(j))) continue;
            // sum up overlap over all histograms
            for(int h: util::size_range(per_attribute_histograms[i])){
                for(int b: util::size_range(per_attribute_histograms[i][h].get().smoothed_values))
                    hist_overlaps[i][j] += std::min(per_attribute_histograms[i][h].get().smoothed_values[b], per_attribute_histograms[j][h].get().smoothed_values[b]);
            }
        }
    }

    // Now the histograms are moved to different sizes of the middle violin
    // Goal is to put the most similar (highes hist_overlap) to different sides of the violin plots
    // placed_attributes contains the attributes in order they should be rendered, from back to front
    std::vector<uint32_t> placed_attributes;
    while(true){
        float cur_max = -1;
        int i_max = -1;
        int j_max = -1;

        for(int i: util::size_range(hist_overlaps)){
            if(placed_attributes | util::contains(uint32_t(i)))
                continue;
            for(int j: util::i_range(i + 1, hist_overlaps.size())){
                if(placed_attributes | util::contains(uint32_t(j)))
                    continue;
                if(hist_overlaps[i][j] > cur_max){
                    cur_max = hist_overlaps[i][j];
                    i_max = i;
                    j_max = j;
                }
            }
        }

        if(i_max == -1 || j_max == -1) break;

        // for following positioning we have to consider the already positioned sides
        if(placed_attributes.empty()){
            violin_positions[i_max].dir = structures::violins::violin_dir_t::left;
            violin_positions[j_max].dir = structures::violins::violin_dir_t::right;
        }
        else{
            // checking the overlap on both sides if i is left and if i is right
            // to do this iterate over all placed attributes and adding the histogram overlap to either i_left or i_right
            float overlap_i_left{}, overlap_i_right{};

            for(uint32_t attribute: placed_attributes){
                uint32_t i_min = std::min<uint32_t>(attribute, i_max);
                uint32_t i_maxx = std::max<uint32_t>(attribute, i_max);
                uint32_t j_min = std::min<uint32_t>(attribute, j_max);
                uint32_t j_maxx = std::max<uint32_t>(attribute, j_max);
                if(violin_positions[attribute].dir == structures::violins::violin_dir_t::left){
                    overlap_i_left += hist_overlaps[i_min][i_maxx];
                    overlap_i_right += hist_overlaps[j_min][j_maxx];
                }
                else{
                    overlap_i_left += hist_overlaps[j_min][j_maxx];
                    overlap_i_right += hist_overlaps[i_min][i_maxx];
                }
            }
            if(overlap_i_left < overlap_i_right){
                violin_positions[i_max].dir = structures::violins::violin_dir_t::left;
                violin_positions[j_max].dir = structures::violins::violin_dir_t::right;
            }
            else{
                violin_positions[i_max].dir = structures::violins::violin_dir_t::right;
                violin_positions[j_max].dir = structures::violins::violin_dir_t::left;
            }
        }

        placed_attributes.emplace_back(i_max);
        placed_attributes.emplace_back(j_max);
    }

    // assign the colors to the attributes and creating the order array
    std::vector<structures::attribute_order_info> attribute_order(attribute_count);
    auto rev_attributes = util::rev_iter(placed_attributes);
    for(auto [a, i]: util::enumerate(rev_attributes)){
        attribute_order[i].attribut_index = a;
        attribute_order[i].active = active_attributes | util::contains(a);
        violin_positions[a].color = util::color_brewer::brew_imcol(palette_name, color_palette.max_colors)[i % color_palette.max_colors].Value;
    }

    return {std::move(violin_positions), std::move(attribute_order)};
}
}
}