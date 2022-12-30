#include "k_means.hpp"
#include <ranges.hpp>
#include <random>
#include <algorithm>


using center_t = std::vector<std::vector<float>>;

float distance(const k_means::float_column_views& input, size_t i, const std::vector<float>& center, k_means::distance_method_t mean, bool euqal_layout){
    float s{};
    switch(mean){
    case k_means::distance_method_t::norm:          for(int d: util::size_range(input)){float diff = (euqal_layout ? input[d].cols[0][i]: input[d](i, 0)) - center[d]; s += diff * diff;} return std::sqrt(s);
    case k_means::distance_method_t::squared_norm:  for(int d: util::size_range(input)){float diff = (euqal_layout ? input[d].cols[0][i]: input[d](i, 0)) - center[d]; s += diff * diff;} return s;
    case k_means::distance_method_t::l1_norm:       for(int d: util::size_range(input)){float diff = (euqal_layout ? input[d].cols[0][i]: input[d](i, 0)) - center[d]; s += std::abs(diff);} return s;
    }
    return 0;
}

void calc_center_pos(const k_means::float_column_views& input, const k_means::float_column_views& indices, center_t& centers, bool equal_data, size_t data_size){
    for(auto& c: centers) for(float& f: c) f = 0;
    std::vector<size_t> cluster_counts(centers.size(), 0);
    for(int d: util::size_range(input)){
        for(size_t i: util::i_range(data_size)){
            int cluster_index = int(indices[0].cols[0][i]);
            centers[cluster_index][d] += equal_data ? input[d].cols[0][i]: input[d](i, 0);
            if(d == 0)
                ++cluster_counts[cluster_index];
        }
    }
    for(int i: util::size_range(centers)){
        for(float& v: centers[i]) v /= double(cluster_counts[i]);
    }
}

bool assign_data_to_cluster(const k_means::float_column_views& input, k_means::float_column_views& indices, const center_t& centers, bool equal_data, size_t data_size, k_means::distance_method_t dist){
    bool change{};
    for(size_t i: util::i_range(data_size)){
        int c = std::min_element(centers.begin(), centers.end(), [&](const std::vector<float>& a, const std::vector<float>& b){return distance(input, i, a, dist, equal_data) < distance(input, i, b, dist, equal_data);}) - centers.begin();
        if(equal_data){
            change |= c != int(indices[0].cols[0][i]);
            indices[0].cols[0][i] = float(c);
        }
        else{
            change |= c != int(indices[0](i, 0));
            indices[0](i, 0) = float(c);
        }
    }
    return change;
}

void k_means::run(const float_column_views& input, float_column_views& output, const k_means_settings_t& settings){
    // the clusters are stored in the output buffer as float values
    // if the layout for all input arrays is not the same the input is increased to full size

    bool equal_layout = true;
    for(int i: util::i_range(input.size() - 1))
        equal_layout &= input[i].equalDataLayout(input[i + 1]);
    size_t data_size = equal_layout ? input[0].cols[0].size() : input[0].size();

    center_t cluster_centers(settings.cluster_count, std::vector<float>(input.size(), 0));
    
    // initialisation -----------------------------------------------------------------------
    std::default_random_engine random_engine{std::random_device{}()};
    switch(settings.init_method){
    case init_method_t::forgy:{
        std::uniform_int_distribution<size_t> distribution(0, data_size - 1);
        for(auto& center: cluster_centers){
            size_t index = distribution(random_engine);
            for(int i: util::size_range(input)) center[i] = equal_layout ? input[i].cols[0][index]: input[i](index, 0);
        }
        break;}
    case init_method_t::uniform_random:{
        std::uniform_int_distribution<int> distribution(0, settings.cluster_count - 1);
        for(int i: util::i_range(data_size)) output[0].cols[0][i] = distribution(random_engine);
        calc_center_pos(input, output, cluster_centers, equal_layout, data_size);
        break;}
    case init_method_t::normal_random:{
        std::vector<std::pair<double, double>> averages(input.size(), std::pair<double,double>{});    // first and second order averages
        for(int d: util::size_range(input)){
            for(size_t i: util::i_range(data_size)){
                float v = equal_layout ? input[d].cols[0][i]: input[d](i, 0);
                averages[d].first += v;
                averages[d].second += v * v;
            }
            averages[d].first /= data_size;
            averages[d].second /= data_size;
        }
        for(auto& v: averages) v.second = std::sqrt(v.second - v.first * v.first); // converting second order to standard deviation
        for(int d: util::size_range(input)){
            std::normal_distribution<double> distribution(averages[d].first, averages[d].second);
            for(int i: util::size_range(cluster_centers)) cluster_centers[i][d] = distribution(random_engine);
        }
        break;}
    case init_method_t::plus_plus:{
        std::uniform_int_distribution<size_t> distribution(0, data_size - 1);
        std::uniform_real_distribution<float> f_distribution(0, 1);
        size_t index = distribution(random_engine);
        for(int i: util::size_range(input)) cluster_centers[0][i] = equal_layout ? input[i].cols[0][index]: input[i](index, 0);
        std::vector<float> probabilities(data_size, std::numeric_limits<float>::infinity());
        for(int i: util::i_range(1, settings.cluster_count)){
            double probability_sum{};
            for(size_t j: util::i_range(data_size)){
                probabilities[j] = std::min(probabilities[j], distance(input, j, cluster_centers[i - 1], settings.distance_method, equal_layout));
                probability_sum += probabilities[j];
            }
            if(probability_sum <= 0)
                continue;
            float rand = f_distribution(random_engine) * probability_sum;
            for(size_t j: util::i_range(data_size)){
                if((rand -= probabilities[j]) > 0)
                    continue;
                for(int d: util::size_range(input))
                    cluster_centers[i][d] = equal_layout ? input[d].cols[0][j]: input[d](j, 0);
                break;
            }
        }
        break;}
    }

    // k_means iteration -----------------------------------------------------------------------
    for(int i: util::i_range(settings.max_iteration)){
        if(!assign_data_to_cluster(input, output, cluster_centers, equal_layout, data_size, settings.distance_method))
            break;  // nothing changed, done
        calc_center_pos(input, output, cluster_centers, equal_layout, data_size);
    }
}