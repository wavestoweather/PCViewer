#include "db_scan.hpp"
#include <kd_tree.hpp>

constexpr float cluster_unclassified = 0;
constexpr float cluster_noise = -1;

bool expandCluster(int cur_cluster_index ,size_t index, db_scan::float_column_view& output, const structures::kd_tree& kd_tree, const db_scan::db_scans_settings_t& settings){
    std::vector<size_t> neighbours = kd_tree.neighborhood(index, settings.epsilon);

    if(neighbours.size() < settings.min_points){
        if(kd_tree.same_layout) output[0].cols[0][index] = cluster_noise;
        else                    output[0](index, 0) = cluster_noise;
        return false;
    }
    else{
        if(kd_tree.same_layout) output[0].cols[0][index] = as<float>(cur_cluster_index);
        else                    output[0](index, 0) = as<float>(cur_cluster_index);
        for(size_t n: neighbours){
            if(kd_tree.same_layout) output[0].cols[0][n] = as<float>(cur_cluster_index);
            else                    output[0](n, 0) = as<float>(cur_cluster_index);
        }

        while(neighbours.size()){
            auto neigh = kd_tree.neighborhood(neighbours.back(), settings.epsilon);
            neighbours.pop_back();

            if(neigh.size() > settings.min_points){
                for(size_t n2: neigh){
                    if(kd_tree.same_layout && output[0].cols[0][n2] <= cluster_unclassified){
                        output[0].cols[0][n2] = as<float>(cur_cluster_index);
                        neighbours.push_back(n2);
                    }
                    if(!kd_tree.same_layout && output[0](n2, 0) <= cluster_unclassified){
                        output[0](n2, 0) = as<float>(cur_cluster_index);
                        neighbours.push_back(n2);
                    }
                }
            }
        }
    }
    return true;
}

void db_scan::run(const float_column_view& input, float_column_view& output, const db_scans_settings_t& settings){
    const structures::kd_tree tree(input);
    int cur_cluster_id{1};
    for(size_t i: util::i_range(tree.data_size)){
        if(tree.same_layout && output[0].cols[0][i] == cluster_unclassified && expandCluster(cur_cluster_id, i, output, tree, settings) ||
            !tree.same_layout && output[0](i, 0) == cluster_unclassified && expandCluster(cur_cluster_id, i, output, tree, settings))
            ++cur_cluster_id;
    }
    // thtats it
}