#pragma once
#include<vector>
#include<inttypes.h>
#include<Eigen/Dense>

class HClustering{
    public:
    struct Settings{

    };
    enum class Distance{
        Euclidean,
        SqEuclidean,
        Manhattan,
        correlation
    };

    HClustering(Eigen::MatrixXf& data, std::vector<std::vector<uint32_t>> cluster, Settings& settings):
    data(data),
    cluster(cluster),
    settings(settings),
    distances(data.rows() * (data.rows() - 1) / 2)
    {
        computeDistanceMatrix();
    }

    protected:
    Eigen::MatrixXf& data;
    std::vector<std::vector<uint32_t>> &cluster;
    Settings& settings;
    std::vector<float> distances;

    uint32_t idx(uint32_t x, uint32_t y){
        uint32_t n = data.rows();
        return x * (2 * n - 1 - x) / 2 + (y - x - 1);
    }

    void computeDistanceMatrix(){
        for(uint32_t y = 0; y < data.rows(); ++y){
            for(uint32_t x = y + 1; x < data.rows(); ++x){
                const uint32_t index = idx(x, y);
                const Eigen::VectorXf vecX = data.row(x);
                const Eigen::VectorXf vecY = data.row(y);

                double measure = 0;
                switch(){
                    
                }
            }
        }
    }

};