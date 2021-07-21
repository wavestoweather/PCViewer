#pragma once

#include<iostream>
#include<vector>
#include<future>
#include<random>
#include<utility>
#include<Eigen/Dense>
#include "Data.hpp"
#include "DBScan.hpp"


//Single Class for data clustering. In the constructor different Clustering methods can be inserted to change the clustering
class DataClusterer{
public:
    enum class Method: int{
        KMeans,
        DBScan,
        Hirarchical
    };

    enum class DistanceMetric: int{
        Norm,
        SquaredNorm,
        L1Norm
    };

    enum class InitMethod: int{
        Forgy,
        UniformRandom,
        NormalRandom,
        PlusPlus
    };

    enum class KMethod: int{
        Mean,
        Median,
        Mediod
    };

    enum class HClusteringLinkage: int{
        Single,
        Complete,
        Weighted,
        Median,
        Average,
        Ward,
        Centroid
    };

    //contains settings for all clustering methods, the correct settings are read out
    struct ClusterSettings{
        //general settings
        DistanceMetric distanceMetric;

        //KMeans
        int kmeansClusters;
        KMethod kmeansMethod;
        InitMethod kmeansInitMethod;
        int maxIterations;

        //DBScan
        int dbscanMinPoints;
        float dbscanEpsilon;

        //Hirarchical
        HClusteringLinkage hclusteringLinkage;
        int hclusteringClusters;
    };

    DataClusterer(const Eigen::MatrixXf& points, Method method, const ClusterSettings& settings):settings(settings), points(points), method(method){
        future = std::async(runAsync, this);
    }

    static void runAsync(DataClusterer* c){
        c->run();
    }

    void run(){
        switch(method){
            case Method::KMeans:{
                execKMeans();
            }break;
            case Method::DBScan:{
                execDBScan();
            }break;
            case Method::Hirarchical:{
                execHirarchical();
            }break;
        }
    }

    std::vector<std::vector<uint32_t>> clusters;
    std::vector<Eigen::VectorXf> clusterMeans;
    std::future<void> future;
    float progress = 0;
    bool clustered = false;

protected:
    ClusterSettings settings;
    const Eigen::MatrixXf& points;
    bool clusterUpdated;
    Method method;
    std::vector<uint32_t> clusterAssignments;

    float distance(const Eigen::VectorXf& a,const Eigen::VectorXf& b){
        switch(settings.distanceMetric){
            case DistanceMetric::Norm: return (a - b).norm();
            case DistanceMetric::SquaredNorm: return (a - b).squaredNorm();
            case DistanceMetric::L1Norm: return (a - b).lpNorm<1>();
        }
    }

    void updateStepKMeans(){
        for(auto& vec: clusterMeans) vec.setZero();
        int c = 0;
        for(auto& cluster: clusters){
            for(uint32_t p: cluster){
                clusterMeans[c] += points.row(p);
            }
            clusterMeans[c] /= static_cast<float>(cluster.size());
            ++c;
        }
    }

    void assignmentStepKMeans(){
        clusters = std::vector<std::vector<uint32_t>>(clusters.size()); //erasing all assignments
        for(int i = 0; i < points.rows(); ++i){
            int clusterID = 0;
            float minDist = distance(points.row(i), clusterMeans[0]);
            for(int j = 1; j < settings.kmeansClusters; ++j){
                float dist = distance(points.row(i), clusterMeans[j]);
                if(dist < minDist){
                    clusterID = j;
                    minDist = dist;
                }
            }
            clusters[clusterID].push_back(i);
            clusterUpdated |= clusterAssignments[i] != clusterID;
            clusterAssignments[i] = clusterID; 
        }
    }

    void execKMeans(){
        //initialization
        clusterAssignments = std::vector<uint32_t>(points.rows(), 0);
        std::mt19937 engine;
        engine.seed(static_cast<unsigned long>(time(nullptr)));
        std::uniform_int_distribution<uint> intDistCluster(0, settings.kmeansClusters - 1);
        std::uniform_int_distribution<uint> intDist(0, points.rows());
        std::uniform_real_distribution<float> realDist(0, 1);
        clusterMeans = std::vector<Eigen::VectorXf>(settings.kmeansClusters);
        clusters.resize(settings.kmeansClusters);
        
        switch(settings.kmeansInitMethod){
            case InitMethod::Forgy:{
                for(auto& clusterMean: clusterMeans) clusterMean = points.row(intDist(engine));
                break;
            }
            case InitMethod::UniformRandom:{
                for(int i = 0; i < points.rows(); ++i) clusters[intDistCluster(engine)].push_back(i);
                for(auto& mean: clusterMeans) mean = Eigen::VectorXf::Zero(points.cols());
                updateStepKMeans();
                break;
            }
            case InitMethod::NormalRandom:{
                Eigen::RowVectorXf means = points.colwise().mean();
                Eigen::MatrixXf centered = points.rowwise() - means;
                Eigen::VectorXf stdDevs = (centered.colwise().squaredNorm() / static_cast<float>(points.rows()));
                stdDevs = stdDevs.array().sqrt();
                std::vector<float> randData(points.cols(), 0);
                std::vector<std::normal_distribution<float>> normalDists(points.cols());
                int i = 0;
                for(auto& dist: normalDists) dist = std::normal_distribution<float>(means[i++], stdDevs[i]);

                for(auto& mean: clusterMeans){
                    mean = Eigen::VectorXf::Zero(points.cols());
                    for(i = 0; i < points.cols(); ++i) mean[i] = normalDists[i](engine);
                }
                break;
            }
            case InitMethod::PlusPlus:{
                clusterMeans[0] = points.row(intDist(engine));
                Eigen::VectorXf probabilities(points.rows());
                std::fill_n(probabilities.data(), points.rows(), std::numeric_limits<float>::infinity());
                for(int i = 1; i < settings.kmeansClusters; ++i){
                    //updating all probabilities with the new found cluster
                    for(int j = 0; j < points.rows(); ++j){
                        float dist = distance(points.row(j), clusterMeans[i - 1]);
                        probabilities(j) = std::min(probabilities(j), dist);
                    }
                    float summedProbability = probabilities.sum();
                    if(summedProbability != 0){
                        float rand = realDist(engine) * summedProbability;
                        int j = 0;
                        for(; j < points.rows(); ++j){
                            rand -= probabilities(j);
                            if(rand < 0) break;
                        }
                        clusterMeans[i] = points.row(j);
                    }
                }
                break;
            }
        }
        progress = .1f;

        //k means iteration
        clusterUpdated = true;
        for(int counter = 0; counter < settings.maxIterations && clusterUpdated; ++counter){
            clusterUpdated = false;
            progress = counter / float(settings.maxIterations) * .9 + .1;

            assignmentStepKMeans();
            updateStepKMeans();
        }
        progress = 1;
        clustered = true;
    }
    void execDBScan(){
        DBScan dbscan(settings.dbscanMinPoints, settings.dbscanEpsilon, points, clusters, &progress);
        progress = 1;
        clustered = true;
    }

    // Hirarchical clustering-------------------------------------------------------------------
    struct Node{
        size_t left, right;
        float distance;
        size_t size;
    };
    struct LanceWilliamsCoeffs
    {
        double ai;
        double aj;
        double beta;
        double gamma;
    };

    std::vector<float> h_distances;
    std::vector<std::pair<uint32_t, uint32_t>> h_indices;
    std::vector<Node> h_nodes;
    std::vector<size_t> h_nodeIDs;
    
    inline uint32_t idx(uint32_t x, uint32_t y){
        uint32_t n = points.rows();
        return x * (2 * n - 1 - x) / 2 + (y - x - 1);
    }
    
    void computeDistanceMatrix(){
        for(uint32_t y = 0; y < points.rows(); ++y){
            for(uint32_t x = y + 1; x < points.rows(); ++x){
                const uint32_t index = idx(x, y);
                const Eigen::VectorXf vecX = points.row(x);
                const Eigen::VectorXf vecY = points.row(y);

                double measure = distance(vecX, vecY);
                h_distances[index] = measure;
                h_indices[index] = std::make_pair(y,x);
            }
        }
    }

    LanceWilliamsCoeffs computeCoefficients(size_t i, size_t j, size_t k){
        size_t nI, nJ, nK;
        size_t nodeID = 0;
        switch(settings.hclusteringLinkage){
            case HClusteringLinkage::Single:
                return {.5, .5, 0, -.5f};
            case HClusteringLinkage::Complete:
                return {.5, .5, 0, .5};
            case HClusteringLinkage::Weighted:
                return {.5, .5, 0, 0};
            case HClusteringLinkage::Median:
                return {.5, .5, -.25, 0};
            case HClusteringLinkage::Average:{
                nodeID = h_nodeIDs[i] - points.rows();
                nI = (nodeID >= 0) ? h_nodes[nodeID].size : 1;
                nodeID = h_nodeIDs[j] - points.rows();
                nJ = (nodeID >= 0) ? h_nodes[nodeID].size : 1;

                const double sumN = static_cast<double>(nI + nJ);
                const double ai = nI / sumN;
                const double aj = nJ / sumN;

                return { ai, aj, 0.0, 0.0 };
            }
            case HClusteringLinkage::Ward:{
                nodeID = h_nodeIDs[i] - points.rows();
                nI = (nodeID >= 0) ? h_nodes[nodeID].size : 1;
                nodeID = h_nodeIDs[j] - points.rows();
                nJ = (nodeID >= 0) ? h_nodes[nodeID].size : 1;
                nodeID = h_nodeIDs[k] - points.rows();
                nK = (nodeID >= 0) ? h_nodes[nodeID].size : 1;

                const double sumN = static_cast<double>(nI + nJ + nK);
                const double ai = (nI + nK) / sumN;
                const double aj = (nJ + nK) / sumN;
                const double beta = nK / sumN * (-1);

                return { ai, aj, beta, 0.0 };
            }
            case HClusteringLinkage::Centroid:{
                nodeID = h_nodeIDs[i] - points.rows();
                nI = (nodeID >= 0) ? h_nodes[nodeID].size : 1;
                nodeID = h_nodeIDs[j] - points.rows();
                nJ = (nodeID >= 0) ? h_nodes[nodeID].size : 1;

                const double sumN = static_cast<double>(nI + nJ);
                const double ai = nI / sumN;
                const double aj = nJ / sumN;
                const double beta = (-nI * nJ) / (sumN * sumN);

                return { ai, aj, beta, 0.0 };
            }
        }
        return {};
    }

    // Lance-Williams algorithm
    float computeNewDistance(float dKI, float dKJ, size_t i, size_t j, size_t k){
        const LanceWilliamsCoeffs c = computeCoefficients(i, j, k);
        
        double betaTerm = 0, gammaTerm = 0;

        if(c.beta > 0){
            const double valIJ = h_distances[idx(i, j)];
            betaTerm = c.beta * valIJ;
        }
        if(c.gamma > 0){
            const double absD = std::abs(dKI - dKJ);
            gammaTerm = c.gamma * absD;
        }

        return c.ai * dKI + c.aj * dKJ + betaTerm + gammaTerm;
    }

    void execHirarchical(){
        std::cout << "Not implemented yet" << std::endl;
        h_distances.resize(points.rows() * (points.rows() - 1) / 2);
        h_indices.resize(h_distances.size());
        h_nodes.resize(points.rows());
        h_nodeIDs.resize(points.rows());
        computeDistanceMatrix();

        const size_t numRuns = points.rows() - 1;

        const float maxValue = std::numeric_limits<float>::max();
        std::iota(h_nodeIDs.begin(), h_nodeIDs.end(), 0);   //filling with 0 to i

        for(size_t counter = 0; counter < numRuns; ++counter){
            auto minIt = std::min_element(h_distances.begin(), h_distances.end());
            float minDist = *minIt;
            size_t i = minIt - h_distances.begin();

            const auto& indices = h_indices[i];
            const size_t& indexI = indices.first, & indexJ = indices.second;

            for(size_t ii = 0; ii < indexI; ++i){
                const float dKI = h_distances[idx(ii, indexI)];
                const float dKJ = h_distances[idx(ii, indexJ)];

                h_distances[idx(ii, indexI)] = computeNewDistance(dKI, dKJ, indexI, indexJ, ii);
                h_distances[idx(ii, indexJ)] = maxValue;
            }

            for(size_t ii = indexI + 1; ii < indexJ; ++i){
                const float dKI = h_distances[idx(indexI, ii)];
                const float dKJ = h_distances[idx(ii, indexJ)];

                h_distances[idx(indexI, ii)] = computeNewDistance(dKI, dKJ, indexI, indexJ, ii);
                h_distances[idx(ii, indexJ)] = maxValue;
            }

            for(size_t ii = indexJ + 1; ii < points.rows(); ++ii){
                const float dKI = h_distances[idx(indexI, ii)];
                const float dKJ = h_distances[idx(indexJ, ii)];

                h_distances[idx(indexI, ii)] = computeNewDistance(dKI, dKJ, indexI, indexJ, ii);
                h_distances[idx(indexJ, ii)] = maxValue;
            }

            h_distances[idx(indexI, indexJ)] = maxValue;

            auto& node = h_nodes[counter];

            node.distance = minDist;
            node.left = size_t(h_nodeIDs[indexI]);
            node.right = size_t(h_nodeIDs[indexJ]);
            node.size = (node.left >= points.rows()) ? h_nodes[node.left - points.rows()].size : 1;
            node.size += (node.right >= points.rows()) ? h_nodes[node.right - points.rows()].size : 1;

            h_nodeIDs[indexI] = counter + points.rows();
        }

        //cutting the node trees
        clusters.resize(settings.hclusteringClusters);
        uint32_t cluster = 0;
        std::fill(h_nodeIDs.begin(), h_nodeIDs.end(), 0);
        for(size_t i = points.rows() - 2; i >= 0; --i){
            const Node& node = h_nodes[i];
            size_t& nodeID = h_nodeIDs[i];

            if(cluster < settings.hclusteringClusters){
                if(node.left < points.rows()) clusters[nodeID].push_back(node.left);
                else h_nodeIDs[node.left - points.rows()] = nodeID;

                cluster++;

                if (node.right < points.rows()) clusters[cluster].push_back(node.right);
                else h_nodeIDs[node.right - points.rows()] = cluster;
                continue;
            }

            const size_t leftID = node.left;
            if (leftID >= points.rows()) h_nodeIDs[leftID - points.rows()] = nodeID;
            else clusters[nodeID].push_back(leftID);

            const size_t rightID = node.right;
            if (rightID >= points.rows()) h_nodeIDs[rightID - points.rows()] = nodeID;
            else clusters[nodeID].push_back(rightID);
        }
    }
};