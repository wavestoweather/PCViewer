#pragma once

#include<iostream>
#include<vector>
#include<future>
#include<random>
#include<Eigen/Dense>
#include "Data.hpp"


//Single Class for data clustering. In the constructor different Clustering methods can be inserted to change the clustering
class DataClusterer{
public:
    enum Method{
        KMeans,
        DBScan,
        Hirarchical
    };

    enum DistanceMetric{
        Norm,
        SquaredNorm,
        L1Norm
    };

    enum InitMethod{
        Forgy,
        UniformRandom,
        NormalRandom,
        PlusPlus
    };

    enum KMethod{
        Mean,
        Median,
        Mediod
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

        //Hirarchical

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

protected:
    ClusterSettings settings;
    std::future<void> future;
    const Eigen::MatrixXf& points;
    std::vector<uint32_t> clusterAssignments;
    bool clusterUpdated;
    Method method;

    float distance(const Eigen::MatrixXf& a,const Eigen::MatrixXf& b){
        switch(settings.distanceMetric){
            case DistanceMetric::Norm: return (a - b).norm();
            case DistanceMetric::SquaredNorm: return (a - b).squaredNorm();
            case DistanceMetric::L1Norm: return (a - b).lpNorm<1>();
        }
    }

    float updateStepKMeans(){
        for(auto& vec: clusterMeans) vec.setZero();
        int c = 0;
        for(auto& cluster: clusters){
            for(uint32_t p: cluster){
                clusterMeans[c] += points.col(p);
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

        //k means iteration
        clusterUpdated = true;
        for(int counter = 0; counter < settings.maxIterations && clusterUpdated; ++counter){
            clusterUpdated = false;

            assignmentStepKMeans();
            updateStepKMeans();
        }
    }
    void execDBScan(){
        std::cout << "Not implemented yet" << std::endl;
    }
    void execHirarchical(){
        std::cout << "Not implemented yet" << std::endl;
    }
};