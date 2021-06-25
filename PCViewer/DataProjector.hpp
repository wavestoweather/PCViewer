#pragma once
#include <future>
#include <iostream>
#include <Eigen/Dense>
#include "Data.hpp"

//Single class for data projection, differnet methods can be set in the constructor
class DataProjector{
public:
    enum Method{
        PCA,
        TSNE
    };

    struct ProjectionSettings{
        //settings for PCA

        //settings for TSNE
        double perplexity, theta;
        int randSeed, maxIter, stopLyingIter, momSwitchIter;
        bool skipRandomInit;
    };

    DataProjector(const Data& data, int reducedDimensionSize, Method projectionMethod, const ProjectionSettings& settings, std::vector<uint32_t> indices = {}):data(data), reducedDimensionSize(reducedDimensionSize), projectionMethod(projectionMethod){
        if(indices.empty()){
            this->indices.resize(data.size());
            for(uint32_t i = 0; i < indices.size(); ++i) this->indices[i] = i;
        }
        else this->indices = indices;
        this->settings = settings;
        future = std::async(runAsync, this);
    }

    static void runAsync(DataProjector* p){
        p->run();
    }

    void run(){
        switch(projectionMethod){
        case Method::PCA:{
            execPCA();
        } break;
        case Method::TSNE:{
            execTSNE();
        }break;
        }
    }

    bool projected = false;
    float progress = .0f;
    int reducedDimensionSize;
    Eigen::MatrixXf projectedPoints;

protected:
    std::future<void> future;
    const Data& data;
    std::vector<uint32_t> indices;
    ProjectionSettings settings;
    Method projectionMethod;
    void execPCA(){
        //data to eigen matrix
        Eigen::MatrixXf d(data.size(), data.columns.size());
        for(int i = 0; i < data.size(); ++i){
            for(int  c = 0; c < data.columns.size(); ++c){
                d(i,c) = data(i,c);
            }
        }
        progress = .33f;
        //zero center data + perfrom svd
        Eigen::RowVectorXf meanCols = d.colwise().mean();
        d = d.rowwise() - meanCols;
        Eigen::BDCSVD<Eigen::MatrixXf> svd(d, Eigen::ComputeThinU | Eigen::ComputeThinV);
        progress = .66f;
        //convert points to pc scores
        auto u = svd.matrixU().real();
        projectedPoints = u * svd.singularValues().real().asDiagonal();
        //drop unused scores
        projectedPoints.conservativeResize(Eigen::NoChange, reducedDimensionSize);
        progress = 1;
        projected = true;
    }
    void execTSNE(){
        std::cout << "Not yet implemented" <<std::endl;
    }
};