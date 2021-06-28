#pragma once
#include <future>
#include <iostream>
#include <Eigen/Dense>
#include "Data.hpp"

//Single class for data projection, differnet methods can be set in the constructor
//When created the data is projected in a separate thread
//On completion the "projected" member is set to true
//Call wait on the future object to sync
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

    Eigen::MatrixXf projectedPoints;
    bool projected = false;
    float progress = .0f;
    int reducedDimensionSize;
    std::future<void> future;

protected:
    const Data& data;
    std::vector<uint32_t> indices;
    ProjectionSettings settings;
    const float eps = 1e-6;
    Method projectionMethod;
    Eigen::MatrixXf getDataMatrix(){
        //data to eigen matrix
        Eigen::MatrixXf d(data.size(), indices.size());
        for(int i = 0; i < data.size(); ++i){
            for(int  c = 0; c < indices.size(); ++c){
                d(i,c) = data(i,indices[c]);
            }
        }
        //zero center and normalize data + perfrom svd
        Eigen::RowVectorXf meanCols = d.colwise().mean();
        d = d.rowwise() - meanCols;
        Eigen::RowVectorXf mins = d.colwise().minCoeff(), diff = d.colwise().maxCoeff() - mins;
        diff.array() += eps;
        d.array().rowwise() /= diff.array();
        return d;
        progress = .33f;
    }

    void execPCA(){
        Eigen::MatrixXf d = getDataMatrix();
        Eigen::BDCSVD<Eigen::MatrixXf> svd(d, Eigen::ComputeThinU | Eigen::ComputeThinV);
        progress = .66f;
        //convert points to pc scores
        auto u = svd.matrixU().real();
        projectedPoints = u * svd.singularValues().real().asDiagonal();
        //drop unused scores and normalizing the points
        projectedPoints.conservativeResize(Eigen::NoChange, reducedDimensionSize);
        Eigen::RowVectorXf min = projectedPoints.colwise().minCoeff();
        Eigen::RowVectorXf diff = projectedPoints.colwise().maxCoeff() - min;
        projectedPoints.rowwise() -= min;
        projectedPoints.array().rowwise() /= diff.array();
        progress = 1;
        projected = true;
    }

    Eigen::MatrixXf squaredEuclideanDistance(const Eigen::MatrixXf& m){
        return ((-2 * m * m.transpose()).colwise() + m.rowwise().squaredNorm()).rowwise() + m.rowwise().squaredNorm().transpose(); 
    }

    Eigen::MatrixXf gaussianPerplexity(const Eigen::MatrixXf& M, double perplexity){
        auto dd = squaredEuclideanDistance(M);
        Eigen::MatrixXf p = Eigen::MatrixXf::Zero(dd.rows(), dd.cols());
        for(int n = 0; n < M.rows(); ++n){
            bool found = false;
            double beta = 1;
            double minBeta = -DBL_MAX;
            double maxBeta = DBL_MAX;
            double tol = 1e-5;
            double sumP;

            int iter = 0;
            while(!found && iter < 200){
                sumP = DBL_MIN;
                double H = 0;
                for(int m = 0; m < M.rows(); ++m) {
                    p(n, m) = exp(-beta * dd(n,m)); 
                    if(n != m) {
                        sumP += p(n,m);
                        H += beta * p(n,m) * dd(n,m);
                    }
                }
                p(n,n) = DBL_MIN;
                H = (H / sumP) + log(sumP);

                double Hdiff = H - log(perplexity);
                if(Hdiff < tol && -Hdiff < tol) found = true;
                else{
                    if(Hdiff > 0){
                        minBeta = beta;
                        if(maxBeta == DBL_MAX || maxBeta == -DBL_MAX)
                            beta *= 2;
                        else
                            beta = (beta + maxBeta) / 2.0;
                    }
                    else{
                        maxBeta = beta;
                        if(minBeta == -DBL_MAX || minBeta == DBL_MAX){
                            if(beta < 0)
                                beta *= 2;
                            else
                                beta = beta <= 1.0 ? -.5 : beta / 2;
                        }
                        else{
                            beta = (beta + minBeta) / 2.0;
                        }
                    }
                }
                ++iter;
            }

            // normalize p
            p.row(n) /= sumP;
        }
        return p;
    }

    void execTSNE(){
        // usage of exact algorithm
        if(data.size() - 1 < 3 * settings.perplexity){
            std::cout << "Perplexity too large for the number of data points!" << std::endl;
            return;
        }
        bool exact = settings.theta == .0f;

        // learning params
        float totalTime = 0;
        double momentum = .5, final_momentum = .8;
        double eta = 200.0;

        Eigen::VectorXd dY = Eigen::VectorXd::Zero(reducedDimensionSize);
        Eigen::VectorXd uY = Eigen::VectorXd::Zero(reducedDimensionSize);
        Eigen::VectorXd gains = Eigen::VectorXd::Ones(reducedDimensionSize);

        Eigen::MatrixXf d = getDataMatrix();

        // Computing input similarities for exact t-SNE
        if(exact){
            auto p = gaussianPerplexity(d, settings.perplexity);
            double sumP = p.sum();
            p /= sumP;
        }
        else{   // input similarities for approxiamte t-SNE
            
        }
    }
};