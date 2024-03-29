#pragma once

#include <vector>
#include <cassert>
#include <math.h>
#include <Eigen/Dense>

#define TINY 1.0e-100

class MultivariateGauss {
public:
    struct MultivariateBrush {
        std::vector<float> mean;                    //mean vector
        std::vector<std::vector<float>> invCov;        //inverse matrix of covariance
        Eigen::MatrixXd cov;
        std::vector<uint32_t> pcInd;                //principal component indices which are above threshold
        std::vector<std::pair<float, float>> pcBounds; //principal component bounds for axes where the singular values are too small
        Eigen::MatrixXd pc;                            //principal komponents
        Eigen::VectorXd sv;                            //singular values divided by sqrt(m-1) (this yields the standard deviations)
        float detCov;                                //determinant of the covariance matrix
    };

    static void compute_covariance_matrix(const std::vector<std::vector<double>>& d, std::vector<std::vector<double>>& covar_matrix);
    static void compute_average_vector(std::vector<std::vector<double>>& d, std::vector<double>& mean);
    static void compute_matrix_inverse(const std::vector<std::vector<double>>& d, std::vector<std::vector<double>>& dst);
    static void compute_matrix_times_matrix(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b, std::vector<std::vector<double>>& dst);
    static void compute_matrix_determinant(const std::vector<std::vector<float>>& a, float& det);
    static void compute_matrix_determinant(const std::vector<std::vector<double>>& a, float& det);
 private:
    static double compute_covariance(const std::vector<std::vector<double>>& d, int i, int j);
    static bool ludcmp(std::vector<std::vector<double>>& a, std::vector<uint32_t>& idx, float& d);
    static void lubksb(std::vector<std::vector<double>>& a, std::vector<uint32_t>& idx, std::vector<double>& sol);
};