#pragma once

#include <vector>
#include <cassert>
#include <math.h>

#define TINY 1.0e-20

class MultivariateGauss {
public:
	static void compute_covariance_matrix(const std::vector<std::vector<double>>& d, std::vector<std::vector<double>>& covar_matrix);
	static void compute_average_vector(std::vector<std::vector<double>>& d, std::vector<double>& mean);
	static void compute_matrix_inverse(const std::vector<std::vector<double>>& d, std::vector<std::vector<double>>& dst);
	static void compute_matrix_times_matrix(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b, std::vector<std::vector<double>>& dst);
 private:
	static double compute_covariance(const std::vector<std::vector<double>>& d, int i, int j);
	static bool ludcmp(std::vector<std::vector<double>>& a, std::vector<uint32_t>& idx, float& d);
	static void lubksb(std::vector<std::vector<double>>& a, std::vector<uint32_t>& idx, std::vector<double>& sol);
};