#pragma once

#include <vector>
#include <cassert>

class MultivariateGauss {
public:
	static void compute_covariance_matrix(const std::vector<std::vector<double>>& d, std::vector<std::vector<double>>& covar_matrix);
	static void compute_average_vector(std::vector<std::vector<double>>& d, std::vector<double>& mean);
 private:
	 static double compute_covariance(const std::vector<std::vector<double>>& d, int i, int j);
};