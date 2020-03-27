#pragma once

#include<vector>

namespace MultivariateGauss {
	static void compute_covariance_matrix(std::vector<std::vector<double>>& d, std::vector<std::vector<double>>& dst);

	namespace Intern {
		static void outer_product(std::vector<double>& row, std::vector<double>& col, std::vector<std::vector<double>>& dst);
		static void subtract(std::vector<double>& row, double val, std::vector<double>& dst);
		static void add(std::vector<std::vector<double>>& m, std::vector<std::vector<double>>& m2, std::vector<std::vector<double>>& dst);
		static double mean(std::vector<double>& data);
		static void scale(std::vector<std::vector<double>>& d, double alpha);
	}
}