#include "MultivariateGauss.h"

void MultivariateGauss::compute_covariance_matrix(const std::vector<std::vector<double>>& d, std::vector<std::vector<double>>& covar_matrix)
{
	int dim = d[0].size();
	assert(dim == covar_matrix.size());
	assert(dim == covar_matrix[0].size());
	for (int i = 0; i < dim; ++i) {
		for (int j = i; j < dim; ++j) {
			covar_matrix[i][j] = compute_covariance(d, i, j);
		}
	}


	// fill the Left triangular matrix
	for (int i = 1; i < dim; i++) {
		for (int j = 0; j < i; ++j) {
			covar_matrix[i][j] = covar_matrix[j][i];
		}
	}

}

void MultivariateGauss::compute_average_vector(std::vector<std::vector<double>>& d, std::vector<double>& mean)
{
	for (int i = 0; i < d.size(); ++i) {
		for (int j = 0; j < d[i].size(); ++j) {
			mean[j] += d[i][j];
		}
	}
	for (int i = 0; i < mean.size(); ++i) {
		mean[i] /= d.size();
	}
	for (int i = 0; i < d.size(); ++i) {
		for (int j = 0; j < mean.size(); ++j) {
			d[i][j] -= mean[j];
		}
	}
}

double MultivariateGauss::compute_covariance(const std::vector<std::vector<double>>& d, int i, int j)
{
	double cov = 0;
	for (int k = 0; k < d.size(); ++k) {
		cov += d[k][i] * d[k][j];
	}

	return cov / (d.size() - 1);
}
