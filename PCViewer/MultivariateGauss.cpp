#include "MultivariateGauss.h"

void MultivariateGauss::Intern::outer_product(std::vector<double>& row, std::vector<double>& col, std::vector<std::vector<double>>& dst)
{
	for (unsigned i = 0; i < row.size(); i++) {
		for (unsigned j = 0; j < col.size(); j++) {
			dst[i][j] = row[i] * col[j];
		}
	}
}

void MultivariateGauss::Intern::subtract(std::vector<double>& row, double val, std::vector<double>& dst)
{
	for (unsigned i = 0; i < row.size(); i++) {
		dst[i] = row[i] - val;
	}
}

void MultivariateGauss::Intern::add(std::vector<std::vector<double>>& m, std::vector<std::vector<double>>& m2, std::vector<std::vector<double>>& dst)
{
	for (unsigned i = 0; i < m.size(); i++) {
		for (unsigned j = 0; j < m[i].size(); j++) {
			dst[i][j] = m[i][j] + m2[i][j];
		}
	}
}

double MultivariateGauss::Intern::mean(std::vector<double>& data)
{
	double mean = 0.0;

	for (unsigned i = 0; (i < data.size()); i++) {
		mean += data[i];
	}

	mean /= data.size();
	return mean;
}

void MultivariateGauss::Intern::scale(std::vector<std::vector<double>>& d, double alpha)
{
	for (unsigned i = 0; i < d.size(); i++) {
		for (unsigned j = 0; j < d[i].size(); j++) {
			d[i][j] *= alpha;
		}
	}
}

void MultivariateGauss::compute_covariance_matrix(std::vector<std::vector<double>>& d, std::vector<std::vector<double>>& dst)
{
	for (unsigned i = 0; i < d.size(); i++) {
		double y_bar = Intern::mean(d[i]);
		std::vector<double> d_d_bar(d[i].size());
		Intern::subtract(d[i], y_bar, d_d_bar);
		std::vector<std::vector<double>> t(d.size());
		Intern::outer_product(d_d_bar, d_d_bar, t);
		Intern::add(dst, t, dst);
	}
	Intern::scale(dst, 1 / (d.size() - 1));
}
