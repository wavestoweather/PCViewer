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

void MultivariateGauss::compute_matrix_inverse(const std::vector<std::vector<double>>& d, std::vector<std::vector<double>>& dst)
{
	std::vector<uint32_t> idx(d.size());
	std::vector<double> col(d[0].size());
	std::vector<std::vector<double>> a = d;
	float dd;
	if (!ludcmp(a, idx, dd)) {
		dst = std::vector<std::vector<double>>(d.size(), std::vector<double>(d[0].size(), 0));
		return;
	}
	for (int j = 0; j < d[0].size(); ++j) {
		for (int i = 0; i < d[0].size(); ++i) col[i] = 0;
		col[j] = 1;
		lubksb(a, idx, col);
		for (int i = 0; i < d[0].size(); ++i) dst[i][j] = col[i];
	}
}

void MultivariateGauss::compute_matrix_times_matrix(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b, std::vector<std::vector<double>>& dst)
{
	for (int i = 0; i < a.size(); ++i) {
		for (int j = 0; j < b[0].size(); ++j) {
			for (int k = 0; k < a[0].size(); ++k) {
				dst[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}

void MultivariateGauss::compute_matrix_determinant(const std::vector<std::vector<float>>& a, float& det)
{
	std::vector<std::vector<double>> b(a.size(),std::vector<double>(a[0].size()));
	for (int i = 0; i < a.size(); ++i) {
		for (int j = 0; j < a[i].size(); ++j) {
			b[i][j] = a[i][j];
		}
	}

	std::vector<uint32_t> idx(a[0].size());
	ludcmp(b, idx, det);
	for (int i = 0; i < b.size(); ++i) det *= b[i][i];
}

bool MultivariateGauss::ludcmp(std::vector<std::vector<double>>& a, std::vector<uint32_t>& idx, float& d)
{
	int i, imax, j, k;
	double big, dum, sum, temp;
	std::vector<double> vv(a[0].size(),1);

	d = 1;
	for (i = 0; i < a[0].size(); ++i) {
		big = 0;
		for (j = 0; j < a[0].size(); ++j) {
			if ((temp = fabs(a[i][j])) > big) big = temp;
		}
		if (big == 0) return false;			//singular matrix
		vv[i] = 1.0 / big;
	}
	for (j = 0; j < a[0].size(); ++j) {
		for (i = 0; i < j; ++i) {
			sum = a[i][j];
			for (k = 0; k < i; ++k) sum -= a[i][k] * a[k][j];
			a[i][j] = sum;
		}
		big = 0;
		for (i = j; i < a[0].size(); ++i) {
			sum = a[i][j];
			for (k = 0; k < j; ++k) sum -= a[i][k] * a[k][j];
			a[i][j] = sum;
			if ((dum = vv[i] * fabs(sum)) >= big) {
				big = dum;
				imax = i;
			}
		}
		if (j != imax) {
			for (k = 0; k < a[0].size(); ++k) {
				dum = a[imax][k];
				a[imax][k] = a[j][k];
				a[j][k] = dum;
			}
			d = -d;
			vv[imax] = vv[j];
		}
		idx[j] = imax;
		if (a[j][j] == 0) a[j][j] = TINY;
		if (j != a[0].size() - 1) {
			dum = 1.0 / (a[j][j]);
			for (i = j + 1; i < a[0].size(); ++i) a[i][j] *= dum;
		}
	}
	return true;
}

void MultivariateGauss::lubksb(std::vector<std::vector<double>>& a, std::vector<uint32_t>& idx, std::vector<double>& b)
{
	int i, ii = 0, ip, j;
	double sum;

	for (i = 0; i < a[0].size(); ++i) {
		ip = idx[i];
		sum = b[ip];
		b[ip] = b[i];
		if (ii)
			for (j = ii - 1; j <= i - 1; ++j) sum -= a[i][j] * b[j];
		else if (sum) ii = i + 1;
		b[i] = sum;
	}
	for (i = a[0].size() - 1; i >= 0; --i) {
		sum = b[i];
		for (j = i + 1; j < a[0].size(); ++j) sum -= a[i][j] * b[j];
		b[i] = sum / a[i][i];
	}
}
