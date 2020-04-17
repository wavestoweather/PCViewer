#ifndef kd_tree_H
#define kd_tree_H
#include <vector>
#include <limits>
#include "MultivariateGauss.h"

class KdTree {
public:
	enum BoundsBehaviour {
		KdTree_Bounds_Static,				//no border adjustment
		KdTree_Bounds_Pull_In_Outer_Border,	//pull in the outer border at splits
		KdTree_Bounds_Pull_In_Both_Borders	//pull in bot borders
	};

	KdTree() {};
	KdTree(std::vector<uint32_t>& indices, std::vector<float*>& data, std::vector<int>& attributes, std::vector<std::vector<std::pair<float, float>>> initialBounds, int recursionDepth, BoundsBehaviour adjustBounds) {
		//building the kd tree;
		this->adjustBounds = adjustBounds;
		this->attributes = attributes;
		this->origBounds = std::vector<std::pair<float, float>>(attributes.size(), std::pair<float,float>( std::numeric_limits<float>::max(),std::numeric_limits<float>::min() ));
		for (int i = 0; i < initialBounds.size();++i) {
			for (auto& pair : initialBounds[i]) {
				if (origBounds[i].first > pair.first)origBounds[i].first = pair.first;
				if (origBounds[i].second < pair.second)origBounds[i].second = pair.second;
			}
		}

		std::vector<std::vector<int>> curIndices;
		std::vector<std::vector<int>> backIndices;
		for (int i = 0; i < initialBounds.size(); ++i) {
			if (!curIndices.size()) {
				for (int neww = 0; neww < initialBounds[i].size(); ++neww) {
					backIndices.push_back({ neww });
				}
			}
			else {
				for (int line = 0; line < curIndices.size(); ++line) {
					for (int neww = 0; neww < initialBounds[i].size(); ++neww) {
						backIndices.push_back({});
						for (int j = 0; j <= i; ++j) {
							if (j < i) {
								backIndices.back().push_back(curIndices[line][j]);
							}
							else {
								backIndices.back().push_back(neww);
							}
						}
					}
				}
			}
			curIndices = backIndices;
			backIndices.clear();
		}
		//PCUtil::matrixdump(curIndices);

		int headSize = ceil(log2(curIndices.size()));
		std::vector<int> curKds;
		for (int line = 0; line < curIndices.size(); ++line) {
			std::vector<std::pair<float, float>> bounds;
			for (int bound = 0; bound < curIndices[line].size(); ++bound) {
				bounds.push_back(initialBounds[bound][curIndices[line][bound]]);
			}
			curKds.push_back(buildRec(0, indices, data, attributes, bounds, recursionDepth - headSize));
		}
		std::vector<int> backKds;
		while (curKds.size() > 1) {
			for (int i = 0; i < curKds.size() - 1; i += 2) {
				Node curNode{};
				curNode.leftChild = curKds[i];
				curNode.rightChild = curKds[i + 1];
				for (int bound = 0; bound < nodes[curNode.leftChild].bounds.size(); ++bound) {
					std::pair<float, float> bounds;
					bounds.first = (nodes[curNode.leftChild].bounds[bound].first < nodes[curNode.rightChild].bounds[bound].first) ? nodes[curNode.leftChild].bounds[bound].first : nodes[curNode.rightChild].bounds[bound].first;
					bounds.second = (nodes[curNode.leftChild].bounds[bound].second > nodes[curNode.rightChild].bounds[bound].second) ? nodes[curNode.leftChild].bounds[bound].second : nodes[curNode.rightChild].bounds[bound].second;
					curNode.bounds.push_back(bounds);
				}
				curNode.split = (nodes[curNode.leftChild].split - 1 + attributes.size()) % attributes.size();
				std::vector<uint32_t> activeInd = getActiveIndices(attributes, data, indices, curNode.bounds);
				curNode.rank = activeInd.size();
				if (attributes.size() <= activeInd.size())
					curNode.multivariate = calcMultivariateBrush(attributes, data, activeInd);
				nodes.push_back(curNode);
				backKds.push_back(nodes.size() - 1);
			}
			if (curKds.size() & 1) {
				Node curNode{};
				curNode.leftChild = curKds.back();
				curNode.rightChild = -1;
				curNode.split = (nodes[curNode.leftChild].split - 1 + attributes.size()) % attributes.size();
				curNode.bounds = nodes[curNode.leftChild].bounds;
				std::vector<uint32_t> activeInd = getActiveIndices(attributes, data, indices, curNode.bounds);
				curNode.rank = activeInd.size();
				if (attributes.size() <= activeInd.size())
					curNode.multivariate = calcMultivariateBrush(attributes, data, activeInd);
				nodes.push_back(curNode);
				backKds.push_back(nodes.size() - 1);
			}
			curKds = backKds;
			backKds.clear();
		}

		root = curKds[0];
	};
	~KdTree() {};

	std::vector<std::vector<std::pair<float, float>>> getBounds(int recursionDepth) {
		return getBoundsRec(nodes[root], recursionDepth, 0);
	};

	std::vector<std::vector<std::pair<float, float>>> getBounds(int recursionDepth, int minRank) {
		return getBoundsRec(nodes[root], recursionDepth, minRank);
	};

	std::vector<MultivariateGauss::MultivariateBrush> getMultivariates(int recursionDepth) {
		return getMulBrushesRec(nodes[root], recursionDepth, attributes.size());
	};

	std::vector<std::pair<float, float>>& getOriginalBounds() {
		return origBounds;
	}

private:
	struct Node {
		int split;
		int rank;							//amount of indices in this node
		MultivariateGauss::MultivariateBrush multivariate;
		std::vector<std::pair<float, float>> bounds;
		int leftChild;						//index at which the childs are lying
		int rightChild;
	};

	//contains all nodes. All nodes are identified by their index in this vector.
	std::vector<Node> nodes;
	std::vector<int> attributes;
	std::vector<std::pair<float, float>> origBounds;
	float minBoundsRatio = .01f;
	int root;								//root index
	BoundsBehaviour adjustBounds;

	std::vector<uint32_t> getActiveIndices(std::vector<int>& attributes, std::vector<float*>& data, std::vector<uint32_t>& indices, std::vector<std::pair<float, float>>& bounds) {
		std::vector<uint32_t> res;
		for (uint32_t i : indices) {
			bool active = true;
			for (int atb = 0; atb < attributes.size(); ++atb) {
				if (data[i][attributes[atb]] < bounds[atb].first || data[i][attributes[atb]] > bounds[atb].second) {
					active = false;
					break;
				}
			}
			if (active) {
				res.push_back(i);
			}
		}
		return res;
	}

	std::vector<int> vectorUnion(std::vector<int>& a, std::vector<int>& b) {
		std::sort(a.begin(), a.end());
		std::sort(b.begin(), b.end());
		std::vector<int> res(a.size() + b.size());
		std::set_union(a.begin(), a.end(), b.begin(), b.end(), res.begin());
		res.erase(std::unique(res.begin(), res.end()), res.end());
		return res;
	};

	MultivariateGauss::MultivariateBrush calcMultivariateBrush(std::vector<int>& attributes, std::vector<float*>& data, std::vector<uint32_t>& indices) {
		std::vector<std::vector<double>> dataMatrix(indices.size(), std::vector<double>(attributes.size()));
		std::vector<std::vector<double>> covariance(attributes.size(), std::vector<double>(attributes.size(), 0));
		std::vector<std::vector<double>> invCov(attributes.size(), std::vector<double>(attributes.size(), 0));
		std::vector<double> mean(attributes.size(), 0);
		Eigen::MatrixXd cov(attributes.size(), attributes.size());
		std::vector<float> brushDiff(attributes.size());
		for (int i = 0; i < brushDiff.size(); ++i) brushDiff[i] = 1. / (origBounds[i].second - origBounds[i].first);
		for (int i = 0; i < indices.size(); ++i) {
			for (int j = 0; j < attributes.size(); ++j) {
				//normalizing the data to the brush ot have a more stabel covariance matrix
				dataMatrix[i][j] = (data[indices[i]][attributes[j]] - origBounds[j].first) * brushDiff[j];
			}
		}
		MultivariateGauss::compute_average_vector(dataMatrix, mean);
		Eigen::MatrixXd dataMat(dataMatrix.size(), dataMatrix[0].size());
		for (int i = 0; i < dataMatrix.size(); ++i) {
			for (int j = 0; j < dataMatrix[i].size(); ++j) {
				dataMat(i, j) = dataMatrix[i][j];
			}
		}
		Eigen::BDCSVD<Eigen::MatrixXd> svd(dataMat, Eigen::ComputeThinU | Eigen::ComputeThinV);
		MultivariateGauss::compute_covariance_matrix(dataMatrix, covariance);
		for (int i = 0; i < covariance.size(); ++i) {
			for (int j = 0; j < covariance[i].size(); ++j) {
				cov(i, j) = covariance[i][j];
			}
			covariance[i][i] += TINY;
		}
		MultivariateGauss::compute_matrix_inverse(covariance, invCov);
		MultivariateGauss::MultivariateBrush multBrush{};
		multBrush.mean = std::vector<float>(mean.size());
		multBrush.invCov = std::vector<std::vector<float>>(attributes.size(), std::vector<float>(attributes.size()));
		multBrush.cov = cov.inverse();
		std::vector<uint32_t> pcInd;
		Eigen::VectorXd singularVals = svd.singularValues() / std::sqrt(indices.size() - 1);
		for (int i = 0; i < singularVals.size(); ++i) {
			if (singularVals(i) > 1e-20) {
				pcInd.push_back(i);
			}
			else {
				std::pair<float, float> b(std::numeric_limits<float>::max(), std::numeric_limits<float>::min());
				for (int j = 0; j < indices.size(); ++j) {
					float v = 0;
					for (int k = 0; k < singularVals.size(); ++k) {
						v += data[indices[j]][attributes[k]] * svd.matrixV()(k, i);
					}
					if (v < b.first)b.first = v;
					if (v > b.second)b.second = v;
				}
				multBrush.pcBounds.push_back(b);
			}
		}
		multBrush.pcInd = pcInd;
		multBrush.pc = svd.matrixV();
		multBrush.sv = singularVals;
		for (int i = 0; i < mean.size(); ++i) multBrush.mean[i] = mean[i];
		for (int i = 0; i < invCov.size(); ++i) {
			for (int j = 0; j < invCov.size(); ++j) {
				multBrush.invCov[i][j] = invCov[i][j];
			}
		}
		MultivariateGauss::compute_matrix_determinant(covariance, multBrush.detCov);

		//checking the exponent values for each datum(datums are already mean centered after compute_average_vector)
		//for (auto& datum : dataMatrix) {
		//	float s = 0;
		//	for (int c = 0; c < datum.size(); ++c) {
		//		float m = 0;
		//		for (int c1 = 0; c1 < datum.size(); ++c1) {
		//			m += datum[c1] * multBrush.invCov[c][c1];
		//		}
		//
		//		s += datum[c] * m;
		//	}
		//	float x = s;
		//}

		return multBrush;
	}

	int buildRec(int split, std::vector<uint32_t>& indices, std::vector<float*>& data, std::vector<int>& attributes, std::vector<std::pair<float,float>>& bounds, int recDepth) {
		if (!indices.size() || !recDepth) return -1;
		Node n = {};
		n.bounds = bounds;
		n.split = split;
		n.rank = indices.size();
		int s2 = (split + 1) % attributes.size();

		//multivariate gauss calculation
		if(attributes.size() <= indices.size())
			n.multivariate = calcMultivariateBrush(attributes, data, indices);

		//check if bounds are too small already
		if ((bounds[split].second - bounds[split].first) / (origBounds[split].second - origBounds[split].first) <= minBoundsRatio) {
			n.rightChild = -1;
			n.leftChild = buildRec(s2, indices, data, attributes, bounds, recDepth - 1);
			nodes.push_back(n);
			return nodes.size() - 1;
		}

		//splitting the bounding box in the middle
		std::vector<std::pair<float, float>> leftBounds(bounds), rightBounds(bounds);
		float mid = (leftBounds[split].first + leftBounds[split].second) / 2;
		leftBounds[split].second = mid;
		rightBounds[split].first = mid;
		switch (adjustBounds) {
		case KdTree_Bounds_Static: break;
		case KdTree_Bounds_Pull_In_Outer_Border: 
			leftBounds[split].first = leftBounds[split].second;
			rightBounds[split].second = rightBounds[split].first;
			break;
		case KdTree_Bounds_Pull_In_Both_Borders:
			float tmp = leftBounds[split].first;
			leftBounds[split].first = leftBounds[split].second;
			leftBounds[split].second = tmp;
			tmp = rightBounds[split].first;
			rightBounds[split].first = rightBounds[split].second;
			rightBounds[split].second = tmp;
			break;
		}

		//assining the points to the left and right bounding box. also get the maximum extent of the points in both directions if wanted
		std::vector<uint32_t> leftPts, rightPts;
		for (int i : indices) {
			float val = data[i][attributes[split]];
			if (val < mid) {
				switch (adjustBounds) {
				case KdTree_Bounds_Static: break;
				case KdTree_Bounds_Pull_In_Outer_Border:
					if (val < leftBounds[split].first && val >= bounds[split].first) leftBounds[split].first = val;
					break;
				case KdTree_Bounds_Pull_In_Both_Borders:
					if (val < leftBounds[split].first && val >= bounds[split].first) leftBounds[split].first = val;
					if (val > leftBounds[split].second && val <= bounds[split].second) leftBounds[split].second = val;
					break;
				}
				if(val >= bounds[split].first)
					leftPts.push_back(i);
			}
			else {
				switch (adjustBounds) {
				case KdTree_Bounds_Static: break;
				case KdTree_Bounds_Pull_In_Outer_Border:
					if (val > rightBounds[split].second && val <= bounds[split].second) rightBounds[split].second = val;
					break;
				case KdTree_Bounds_Pull_In_Both_Borders:
					if (val > rightBounds[split].second && val <= bounds[split].second) rightBounds[split].second = val;
					if (val < rightBounds[split].first && val >= bounds[split].first) rightBounds[split].first = val;
					break;
				}
				if(val <= bounds[split].second)
					rightPts.push_back(i);
			}
		}
		
		//creating the childs recursiveley
		n.leftChild = buildRec(s2, leftPts, data, attributes, leftBounds, recDepth -1);
		n.rightChild = buildRec(s2, rightPts, data, attributes, rightBounds, recDepth -1);
		nodes.push_back(n);
		return nodes.size() - 1;
	};

	std::vector<std::vector<std::pair<float,float>>> getBoundsRec(Node& n, int recDepth, int minRank) {
		if (!recDepth) {
			std::vector<std::vector<std::pair<float,float>>> r;
			r.push_back(n.bounds);
			return r;
		}

		if (n.rank < minRank) return { };

		//getting the bounds from the left and right child and appending the vectors
		std::vector<std::vector<std::pair<float, float>>> left = (n.leftChild >= 0)?getBoundsRec(nodes[n.leftChild], recDepth - 1, minRank):std::vector<std::vector<std::pair<float, float>>>(), 
			right = (n.rightChild >= 0)?getBoundsRec(nodes[n.rightChild], recDepth - 1, minRank): std::vector<std::vector<std::pair<float, float>>>();
		left.reserve(left.size() + right.size());
		left.insert(left.end(), right.begin(), right.end());
		return left;
	};

	std::vector<MultivariateGauss::MultivariateBrush> getMulBrushesRec(Node& n, int recDepth, int minRank) {
		if (n.rank < minRank) return { };
		
		if (!recDepth) {
			std::vector<MultivariateGauss::MultivariateBrush> r;
			r.push_back(n.multivariate);
			return r;
		}

		std::vector<MultivariateGauss::MultivariateBrush> left = (n.leftChild >= 0) ? getMulBrushesRec(nodes[n.leftChild], recDepth - 1, minRank) : std::vector<MultivariateGauss::MultivariateBrush>(),
			right = (n.rightChild >= 0) ? getMulBrushesRec(nodes[n.rightChild], recDepth - 1, minRank) : std::vector<MultivariateGauss::MultivariateBrush>();
		left.reserve(left.size() + right.size());
		left.insert(left.end(), right.begin(), right.end());
		return left;
	}
};

#endif // !kd_tree_H