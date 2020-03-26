#ifndef kd_tree_H
#define kd_tree_H
#include <vector>

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

		bool done = false;
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
		PCUtil::matrixdump(curIndices);

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
				nodes.push_back(curNode);
				backKds.push_back(nodes.size() - 1);
			}
			if (curKds.size() & 1) {
				Node curNode{};
				curNode.leftChild = curKds.back();
				curNode.rightChild = -1;
				curNode.split = (nodes[curNode.leftChild].split - 1 + attributes.size()) % attributes.size();
				curNode.bounds = nodes[curNode.leftChild].bounds;
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
		return getBoundsRec(nodes[root], recursionDepth);
	};

private:
	struct Node {
		int split;
		std::vector<std::pair<float, float>> bounds;
		int leftChild;						//index at which the childs are lying
		int rightChild;
	};

	//contains all nodes. All nodes are identified by their index in this vector.
	std::vector<Node> nodes;
	std::vector<int> attributes;
	int root;								//root index
	BoundsBehaviour adjustBounds;

	int buildRec(int split, std::vector<uint32_t>& indices, std::vector<float*>& data, std::vector<int> attributes, std::vector<std::pair<float,float>>& bounds, int recDepth) {
		if (!indices.size() || !recDepth) return -1;
		Node n = {};
		n.bounds = bounds;
		n.split = split;

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
		int s2 = (split + 1) % attributes.size();
		n.leftChild = buildRec(s2, leftPts, data, attributes, leftBounds, recDepth -1);
		n.rightChild = buildRec(s2, rightPts, data, attributes, rightBounds, recDepth -1);
		nodes.push_back(n);
		return nodes.size() - 1;
	};

	std::vector<std::vector<std::pair<float,float>>> getBoundsRec(Node& n, int recDepth) {
		if (!recDepth) {
			std::vector<std::vector<std::pair<float,float>>> r;
			r.push_back(n.bounds);
			return r;
		}

		//getting the bounds from the left and right child and appending the vectors
		std::vector<std::vector<std::pair<float, float>>> left = (n.leftChild >= 0)?getBoundsRec(nodes[n.leftChild], recDepth - 1):std::vector<std::vector<std::pair<float, float>>>(), 
			right = (n.rightChild >= 0)?getBoundsRec(nodes[n.rightChild], recDepth - 1): std::vector<std::vector<std::pair<float, float>>>();
		left.reserve(left.size() + right.size());
		left.insert(left.end(), right.begin(), right.end());
		return left;
	};
};


#endif // !kd_tree_H