#ifndef kd_tree_H
#define kd_tree_H
#include <vector>

class KdTree {
public:

	KdTree() {};
	KdTree(std::vector<int>& indices, std::vector<float*>& data, std::vector<int>& attributes, std::vector<std::pair<float, float>> initialBounds, int recursionDepth, bool adjustBounds) {
		//building the kd tree;
		this->adjustBounds = adjustBounds;
		this->attributes = attributes;
		nodes.reserve(pow(2,recursionDepth + 1)-1);
		root = buildRec(0, indices, data, attributes, initialBounds, recursionDepth);
	};
	~KdTree() {};

	std::vector<std::vector<std::pair<float, float>>> getBounds(int recursionDepth) {
		return getBoundsRec(root, recursionDepth);
	};

private:
	struct Node {
		int split;
		std::vector<std::pair<float, float>> bounds;
		Node* leftChild;
		Node* rightChild;
	};

	//contains all nodes
	std::vector<Node> nodes;
	std::vector<int> attributes;
	Node* root;
	bool adjustBounds;

	Node* buildRec(int split, std::vector<int>& indices, std::vector<float*>& data, std::vector<int> attributes, std::vector<std::pair<float,float>>& bounds, int recDepth) {
		if (!indices.size() || !recDepth) return nullptr;
		Node n = {};
		n.bounds = bounds;
		n.split = split;

		//splitting the bounding box in the middle
		std::vector<std::pair<float, float>> leftBounds(bounds), rightBounds(bounds);
		leftBounds[split].second = (leftBounds[split].first + leftBounds[split].second) / 2;
		rightBounds[split].first = leftBounds[split].second;

		//assining the points to the left and right bounding box. also get the maximum extent of the points in both directions if wanted
		float left = leftBounds[split].second, right = leftBounds[split].second;
		std::vector<int> leftPts, rightPts;
		for (int i : indices) {
			float val = data[i][attributes[split]];
			if (val < leftBounds[split].second) {
				if (val < left) left = val;
				leftPts.push_back(i);
			}
			else {
				if (val > right) right = val;
				rightPts.push_back(i);
			}
		}
		if (adjustBounds) {
			leftBounds[split].first = left;
			rightBounds[split].second = right;
		}
		
		//creating the childs recursiveley
		int s2 = (split + 1) % attributes.size();
		n.leftChild = buildRec(s2, leftPts, data, attributes, leftBounds, recDepth -1);
		n.rightChild = buildRec(s2, rightPts, data, attributes, rightBounds, recDepth -1);
		nodes.push_back(n);
		return &nodes.back();
	};

	std::vector<std::vector<std::pair<float,float>>> getBoundsRec(Node* n, int recDepth) {
		if (!recDepth) {
			std::vector<std::vector<std::pair<float,float>>> r;
			r.push_back(n->bounds);
			return r;
		}

		//getting the bounds from the left and right child and appending the vectors
		std::vector<std::vector<std::pair<float, float>>> left = (n->leftChild)?getBoundsRec(n->leftChild, recDepth - 1):std::vector<std::vector<std::pair<float, float>>>(), right = (n->rightChild)?getBoundsRec(n->rightChild, recDepth - 1): std::vector<std::vector<std::pair<float, float>>>();
		left.reserve(left.size() + right.size());
		left.insert(left.end(), right.begin(), right.end());
		return left;
	};
};


#endif // !kd_tree_H
