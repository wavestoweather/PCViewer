#pragma once
#include<Eigen/Dense>
#include<vector>
#include<algorithm>

class DBScan {
public:    
    DBScan(unsigned int minPts, float eps, const Eigen::MatrixXf& points, std::vector<std::vector<uint32_t>>& cluster, float* progress): 
    minPoints(minPts),
    epsilon(eps * eps),
    tree(points),
    progress(progress),
    points(points),
    cluster(cluster)
    {
        pointCluster = std::vector<int>(points.rows(), ClusterId::Unclassified);
        visited = std::vector<bool>(points.rows(), false);
        execClustering();
        //converting to standard cluster datastructure
        cluster.resize(maxCluster);
        for(size_t i = 0; i < pointCluster.size(); ++i){
            if(pointCluster[i] == ClusterId::Noise) cluster.front().push_back(i);
            else cluster[pointCluster[i]].push_back(i);
        }
    }
    
    unsigned int minPoints;
    float epsilon;
    float* progress;
    
protected:
    enum ClusterId: int{
        Unclassified = -1,
        CorePoint = 1,
        BorderPoint = 2,
        Noise = -2
    };

    struct KdNode{
        size_t dataIndex;
        size_t left, right;
    };

    class KdTree{
    public:
        std::vector<KdNode> nodes;
        size_t root;
        KdTree(const Eigen::MatrixXf& points): points(points){
            std::vector<size_t> indices(points.rows());
            for(int i = 0; i < indices.size(); ++i) indices[i] = i;
            root = buildTree(indices.begin(), indices.end(), 0);
        };

        std::vector<size_t> neighborhood(const Eigen::RowVectorXf& p, const double& squaredRadius){
            return neighborhoodRec(root, p, squaredRadius, 0);
        };
    protected:
        const Eigen::MatrixXf& points;
        const size_t noChild = size_t(-1);
        size_t buildTree(const std::vector<size_t>::iterator& begin, const std::vector<size_t>::iterator& end, uint32_t level){
            assert(begin != end);     //if indices size is 0, something went wrong
            assert(begin < end);
            if(end - begin == 1){        //creating a leaf node
                nodes.push_back({*begin, noChild, noChild});
                return nodes.size() - 1;
            }
            //sorting the current indices
            std::sort(begin, end, [&](size_t left, size_t right){return points(left, level) < points(right, level);});
            auto lbegin = begin;
            auto lend = begin + (end - begin) / 2;
            auto rbegin = lend + 1;
            auto rend = end;

            size_t ret = nodes.size();
            nodes.push_back({*lend});
            nodes[ret].left = buildTree(lbegin, lend, (level + 1) % points.cols());
            if(rbegin != rend)
                nodes[ret].right = buildTree(rbegin, rend, (level + 1) % points.cols());
            else
                nodes[ret].right = noChild;

            return ret;
        }

        std::vector<size_t> neighborhoodRec(size_t node, const Eigen::RowVectorXf& p, const double& radius,const size_t& level){
            KdNode& curNode = nodes[node];
            if(curNode.left == noChild){
                if((points.row(curNode.dataIndex) - p).squaredNorm() < radius) return {curNode.dataIndex};
                else return {};
            }

            float d, dx, dx2;

            d = (points.row(curNode.dataIndex) - p).squaredNorm();
            dx = points(curNode.dataIndex, level) - p(level);
            dx2 = dx * dx;

            std::vector<size_t> result;
            if(d <= radius){
                result.push_back(curNode.dataIndex);
            }

            size_t section, other;
            if(dx > 0){
                section = curNode.left;
                other = curNode.right;
            }
            else{
                section = curNode.right;
                other = curNode.left;
            }

            std::vector<size_t>  nbh;
            if(section != noChild) 
                nbh = neighborhoodRec(section, p, radius, (level + 1) % points.cols());
            result.insert(result.end(), nbh.begin(), nbh.end());
            if(dx2 < radius && other != noChild){   //cannot exclude the second branch
                nbh = neighborhoodRec(other, p, radius, (level + 1) % points.cols());
                result.insert(result.end(), nbh.begin(), nbh.end());
            }

            return result;
        }
    }tree;

    const Eigen::MatrixXf& points;
    std::vector<int> pointCluster;
    std::vector<std::vector<uint32_t>>& cluster;
    std::vector<bool> visited;
    size_t visitedCounter = 0;
    int maxCluster = 1;

    void execClustering(){
        int clusterID = 1;
        for(size_t i = 0; i < points.rows(); ++i){
            if(pointCluster[i] == ClusterId::Unclassified && expandCluster(i, clusterID)){
                ++clusterID;
            }
        }
        maxCluster = clusterID;
    }

    bool expandCluster(int p, int clusterID){
        std::vector<size_t> neighbours = getNeighbours(p);

        if(neighbours.size() < minPoints){
            *progress = float(++visitedCounter) / points.rows();
            pointCluster[p] = ClusterId::Noise;
            return false;
        }
        else{
            pointCluster[p] = clusterID;
            *progress = float(++visitedCounter) / points.rows();
            for(size_t n: neighbours){
                pointCluster[n] = clusterID;
                *progress = float(++visitedCounter) / points.rows();
            }

            for(size_t i = 0, n = neighbours.size(); i < n; ++i){
                std::vector<size_t> neigh = getNeighbours(neighbours[i]);

                if(neigh.size() >= minPoints){
                    for(size_t n2: neigh){
                        if(pointCluster[n2] == ClusterId::Unclassified || pointCluster[n2] == ClusterId::Noise){
                            if(!visited[n2]){
                                visited[n2] = true;
                                neighbours.push_back(n2);
                                n = neighbours.size();
                            }
                            pointCluster[n2] = clusterID;
                            *progress = float(++visitedCounter) / points.rows();
                        }
                    }
                }
            }
        }
        return true;
    }

    //neighbours exclude the point for which neighbours are searched
    std::vector<size_t> getNeighbours(int p){
        //std::vector<int> neighbours;
        //const Eigen::RowVectorXf& point = points.row(p);
        //for(int i = 0; i < points.rows(); ++i){
        //    if(i != p && (points.row(i) - point).norm() <= epsilon){
        //        neighbours.push_back(i);
        //    }
        //}
        //return neighbours;

        return tree.neighborhood(points.row(p), epsilon);
    }
};