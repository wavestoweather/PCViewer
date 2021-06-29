#pragma once
#include<Eigen/Dense>
#include<vector>

class DBScan {
public:    
    DBScan(unsigned int minPts, float eps, const Eigen::MatrixXf& points, std::vector<std::vector<uint32_t>>& cluster, float* progress): 
    minPoints(minPts),
    epsilon(eps),
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

    const Eigen::MatrixXf& points;
    std::vector<int> pointCluster;
    std::vector<std::vector<uint32_t>>& cluster;
    std::vector<bool> visited;
    int maxCluster = 1;

    void execClustering(){
        int clusterID = 1;
        for(size_t i = 0; i < points.rows(); ++i){
            *progress = float(i) / points.rows();
            if(pointCluster[i] == ClusterId::Unclassified && expandCluster(i, clusterID)){
                ++clusterID;
            }
        }
        maxCluster = clusterID;
    }

    bool expandCluster(int p, int clusterID){
        std::vector<int> neighbours = getNeighbours(p);

        if(neighbours.size() < minPoints){
            pointCluster[p] = ClusterId::Noise;
            return false;
        }
        else{
            pointCluster[p] = clusterID;
            for(int n: neighbours){
                pointCluster[n] = clusterID;
            }

            for(size_t i = 0, n = neighbours.size(); i < n; ++i){
                std::vector<int> neigh = getNeighbours(neighbours[i]);

                if(neigh.size() >= minPoints){
                    for(int n2: neigh){
                        if(pointCluster[n2] == ClusterId::Unclassified || pointCluster[n2] == ClusterId::Noise){
                            if(!visited[n2]){
                                visited[n2] = true;
                                neighbours.push_back(n2);
                                n = neighbours.size();
                            }
                            pointCluster[n2] = clusterID;
                        }
                    }
                }
            }
        }
        return true;
    }

    //neighbours exclude the point for which neighbours are searched
    std::vector<int> getNeighbours(int p){
        std::vector<int> neighbours;
        const Eigen::RowVectorXf& point = points.row(p);
        for(int i = 0; i < points.rows(); ++i){
            if(i != p && (points.row(i) - point).norm() <= epsilon){
                neighbours.push_back(i);
            }
        }
        return neighbours;
    }
};