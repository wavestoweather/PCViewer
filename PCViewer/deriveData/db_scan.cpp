#include "db_scan.hpp"
#include <ranges.hpp>
#include <numeric>
#include <radix.hpp>

constexpr float cluster_unclassified = 0;
constexpr float cluster_noise = -1;

class KdTree{
public:
    bool same_layout{true};
    size_t data_size{};

    KdTree(const db_scan::float_column_view& points): points(points){
        for(size_t i: util::i_range(size_t(1), points.size() - 1))
            same_layout &= points[i - 1].equalDataLayout(points[i]);
            
        data_size = same_layout ? points[0].cols[0].size() : points[0].size();
        std::vector<size_t> indices(data_size);
        std::iota(indices.begin(), indices.end(), 0);

        // tmp buffer only needed for tree building
        tmp.resize(data_size);
        root = buildTree(indices.begin(), indices.end(), 0);
        tmp = {};
    };

    std::vector<size_t> neighborhood(size_t p, const double& squaredRadius) const{
        return neighborhoodRec(root, p, squaredRadius, 0);
    };
protected:
    struct KdNode{
        size_t dataIndex;
        size_t left, right;
    };
    std::vector<KdNode> nodes;
    size_t root;

    const db_scan::float_column_view& points;
    const size_t noChild = size_t(-1);

    std::vector<size_t> tmp;
    size_t buildTree(const std::vector<size_t>::iterator& begin, const std::vector<size_t>::iterator& end, uint32_t level){
        assert(begin != end);     //if indices size is 0, something went wrong
        assert(begin < end);
        if(end - begin == 1){        //creating a leaf node
            nodes.push_back({*begin, noChild, noChild});
            return nodes.size() - 1;
        }
        //sorting the current indices
        radix::return_t<std::vector<size_t>::iterator> r;
        if(same_layout)
            r = radix::sort_indirect(begin, end, tmp.begin(), [&](size_t index){return points[level].cols[0][index];});
        else
            r = radix::sort_indirect(begin, end, tmp.begin(), [&](size_t index){return points[level](index, 0);});
        if(&*r.begin != &*begin)
            std::copy(r.begin, r.end, begin);
        
        auto lbegin = begin;
        auto lend = begin + (end - begin) / 2;
        auto rbegin = lend + 1;
        auto rend = end;

        size_t ret = nodes.size();
        nodes.push_back({*lend});
        nodes[ret].left = buildTree(lbegin, lend, (level + 1) % points.size());
        if(rbegin != rend)
            nodes[ret].right = buildTree(rbegin, rend, (level + 1) % points.size());
        else
            nodes[ret].right = noChild;

        return ret;
    }

    double square_distance(size_t a, size_t b) const {
        double d{};
        if(same_layout){
            for(int i: util::size_range(points)){
                float v = points[i].cols[0][a] - points[i].cols[0][b];
                d += v * v;
            } 
        }
        else{
            for(int i: util::size_range(points)){
                float v = points[i](a, 0) - points[i](b, 0);
                d += v * v;
            }
        }
        return d;
    }

    std::vector<size_t> neighborhoodRec(size_t node, size_t p, const double& radius,const size_t& level) const{
        const KdNode& curNode = nodes[node];
        if(curNode.left == noChild){
            if(square_distance(curNode.dataIndex, p) < radius) return {curNode.dataIndex};
            else return {};
        }

        float d, dx, dx2;

        d = square_distance(curNode.dataIndex, p);
        if(same_layout)
            dx = points[level].cols[0][curNode.dataIndex] - points[level].cols[0][p];
        else    
            dx = points[level](curNode.dataIndex, 0) - points[level](p, 0);
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
            nbh = neighborhoodRec(section, p, radius, (level + 1) % points.size());
        result.insert(result.end(), nbh.begin(), nbh.end());
        if(dx2 < radius && other != noChild){   //cannot exclude the second branch
            nbh = neighborhoodRec(other, p, radius, (level + 1) % points.size());
            result.insert(result.end(), nbh.begin(), nbh.end());
        }

        return result;
    }
};

bool expandCluster(int cur_cluster_index ,size_t index, db_scan::float_column_view& output, const KdTree& kd_tree, const db_scan::db_scans_settings_t& settings){
    std::vector<size_t> neighbours = kd_tree.neighborhood(index, settings.epsilon);

    if(neighbours.size() < settings.min_points){
        if(kd_tree.same_layout) output[0].cols[0][index] = cluster_noise;
        else                    output[0](index, 0) = cluster_noise;
        return false;
    }
    else{
        if(kd_tree.same_layout) output[0].cols[0][index] = cur_cluster_index;
        else                    output[0](index, 0) = cur_cluster_index;
        for(size_t n: neighbours){
            if(kd_tree.same_layout) output[0].cols[0][n] = cur_cluster_index;
            else                    output[0](n, 0) = cur_cluster_index;
        }

        while(neighbours.size()){
            auto neigh = kd_tree.neighborhood(neighbours.back(), settings.epsilon);
            neighbours.pop_back();

            if(neigh.size() > settings.min_points){
                for(size_t n2: neigh){
                    if(kd_tree.same_layout && output[0].cols[0][n2] <= cluster_unclassified){
                        output[0].cols[0][n2] = cur_cluster_index;
                        neighbours.push_back(n2);
                    }
                    if(!kd_tree.same_layout && output[0](n2, 0) <= cluster_unclassified){
                        output[0](n2, 0) = cur_cluster_index;
                        neighbours.push_back(n2);
                    }
                }
            }
        }
    }
    return true;
}

void db_scan::run(const float_column_view& input, float_column_view& output, const db_scans_settings_t& settings){
    const KdTree tree(input);
    int cur_cluster_id{1};
    for(size_t i: util::i_range(tree.data_size)){
        if(tree.same_layout && output[0].cols[0][i] == cluster_unclassified && expandCluster(cur_cluster_id, i, output, tree, settings) ||
            !tree.same_layout && output[0](i, 0) == cluster_unclassified && expandCluster(cur_cluster_id, i, output, tree, settings))
            ++cur_cluster_id;
    }
    // thtats it
}