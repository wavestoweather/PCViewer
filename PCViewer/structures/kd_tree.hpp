#pragma once
#include <ranges.hpp>
#include <numeric>
#include <radix.hpp>
#include <memory_view.hpp>
#include <limits>
#include "../deriveData/MemoryView.hpp"

namespace structures{
class kd_tree{
public:
    using float_column_view = std::vector<deriveData::column_memory_view<float>>;

    bool same_layout{true};
    size_t data_size{};

    kd_tree(const float_column_view& points, util::memory_view<const uint32_t> indices = {}): points(points){
        for(size_t i: util::i_range(size_t(1), points.size() - 1))
            same_layout &= points[i - 1].equalDataLayout(points[i]);
            
        data_size = indices ? indices.size(): same_layout ? points[0].cols[0].size() : points[0].size();
        std::vector<size_t> ind;
        if(indices)
            ind = indices | util::to<std::vector<size_t>>();
        else
            ind = util::i_range(data_size) | util::to<std::vector<size_t>>();

        // tmp buffer only needed for tree building
        tmp.resize(data_size);
        root = build_tree(ind.begin(), ind.end(), 0);
        tmp = {};
    };

    std::vector<size_t> neighborhood(size_t p, const double& squaredRadius) const noexcept{
        return neighborhood_rec(root, p, squaredRadius, 0);
    };

    std::tuple<size_t, float> nearest_neighbour(size_t p) const noexcept{
        return nearest_neighbour_rec(root, p, 0);
    }
protected:
    struct KdNode{
        size_t dataIndex;
        size_t left, right;
    };
    std::vector<KdNode> nodes;
    size_t root;

    const float_column_view& points;
    const size_t noChild = size_t(-1);

    std::vector<size_t> tmp;
    size_t build_tree(const std::vector<size_t>::iterator& begin, const std::vector<size_t>::iterator& end, uint32_t level){
        assert(begin != end);     //if indices size is 0, something went wrong
        assert(begin < end);
        if(end - begin == 1){        //creating a leaf node
            nodes.push_back({*begin, noChild, noChild});
            return nodes.size() - 1;
        }
        //sorting the current indices
        radix::return_t<std::vector<size_t>::iterator> r;
        if(end - begin > 1 << 5){
            if(same_layout)
                r = radix::sort_indirect(begin, end, tmp.begin(), [&](size_t index){return points[level].cols[0][index];});
            else
                r = radix::sort_indirect(begin, end, tmp.begin(), [&](size_t index){return points[level](index, 0);});
            if(&*r.begin != &*begin)
                std::copy(r.begin, r.end, begin);
        }
        else{
            if(same_layout)
                std::sort(begin, end, [&](size_t a, size_t b){return points[level].cols[0][a] < points[level].cols[0][b];});
            else
                std::sort(begin, end, [&](size_t a, size_t b){return points[level](a, 0) < points[level](b, 0);});
        }
        
        auto lbegin = begin;
        auto lend = begin + (end - begin) / 2;
        auto rbegin = lend + 1;
        auto rend = end;

        size_t ret = nodes.size();
        nodes.push_back({*lend});
        nodes[ret].left = build_tree(lbegin, lend, (level + 1) % points.size());
        if(rbegin != rend)
            nodes[ret].right = build_tree(rbegin, rend, (level + 1) % points.size());
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

    std::vector<size_t> neighborhood_rec(size_t node, size_t p, const double& radius,const size_t& level) const{
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
            nbh = neighborhood_rec(section, p, radius, (level + 1) % points.size());
        result.insert(result.end(), nbh.begin(), nbh.end());
        if(dx2 < radius && other != noChild){   //cannot exclude the second branch
            nbh = neighborhood_rec(other, p, radius, (level + 1) % points.size());
            result.insert(result.end(), nbh.begin(), nbh.end());
        }

        return result;
    }

    std::tuple<size_t, float> nearest_neighbour_rec(size_t node, size_t p, int level) const {
        if(node == noChild)
            return {0, std::numeric_limits<float>::max()};
        const auto& cur_node = nodes[node];
        const auto cur_node_data_index = cur_node.dataIndex;
        float dist = square_distance(cur_node_data_index, p);
        size_t best;
        float best_dist;
        if(points[level].cols[0][p] < points[level].cols[0][cur_node_data_index])
            std::tie(best, best_dist) = nearest_neighbour_rec(cur_node.left, p, (level + 1) % points.size());
        else
            std::tie(best, best_dist) = nearest_neighbour_rec(cur_node.right, p, (level + 1) % points.size());

        if(dist < best_dist)
            return {cur_node_data_index, dist};
        else
            return {best, best_dist};
    }
};
}