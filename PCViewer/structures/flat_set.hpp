#pragma once
#include <vector>
#include <algorithm>
#include <radix.hpp>

namespace structures{
template<typename T>
class flat_set{
    std::vector<T> _elements{};

public:
    flat_set() = default;
    flat_set(size_t size): _elements(size) {}
    flat_set(std::vector<T>&& e): _elements(std::move(e)) {if(_elements.size() > 1){if constexpr (!std::is_arithmetic_v<T>) std::sort(_elements.begin(), _elements.end()); else radix::sort(_elements);};}
    flat_set(const std::vector<T>& e): _elements(e) {if(_elements.size() > 1) {if constexpr (!std::is_arithmetic_v<T>) std::sort(_elements.begin(), _elements.end()); else radix::sort(_elements);};}
    template<typename U>
    flat_set(U begin, U end): _elements(begin, end) {if(_elements.size() > 1) {if constexpr (!std::is_arithmetic_v<T>) std::sort(_elements.begin(), _elements.end()); else radix::sort(_elements);};}

    size_t size() const             {return _elements.size();}
    decltype(_elements.begin()) begin(){return _elements.begin();}
    decltype(_elements.end())   end()  {return _elements.end();}
    bool contains(const T& v) const {return std::binary_search(_elements.begin(), _elements.end(), v);}

    flat_set<T>& operator&=(const flat_set<T>& o) {std::vector<T> res(_elements.size() + o._elements.size()); auto it = std::set_intersection(_elements.begin(), _elements.end(), o._elements.begin(), o._elements.end(), res.begin()); res.resize(it - res.begin()); _elements = std::move(res); return *this;}
    flat_set<T>& operator|=(const flat_set<T>& o) {std::vector<T> res(_elements.size() + o._elements.size()); auto it = std::set_union(_elements.begin(), _elements.end(), o._elements.begin(), o._elements.end(), res.begin()); res.resize(it - res.begin()); _elements = std::move(res); return *this;}
    flat_set<T>& operator/=(const flat_set<T>& o) {std::vector<T> res(_elements.size() + o._elements.size()); auto it = std::set_difference(_elements.begin(), _elements.end(), o._elements.begin(), o._elements.end(), res.begin()); res.resize(it - res.begin()); _elements = std::move(res); return *this;}
    flat_set<T>& operator^=(const flat_set<T>& o) {std::vector<T> res(_elements.size() + o._elements.size()); auto it = std::set_symmetric_difference(_elements.begin(), _elements.end(), o._elements.begin(), o._elements.end(), res.begin()); res.resize(it - res.begin()); _elements = std::move(res); return *this;}

    flat_set<T> operator&(const flat_set<T>& o) const {flat_set<T> res(_elements.size() + o._elements.size()); auto it = std::set_intersection(_elements.begin(), _elements.end(), o._elements.begin(), o._elements.end(), res._elements.begin()); res._elements.resize(it - res.begin()); return res;}
    flat_set<T> operator|(const flat_set<T>& o) const {flat_set<T> res(_elements.size() + o._elements.size()); auto it = std::set_union(_elements.begin(), _elements.end(), o._elements.begin(), o._elements.end(), res._elements.begin()); res._elements.resize(it - res.begin()); return res;}
    flat_set<T> operator/(const flat_set<T>& o) const {flat_set<T> res(_elements.size() + o._elements.size()); auto it = std::set_difference(_elements.begin(), _elements.end(), o._elements.begin(), o._elements.end(), res._elements.begin()); res._elements.resize(it - res.begin()); return res;}
    flat_set<T> operator^(const flat_set<T>& o) const {flat_set<T> res(_elements.size() + o._elements.size()); auto it = std::set_symmetric_difference(_elements.begin(), _elements.end(), o._elements.begin(), o._elements.end(), res._elements.begin()); res._elements.resize(it - res.begin()); return res;}
};
}