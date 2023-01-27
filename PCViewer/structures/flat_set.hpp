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
    flat_set(std::vector<T>&& e, bool sorted = false): _elements(std::move(e))  {if(_elements.size() > 1 && !sorted){if constexpr (!std::is_arithmetic_v<T>) std::sort(_elements.begin(), _elements.end()); else radix::sort(_elements);};}
    flat_set(const std::vector<T>& e, bool sorted = false): _elements(e)        {if(_elements.size() > 1 && !sorted) {if constexpr (!std::is_arithmetic_v<T>) std::sort(_elements.begin(), _elements.end()); else radix::sort(_elements);};}
    template<typename U>
    flat_set(U begin, U end, bool sorted = false): _elements(begin, end)        {if(_elements.size() > 1 && !sorted) {if constexpr (!std::is_arithmetic_v<T>) std::sort(_elements.begin(), _elements.end()); else radix::sort(_elements);};}

    const T* data() const           {return _elements.data();}
    size_t size() const             {return _elements.size();}
    decltype(_elements.begin()) begin(){return _elements.begin();}
    decltype(_elements.end())   end()  {return _elements.end();}
    decltype(_elements.cbegin()) begin() const {return _elements.cbegin();}
    decltype(_elements.cend())   end() const {return _elements.cend();}
    bool contains(const T& v) const {return std::binary_search(_elements.begin(), _elements.end(), v);}
    void erase(const T& v)          {if(contains(v)) _elements.erase(std::find(_elements.begin(), _elements.end(), v));}

    flat_set<T>& operator&=(const flat_set<T>& o) {std::vector<T> res(_elements.size() + o._elements.size()); auto it = std::set_intersection(_elements.begin(), _elements.end(), o._elements.begin(), o._elements.end(), res.begin()); res.resize(it - res.begin()); _elements = std::move(res); return *this;}
    flat_set<T>& operator|=(const flat_set<T>& o) {std::vector<T> res(_elements.size() + o._elements.size()); auto it = std::set_union(_elements.begin(), _elements.end(), o._elements.begin(), o._elements.end(), res.begin()); res.resize(it - res.begin()); _elements = std::move(res); return *this;}
    flat_set<T>& operator/=(const flat_set<T>& o) {std::vector<T> res(_elements.size() + o._elements.size()); auto it = std::set_difference(_elements.begin(), _elements.end(), o._elements.begin(), o._elements.end(), res.begin()); res.resize(it - res.begin()); _elements = std::move(res); return *this;}
    flat_set<T>& operator^=(const flat_set<T>& o) {std::vector<T> res(_elements.size() + o._elements.size()); auto it = std::set_symmetric_difference(_elements.begin(), _elements.end(), o._elements.begin(), o._elements.end(), res.begin()); res.resize(it - res.begin()); _elements = std::move(res); return *this;}

    flat_set<T>& operator|=(const T& o) {std::vector<T> res(_elements.size() + 1); auto it = std::set_union(_elements.begin(), _elements.end(), &o, (&o) + 1, res.begin()); res.resize(it - res.begin()); _elements = std::move(res); return *this;}

    flat_set<T> operator&(const flat_set<T>& o) const {flat_set<T> res(_elements.size() + o._elements.size()); auto it = std::set_intersection(_elements.begin(), _elements.end(), o._elements.begin(), o._elements.end(), res._elements.begin()); res._elements.resize(it - res.begin()); return res;}
    flat_set<T> operator|(const flat_set<T>& o) const {flat_set<T> res(_elements.size() + o._elements.size()); auto it = std::set_union(_elements.begin(), _elements.end(), o._elements.begin(), o._elements.end(), res._elements.begin()); res._elements.resize(it - res.begin()); return res;}
    flat_set<T> operator/(const flat_set<T>& o) const {flat_set<T> res(_elements.size() + o._elements.size()); auto it = std::set_difference(_elements.begin(), _elements.end(), o._elements.begin(), o._elements.end(), res._elements.begin()); res._elements.resize(it - res.begin()); return res;}
    flat_set<T> operator^(const flat_set<T>& o) const {flat_set<T> res(_elements.size() + o._elements.size()); auto it = std::set_symmetric_difference(_elements.begin(), _elements.end(), o._elements.begin(), o._elements.end(), res._elements.begin()); res._elements.resize(it - res.begin()); return res;}

    bool operator==(const flat_set<T>& o) const {return _elements == o._elements;}
};
}