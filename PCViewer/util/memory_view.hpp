#pragma once
#include <vector>
#include <type_traits>
#include <ranges.hpp>
#include <iostream>
#include <array>
#include <std_util.hpp>
#include <functional>

namespace util{
// writeable memory view
template<class T = uint32_t>
class memory_view{
    T* _data{};
    size_t _size{};
public:
    static constexpr size_t n_pos{size_t(-1)};

    memory_view(){};
    memory_view(T& d): _data(&d), _size(1){};
    memory_view(std::vector<T>& v): _data(v.data()), _size(v.size()){};
    memory_view(const std::vector<typename std::remove_const<T>::type>& v): _data(v.data()), _size(v.size()){};
    template<size_t size>
    memory_view(std::array<T, size>& a): _data(a.data()), _size(size){};
    template<size_t size>
    memory_view(const std::array<typename std::remove_const<T>::type, size>& a): _data(a.data()), _size(a.size()){};
    memory_view(T* data, size_t size): _data(data), _size(size){};
    template<class U>
    memory_view(memory_view<U> m): _data(reinterpret_cast<T*>(m.data())), _size(m.size() * sizeof(U) / sizeof(T)){
        static_assert(sizeof(U) % sizeof(T) == 0 || sizeof(T) % sizeof(U) == 0);   // debug assert to check if the memory views can be converted to each other, e.g. if the element sizes align
    }
    memory_view(std::initializer_list<typename std::remove_const<T>::type> l): _data(l.begin()), _size(l.size()){}
    memory_view(const memory_view&) = default;
    memory_view(memory_view&&) = default;
    memory_view& operator=(const memory_view&) = default;
    memory_view& operator=(memory_view&&) = default;

    T* data()               {return _data;};
    const T* data() const   {return _data;};
    size_t size() const     {return _size;};
    size_t byte_size() const {return _size * sizeof(T);};
    bool empty() const      {return _size == 0;};
    T& operator[](size_t i){
        assert(i < _size);   // debug assert for in bounds check
        return _data[i];
    }
    const T& operator[](size_t i) const{
        assert(i < _size);
        return _data[i];
    }
    T* operator->(){
        assert(_size);
        return _data;
    }
    const T* operator->() const {
        assert(_size);
        return _data;
    }
    T& operator*(){
        assert(_size);
        return *_data;
    }
    bool operator==(const memory_view& o) const{
        return _data == o._data && _size == o._size;
    }
    operator bool() const {return _data && _size;};

    bool equal_data(const memory_view& o) const{
        if(_size != o._size)
            return false;
        for(auto i: util::i_range(_size)){
            if(_data[i] != o._data[i])
                return false;
        }
        return true;
    }

    size_t data_hash() const{
        size_t seed{};
        std::hash<typename std::remove_const<T>::type> hasher;
        for(T* b = begin(); b != end(); ++b)
            seed = std::hash_combine(seed, hasher(*b));
        return seed;
    }

    T* find(const T& e){
        for(auto &el: *this)
            if(el == e)
                return &el;
        return end();
    }
    const T* find(const T& e) const {
        return find(e);
    }

    T& find(std::function<bool(const T& e)> f){
        for(auto &e: *this)
            if(f(e))
                return e;
        throw std::runtime_error{"util::memory_view::find() Element does not exist."};
    }
    const T& find(std::function<bool(const T& e)> f) const{
        return find(f);
    }
    bool contains(const T& t) const{
        for(const auto& e: *this)
            if(e == t)
                return true;
        return false;
    }
    template<typename F>
    bool contains(F f) const {
        for(const auto& e: *this)
            if(f(e))
                return true;
        return false;
    }
    size_t index_of(const T& t) const{
        for(size_t i: i_range(_size))
            if(_data[i] == t)
                return i;
        return n_pos;
    }
    template<typename f>
    size_t index_of(f functor){
        for(size_t i: i_range(_size))
            if(functor(_data[i]))
                return i;
        return n_pos;
    }

    T& front(){
        return *_data;
    }
    const T& front() const{
        return *_data;
    }
    T& back(){
        return *(_data + _size - 1);
    }
    const T& back() const{
        return *(_data + _size - 1);
    }

    T* begin() {return _data;};
    T* end() {return _data + _size;};
    const T* begin() const {return _data;};
    const T* end() const {return _data + _size;};

};

template<class T>
struct column_memory_view{ // holds one or more columns (done to also be able to hold vectors)
    memory_view<uint32_t> dimensionSizes{};
    memory_view<uint32_t> columnDimensionIndices{};
    std::vector<memory_view<T>> cols{};

    column_memory_view() = default;
    column_memory_view(memory_view<T> data, memory_view<uint32_t> dimensionSizes = {}, memory_view<uint32_t> columnDimensionIndices = {}):
        dimensionSizes(dimensionSizes),
        columnDimensionIndices(columnDimensionIndices)
        {
            // checking for column or single row data in case of a constant
            if(dimensionSizes.empty()){  // row data
                for(int i: util::size_range(data))
                    cols.push_back(memory_view(data.data() + i, 1));            
            }
            else{
                cols = {data};
            }
        };
    column_memory_view(std::vector<memory_view<T>> dataVec, memory_view<uint32_t> dimensionSizes = {}, memory_view<uint32_t> columnDimensionIndices = {}):
        dimensionSizes(dimensionSizes),
        columnDimensionIndices(columnDimensionIndices),
        cols(dataVec){};

    
    // returns the amount of elements in this column_memory_view
    // Note: diemnsionSizes.empty() indicates a constant in which case the size = 1
    uint64_t size() const{
        uint64_t ret{1};
        for(auto s: util::size_range(dimensionSizes)) ret *= dimensionSizes[s];
        return ret;
    }
    uint64_t column_size() const{
        uint64_t ret{1};
        for(auto s: util::size_range(columnDimensionIndices)) ret *= dimensionSizes[columnDimensionIndices[s]];
        return ret;
    }
    // returns if the columns span all dimensions
    bool full() const{
        return size() == cols[0].size();
    }

    bool operator==(const column_memory_view& o) const{
        return dimensionSizes == o.dimensionSizes && columnDimensionIndices == o.columnDimensionIndices && cols == o.cols;
    }
    bool equal_data(const column_memory_view& o) const{
        if(cols.size() != o.cols.size())
            return false;
        if(!dimensionSizes.equal_data(o.dimensionSizes))
            return false;
        if(!columnDimensionIndices.equal_data(o.columnDimensionIndices))
            return false;
        for(auto c: util::size_range(cols)){
            if(!cols[c].equal_data(o.cols[c]))
                return false;
        }
        return true;
    }
    // only checks dimension Sizes
    bool equal_dimensions(const column_memory_view& o) const{
        if(!dimensionSizes.equal_data(o.dimensionSizes))
            return false;
        return true;
    }
    // only checks dimensionSizes and columnDimensionIndices for equality
    bool equal_data_layout(const column_memory_view& o) const{
        if(!dimensionSizes.equal_data(o.dimensionSizes))
            return false;
        if(!columnDimensionIndices.equal_data(o.columnDimensionIndices))
            return false;
        return true;
    }

    T& operator()(uint64_t index, uint32_t column){
        auto cI = columnIndex(index);
        assert(cI < cols[column].size());
        return cols[column][cI];
    }
    const T& operator()(uint64_t index, uint32_t column) const{
        auto cI = columnIndex(index);
        assert(cI < cols[column].size());
        return cols[column][cI];
    }

    operator bool() const{ return cols.size();};
private:
    uint64_t dimensionIndex(const std::vector<uint64_t>& dimensionIndices) const{
        uint32_t columnIndex = 0;
        for(int d = 0; d < columnDimensionIndices.size(); ++d){
            uint32_t factor = 1;
            for(int i = d + 1; i < columnDimensionIndices.size(); ++i){
                factor *= dimensionSizes[columnDimensionIndices[i]];
            }
            columnIndex += factor * dimensionIndices[columnDimensionIndices[d]];
        }
        return columnIndex;
    }
    uint64_t columnIndex(uint64_t index) const{
        std::vector<uint64_t> dimensionIndices(dimensionSizes.size());
        for(int i = dimensionSizes.size() - 1; i >= 0; --i){
            dimensionIndices[i] = index % dimensionSizes[i];
            index /= dimensionSizes[i];
        }
        return dimensionIndex(dimensionIndices);
    }
};  
}

template<class T>
std::ostream& operator<<(std::ostream &stream, util::memory_view<T> var) {
    stream << "[ ";
    for(int i: util::size_range(var)){
        stream << var[i] << ", ";
    }
    return stream << "]";
}