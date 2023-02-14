#pragma once
#include <vector>
#include <type_traits>
#include <array>
#include <cstdint>
#include <string_view>
#include "../range.hpp"

namespace deriveData{
// writeable memory view
template<class T>
class memory_view{
    T* _data{};
    size_t _size{};
public:
    memory_view(){};
    memory_view(T& d): _data(&d), _size(1){};
    memory_view(std::vector<T>& v): _data(v.data()), _size(v.size()){};
    //memory_view(const std::vector<std::remove_const<T>::type>& v): _data(v.data()), _size(v.size()){static_assert(std::is_const<T>::value);}
    template<size_t size>
    memory_view(std::array<T, size>& a): _data(a.data()), _size(size){};
    memory_view(T* data, size_t size): _data(data), _size(size){};
    template<class U>
    memory_view(memory_view<U> m): _data(reinterpret_cast<T*>(m.data())), _size(m.size() * sizeof(U) / sizeof(T)){
        assert(m.size() * sizeof(U) == _size * sizeof(T));   // debug assert to check if the memory views can be converted to each other, e.g. if the element sizes align
    }
    memory_view(const memory_view&) = default;
    memory_view(memory_view&&) = default;
    memory_view& operator=(const memory_view&) = default;
    memory_view& operator=(memory_view&&) = default;

    T* data(){return _data;};
    const T* data() const {return _data;};
    size_t size() const {return _size;};
    bool empty() const {return _size == 0;};
    T& operator[](size_t i){ assert(i < _size); return _data[i];}
    const T& operator[](size_t i) const {assert(i < _size); return _data[i];}
    bool operator==(const memory_view& o) const{ return _data == o._data && _size == o._size;}
    bool operator!=(const memory_view& o) const{ return !(*this == o);}
    operator bool() const {return _data && _size;};

    bool equalData(const memory_view& o) const{
        if(_size != o._size)
            return false;
        for(auto i: irange(static_cast<unsigned long>(_size))){
            if(_data[i] != o._data[i])
                return false;
        }
        return true;
    }

    T* begin() {return _data;};
    T* end() {return _data + _size;};
    const T* begin() const {return _data;};
    const T* end() const {return _data + _size;};
};

template<class T>
struct column_memory_view{ // holds one or more columns (done to also be able to hold vectors)
    std::vector<std::string_view> dimensionNames{};
    memory_view<uint32_t> dimensionSizes{};
    memory_view<uint32_t> columnDimensionIndices{};
    std::vector<memory_view<T>> cols{};

    column_memory_view() = default;
    column_memory_view(memory_view<uint32_t> sizes, memory_view<uint32_t> indices, std::vector<memory_view<T>> c): dimensionSizes(sizes), columnDimensionIndices(indices), cols(c){}
    column_memory_view(memory_view<T> data, memory_view<uint32_t> dimensionSizes = {}, memory_view<uint32_t> columnDimensionIndices = {}):
        dimensionSizes(dimensionSizes),
        columnDimensionIndices(columnDimensionIndices)
        {
            // checking for column or single row data in case of a constant
            if(dimensionSizes.empty()){  // row data
                for(int i: irange(static_cast<unsigned long>(data.size())))
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
        for(auto s: irange(static_cast<unsigned long>(dimensionSizes.size()))) ret *= dimensionSizes[s];
        return ret;
    }
    uint64_t columnSize() const{
        uint64_t ret{1};
        for(auto s: irange(static_cast<unsigned long>(columnDimensionIndices.size()))) ret *= dimensionSizes[columnDimensionIndices[s]];
        return ret;
    }
    // returns if the columns span all dimensions
    bool full() const{
        return size() == cols[0].size();
    }
    bool is_constant() const{
        return dimensionSizes.size() == 0;
    }

    bool operator==(const column_memory_view& o) const{
        return dimensionSizes == o.dimensionSizes && columnDimensionIndices == o.columnDimensionIndices && cols == o.cols;
    }
    bool equalData(const column_memory_view& o) const{
        if(cols.size() != o.cols.size())
            return false;
        if(!dimensionSizes.equalData(o.dimensionSizes))
            return false;
        if(!columnDimensionIndices.equalData(o.columnDimensionIndices))
            return false;
        for(auto c: irange(cols)){
            if(!cols[c].equalData(o.cols[c]))
                return false;
        }
        return true;
    }
    // only checks dimension Sizes
    bool equalDimensions(const column_memory_view& o) const{
        if(!dimensionSizes.equalData(o.dimensionSizes))
            return false;
        return true;
    }
    // only checks dimensionSizes and columnDimensionIndices for equality
    bool equalDataLayout(const column_memory_view& o) const{
        if(!dimensionSizes.equalData(o.dimensionSizes))
            return false;
        if(!columnDimensionIndices.equalData(o.columnDimensionIndices))
            return false;
        return true;
    }
    // check less equal
    bool dataLayoutLE(const column_memory_view& o) const{
        if(size() < o.size())
            return true;
        return false;
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

    std::vector<uint64_t> columnIndexToDimensionIndices(uint64_t index) const{
        std::vector<uint64_t> dimensionIndices(dimensionSizes.size());
        for(int i = static_cast<int>(columnDimensionIndices.size()) - 1; i >= 0; --i){
            uint32_t dim  = columnDimensionIndices[i];
            dimensionIndices[dim] = index % dimensionSizes[dim];
            index /= dimensionSizes[dim];
        }
        return dimensionIndices;
    }

    uint64_t dimensionIndicesToColumnIndex(const std::vector<uint64_t>& dimensionIndices) const{
        return dimensionIndex(dimensionIndices);
    }

    float atDimensionIndices(const std::vector<uint64_t>& indices, int col = 0) const{
        return cols[col][dimensionIndex(indices)];
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
        for(int i = static_cast<int>(dimensionSizes.size()) - 1; i >= 0; --i){
            dimensionIndices[i] = index % dimensionSizes[i];
            index /= dimensionSizes[i];
        }
        return dimensionIndex(dimensionIndices);
    }
}; 

template<typename T>
inline bool equalDataLayouts(const std::vector<column_memory_view<float>>& input){
    for(int i = 0; i < input.size() - 1; ++i)
        if(!input[i].equalDataLayout(input[i + 1]))
            return false;
    return true;
}
}