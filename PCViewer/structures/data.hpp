#pragma once

#include<vector>
#include<set>
#include<stdint.h>
#include<algorithm>
#include<cassert>
#include<limits>
#include<array_struct.hpp>

/*  This class holds the data in optimized format
*
*   The data is stored dimension bound, meaning that for each attribute/column
*   it is stored for which dimension it is defined
*
*   A dimension is simply a size, describin how large a dimension is.
+
+   The overall size of a dataset is the multiplication of all dimensions
*
*   The data is stored in column format, allowing also for each collumn
*   to be dpendant on only a subset of the dimensions. The fastest varying dimension is the last
*   in the dependant dimensions array.
*
*   The class provides a function to save a linearized packed form of the data at a pointer position.
*   The required size for the array at the pointer can be querried with the packedByteSize function
*
*   The packed header layout is the follwoing:
*   float dimensionCount
*   float columnCount
*   float dimension_sizes[dimensionCount]
*   float column_dimensionsCounts[columnCount]
*   float columnDimensionOffsets[columnCount]           //points to the index where the column dimension struct begins, offset in float array offset
*   float dataOffsets[columnCount]                      //poitns to the index where the data for each column begins
*   ColumnDimension column_dimensions[columnCount]
*       ---- ColumnDimension:
*               float column_dimensions[columnDimensionCount]
*/  
namespace structures{
template<class T = float>
class data{
    public:
    
    std::vector<uint32_t> dimension_sizes;
    std::vector<std::vector<uint32_t>> column_dimensions;        // for constant columns their corresponding vector here is empty
    std::vector<std::vector<T>> columns;

    data(){};
    // suggested way is to use default constructor and directly fill the vectors. Thus no copy constructor for the vectors is invoked
    data(const std::vector<uint32_t>& dimension_sizes,const std::vector<std::vector<uint32_t>>& column_dimensions,const std::vector<std::vector<T>>& columns):
        dimension_sizes(dimension_sizes), column_dimensions(column_dimensions), columns(columns){};

    void clear() {
        dimension_sizes.clear();
        column_dimensions.clear();
        columns.clear();
    }

    uint64_t size() const{
        if(dimension_sizes.empty()) return 0;
        uint64_t ret = 1;
        for(int i: dimension_sizes) ret *= i;
        return ret;
    };

    // access data by an index \in[0, cross(dimension_sizes)] and a column
    T& operator()(uint32_t index, uint32_t column){
        uint64_t colIndex = columnIndex(index, column);
        return columns[column][colIndex];
    }
    // const data access
    const T& operator()(uint32_t index, uint32_t column) const{
        uint64_t colIndex = columnIndex(index, column);
        return columns[column][colIndex];
    }
    // access data by dimension indices
    T& operator()(const std::vector<uint32_t>& dimensionIndices, uint32_t column){
        return columns[column][this->index(dimensionIndices, column)];
    }
    // const access data by dimension indices
    const T& operator()(const std::vector<uint32_t>& dimensionIndices, uint32_t column) const {
        return columns[column][this->index(dimensionIndices, column)];
    }
    // converts data index to column index (needed for non full dimensional columns)
    uint64_t columnIndex(uint32_t index, uint32_t column) const{
        std::vector<uint32_t> dimensionIndices(dimension_sizes.size());
        for(int i = dimension_sizes.size() - 1; i >= 0; --i){
            dimensionIndices[i] = index % dimension_sizes[i];
            index /= dimension_sizes[i];
        }
        return this->index(dimensionIndices, column);
    }

    // shrinks all vectors to fit the data and removes unused dimensions to avoid unnesecary index accessing
    void compress(){
        dimension_sizes.shrink_to_fit();
        column_dimensions.shrink_to_fit();
        for(auto& column: column_dimensions) column.shrink_to_fit();
        columns.shrink_to_fit();
        for(auto& column: columns) column.shrink_to_fit();
        removeUnusedDimension();
    }

    // subsample a dimension
    void subsampleTrim(const std::vector<uint32_t>& samplingRates, const std::vector<std::pair<uint32_t, uint32_t>>& trimIndices){    
        std::vector<uint32_t> normalSampling(samplingRates.size(), 1);
        std::vector<std::pair<uint32_t, uint32_t>> noTrim;
        for(auto dim: dimension_sizes) noTrim.push_back({0, dim});
        if(samplingRates == normalSampling && trimIndices == noTrim) return;        //nothing has to be done

        std::vector<uint32_t> reducedDimensions(dimension_sizes.size());
        for(int d = 0; d < dimension_sizes.size(); ++d){
            reducedDimensions[d] = trimIndices[d].second - trimIndices[d].first;
            reducedDimensions[d] += samplingRates[d] - 1;
            reducedDimensions[d] /= samplingRates[d];
        }

        for(int c = 0; c < columns.size(); ++c){
            bool trimmedSubsampled = false; 
            for(auto dim: column_dimensions[c]){
                if(samplingRates[dim] != 1 || trimIndices[dim] != std::pair<uint32_t, uint32_t>(0, dimension_sizes[dim])){
                    trimmedSubsampled = true;
                    break;
                }
            }
            if(!trimmedSubsampled) continue;            //discarding columns which dont depend on any subsampled/trimmed dimension

            std::vector<uint32_t> redDimIndices(column_dimensions[c].size(), 0);
            std::vector<uint32_t> redDimIncrements(column_dimensions[c].size(), 1);
            std::vector<uint32_t> redDimStarts(column_dimensions[c].size());
            std::vector<uint32_t> redDimStops(column_dimensions[c].size());
            for(int cd = 0; cd < column_dimensions[c].size(); ++cd){
                redDimIndices[cd] = trimIndices[column_dimensions[c][cd]].first;
                redDimIncrements[cd] = samplingRates[column_dimensions[c][cd]];
                redDimStarts[cd] = redDimIndices[cd];
                redDimStops[cd] = trimIndices[column_dimensions[c][cd]].second;
            }
            uint32_t redColumnSize = 1;
            for(auto d: column_dimensions[c]) redColumnSize *= reducedDimensions[d];
            std::vector<T> redData(redColumnSize);
            uint32_t redDataCur = 0;
            while(redDimIndices[0] < redDimStops[0]){
                //copy value
                redData[redDataCur++] = columns[c][indexReducedDimIndices(redDimIndices, c)];
                //increase dimension itertor
                redDimIndices.back() += redDimIncrements.back();
                for(int d = redDimIndices.size() - 1; d > 0; --d){
                    if(redDimIndices[d] >= redDimStops[d]){
                        redDimIndices[d] = redDimStarts[d];
                        redDimIndices[d - 1] += redDimIncrements[d -1];
                    }
                }
            }
            assert(redDataCur == redColumnSize);
            columns[c] = redData;
        }

        dimension_sizes = reducedDimensions;
    }

    // remove dimension by slicing at one index
    void removeDim(uint32_t dimension, uint32_t slice){
        // currently uses lazy implementation based on subsampleTrim, should be made more efficient
        std::vector<uint32_t> samplingRates(dimension_sizes.size(), 1);   //no subsampling
        std::vector<std::pair<uint32_t, uint32_t>> trimIndices(dimension_sizes.size());
        for(int d = 0; d < trimIndices.size(); ++d){
            if(d == dimension) trimIndices[d] = {slice, slice + 1};
            else trimIndices[d] = {0, dimension_sizes[d]};
        }
        subsampleTrim(samplingRates, trimIndices);

        //removing the dimension, updating column dimensions(indices might have to be decremented)
        dimension_sizes.erase(dimension_sizes.begin() + dimension);
        for(int c = 0; c < column_dimensions.size(); ++c){
            int dimIndex = -1;
            for(int d = 0; d < column_dimensions[c].size(); ++d){
                int dim = column_dimensions[c][d];
                if(dim == dimension) dimIndex = d;
                if(dim > dimension) --column_dimensions[c][d];
            }
            if(dimIndex >= 0)
                column_dimensions[c].erase(column_dimensions[c].begin() + dimIndex);
        }
    }

    void removeColumn(uint32_t column){
        columns.erase(columns.begin() + column);
        column_dimensions.erase(column_dimensions.begin() + column);
    }

    void removeUnusedDimension(){
        std::set<uint32_t> usedDims;
        for(auto& columnDims: column_dimensions) for(auto& dim: columnDims) usedDims.insert(dim);
        std::vector<uint32_t> unusedDims;
        if(usedDims.size() < dimension_sizes.size()){  //unused dims exist
            for(int d = 0; d < dimension_sizes.size(); ++d){
                if(usedDims.find(d) == usedDims.end())
                    unusedDims.push_back(d);
            }
            for(int c = 0; c < column_dimensions.size(); ++c){
                for(int d = 0; d < column_dimensions[c].size(); ++d){
                    int decrement = 0;
                    for(auto unused: unusedDims){
                        if(column_dimensions[c][d] > unused) ++decrement;
                    }
                    column_dimensions[c][d] -= decrement;
                }
            }
        }
    }

    void linearizeColumn(int column){
        std::set<float> elements(columns[column].begin(), columns[column].end());
        float min = *elements.begin();
        float max = *elements.rbegin();
        for(int d = 0; d < columns[column].size(); ++d){
            int i = std::distance(elements.begin(), elements.find(columns[column][d]));
            float alpha = i / float(elements.size() - 1);
            columns[column][d] = (1 - alpha) * min + alpha * max;
        }
    }

private:
    // returns the index for a column given the diemension indices
    uint32_t index(const std::vector<uint32_t>& dimensionIndices, uint32_t column) const{
        uint32_t columnIndex = 0;
        for(int d = 0; d < column_dimensions[column].size(); ++d){
            uint32_t factor = 1;
            for(int i = d + 1; i < column_dimensions[column].size(); ++i){
                factor *= dimension_sizes[column_dimensions[column][i]];
            }
            columnIndex += factor * dimensionIndices[column_dimensions[column][d]];
        }
        return columnIndex;
    }

    //  returns the index for a column given the dimension indices. Dimension indices only include indices for the current dimension, no mapping needed
    uint32_t indexReducedDimIndices(const std::vector<uint32_t>& dimensionIndices, uint32_t column) const{
        uint32_t columnIndex = 0;
        for(int d = 0; d < column_dimensions[column].size(); ++d){
            uint32_t factor = 1;
            for(int i = d + 1; i < column_dimensions[column].size(); ++i){
                factor *= dimension_sizes[column_dimensions[column][i]];
            }
            columnIndex += factor * dimensionIndices[d];
        }
        return columnIndex;
    }
};
}