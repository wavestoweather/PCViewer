#pragma once

#include<vector>
#include<set>
#include<stdint.h>
#include<algorithm>
#include<cassert>
#include<limits>

/*  This class holds the data in optimized format
*
*   The data is stored dimension bound, meaning that for each attribute/column
*   it is stored for which dimension it is defined
*
*   A dimension is simply an size, describin how large a dimension is.
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
*   float dimensionSizes[dimensionCount]
*   float columnDimensionsCounts[columnCount]
*   float columnDimensionOffsets[columnCount]           //points to the index where the column dimension struct begins, offset in float array offset
*   float dataOffsets[columnCount]                      //poitns to the index where the data for each column begins
*   ColumnDimension columnDimensions[columnCount]
*       ---- ColumnDimension:
*               float columnDimensions[columnDimensionCount]
*/  
class Data{
    public:
    
    std::vector<uint32_t> dimensionSizes;
    std::vector<std::vector<uint32_t>> columnDimensions;        // for constant columns their corresponding vector here is empty
    std::vector<std::vector<float>> columns;

    Data(){};
    // suggested way is to use default constructor and directly fill the vectors. Thus no copy constructor for the vectors is invoked
    Data(const std::vector<uint32_t>& dimensionSizes,const std::vector<std::vector<uint32_t>>& columnDimensions,const std::vector<std::vector<float>>& columns):
        dimensionSizes(dimensionSizes), columnDimensions(columnDimensions), columns(columns){};

    uint32_t size() const{
        uint32_t ret = 1;
        for(int i: dimensionSizes) ret *= i;
        return ret;
    };

    uint64_t packedByteSize() const{
        uint64_t headerSize = calcHeaderSize();
        uint64_t dataSize = calcDataSize();
        return headerSize + dataSize;
    };

    // packs all data for the gpu in float format(also indexing information in the header so it can be used by standard cast to int)
    // handle over the mapped memory address of the data buffer to instantly upload to gpu
    void packData(void* dst) const{
        uint64_t headerSize = calcHeaderSize();
        uint64_t dataSize = calcDataSize();
        std::vector<uint8_t> data(headerSize + dataSize);       //byte vector
        createPackedHeaderData(data);
        createPackedData(data, headerSize);
        std::copy(data.begin(), data.end(), (uint8_t*)dst);
    };

    // shrinks all vectors to fit the data and removes unused dimensions to avoid unnesecary index accessing
    void compress(){
        dimensionSizes.shrink_to_fit();
        columnDimensions.shrink_to_fit();
        for(auto& column: columnDimensions) column.shrink_to_fit();
        columns.shrink_to_fit();
        for(auto& column: columns) column.shrink_to_fit();
        removeUnusedDimension();
    }

    // subsample a dimension
    void subsampleTrim(const std::vector<uint32_t>& samplingRates, const std::vector<std::pair<uint32_t, uint32_t>>& trimIndices){    
        std::vector<uint32_t> normalSampling(samplingRates.size(), 1);
        std::vector<std::pair<uint32_t, uint32_t>> noTrim;
        for(auto dim: dimensionSizes) noTrim.push_back({0, dim});
        if(samplingRates == normalSampling && trimIndices == noTrim) return;        //nothing has to be done

        std::vector<uint32_t> reducedDimensions(dimensionSizes.size());
        for(int d = 0; d < dimensionSizes.size(); ++d){
            reducedDimensions[d] = trimIndices[d].second - trimIndices[d].first;
            reducedDimensions[d] += samplingRates[d] - 1;
            reducedDimensions[d] /= samplingRates[d];
        }

        for(int c = 0; c < columns.size(); ++c){
            bool trimmedSubsampled = false; 
            for(auto dim: columnDimensions[c]){
                if(samplingRates[dim] != 1 || trimIndices[c] != std::pair<uint32_t, uint32_t>(0, dimensionSizes[c])){
                    trimmedSubsampled = true;
                    break;
                }
            }
            if(!trimmedSubsampled) continue;            //discarding columns which dont depend on any subsampled/trimmed dimension

            std::vector<uint32_t> redDimIndices(columnDimensions[c].size(), 0);
            std::vector<uint32_t> redDimIncrements(columnDimensions[c].size(), 1);
            std::vector<uint32_t> redDimStarts(columnDimensions[c].size());
            std::vector<uint32_t> redDimStops(columnDimensions[c].size());
            for(int cd = 0; cd < columnDimensions[c].size(); ++cd){
                redDimIndices[cd] = trimIndices[columnDimensions[c][cd]].first;
                redDimIncrements[cd] = samplingRates[columnDimensions[c][cd]];
                redDimStarts[cd] = redDimIndices[cd];
                redDimStops[cd] = trimIndices[columnDimensions[c][cd]].second;
            }
            uint32_t redColumnSize = 1;
            for(auto d: columnDimensions[c]) redColumnSize *= reducedDimensions[d];
            std::vector<float> redData(redColumnSize);
            uint32_t redDataCur = 0;
            while(redDimIndices[0] < dimensionSizes[columnDimensions[c][0]]){
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

        dimensionSizes = reducedDimensions;
    }

    // remove dimension by slicing at one index
    void removeDim(uint32_t dimension, uint32_t slice){
        // currently uses lazy implementation based on subsampleTrim, should be made more efficient
        std::vector<uint32_t> samplingRates(dimensionSizes.size(), 1);   //no subsampling
        std::vector<std::pair<uint32_t, uint32_t>> trimIndices(dimensionSizes.size());
        for(int d = 0; d < trimIndices.size(); ++d){
            if(d == dimension) trimIndices[d] = {slice, slice + 1};
            else trimIndices[d] = {0, dimensionSizes[d]};
        }
        subsampleTrim(samplingRates, trimIndices);

        //removing the dimension, updating column dimensions(indices might have to be decremented)
        dimensionSizes.erase(dimensionSizes.begin() + dimension);
        for(int c = 0; c < columnDimensions.size(); ++c){
            int dimIndex = -1;
            for(int d = 0; d < columnDimensions[c].size(); ++d){
                int dim = columnDimensions[c][d];
                if(dim == dimension) dimIndex = d;
                if(dim > dimension) --columnDimensions[c][d];
            }
            if(dimIndex >= 0)
                columnDimensions[c].erase(columnDimensions[c].begin() + dimIndex);
        }
    }

    void removeColumn(uint32_t column){
        columns.erase(columns.begin() + column);
        columnDimensions.erase(columnDimensions.begin() + column);
    }

    void removeUnusedDimension(){
        std::set<uint32_t> usedDims;
        for(auto& columnDims: columnDimensions) for(auto& dim: columnDims) usedDims.insert(dim);
        std::vector<uint32_t> unusedDims;
        if(usedDims.size() < dimensionSizes.size()){  //unused dims exist
            for(int d = 0; d < dimensionSizes.size(); ++d){
                if(usedDims.find(d) == usedDims.end())
                    unusedDims.push_back(d);
            }
            for(int c = 0; c < columnDimensions.size(); ++c){
                for(int d = 0; d < columnDimensions[c].size(); ++d){
                    int decrement = 0;
                    for(auto unused: unusedDims){
                        if(columnDimensions[c][d] > unused) ++decrement;
                    }
                    columnDimensions[c][d] -= decrement;
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

    // access data by an index \in[0, cross(dimensionSizes)] and a column
    float& operator()(uint32_t index, uint32_t column){
        std::vector<uint32_t> dimensionIndices(dimensionSizes.size());
        for(int i = dimensionSizes.size() - 1; i >= 0; --i){
            dimensionIndices[i] = index % dimensionSizes[i];
            index /= dimensionSizes[i];
        }
        uint32_t columnIndex = this->index(dimensionIndices, column);
        return columns[column][columnIndex];
    }
    // const data access
    const float& operator()(uint32_t index, uint32_t column) const{
        std::vector<uint32_t> dimensionIndices(dimensionSizes.size());
        for(int i = dimensionSizes.size() - 1; i >= 0; --i){
            dimensionIndices[i] = index % dimensionSizes[i];
            index /= dimensionSizes[i];
        }
        uint32_t columnIndex = this->index(dimensionIndices, column);
        return columns[column][columnIndex];
    }

private:
    // header size in bytes
    uint64_t calcHeaderSize() const{
        uint64_t columnDimensionSize = 0;
        for(auto& cd: columnDimensions) columnDimensionSize += cd.size();
        return (2 + dimensionSizes.size() + 3 * columns.size() + columnDimensionSize) * sizeof(float);
    }
    // data size in bytes
    uint64_t calcDataSize() const{
        uint64_t dataSize = 0;
        for(auto& column: columns){
            dataSize += column.size() * sizeof(column[0]);
        }
        return dataSize;
    }
    // returns the index for a column given the diemension indices
    uint32_t index(const std::vector<uint32_t>& dimensionIndices, uint32_t column) const{
        uint32_t columnIndex = 0;
        for(int d = 0; d < columnDimensions[column].size(); ++d){
            uint32_t factor = 1;
            for(int i = d + 1; i < columnDimensions[column].size(); ++i){
                factor *= dimensionSizes[columnDimensions[column][i]];
            }
            columnIndex += factor * dimensionIndices[columnDimensions[column][d]];
        }
        return columnIndex;
    }
    //  returns the index for a column given the dimension indices. Dimension indices only include indices for the current dimension, no mapping needed
    uint32_t indexReducedDimIndices(const std::vector<uint32_t>& dimensionIndices, uint32_t column) const{
        uint32_t columnIndex = 0;
        for(int d = 0; d < columnDimensions[column].size(); ++d){
            uint32_t factor = 1;
            for(int i = d + 1; i < columnDimensions[column].size(); ++i){
                factor *= dimensionSizes[columnDimensions[column][i]];
            }
            columnIndex += factor * dimensionIndices[d];
        }
        return columnIndex;
    }

    // puts header data into the beginning of the dst vector(cast to floats)
    void createPackedHeaderData(std::vector<uint8_t>& dst) const{
        uint32_t headerSize = calcHeaderSize();
        uint32_t curPos = 0;
        *reinterpret_cast<float*>(&dst[curPos]) = dimensionSizes.size();
        curPos += 4;
        *reinterpret_cast<float*>(&dst[curPos]) = columns.size();
        curPos += 4;
        for(int i = 0; i < dimensionSizes.size(); ++i){ //diemnsion sizes
            *reinterpret_cast<float*>(&dst[curPos]) = dimensionSizes[i];
            curPos += 4;
        }
        std::vector<uint32_t> columnDimensionOffsets(columns.size());
        uint32_t baseDimensionOffset = 2 + dimensionSizes.size() + columns.size() * 3;
        for(int i = 0; i < columns.size(); ++i){        //column dimension counts
            *reinterpret_cast<float*>(&dst[curPos]) = columnDimensions[i].size();
            curPos += 4;
            if(i == 0)
                columnDimensionOffsets[i] = baseDimensionOffset;
            else
                columnDimensionOffsets[i] = columnDimensionOffsets[i - 1] + columnDimensions[i - 1].size();
        }
        for(int i = 0; i < columnDimensionOffsets.size(); ++i){ //column dimensions offsets
            *reinterpret_cast<float*>(&dst[curPos]) = columnDimensionOffsets[i];
            curPos += 4;
        }
        uint32_t curOffset = headerSize / sizeof(float);
        for(int i = 0; i < columns.size(); ++i){        // data offsets
            *reinterpret_cast<float*>(&dst[curPos]) = curOffset;
            curPos += 4;
            curOffset += columns[i].size();
        }
        for(int i = 0; i < columnDimensions.size(); ++i){   // column dimebnsions information
            for(int j = 0; j < columnDimensions[i].size(); ++j){
                *reinterpret_cast<float*>(&dst[curPos]) = columnDimensions[i][j];
                curPos += 4;
            }
        }
        assert(curPos == headerSize);           //safety check
    }

    void createPackedData(std::vector<uint8_t>& dst, uint32_t startOffset) const{
        uint64_t curPos = startOffset;
        for(int i = 0; i < columns.size(); ++i){
            for(int j =0 ; j < columns[i].size(); ++j){
                *reinterpret_cast<float*>(&dst[curPos]) = columns[i][j];
                curPos += 4;
            }
        }
        assert(curPos - startOffset == calcDataSize()); //safety check
    }
};