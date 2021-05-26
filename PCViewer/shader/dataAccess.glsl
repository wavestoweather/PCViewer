const uint maxAmtOfColumns = 50;
const uint dimensionSizesOffset = 2;

float getPackedData(uint index, uint column){
    uint dimensionIndices[maxAmtOfColumns];
    int dimensionCount = int(data.d[0]);
    int columnCount = int(data.d[1]);
    for(int i = dimensionCount - 1; i >= 0; --i){
        int dimensionSize = int(data.d[dimensionSizesOffset + i]);
        dimensionIndices[i] = index % dimensionSize;
        index /= dimensionSize;
    }
    uint columnIndex = 0;
    int columnDimensionsCount = int(data.d[2 + dimensionCount + column]);
    int baseColumnDimensionsOffset = int(data.d[2 + dimensionCount + columnCount + column]);
    for(int d = 0; d < columnDimensionsCount; ++d){
        uint factor = 1;
        for(int i = d + 1; i < columnDimensionsCount; ++i){
            int dim = int(data.d[baseColumnDimensionsOffset + i]);
            factor *= uint(data.d[dimensionSizesOffset + dim]);
        }
        int dim = int(data.d[baseColumnDimensionsOffset + d]);
        columnIndex += factor * dimensionIndices[dim];
    }

    int columnBaseOffset = int(data.d[2 + dimensionCount + 2 * columnCount + column]);
    return data.d[columnBaseOffset + columnIndex];
}