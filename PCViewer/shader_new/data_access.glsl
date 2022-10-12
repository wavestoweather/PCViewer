layout(buffer_reference, scalar) buffer DataHeader{
    uint dimension_count;
    uint column_count;
    uint data_address_offset;
    uint _;
    uint data[];
};

layout(buffer_reference, scalar) buffer Data{
    float d[];
};

const uint maxAmtOfColumns = 50;
#define nan uintBitsToFloat(0x7ff80000);

float get_packed_data(uint index, uint column){
    DataHeader data_header = DataHeader(data_header_address);
    if(data_header.dimension_count <= 1){
        Data data = Data(uvec2(data_header.data[data_header.data_address_offset + 2 * column], data_header.data[data_header.data_address_offset + 2 * column + 1]));
        if(data_header.dimension_count == 0)
            return data.d[0];
        else
            return data.d[index];
    }
    uint dimensionIndices[maxAmtOfColumns];
    uint dimensionCount = data_header.dimension_count;
    uint columnCount = data_header.column_count;
    for(int i = int(dimensionCount) - 1; i >= 0; --i){
        uint dimensionSize = uint(data_header.data[i]);
        dimensionIndices[i] = index % dimensionSize;
        index /= dimensionSize;
    }
    uint columnIndex = 0;
    uint columnDimensionsCount = data_header.data[dimensionCount + column];
    uint baseColumnDimensionsOffset = data_header.data[dimensionCount + columnCount + column];
    for(int d = 0; d < columnDimensionsCount; ++d){
        uint factor = 1;
        for(uint i = d + 1; i < columnDimensionsCount; ++i){
            uint dim = data_header.data[baseColumnDimensionsOffset + i];
            factor *= data_header.data[dim];
        }
        uint dim = data_header.data[baseColumnDimensionsOffset + d];
        columnIndex += factor * dimensionIndices[dim];
    }
    Data data = Data(uvec2(data_header.data[data_header.data_address_offset + 2 * column], data_header.data[data_header.data_address_offset + 2 * column + 1]));
    return data.d[columnIndex];
}