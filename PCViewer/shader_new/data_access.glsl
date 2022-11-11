layout(buffer_reference, scalar) buffer DataHeader{
    uint dimension_count;
    uint column_count;
    uint data_address_offset;
    uint data_transform_offset;
    uint data[];
};

const uint float_type = 0;
const uint half_type = 1;
const uint uint_type = 2;
const uint ushort_type = 3;

layout(constant_id = 0) const uint data_type = float_type;
layout(buffer_reference, scalar) buffer Data{
    float d[];
};
layout(buffer_reference, scalar) buffer Half{
    float16_t d[];
};
layout(buffer_reference, scalar) buffer UInt{
    uint d[];
};
layout(buffer_reference, scalar) buffer UShort{
    uint16_t d[];
};

const uint maxAmtOfColumns = 50;
const float inf = 1./0.;
#define nan uintBitsToFloat(0x7ff80000);

DataHeader data_header = DataHeader(data_header_address);

float get_raw_data(uint index, uint column){
    if(data_header.dimension_count <= 1){
        uvec2 address = uvec2(data_header.data[data_header.data_address_offset + 2 * column], data_header.data[data_header.data_address_offset + 2 * column + 1]);
        switch(data_type){
        case float_type:{
            Data data = Data(address);
            if(data_header.dimension_count == 0)
                return data.d[0];
            else
                return data.d[index];
            break;
        }
        case half_type:{
            Half data = Half(address);
            if(data_header.dimension_count == 0)
                return float(data.d[0]);
            else
                return float(data.d[index]);
            break;
        }
        case uint_type:{
            UInt data = UInt(address);
            if(data_header.dimension_count == 0)
                return float(data.d[0]) / float(0xffffffffu);
            else
                return float(data.d[index]) / float(0xffffffffu);
            break;
        }
        case ushort_type:{
            UShort data = UShort(address);
            if(data_header.dimension_count == 0)
                return float(data.d[0]) / float(0xffffu);
            else
                return float(data.d[index]) / float(0xffffu);
            break;
        }
        }
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

    uvec2 address = uvec2(data_header.data[data_header.data_address_offset + 2 * column], data_header.data[data_header.data_address_offset + 2 * column + 1]);
    switch(data_type){
    case float_type:{
        Data data = Data(address);
        return data.d[columnIndex];
    }
    case half_type:{
        Half data = Half(address);
        return float(data.d[columnIndex]);
    }
    case uint_type:{
        UInt data = UInt(address);
        return float(data.d[columnIndex]) / float(0xffffffffu);
    }
    case ushort_type:{
        UShort data = UShort(address);
        return float(data.d[columnIndex]) / float(0xffffu);
    }
    }
}

float get_packed_data(uint index, uint column){
    float d = get_raw_data(index, column);
    if(data_header.data_transform_offset != 0){
        // data has to be transformed for the final value
        uint offset_base = data_header.data_transform_offset + 2 * column;
        float scale = uintBitsToFloat(data_header.data[offset_base]);
        float offset = uintBitsToFloat(data_header.data[offset_base + 1]);
        d = d * scale + offset;
    }
    return d;
}