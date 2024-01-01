layout(buffer_reference, scalar) buffer vec{
    float[] data;
};

layout(buffer_reference, scalar) buffer uvec{
    uint[] data;
};

layout(buffer_reference, scalar) buffer ivec{
    int[] data;
};

#ifdef UINT8_DEFINED
layout(buffer_reference, scalar) buffer uchar_vec{
    uint8_t[] data;
};
#endif