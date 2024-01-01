#extension GL_EXT_buffer_reference: require
#extension GL_EXT_buffer_reference2: require
#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, scalar) buffer Vec{float d[];};
layout(buffer_reference, scalar) buffer UVec{uint d[];};

float squared_cauchy_2d(float x1, float x2, float y1, float y2)
{
    return pow(1.0 + pow(x1 - y1, 2) + pow(x2 - y2, 2), -2);
}