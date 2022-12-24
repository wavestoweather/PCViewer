#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_NV_shader_subgroup_partitioned : enable
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_shader_subgroup_ballot: enable
#extension GL_EXT_buffer_reference2: require
#extension GL_EXT_buffer_reference_uvec2: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#include "scatterplot_forms.glsl"

layout(push_constant) uniform PC{
    uint64_t        data_header_address;
    uint64_t        index_buffer_address;
    uint64_t        activation_bitset_address;
    uint            attribute_a;
    uint            attribute_b;
    float           a_min;
    float           a_max;
    float           b_min;
    float           b_max;
    uint            flip_axes;
    uint            form;
    float           radius;
    uint            fill;
    vec4            color;
};

#include "standard_buffer_refs.glsl"
#include "data_access.glsl"

layout(location = 0) out vec4 out_color;
layout(location = 1) out uint out_form;

void main(){
    uint data_index = gl_VertexIndex;

    if(index_buffer_address != 0){
        uvec index_buffer = uvec(index_buffer_address);
        data_index = index_buffer.data[data_index];
    }

    uvec activation_bitset = uvec(activation_bitset_address);
    bool act = (activation_bitset.data[data_index / 32] & (1 << (data_index % 32))) > 0;
    if(!act){
        gl_Position = vec4(-2);
        return;
    }

    float a = (get_packed_data(data_index, attribute_a) - a_min) / (a_max - a_min);
    float b = (get_packed_data(data_index, attribute_b) - b_min) / (b_max - b_min);
    a = a * 2 - 1;
    b = b * 2 - 1;
    if(flip_axes == 0)
        gl_Position = vec4(a, b, 0, 1);
    else
        gl_Position = vec4(b, a, 0, 1);
    gl_Position.y = -gl_Position.y;
    
    out_color = color;
    out_form = form;
    gl_PointSize = radius;
}