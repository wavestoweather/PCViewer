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

#include "standard_buffer_refs.glsl"
#include "scatterplot_forms.glsl"

layout(set = 0, binding = 0) uniform sampler2D color_transfer_texture;

layout(push_constant) uniform PC{
    uint64_t        counts_address;
    uint64_t        ordering_address;
    uint            flip_axes;
    uint            bin_size;
    uint            form;
    float           radius;
    uint            priority_rendering;
    uint            f,i,ll;
    vec4            color;
};

layout(location = 0) out vec4 out_color;
layout(location = 1) out uint out_form;

void main(){
    uint hist_index = gl_VertexIndex;

    if(ordering_address != 0){
        uvec o = uvec(ordering_address);
        hist_index = o.data[hist_index];
    }

    uvec counts = uvec(counts_address);
    uint count = counts.data[hist_index];
    if(count == 0){
        gl_Position = vec4(-2);
        return;
    }
    float a = (float(hist_index / bin_size) + .5) / bin_size;
    float b = (float(hist_index % bin_size) + .5) / bin_size;
    a = a * 2 - 1;
    b = b * 2 - 1;

    if(flip_axes == 0)
        gl_Position = vec4(a, b, 0, 1);
    else 
        gl_Position = vec4(b, a, 0, 1);
    gl_Position.y = -gl_Position.y;

    gl_PointSize = radius;
    out_form = form;
    if(priority_rendering > 0){
        float norm = float(count) / 255.f;
        out_color.xyz = texture(color_transfer_texture, vec2(norm, .5f)).xyz;
        out_color.w = color.w;
    }
    else
        out_color = vec4(color.xyz,  1. - pow(1. - color.a, count));
}