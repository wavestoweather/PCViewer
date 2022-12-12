#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2: require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#include "standard_buffer_refs.glsl"

const uint mapping_color_density = 1;
const uint mapping_gray_density = 2;
const uint mapping_transfer_func = 3;

layout(set = 0, binding = 0) uniform sampler2D color_transfer_texture;

layout(push_constant) uniform PC{
    layout(offset = 16) uint64_t histogram_address;
    uint     bin_count;
    uint     mapping_type;
    vec4     color;
    float    blur_radius;
};

layout(location = 0) out vec4 out_color;

void main(){
    uvec bins = uvec(histogram_address);
    int fragment_index = int(bin_count) - 1 - int(gl_FragCoord.y);//int((gl_FragCoord.y * .5 + .5) * bin_count);

    int pixel_radius = int(blur_radius * bin_count);
    float sdev = blur_radius * bin_count / 3;  // pixel radius should be at 3 sigma, so sigma/stddev is a third
    sdev = sdev * sdev;
    float count = 0;
    float divider = 0;
    for(int i = max(fragment_index - pixel_radius, 0); i <= min(fragment_index + pixel_radius, bin_count - 1); ++i){
        float cur_count = float(bins.e[i]);
        float gaussian_fac = exp(-(pow(i - fragment_index, 2)/sdev));
        count += gaussian_fac * cur_count;
        divider += gaussian_fac;
    }
    count /= divider;
    float opacity = 1. - pow(1. - color.a, count);

    switch(mapping_type){
    case mapping_color_density: out_color = vec4(color.xyz * opacity, 1); break;
    case mapping_gray_density: out_color = vec4(vec3(1) * opacity, 1); break;
    case mapping_transfer_func: out_color = texture(color_transfer_texture, vec2(opacity, .5)); break;
    default:
        out_color = vec4(1,0,0,1);
    }
}