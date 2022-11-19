#version 450

#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_buffer_reference2: require
#extension GL_EXT_buffer_reference_uvec2: require
#extension GL_EXT_shader_explicit_arithmetic_types: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, scalar) buffer AttributeInfos{
    uint     attribute_count;               // amount of active attributes
    uint     _, __;
    uint     data_flags;                    // can contain additional data flags
    vec4     vertex_transformations[];      // x holds the attribute index, y and z hold the lower and the upper bound respectivley for the first amtOfAttributes positions (in x axis position to variable is stored)
};
layout(buffer_reference, scalar) buffer Histogram{
    uint v[];
};
layout(buffer_reference, scalar) buffer Ordering{
    uint i[];
};

layout(set = 0, binding = 0) uniform sampler2D color_transfer_texture;

layout(push_constant) uniform PCs{
    uint64_t    attribute_info_address;
    uint64_t    histogram_address;
    uint64_t    ordering_address;           // needed for order dependant rendering (eg. priority rendering)
    int         a_axis;                     // holds the final axis position index (after activation, reordering) for the primary histogram axis
    int         b_axis;                     // holds the final axis position index (after activation, reordering) for the secondary histogram axis
    int         c_axis;
    int         d_axis;

    uint        a_size;
    uint        b_size;
    uint        c_size;
    uint        d_size;
    float       padding;
    uint        priority_rendering;         // 1 indicates priority rendering should be used
    vec4        color;
    uint        line_verts;
};

layout(location = 0) out vec4 out_color;

#include "splines.glsl"

void swap(inout vec2 a, inout vec2 b){
    vec2 t = a;
    a = b;
    b = t;
}

void main(){
    uint divider = (line_verts - 1) * 2;
    uint hist_index = gl_VertexIndex / divider;
    uint line_index = gl_VertexIndex % divider;
    line_index = (line_index / 2) + (line_index & 1);
    
    Histogram hist = Histogram(histogram_address);
    uint count = hist.v[hist_index];
    if(count < 1){
        out_color.a = 0;
        gl_Position = vec4(-2);
    }

    float x, y;
    AttributeInfos attr_infos = AttributeInfos(attribute_info_address);
    float gap = 2.f / (attr_infos.attribute_count - 1.f);
    if(line_verts == 2){
        bool is_b_axis = line_index == 1;
        uint axis_index = is_b_axis ? b_axis: a_axis;

        if(ordering_address != 0){
            Ordering o = Ordering(ordering_address);
            hist_index = o.i[hist_index];
        }

        x = -1. + axis_index * gap;

        if(is_b_axis)
            y = (float(hist_index % b_size) + .5f) / b_size;
        else
            y = (float(hist_index / b_size) + .5f) / a_size;
    }
    else{
        vec2 p[4];
        uint bin_size = a_size * b_size * c_size;
        p[3].y = (float(hist_index / bin_size) + .5) / d_size;
        uint rest_index = hist_index % bin_size;
        bin_size = a_size * b_size;
        p[2].y = (float(rest_index / bin_size) + .5) / c_size;
        rest_index = rest_index % bin_size;
        bin_size = a_size;
        p[1].y = (float(rest_index / bin_size) + .5) / b_size;
        rest_index = rest_index % bin_size;
        p[0].y = (float(rest_index) + .5) / a_size;

        p[0].x = -1 + a_axis * gap;
        p[1].x = -1 + b_axis * gap;
        p[2].x = -1 + c_axis * gap;
        p[3].x = -1 + d_axis * gap;
        if(p[0].x>p[1].x) swap(p[0], p[1]);
        if(p[2].x>p[3].x) swap(p[2], p[3]);
        if(p[0].x>p[2].x) swap(p[0], p[2]);
        if(p[1].x>p[3].x) swap(p[1], p[3]);
        if(p[1].x>p[2].x) swap(p[1], p[2]);

        vec2 c = get_spline_pos(p[0], p[1], p[2], p[3], float(line_index) / float(line_verts - 1));
        x = c.x;
        y = c.y;
    }

    y = y * 2 - 1;
    gl_Position = vec4(x, y * -1.f, 0, 1);

    out_color = color;

    if(priority_rendering == 1){
        float norm = float(count) / 65535.f;
        norm = .98 * norm + .02;
        out_color.xyz = texture(color_transfer_texture, vec2(norm, .5f)).xyz;
    }
    else{
        out_color.a = 1. - pow(1. - color.a, count);
    }
}