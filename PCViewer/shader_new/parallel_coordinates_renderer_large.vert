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
    uint        _, __;
    uint        a_axis;                     // holds the final axis position index (after activation, reordering) for the primary histogram axis
    uint        b_axis;                     // holds the final axis position index (after activation, reordering) for the secondary histogram axis
    uint        a_size;
    uint        b_size;
    uint        vertex_count_per_line;      // is at least as high as attribute_count (when equal, polyline rendering)
    float       padding;
    uint        priority_rendering;         // 1 indicates priority rendering should be used
    uint        ___;
    vec4        color;
};

layout(location = 0) out vec4 out_color;

void main(){
    uint hist_index = gl_VertexIndex >> 1;
    bool is_b_axis = bool(gl_VertexIndex & 1);
    uint axis_index = is_b_axis ? b_axis: a_axis;

    if(ordering_address != 0){
        Ordering o = Ordering(ordering_address);
        hist_index = o.i[hist_index];
    }

    AttributeInfos attr_infos = AttributeInfos(attribute_info_address);

    float gap = 2.f / (vertex_count_per_line - 1.f);
    float x = -1. + axis_index * gap;

    float y;
    if(is_b_axis)
        y = (float(hist_index % b_size) + .5f) / b_size;
    else
        y = (float(hist_index / b_size) + .5f) / a_size;

    // TODO activate axis transformation
    //y = (y * attr_infos.vertex_transformations[axis_index].y) / (attr_infos.vertex_transformations[vertex_index].z - attr_infos.vertex_transformations[vertex_index].y);
    gl_Position = vec4(x, y * -1.f, 0, 1);

    out_color = color;

    Histogram hist = Histogram(histogram_address);
    uint count = hist.v[hist_index];
    if(priority_rendering == 1){
        float norm = float(count) / 65535.f;
        norm = .98 * norm + .02;
        out_color.xyz = texture(color_transfer_texture, vec2(norm, .5f)).xyz;
        if(count < 2){
            out_color.a = 0;
            gl_Position = vec4(-2);
        }
    }
    else{
        out_color.a = 1. - pow(1. - color.a, count);
    }
}