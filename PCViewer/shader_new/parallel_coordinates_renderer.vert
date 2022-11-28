#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference2: require
#extension GL_EXT_buffer_reference_uvec2: require
#extension GL_EXT_shader_explicit_arithmetic_types: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(buffer_reference, scalar) buffer AttributeInfos{
    uint     attribute_count;               // amount of active attributes
    uint     _, __;
    uint     data_flags;                    // can contain additional data flags
    vec4     vertex_transformations[];      // x holds the attribute index, y and z hold the lower and the upper bound respectivley for the first amtOfAttributes positions (in x axis position to variable is stored)
};

layout(buffer_reference, scalar) buffer IndexBuffer{
    uint i[];
};

layout(buffer_reference, scalar) buffer ActivationBitset{
    uint i[];
};

layout(buffer_reference, scalar) buffer PriorityValues{
    uint8_t vals[];
};

layout(set = 0, binding = 0) uniform sampler2D color_transfer_texture;

layout(push_constant) uniform PCs{
    uint64_t    attribute_info_address;
    uint64_t    data_header_address;
    uint64_t    priorities_address;
    uint64_t    index_buffer_address;
    uint64_t    index_order_address;
    uint64_t    activation_bitset_address;
    uint        vertex_count_per_line;        // is at least as high as attribute_count (when equal, polyline rendering)
    float       padding;
    uint        spa,ce;
    vec4        color;
};

layout(location = 0) out vec4 out_color;

#include "data_access.glsl"
#include "splines.glsl"

void main() {
    int data_index = gl_InstanceIndex;
    int vertex_index = gl_VertexIndex;

    if(index_order_address != 0){
        IndexBuffer order = IndexBuffer(index_order_address);
        data_index = int(order.i[data_index]);
    }

    ActivationBitset activation_bitset = ActivationBitset(activation_bitset_address);
    bool act = (activation_bitset.i[data_index / 32] & (1 << (data_index % 32))) > 0;
    if(!act){
        out_color = vec4(0);
        gl_Position = vec4(-2);
        return;
    }

    AttributeInfos attr_infos = AttributeInfos(attribute_info_address);
    
    if(index_buffer_address != 0){
        IndexBuffer index_buffer = IndexBuffer(index_buffer_address);
        vertex_index = int(index_buffer.i[vertex_index]);
    }

    uint in_between = (vertex_count_per_line - 1) / (attr_infos.attribute_count - 1);
    uint base_x_index = vertex_index / in_between;
    uint in_between_index = vertex_index % in_between;
    float gap = 2.0f/(vertex_count_per_line - 1.0f);
    float full_gap = in_between * gap;
    float x = -1.0 + vertex_index * gap;
    float x_base = -1.0 + base_x_index * full_gap;
    //addding the padding to x
    x *= 1.f - padding;
    float y;
    if(in_between_index == 0){    // vertex exactly at an axis
        uint attribute_index = uint(attr_infos.vertex_transformations[base_x_index].x);
        float val = get_packed_data(data_index, attribute_index);
        y = (val - attr_infos.vertex_transformations[base_x_index].y) / (attr_infos.vertex_transformations[base_x_index].z - attr_infos.vertex_transformations[base_x_index].y);
        y = y * 2 - 1;
    }
    else{       // reading out neighbouring data for spline calculation
        uint left = base_x_index;
        uint left_left = max(int(left) - 1, 0);
        uint right = left + 1;
        uint right_right = min(right + 1, attr_infos.attribute_count - 1);
        float left_val, left_left_val, right_val, right_right_val;
        uint attribute_index = uint(attr_infos.vertex_transformations[left].x);
        left_val = get_packed_data(data_index, attribute_index);
        attribute_index = uint(attr_infos.vertex_transformations[right].x);
        right_val = get_packed_data(data_index, attribute_index);
        if(left == left_left)
            left_left_val = left_val;
        else{
            attribute_index = uint(attr_infos.vertex_transformations[left_left].x);
            left_left_val = get_packed_data(data_index, attribute_index);
        }
        if(right == right_right)
            right_right_val = right_val;
        else{
            attribute_index = uint(attr_infos.vertex_transformations[right_right].x);
            right_right_val = get_packed_data(data_index, attribute_index);
        }
        left_left_val = (left_left_val - attr_infos.vertex_transformations[left_left].y) / (attr_infos.vertex_transformations[left_left].z - attr_infos.vertex_transformations[left_left].y);
        left_val = (left_val - attr_infos.vertex_transformations[left].y) / (attr_infos.vertex_transformations[left].z - attr_infos.vertex_transformations[left].y);
        right_val = (right_val - attr_infos.vertex_transformations[right].y) / (attr_infos.vertex_transformations[right].z - attr_infos.vertex_transformations[right].y);
        right_right_val = (right_right_val - attr_infos.vertex_transformations[right_right].y) / (attr_infos.vertex_transformations[right_right].z - attr_infos.vertex_transformations[right_right].y);
        // calculating interpolated value
        vec2 p0 = vec2(x_base - full_gap, left_left_val);
        vec2 p1 = vec2(x_base, left_val);
        vec2 p2 = vec2(x_base + full_gap, right_val);
        vec2 p3 = vec2(x_base +  2 * full_gap, right_right_val);
        vec2 c = get_spline_pos(p0, p1, p2, p3, float(in_between_index) / float(in_between));
        x = c.x;
        y = c.y * 2 - 1;
    }

    gl_Position = vec4( x, y * -1.0f, 0.0, 1.0);

    out_color = color;
    if(priorities_address != 0){
        PriorityValues priorities = PriorityValues(priorities_address);
        uint i = gl_VertexIndex / attr_infos.attribute_count;
        out_color.xyz = texture(color_transfer_texture, vec2(float(priorities.vals[data_index]) / 255.f,.5f)).xyz;
    }
}