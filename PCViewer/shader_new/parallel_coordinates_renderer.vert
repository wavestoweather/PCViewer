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
    uint64_t    activation_bitset_address;
    uint        vertex_count_per_line;        // is at least as high as attribute_count (when equal, polyline rendering)
    float       padding;
    vec4        color;
};

layout(location = 0) out vec4 out_color;

#include "data_access.glsl"

const float alpha = .5;
float get_t(float t, in vec2 p0, in vec2 p1){
    float a = pow((p1.x-p0.x), 2.0f) + pow((p1.y-p0.y), 2.0f);
    float b = pow(a, .5f);
    float c = pow(b,alpha);
    return c+t;
}

void main() {
    int data_index = gl_InstanceIndex;
    int vertex_index = gl_VertexIndex;


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

    uint in_between = vertex_count_per_line - 1 / (attr_infos.attribute_count - 1);
    float gap = 2.0f/(vertex_count_per_line - 1.0f);
    float x_base = -1.0f + vertex_index / in_between * gap;
    float in_between_ratio = float(vertex_count_per_line % in_between) / in_between;
    float x = x_base + in_between_ratio * gap;
    //addding the padding to x
    x *= 1.f - padding;
    float y;
    if(vertex_index % in_between == 0){    // vertex exactly at an axis
        uint attribute_index = uint(attr_infos.vertex_transformations[vertex_index / in_between]);
        float val = get_packed_data(data_index, attribute_index);
        y = (val - attr_infos.vertex_transformations[vertex_index / in_between].y) / (attr_infos.vertex_transformations[vertex_index / in_between].z - attr_infos.vertex_transformations[vertex_index / in_between].y);
        y = y * 2 - 1;
    }
    else{       // reading out neighbouring data for spline calculation
        uint left = vertex_index / in_between;
        uint left_left = max(left - 1, 0);
        uint right = left + 1;
        uint right_right = min(right + 1, attr_infos.attribute_count - 1);
        float left_val, left_left_val, right_val, right_right_val;
        uint attribute_index = uint(attr_infos.vertex_transformations[left]);
        left_val = get_packed_data(data_index, attribute_index);
        attribute_index = uint(attr_infos.vertex_transformations[right]);
        right_val = get_packed_data(data_index, attribute_index);
        if(left == left_left)
            left_left_val = left_val;
        else{
            attribute_index = uint(attr_infos.vertex_transformations[left_left]);
            left_left_val = get_packed_data(data_index, attribute_index);
        }
        if(right == right_right)
            right_right_val = right_val;
        else{
            attribute_index = uint(attr_infos.vertex_transformations[right_right]);
            right_right_val = get_packed_data(data_index, attribute_index);
        }
        left_left_val = (left_left_val - attr_infos.vertex_transformations[left_left].y) / (attr_infos.vertex_transformations[left_left].z - attr_infos.vertex_transformations[left_left].y);
        left_val = (left_val - attr_infos.vertex_transformations[left].y) / (attr_infos.vertex_transformations[left].z - attr_infos.vertex_transformations[left].y);
        right_val = (right_val - attr_infos.vertex_transformations[right].y) / (attr_infos.vertex_transformations[right].z - attr_infos.vertex_transformations[right].y);
        right_right_val = (right_right_val - attr_infos.vertex_transformations[right_right].y) / (attr_infos.vertex_transformations[right_right].z - attr_infos.vertex_transformations[right_right].y);
        // calculating interpolated value
        vec2 p0 = vec2(x_base - gap, left_left_val);
        vec2 p1 = vec2(x_base, left_val);
        vec2 p2 = vec2(x_base + gap, right_val);
        vec2 p3 = vec2(x_base +  2 * gap, right_right_val);
        float t0 = 0;
        float t1 = get_t(t0, p0, p1);
        float t2 = get_t(t1, p1, p2);
        float t3 = get_t(t2, p2, p3);

        float t = mix(t1, t2, in_between_ratio);
        vec2 a1 = ( t1-t )/( t1-t0 )*p0 + ( t-t0 )/( t1-t0 )*p1;
        vec2 a2 = ( t2-t )/( t2-t1 )*p1 + ( t-t1 )/( t2-t1 )*p2;
        vec2 a3 = ( t3-t )/( t3-t2 )*p2 + ( t-t2 )/( t3-t2 )*p3;
        vec2 b1 = ( t2-t )/( t2-t0 )*a1 + ( t-t0 )/( t2-t0 )*a2;
        vec2 b2 = ( t3-t )/( t3-t1 )*a2 + ( t-t1 )/( t3-t1 )*a3;
        vec2 c = ( t2-t )/( t2-t1 )*b1 + ( t-t1 )/( t2-t1 )*b2;
        x = c.x;
        y = c.y;
    }

    gl_Position = vec4( x, y * -1.0f, 0.0, 1.0);

    out_color = color;
    if(int(attr_infos.vertex_transformations[0].a) == 1){
        PriorityValues priorities = PriorityValues(priorities_address);
        uint i = gl_VertexIndex / attr_infos.attribute_count;
        out_color.xyz = texture(color_transfer_texture, vec2(float(priorities.vals[data_index]) / 255.f,.5f)).xyz;
    }
}