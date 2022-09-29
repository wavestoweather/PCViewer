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
	uint 	attribute_count;				// amount of active attributes
	uint 	_, __;
	uint 	data_flags;						// can contain additional data flags
	vec4 	vertex_transformations[];		// x holds the attribute index, y and z hold the lower and the upper bound respectivley for the first amtOfAttributes positions (in x axis position to variable is stored)
};

layout(buffer_reference, scalar) buffer PriorityValues{
	uint8_t vals[];
};

layout(set = 0, binding = 0) uniform sampler2D color_transfer_texture;

layout(push_constant) uniform PCs{
	uint64_t 	attribute_info_address;
	uint64_t 	data_header_address;
	uint64_t	priorities_address;
	uint		vertex_count_per_line;		// is at least as high as attribute_count (when equal, polyline rendering)
	float 		padding;
	vec4 		color;
};

layout(location = 0) out vec4 out_color;

#include "data_access.glsl"

void main() {
	int data_index = gl_InstanceIndex;
	int vertex_index = gl_VertexIndex;

	AttributeInfos attr_infos = AttributeInfos(attribute_info_address);

	float gap = 2.0f/(vertex_count_per_line - 1.0f);
	float x = -1.0f + vertex_index * gap;
	//addding the padding to x
	x *= 1.f - padding;
	float y;
	if(vertex_count_per_line == attr_infos.attribute_count){	// polyline rendering
		uint attribute_index = uint(attr_infos.vertex_transformations[vertex_index]);
		float val = get_packed_data(data_index, attribute_index);
		y = (val - attr_infos.vertex_transformations[vertex_index].x) / (attr_infos.vertex_transformations[vertex_index].z - attr_infos.vertex_transformations[vertex_index].y);
		y = y * 2 - 1;
	}
	else{
		//uint left_attribute;
	}

    gl_Position = vec4( x, y * -1.0f, 0.0, 1.0);

	out_color = color;
	if(int(attr_infos.vertex_transformations[0].a) == 1){
		PriorityValues priorities = PriorityValues(priorities_address);
		uint i = gl_VertexIndex / attr_infos.attribute_count;
		out_color.xyz = texture(color_transfer_texture, vec2(float(priorities.vals[data_index]) / 255.f,.5f)).xyz;
	}
}