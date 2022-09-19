#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

layout(binding = 0) buffer UniformBufferObject{
	float alpha;
	uint amtOfVerts;
	uint amtOfAttributes;
	float padding;
	uint dataFlags;						//contains additional data flags
	uint fill, fill1, fill2;
	vec4 color;
	vec4 vertexTransformations[];		//x holds the x position, y and z hold the lower and the upper bound respectivley for the first amtOfAttributes positions
} ubo;

layout(std430, binding = 1) buffer Priorities{
	uint8_t v;
}priorities;

layout (set = 0, binding = 2) uniform sampler2D color_map;

layout (binding = 0, set = 1) buffer Data{
	float d[];
}data;

layout(location = 0) out vec4 color;

#include "dataAccess.glsl"

void main() {
	float gap = 2.0f/(ubo.amtOfVerts - 1.0f); //gap is tested, and is correct

	uint index = gl_VertexIndex / ubo.amtOfAttributes;
	uint i = gl_VertexIndex % ubo.amtOfAttributes;
	float x = -1.0f + ubo.vertexTransformations[i].x * gap;
	//addding the padding to x
	x *= 1-ubo.padding;
	
	float inPosition = getPackedData(index, i);
	float y = inPosition - ubo.vertexTransformations[i].y;
	y /= (ubo.vertexTransformations[i].z - ubo.vertexTransformations[i].y);
	y *= 2;
	y -= 1;

    gl_Position = vec4( x, y * -1.0f, 0.0, 1.0);

	color = ubo.color;
	if(int(ubo.vertexTransformations[0].a) == 1){
		uint i = gl_VertexIndex / ubo.amtOfAttributes;
		color.xyz = texture(ironMap, vec2(priorities.v[i],.5f)).xyz;
	}
}