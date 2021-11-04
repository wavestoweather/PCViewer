#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

layout(binding = 0) buffer UniformBufferObject{
	float alpha;
	uint amtOfVerts;
	uint amtOfAttributes;
	float padding;
	vec4 color;
	vec4 vertexTransformations[];		//x holds the x position, y and z hold the lower and the upper bound respectivley
} ubo;

layout(std430, binding = 1) buffer PriorityColors{
	vec4 colors[];
}pCol;

layout (binding = 2) uniform sampler2D ironMap;

layout (binding = 0, set = 1) buffer Data{
	float d[];
}data;

const uint mask0 = 0xff000000;
const uint mask1 = 0x00ff0000;
const uint mask2 = 0x0000ff00;
const uint mask3 = 0x000000ff;

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
		uint major = i >> 2;
		uint minor = i & 0x3;
		color.xyz = texture(ironMap, vec2(pCol.colors[major][minor],.5f)).xyz;

		/*
		uint t = pCol.colors[majormajor][major];
		switch(minor){
		case 0: color.x = ((t&mask0)>>24)/255.0f; break;
		case 1: color.x = ((t&mask1)>>16)/255.0f; break;
		case 2: color.x = ((t&mask2)>>8)/255.0f; break;
		case 3: color.x = ((t&mask3))/255.0f; break;
		}
		minor++;
		major += minor >> 2;
		majormajor += major >> 2;
		major = major & 3;
		minor = minor & 3;
		t = pCol.colors[majormajor][major];
		switch(minor){
		case 0: color.y = ((t&mask0)>>24)/255.0f; break;
		case 1: color.y = ((t&mask1)>>16)/255.0f; break;
		case 2: color.y = ((t&mask2)>>8)/255.0f; break;
		case 3: color.y = ((t&mask3))/255.0f; break;
		}
		minor++;
		major += minor >> 2;
		majormajor += major >> 2;
		major = major & 3;
		minor = minor & 3;
		t = pCol.colors[majormajor][major];
		switch(minor){
		case 0: color.z = ((t&mask0)>>24)/255.0f; break;
		case 1: color.z = ((t&mask1)>>16)/255.0f; break;
		case 2: color.z = ((t&mask2)>>8)/255.0f; break;
		case 3: color.z = ((t&mask3))/255.0f; break;
		}*/
	}
}