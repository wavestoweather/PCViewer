#version 450
#extension GL_ARB_separate_shader_objects : enable

//layout(binding = 0) uniform UniformBufferObject{
//	float alpha;
//	uint amtOfVerts;
//	uint amtOfAttributes;
//	uint padding;
//	vec4 color;
//	vec4 ordering[50];
//} ubo;


layout(location = 0) in vec4 col;
layout(location = 0) out vec4 outColor;

void main() {
	
    outColor = col;
}