#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject{
	float alpha;
	uint amtOfVerts;
	uint ordering[20];
} ubo;

layout(location = 0) in float inPosition;

void main() {
	float gap = 2/(ubo.amtOfVerts - 1);
	float x = -1 + ubo.ordering[gl_VertexIndex] * gap;

    gl_Position = vec4(x, inPosition, 0.0, 1.0);
}