#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject{
	float alpha;
	uint amtOfVerts;
	uint ordering[20];
} ubo;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(1.0, 1.0, 1.0, ubo.alpha);
}