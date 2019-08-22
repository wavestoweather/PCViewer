#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject{
	vec3 camPos;
	mat4 mvp;
} ubo;

layout(location = 0) in vec3 worldPos;
layout(location = 0) out vec4 outColor;

void main() {
	outColor = vec4(worldPos+.5f,1);
}