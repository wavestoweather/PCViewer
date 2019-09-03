#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject{
	vec3 camPos;
	vec4 color;
	mat4 mvp;
	mat4 worldNormals;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 normal;
layout(location = 0) out vec3 worldPos;
layout(location = 1) out vec3 worldNormal; 

void main() {
	gl_Position = ubo.mvp * vec4(inPosition.xyz,1);
	worldPos = inPosition.xyz;
	worldNormal = (ubo.worldNormals * vec4(normal,0)).xyz;
}