#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject{
    vec3 camPos;
    vec3 cubeSides;
    vec3 lightPos;
    mat4 mvp;
} ubo;

layout(location = 0) in vec4 inPosition;
layout(location = 0) out vec3 worldPos;

void main() {
    gl_Position = ubo.mvp * vec4(inPosition.xyz,1);
    worldPos = inPosition.xyz;
}