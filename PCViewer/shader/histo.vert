#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject{
	float x;
	float width;
	float maxVal;
	float minVal;
	vec4 color;
} ubo;

layout(location = 0) in float inPosition;

void main() {
	float y = (((inPosition - ubo.minVal) / (ubo.maxVal - ubo.minVal)) - .5f) * 2;

    gl_Position = vec4( ubo.x, y * -1.0f, 0.0, 1.0);
}