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
	int multiplyer = gl_VertexIndex & 1;
	float y = (((inPosition - ubo.minVal) / (ubo.maxVal - ubo.minVal)) - 1) * 2 ;

    gl_Position = vec4( ubo.x + (multiplyer * ubo.width), y * -1.0f, 0.0, 1.0);
}