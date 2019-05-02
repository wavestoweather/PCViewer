#version 450
#extension GL_ARB_separate_shader_objects : enable

vec2 positions[6] = vec2[](
    vec2(-0.5, -0.5),
    vec2(-0.3, 0.5),
    vec2(-0.1, -0.5),
	vec2(0.1,0.5),
	vec2(0.3,-0.5),
	vec2(0.5,0.5)
);

layout(binding = 0) buffer UniformBufferObject{
	float alpha;
	uint amtOfVerts;
	uint ordering[20];
} ubo;

layout(location = 0) in float inPosition;

void main() {
	float gap = 2.0f/(ubo.amtOfVerts - 1.0f); //gap is tested, and ist correct
	float x = -1.0f + ubo.ordering[gl_VertexIndex] * gap;
	x = (ubo.ordering[gl_VertexIndex])/6.0f;

    gl_Position = vec4( x,positions[gl_VertexIndex].y, 0.0, 1.0);
}