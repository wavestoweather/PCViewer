#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject{
	float alpha;
	uint amtOfVerts;
	uint amtOfAttributes;
	uint nothing;
	vec4 color;
	vec4 vertexTransformations[20];		//x holds the x position, y and z hold the lower and the upper bound respectivley
} ubo;

layout(location = 0) in float inPosition;

void main() {
	float gap = 2.0f/(ubo.amtOfVerts - 1.0f); //gap is tested, and ist correct

	uint i = gl_VertexIndex % ubo.amtOfAttributes;
	float x = -1.0f + ubo.vertexTransformations[i].x * gap;
	
	float y = inPosition - ubo.vertexTransformations[i].y;
	y *= 2 * (ubo.vertexTransformations[i].z - ubo.vertexTransformations[i].y);
	y -= 1;

    gl_Position = vec4( x, y * -1.0f, 0.0, 1.0);
}