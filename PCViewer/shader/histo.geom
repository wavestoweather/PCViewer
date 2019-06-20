#version 450

layout(binding = 0) uniform UniformBufferObject{
	float x;
	float width;
	float maxVal;
	float minVal;
	vec4 color;
} ubo;

layout (lines) in;
layout (line_strip, max_vertices = 2) out;

void main() {    
    gl_Position = gl_in[0].gl_Position; 
    EmitVertex();

    gl_Position = gl_in[1].gl_Position + vec4( ubo.width, 0.0, 0.0, 0.0);
    EmitVertex();
    
    EndPrimitive();
}