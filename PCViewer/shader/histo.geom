#version 450

layout(binding = 0) uniform UniformBufferObject{
    float x;
    float width;
    float maxVal;
    float minVal;
    uint attributeInd;
    uint amtOfAttributes;
    uint pad;
    uint padding;
    vec4 color;
} ubo;

layout( set = 0, binding = 1, r8 ) uniform imageBuffer act;

layout (points) in;
layout (location = 0) in uint index[];
layout (line_strip, max_vertices = 2) out;

void main() {
    bool a = bool(imageLoad( act, int(index[0])));
    if(!a) return;

    gl_Position = gl_in[0].gl_Position; 
    EmitVertex();

    gl_Position = gl_in[0].gl_Position + vec4( ubo.width, 0.0, 0.0, 0.0);
    EmitVertex();
    
    EndPrimitive();
}