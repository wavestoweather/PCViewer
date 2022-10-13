#version 450
#extension GL_ARB_separate_shader_objects : enable

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

layout(location = 0) out vec4 outColor;

void main() {
    outColor = ubo.color;
}