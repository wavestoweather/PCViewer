#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

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

layout(std430,binding = 2) buffer DataSet{
    float d[];
}data;

#include "dataAccess.glsl"

layout(location = 0) out uint index;
void main() {
    float val = getPackedData(gl_VertexIndex, ubo.attributeInd);
    float y = (((val - ubo.minVal) / (ubo.maxVal - ubo.minVal)) - .5f) * 2;
    index = gl_VertexIndex;
    gl_Position = vec4( ubo.x, y * -1.0f, 0.0, 1.0);
}