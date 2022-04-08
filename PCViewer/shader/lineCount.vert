#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

layout(binding = 0) uniform Infos{
    uint amtOfDataPoints;
    uint aBins, bBins;
    uint padding;
};

layout(location = 0) in float a;
layout(location = 1) in float b;

layout(location = 0) out float increment;

void main(){
    float x = fract(a) * 2 - 1; // transforming from [0,1] to [-1,1]
    float y = fract(b) * 2 - 1;
    //x = float(gl_VertexIndex) / float(amtOfDataPoints) * 2 - 1;
    //y = float(gl_VertexIndex * 2 % amtOfDataPoints) / float(amtOfDataPoints) * 2 - 1;
    //gl_Position.xy = vec2(0, 0);
    increment = 1;
    gl_PointSize = 1;
}