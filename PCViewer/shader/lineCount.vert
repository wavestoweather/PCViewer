#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

layout(binding = 0) uniform Infos{
    uint amtOfDataPoints;
    uint aBins, bBins;
    uint padding;
};

layout(binding = 1) buffer Activations{
    uint activations[];
};

layout(location = 0) in float a;
layout(location = 1) in float b;

layout(location = 0) out float increment;

void main(){
    uint act = activations[gl_VertexIndex / 32];
    if((act & (1 << (gl_VertexIndex & 31))) == 0){
        gl_Position.xy = vec2(-2, -2);  // outside viewport
        gl_PointSize = 0;   // not visible
        increment = 0;
        return;
    }
    // scaling by 2 - 1/aBins to always have the bins fall inside rendertarget
    float aInv = 1.0/float(aBins);
    float bInv = 1.0/float(bBins);
    float x = (a - .5) * (2 - aInv); // transforming from [0,1] to [-1,1]
    float y = (b - .5) * (2 - bInv);
    //x = float(gl_VertexIndex) / float(amtOfDataPoints) * 2 - 1;
    //y = float(gl_VertexIndex * 2 % amtOfDataPoints) / float(amtOfDataPoints) * 2 - 1;
    gl_Position.xy = vec2(x, y);
    increment = 1;
    gl_PointSize = 1;
}