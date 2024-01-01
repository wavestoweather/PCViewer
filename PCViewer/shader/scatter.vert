#version 450
#extension GL_GOOGLE_include_directive : enable

layout(std430, binding = 0) buffer Dat{
    float d[];
} data;

layout(binding = 1, r8) uniform imageBuffer act; 

layout(std430, binding = 2) buffer Ind{
    uint i[];
} indices;

layout(std430, binding = 3) buffer Buffer{
    float spacing;
    float radius;
    uint inactiveSquare;    //last bit inactive, second last square
    uint matrixSize;        //amt of matrices in x and y direction
    vec4 color;
    vec4 inactiveColor;
    float minMax[];
}ubo;

layout(push_constant) uniform PushConstants{
    uint posX;
    uint posY;
    uint xAttr;
    uint yAttr;
    uint discardBits;
}pConst;

layout(location = 0) out vec4 color;
layout(location = 1) out uint square;
layout(location = 2) out uint disc; //uint to indicate if the fragments should be discarded

#include"dataAccess.glsl"

const uint discardNone = 0, discardActive = 1, discardInactive = 2;

void main() {
    uint ind = indices.i[gl_VertexIndex];
    bool lineActive = bool(imageLoad(act, int(ind)));
    square = ubo.inactiveSquare & (1 << 1);
    switch(pConst.discardBits){
    case discardNone: disc = 0; break;
    case discardActive: lineActive ? disc = 1 : disc = 0; break;
    case discardInactive: lineActive ? disc = 0: disc = 1; break;
    }
    if(lineActive || bool(ubo.inactiveSquare & 1)){
        float x = getPackedData(ind, pConst.xAttr);
        float y = getPackedData(ind, pConst.yAttr);
        //normalization
        x = (x - ubo.minMax[2 * pConst.xAttr]) / (ubo.minMax[2 * pConst.xAttr + 1] - ubo.minMax[2 * pConst.xAttr]);
        y = 1 - (y - ubo.minMax[2 * pConst.yAttr]) / (ubo.minMax[2 * pConst.yAttr + 1] - ubo.minMax[2 * pConst.yAttr]);
        gl_PointSize = ubo.radius;
        if(x < 0 || y < 0 || x > 1 || y > 1) gl_Position = vec4(-2,-2,-2,1);
        else{
            float blockTotalWidth = 1.0 / (ubo.matrixSize - 1);
            float blockWidth = blockTotalWidth * (1.0 - ubo.spacing);
            x *= blockWidth;
            x += pConst.posX * blockTotalWidth;
            y *= blockWidth;
            y += pConst.posY * blockTotalWidth;
            x = x * 2 - 1;
            y = y * 2 - 1;
            gl_Position = vec4(x, y, .5,1);
        }
        if(lineActive) color = ubo.color;
        else color = ubo.inactiveColor;
    }
    else{
        gl_Position = vec4(-2,-2,-2,1);
        color = ubo.inactiveColor;
    }
}