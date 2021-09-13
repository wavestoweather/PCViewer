#version 450
#extension GL_GOOGLE_include_directive : enable

layout(std430, binding = 0) buffer Dat{
	float d[];
} data;

layout(binding = 1, r8) uniform imageBuffer act; 

layout(std430, binding = 2) buffer Ind{
    uint i[];
} indices;

layout(std430, binding = 2) buffer Buffer{
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
}pConst;

layout(location = 0) out vec4 color;
layout(location = 1) out uint square;

#include"dataAccess.glsl"

void main() {
    uint ind = indices.i[gl_VertexIndex];
    bool lineActive = bool(imageLoad(act, int(ind)));
    square = ubo.inactiveSquare & (1 << 1);
    if(lineActive || bool(ubo.inactiveSquare & 1)){
        float x = getPackedData(ind, pConst.xAttr);
        float y = getPackedData(ind, pConst.yAttr);
        //normalization
        x = (x - ubo.minMax[2 * pConst.xAttr]) / (ubo.minMax[2 * pConst.xAttr + 1] - ubo.minMax[2 * pConst.xAttr]);
        y = (y - ubo.minMax[2 * pConst.yAttr]) / (ubo.minMax[2 * pConst.yAttr + 1] - ubo.minMax[2 * pConst.yAttr]);
        gl_PointSize = ubo.radius;
        if(x < 0 || y < 0 || x > 1 || y > 1) gl_Position = vec4(-2,-2,-2,1);
        else gl_Position = vec4((x - (ubo.matrixSize - 1) * ubo.spacing) / (ubo.matrixSize) + pConst.posX * (1.0f / (ubo.matrixSize- 1) + ubo.spacing / 2),
                                (y - (ubo.matrixSize - 1) * ubo.spacing) / (ubo.matrixSize) + pConst.posY * (1.0f / (ubo.matrixSize- 1) + ubo.spacing / 2),
                                .5,1);
        if(lineActive) color = ubo.color;
        else color = ubo.inactiveColor;
    }
	else{
        gl_Position = vec4(-2,-2,-2,1);
        color = ubo.inactiveColor;
    }
}