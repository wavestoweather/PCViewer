#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

layout(binding = 0) buffer StorageBuffer{
    float[] d;  //discrete structure of values in LineBundles.hpp
}data;

layout(binding = 1) uniform UniformBufferObject{
    float alpha;
    uint amtOfVerts;
    uint amtOfAttributes;
    float padding;
    vec4 color;
    vec4 vertexTransformations[50];        //x holds the x position, y and z hold the lower and the upper bound respectivley
} ubo;

layout(location = 0) in vec3 posIn;     //min, avg, max
layout(location = 1) in uvec2 ids;      //axis, groupNr
layout(location = 0) out vec4 color;
layout(location = 1) out vec4 posOut;   //same as posIn but transformed to normalized device coords: min, avg, max
layout(location = 2) out uvec2 idsOut;

void main() {
    float gap = 2.0f/(ubo.amtOfVerts - 1.0f); //gap is tested, and is correct

    uint i = ids.x;
    float x = -1.0f + ubo.vertexTransformations[i].x * gap;
    //addding the padding to x
    x *= 1-ubo.padding;
    
    vec3 y = posIn - ubo.vertexTransformations[i].y;
    y /= (ubo.vertexTransformations[i].z - ubo.vertexTransformations[i].y);
    y *= -2;
    y += 1;
    posOut = vec4(y, 1);

    gl_Position = vec4( x, y.y, 0.0, 1.0);

    color.x = data.d[0];
    color.y = data.d[1];
    color.z = data.d[2];
    color.w = data.d[3];
    //retrieving alpha value in geometry shader as both axis are needed

    idsOut = ids;
}