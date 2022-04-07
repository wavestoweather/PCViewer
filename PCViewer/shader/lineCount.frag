#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in float increment;
layout(location = 0) out float oInc;

void main(){
    oInc = increment;   // simple pass through
}