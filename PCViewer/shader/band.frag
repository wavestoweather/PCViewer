#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 col;
layout(location = 1) in vec2 side;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = col;
    float alphInv = dot(side, side);
    outColor.a = max(1 ,(1 - alphInv) * 10);
}