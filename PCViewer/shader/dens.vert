#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 inPosition;
layout(location = 1) out vec2 tex;

void main() {
    gl_Position = vec4(inPosition.xy,0,1);
    tex = inPosition.xy / 2 + .5f;
}