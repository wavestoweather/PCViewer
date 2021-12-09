#version 450

layout(location = 0) in vec4 col;
layout(location = 1) flat in uint square;
layout(location = 2) flat in uint disc;
layout(location = 0) out vec4 outColor;

void main() {
    if(bool(disc)) discard;
    outColor = col;
    vec2 coord = gl_PointCoord - vec2(0.5);  //from [0,1] to [-0.5,0.5]
    if(length(coord) > 0.5 && !bool(square)) //outside of circle radius and not square?
        discard;
}