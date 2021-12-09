#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 col;
layout(location = 1) in vec4 side;
layout(location = 2) in vec4 haloColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = col;
    outColor.a = min(outColor.a, 1.0);
    vec2 t = clamp(side.xy * 10 * abs(side.z) / side.w, vec2(0) , vec2(1));
    float halo = min(t.x,t.y);
    halo = clamp(halo, .0, 1.0);
    if(abs(side.w) == 0) halo = 1;
    outColor = mix( outColor, vec4(haloColor.xyz, outColor.a), 1 - halo);
}