#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 col;
layout(location = 1) in vec3 side;
layout(location = 2) in vec4 haloColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = col;
    outColor.a = min(outColor.a, 1.0);
    float heightS = 10 * sqrt(abs(side.z)); //height is negative -> invert
    float halo = dot(side.xy, side.xy) * heightS - heightS + 1;
    halo = clamp(halo, .0, 1.0);
    outColor = mix( outColor, vec4(haloColor.xyz, outColor.a), halo);
}