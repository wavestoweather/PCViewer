#version 450

#include "scatterplot_forms.glsl"

layout(location = 0) in vec4 color;
layout(location = 1) flat in uint form;
layout(location = 0) out vec4 out_color;

void main(){
    vec2 coord = gl_PointCoord - vec2(.5);
    if(form == form_circle && length(coord) > .5)
        discard;
    out_color = color;
}