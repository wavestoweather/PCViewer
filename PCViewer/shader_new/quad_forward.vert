#version 450

layout(push_constant) uniform PC{
    vec4 span;  // first vec2 left bottom, second vec2 right top, all in normalized device coords
};

void main(){
    if((gl_VertexIndex >> 1) > 0)
        gl_Position.x = span.z;
    else
        gl_Position.x = span.x;
    
    if(gl_VertexIndex == 0 || gl_VertexIndex == 3)
        gl_Position.y = span.y;
    else
        gl_Position.y = span.w;
}