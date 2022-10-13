#version 450

layout(location = 0) in vec4 col;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = col;
    if (true)
    {
        vec2 coord = gl_PointCoord - vec2(0.5);  //from [0,1] to [-0.5,0.5]
        if(length(coord) > 0.5)                  //outside of circle radius?
            discard;
    }
}


//#version 450
//#extension GL_ARB_separate_shader_objects : enable
//
//layout(binding = 0) uniform UniformBufferObject{
//    vec3 camPos;
//    vec4 color;
//    mat4 mvp;
//    mat4 worldNormals;
//} ubo;
//
//layout(location = 0) in vec3 worldPos;
//layout(location = 1) in vec3 worldNormal;
//layout(location = 0) out vec4 outColor;
//
//vec3 l = normalize(vec3(1,1,1));
//vec3 lightCol = vec3(1,1,1);
//float kd = .6f;
//float ka = .4f;
//float ks = .3f;
//int specExp = 20;
//
//void main() {
//    //using phong lighting to light the surface.
//    vec3 specular = ks * pow(clamp(dot(-reflect(normalize(worldNormal),l),normalize(ubo.camPos-worldPos)),0,1),specExp) * lightCol;
//    vec3 diffuse = kd * clamp(dot(normalize(worldNormal),l),0,1) * ubo.color.xyz * lightCol;
//    vec3 ambient = ka * ubo.color.xyz;
//    outColor = vec4(specular + diffuse + ambient,ubo.color.a);
//}