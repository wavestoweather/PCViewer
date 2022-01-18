#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 2, binding = 0, r32ui)uniform uimage2D densityImage;

//layout(binding = 0) uniform UniformBufferObject{
//	float alpha;
//	uint amtOfVerts;
//	uint amtOfAttributes;
//	uint padding;
//	vec4 color;
//	vec4 ordering[50];
//} ubo;


layout(location = 0) in vec4 col;

void main() {
	ivec2 imagePos = ivec2(gl_FragCoord.xy);
    ivec2 size = imageSize(densityImage);
    //if(imagePos.x >= 0 && imagePos.x < size.x && imagePos.y >= 0 && imagePos.y < size.y)
    imageAtomicAdd(densityImage, imagePos, 1);
    //imagePos = ivec2(10,10);
    //imageStore(densityImage, imagePos, ivec4(10));
}