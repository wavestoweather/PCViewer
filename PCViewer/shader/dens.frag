#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform sampler2D texSampler;
layout(binding = 1) uniform UniformBufferObject{
	bool enableMapping;
	float radius;
	int imageHeight;
} ubo;
layout(binding = 2) uniform sampler2D ironMap;

layout(location = 0) out vec4 outColor;
layout(location = 1) in vec2 tex;

void main() {
	//Gaussian blur in y direction
	float sdev = (2*ubo.radius*ubo.imageHeight)/3;
	float prefac = 1/(sqrt(2*3.141593f*pow(sdev,2)));
	float yStep = 1.0/ubo.imageHeight;
	float divider = 0;
	outColor = vec4(0,0,0,0);
	for(float i = (tex.y-ubo.radius)*ubo.imageHeight;i<(tex.y+ubo.radius)*ubo.imageHeight;i+=1){
		if(i>0&&i<ubo.imageHeight){
			vec2 curT = vec2(tex.x,i/ubo.imageHeight);
			vec4 col = texture(texSampler, curT);
			float gaussianFac = prefac*exp(-(pow(i-tex.y*ubo.imageHeight,2)/pow(sdev,2)));
			outColor += col * gaussianFac;
			divider += gaussianFac;
		}
	}
	//normalization
    outColor /= divider;
	//mapping a ironMap Texture to it
	if(ubo.enableMapping){
		outColor = texture(ironMap, vec2(max(max(outColor.x,outColor.y),outColor.z),.5f));
	}
}