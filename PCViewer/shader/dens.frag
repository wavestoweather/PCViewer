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

float gaussianCoeff[21] = float[](0.034250034986995054, 0.037663354261827944, 0.04100473657395996, 0.04419835566427083, 0.0471666741100688, 0.04983350685645082, 0.052127236528799616, 0.05398399351448455, 0.05535060481669371, 0.05618712208380753, 0.056468761205282325, 0.05618712208380753, 0.05535060481669371, 0.05398399351448455, 0.052127236528799616, 0.04983350685645082, 0.0471666741100688, 0.04419835566427083, 0.04100473657395996, 0.037663354261827944, 0.034250034986995054);

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