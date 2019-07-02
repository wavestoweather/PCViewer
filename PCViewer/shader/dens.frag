#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform sampler2D texSampler;
layout(binding = 1) uniform UniformBufferObject{
	float radius;
} ubo;

layout(location = 0) out vec4 outColor;
layout(location = 1) in vec2 tex;

float gaussianCoeff[21] = float[](0.034250034986995054, 0.037663354261827944, 0.04100473657395996, 0.04419835566427083, 0.0471666741100688, 0.04983350685645082, 0.052127236528799616, 0.05398399351448455, 0.05535060481669371, 0.05618712208380753, 0.056468761205282325, 0.05618712208380753, 0.05535060481669371, 0.05398399351448455, 0.052127236528799616, 0.04983350685645082, 0.0471666741100688, 0.04419835566427083, 0.04100473657395996, 0.037663354261827944, 0.034250034986995054);

void main() {
	//Gaussian blur in y direction
	float yStep = ubo.radius/21;
	float divider = 0;
	outColor = vec4(0,0,0,0);
	for(int i = 0;i<21;i++){
		vec2 curT = tex+vec2(0,yStep*(i-10));
		vec4 col = texture(texSampler, curT);
		outColor += col * gaussianCoeff[i];
		divider += float(curT.y<1&&curT.y>0) * gaussianCoeff[i];
	}
	//normalization
    outColor /= divider;
}