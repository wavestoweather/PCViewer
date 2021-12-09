#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform sampler2D texSampler;
layout(binding = 1) uniform UniformBufferObject{
	int enableMapping;
	float radius;
	int imageHeight;
	float gap;
	float compare;
} ubo;
layout(binding = 2) uniform sampler2D ironMap;

layout(location = 0) out vec4 outColor;
layout(location = 1) in vec2 tex;

const float a0 = .05f;
const float densityMultiplier = .2f;

void main() {
	//setting the bool to efficiently subtract in the for loop
	bool sub = ubo.compare != -1.0f;
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
			if(sub){
				outColor -= gaussianFac * texture(texSampler, vec2(int(curT.x/ubo.gap) * ubo.gap + ubo.compare,curT.y));
			}
			divider += gaussianFac;
		}
	}
	//normalization
	if(sub){
		outColor = abs(outColor);
	}
    outColor /= divider;
	//transforming linear density to exponential density
	if((ubo.enableMapping & 2) > 0){
		outColor = vec4(1) - pow(vec4(1-a0),outColor / a0 * densityMultiplier);
	}
	//mapping a ironMap Texture to it
	if((ubo.enableMapping & 1) > 0){
		outColor = texture(ironMap, vec2(max(max(outColor.x,outColor.y),outColor.z),.5f));
	}
	//mapping density to grey values
	if((ubo.enableMapping & 4) > 0){
		outColor = vec4(vec3(max(max(outColor.x,outColor.y),outColor.z)),1);
	}
	outColor = clamp(outColor,vec4(0),vec4(1));
}