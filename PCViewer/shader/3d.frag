#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject{
	vec3 camPos;
	vec3 cubeSides;
	mat4 mvp;
} ubo;
layout(binding = 1) uniform sampler3D texSampler;

layout(location = 0) in vec3 endPos;
layout(location = 0) out vec4 outColor;

void main() {
	vec3 d = endPos-ubo.camPos;
	vec3 dinv = 1/d;

	//calculating the starting position
	vec3 t;
	t = (ubo.cubeSides-ubo.camPos)*dinv;
	t.x = (t.x>.999999)?-1.0/0:t.x;
	t.y = (t.y>.999999)?-1.0/0:t.y;
	t.z = (t.z>.999999)?-1.0/0:t.z;
	
	float tmax = max(t.x,max(t.y,t.z));
	vec3 startPoint = ubo.camPos+clamp(tmax,.05,1)*d;

	float alphaStop = .98f;
	float stepsize = .0001f;
	
	//outColor is calculated with gamma correction
	outColor = vec4(0,0,0,0);
	d = endPos-startPoint;
	float len = length(d);
	d = normalize(d);
	for(float i = 0; i<len;i+=stepsize){
		vec4 tex = texture(texSampler,(startPoint+i*d)+.5f);
		outColor.xyz+=(outColor.a*tex.a)*pow(tex.xyz,vec3(2.2f))*stepsize*100;
		outColor.a+=(1-outColor.a)*tex.a*stepsize*100;
		if(outColor.a>alphaStop){
			break;
		}
	}
	outColor.xyz = pow(outColor.xyz,vec3(1/2.2f));
}