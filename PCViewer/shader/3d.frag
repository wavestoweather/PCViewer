#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject{
	vec3 camPos;
	mat4 mvp;
} ubo;
layout(binding = 1) uniform sampler3D texSampler;

layout(location = 0) in vec3 endPos;
layout(location = 0) out vec4 outColor;

void main() {
	vec3 d = endPos-ubo.camPos;
	vec3 dinv = 1/d;
	//vec3 cubeSides = clamp(normalize(ubo.camPos)+.5,0.0,1.0)-1;
	vec3 cubeSides;
	cubeSides.x=float(ubo.camPos.x>0)-.5f;
	cubeSides.y=float(ubo.camPos.y>0)-.5f;
	cubeSides.z=float(ubo.camPos.z>0)-.5f;
	//calculating the starting position
	vec3 t;
	t = (cubeSides-ubo.camPos)*dinv;
	t.x = (t.x>.999999)?-1.0/0:t.x;
	t.y = (t.y>.999999)?-1.0/0:t.y;
	t.z = (t.z>.999999)?-1.0/0:t.z;
	
	float tmax = max(t.x,max(t.y,t.z));
	vec3 startPoint = ubo.camPos+clamp(tmax,.01,1)*d;
	//outColor = vec4(startPoint+.5,1);
	//return;

	float alphaStop = .98f;
	float stepsize = .005f;
	
	outColor = vec4(0,0,0,0);
	d = endPos-startPoint;
	float len = length(d);
	d = normalize(d);
	for(float i = 0; i<len;i+=stepsize){
		vec4 tex = texture(texSampler,(startPoint+i*d)+.5f);
		outColor.xyz+=(1-outColor.a)*tex.xyz*stepsize*20;
		outColor.a+=(1-outColor.a)*tex.a*stepsize*20;
		if(outColor.a>alphaStop){
			break;
		}
	}

	//outColor.a = 1;
}