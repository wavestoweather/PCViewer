#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject{
	vec3 camPos;
	vec3 cubeSides;
	vec3 lightDir;
	mat4 mvp;
} ubo;
layout(binding = 1) uniform sampler3D texSampler;

layout(location = 0) in vec3 endPos;
layout(location = 0) out vec4 outColor;

float rand(vec3 co)
{
    return fract(sin(dot(co ,vec3(12.9898,78.233, 122.3617))) * 43758.5453);
}

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

	const float alphaStop = .98f;
	const float stepsize = .0013f;
	const int lightSteps = 5;
	const float lightStepIncrease = 0.001f;
	const float beerFactor = 1.0f;
	const float densityMultiplier = 100.0f;
	
	//outColor is calculated with gamma correction
	outColor = vec4(0,0,0,0);
	d = endPos-startPoint;
	float len = length(d);
	int iterations = int(len/stepsize);

	startPoint += .5f;

	vec3 step = normalize(d) * stepsize;
	//insert random displacement to startpositon
	startPoint += step * rand(startPoint);
	vec3 lightStep = normalize(-ubo.lightDir) * .002f;
	float transmittance = 1;
	for(int i = 0; i < iterations; i++){

		vec4 tex = texture(texSampler,startPoint);
		if(tex.a > 0.0001f) {
			//computing lighting
			float lightDens = 0;
			vec3 lightPos = startPoint + lightStep;
			for(int j = 0;j<lightSteps;j++){
				if(lightPos.x>1.0f||lightPos.y>1.0f||lightPos.z>1.0f||lightPos.x<0.0f||lightPos.y<0.0f||lightPos.z<0.0f){
					break;
				}
				lightDens += texture(texSampler,lightPos).a * length(lightStep) * densityMultiplier;
				lightStep += lightStepIncrease;
				lightPos += lightStep;
			}

			//lightDens is now the light intensity
			lightDens = clamp(exp(-beerFactor * lightDens),.1f,1.0f);

			//adding the opacity as density
			float curDensity = tex.a * stepsize * densityMultiplier;

			//tex.a *= stepsize * 100;
			//tex.rgb *= tex.a * lightDens;
			//outColor = (1.0f - outColor.a)*tex + outColor;
			outColor.xyz += transmittance * lightDens * curDensity * tex.xyz;

			//transmittance
			transmittance *= 1 - curDensity;
			
			if(transmittance <= 1 - alphaStop){
				break;
			}
		}
		startPoint += step;
	}
	outColor.a = 1 - transmittance;
}