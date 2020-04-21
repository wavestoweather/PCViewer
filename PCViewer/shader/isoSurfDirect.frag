#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject{
	vec3 camPos;
	vec3 cubeSides;
	vec3 lightDir;
	mat4 mvp;
} ubo;

//currently the maximum amount of brushes attributes is 30!
layout(binding = 1) uniform sampler3D texSampler[30];

layout(std430 ,binding = 2) buffer brushInfos{
	uint amtOfAxis;
	uint shade;
	float stepSize;
	float isoValue;
	float[] brushes;
	//float[] for the colors of the brushes:
	//color brush0[4*float], color brush1[4*float], ... , color brush n[4*float]
}info;

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
	vec3 startPoint = ubo.camPos+clamp(tmax,.05,1.0)*d;

	const float alphaStop = .98f;
	float stepsize = info.stepSize;			//.0013f;
	float curStepsize = stepsize;
	float growth = 1.5f;
	float maxStepsize = stepsize * 8;
	float isoVal = info.isoValue;
	const int refinmentSteps = 8;
	
	outColor = vec4(0,0,0,0);
	d = endPos-startPoint;
	float len = length(d);
	if(len < .0001) return;
	d = normalize(d);

	startPoint += .5f;

	vec3 step = d * stepsize;
	//insert random displacement to startpositon (prevents bad visual effects)
	startPoint += step * rand(startPoint);

	//for every axis/attribute here the last density is stored
	float prevDensity[30] = float[30](0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
	uint brushBits[30] = uint[30](0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff);
	bool brushBorder[30] = bool[30](false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false);
	vec4 brushColor[30] = vec4[30](vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0));
	
	bool br = false;		//bool to break early
	while(startPoint.x >= 0 && startPoint.x <= 1 && startPoint.y >= 0 && startPoint.y <= 1 && startPoint.z >= 0 && startPoint.z <= 1){
		//for every axis/attribute
		for(int axis = 0;axis<info.amtOfAxis && !br;++axis){
			int axisOffset = int(info.brushes[axis]);
			//check if there exists a brush on this axis
			if(bool(info.brushes[axisOffset])){		//amtOfBrushes > 0
				//as there exist brushes we get the density for this attribute
				float density = texture(texSampler[axis],startPoint).x;
				//for every brush
				for(int brush = 0;brush<info.brushes[axisOffset] && !br;++brush){
					int brushOffset = int(info.brushes[axisOffset + 1 + brush]);
					//for every MinMax
					for(int minMax = 0;minMax<info.brushes[brushOffset + 1] && !br;++minMax){
						int minMaxOffset = brushOffset + 6 + 2 * minMax;			//+6 as after 1 the brush index lies, then the amtount of Minmax lies and then the color comes in a vec4
						int brushIndex = int(info.brushes[brushOffset]);
						float mi = info.brushes[minMaxOffset];
						float ma = info.brushes[minMaxOffset + 1];
						bool stepInOut = prevDensity[axis] < mi && density >= mi ||
							prevDensity[axis] > mi && density <= mi ||
							prevDensity[axis] > ma && density <= ma ||
							prevDensity[axis] < ma && density >= ma;
	
						//this are all the things i have to set to test if a surface has to be drawn
						brushBorder[brushIndex] = brushBorder[brushIndex] || stepInOut;
						brushBits[brushIndex] &= (uint((density<mi||density>ma)&&!brushBorder[brushIndex]) << axis) ^ 0xffffffff;
						brushColor[brushIndex] = vec4(info.brushes[brushOffset + 2],info.brushes[brushOffset + 3],info.brushes[brushOffset + 4],info.brushes[brushOffset + 5]);
	
						//the surface calculation is moved to the end of the for loop, as we have to check for every attribute of the brush if it is inside it
						//if(stepInBot^^stepOutBot || stepInTop^^stepOutTop){			//if we stepped in or out of the min max range blend surface color to total color
						//	vec4 surfColor = vec4(bInfo.brushes[brushOffset + 1,brushOffset + 2,brushOffset + 3,brushOffset + 4]);
						//	outColor.xyz += (1-outColor.w) * surfColor.w * surfColor.xyz;
						//	outColor.w += (1-outColor.w) * surfColor.w;
						//	//check for alphaStop
						//	if(outColor.w > alphaStop) br = true;
						//}
					}
				}
				prevDensity[axis] = density;
			}
		}
	
		//surface rendering
		for(int i = 0;i<30;++i){
			if(brushBorder[i] && brushBits[i] == 0xffffffff){		//surface has to be drawn TODO: shading
				outColor.xyz += (1-outColor.w) * brushColor[i].w * brushColor[i].xyz;
				outColor.w += (1-outColor.w) * brushColor[i].w;
				if(outColor.w>alphaStop) br = true;
			}
			//resetting all brush things
			brushBorder[i] = false;
			brushBits[i] = 0xffffffff;
		}
	
		startPoint += step;
	}

	//dividing the outColor by its w component to account for multiplication with w in the output merger
	outColor.xyz /= outColor.w;
}