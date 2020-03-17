#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject{
	vec3 camPos;
	vec3 cubeSides;
	vec3 lightDir;
	mat4 mvp;
} ubo;

//currently the maximum amount of density attributes is 30!
layout(binding = 1) uniform sampler3D texSampler[30];

layout(std430 ,binding = 2) buffer brushInfos{
	uint amtOfAxis;
	uint padding[3];
	float[] brushes;
	//float[] brushes structure (a stands for axis):
	//offset a1, offset a2, ..., offset an, a1, a2, ..., an
	//axis structure:
	//amtOfBrushes, offset b1, offset b2, ..., offset bn, b1, b2, ..., bn
	//brush structure:
	//bIndex, amtOfMinMax, color(vec4), minMax1, minMax2, ..., minMaxN
}bInfo;

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
	const float stepsize = .0013f;		//might change this to a non constant to be changed at runtime
	
	outColor = vec4(0,0,0,0);
	d = endPos-startPoint;
	float len = length(d);
	int iterations = int(len/stepsize);

	startPoint += .5f;

	vec3 step = normalize(d) * stepsize;
	//insert random displacement to startpositon (prevents bad visual effects)
	startPoint += step * rand(startPoint);

	//for every axis/attribute here the last density is stored
	float prevDensity[30] = float[30](0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
	uint brushBits[30] = uint[30](0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff);
	bool brushBorder[30] = bool[30](false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false);
	vec4 brushColor[30] = vec4[30](vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0));

	bool br = false;		//bool to break early
	for(int i = 0; i < iterations && !br; i++){
		//for every axis/attribute
		for(int axis = 0;axis<bInfo.amtOfAxis && !br;++axis){
			int axisOffset = int(bInfo.brushes[axis]);
			//check if there exists a brush on this axis
			if(bool(bInfo.brushes[axisOffset])){		//amtOfBrushes > 0
				//as there exist brushes we get the density for this attribute
				float density = texture(texSampler[axis],startPoint).x;
				//for every brush
				for(int brush = 0;brush<bInfo.brushes[axisOffset] && !br;++brush){
					int brushOffset = int(bInfo.brushes[axisOffset + 1 + brush]);
					//for every MinMax
					for(int minMax = 0;minMax<bInfo.brushes[brushOffset + 1] && !br;++minMax){
						int minMaxOffset = brushOffset + 6 + 2 * minMax;			//+6 as after 1 the brush index lies, then the amtount of Minmax lies and then the color comes in a vec4
						int brushIndex = int(bInfo.brushes[brushOffset]);
						float mi = bInfo.brushes[minMaxOffset];
						float ma = bInfo.brushes[minMaxOffset + 1];
						bool stepInOut = prevDensity[axis] < mi && density >= mi ||
							prevDensity[axis] > mi && density <= mi ||
							prevDensity[axis] > ma && density <= ma ||
							prevDensity[axis] < ma && density >= ma;

						//this are all the things i have to set to test if a surface has to be drawn
						brushBorder[brushIndex] = brushBorder[brushIndex] || stepInOut;
						brushBits[brushIndex] &= (uint((density<mi||density>ma)&&!brushBorder[brushIndex]) << axis) ^ 0xffffffff;
						brushColor[brushIndex] = vec4(bInfo.brushes[brushOffset + 2],bInfo.brushes[brushOffset + 3],bInfo.brushes[brushOffset + 4],bInfo.brushes[brushOffset + 5]);

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
}