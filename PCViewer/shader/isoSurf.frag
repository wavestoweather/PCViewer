#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject{
	vec4 camPos;			//the camPos also includes if grid lines shall be drawn(the float indicates the width of the lines)
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
	float[] colors;
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
	vec3 d = endPos-ubo.camPos.xyz;
	vec3 dinv = 1/d;

	//calculating the starting position
	vec3 t;
	t = (ubo.cubeSides-ubo.camPos.xyz)*dinv;
	t.x = (t.x>.999999)?-1.0/0:t.x;
	t.y = (t.y>.999999)?-1.0/0:t.y;
	t.z = (t.z>.999999)?-1.0/0:t.z;
	
	float tmax = max(t.x,max(t.y,t.z));
	vec3 startPoint = ubo.camPos.xyz+clamp(tmax,.05,1.0)*d;

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

	//check for grid lines
	if(bool(ubo.camPos.w)){
		float distanceX = min(startPoint.x,1-startPoint.x);
		float distanceY = min(startPoint.y,1-startPoint.y);
		float distanceZ = min(startPoint.z,1-startPoint.z);
		float sorted[3] = float[3](-1,-1,-1);
		int xCount = 0;
		if(distanceX>distanceY) xCount++;
		if(distanceX>distanceZ) xCount++;
		int yCount = 0;
		if(distanceY>=distanceX) yCount++;
		if(distanceY>distanceZ) yCount++;
		int zCount = 0;
		if(distanceZ>=distanceX) zCount++;
		if(distanceZ>=distanceY) zCount++;
		sorted[xCount] = distanceX;
		sorted[yCount] = distanceY;
		sorted[zCount] = distanceZ;

		float radius = (sorted[0] * sorted[0] + sorted[1] * sorted[1]) / ( ubo.camPos.w * ubo.camPos.w);
		if(radius <= 1)		//blend in grid line
		{
			vec4 brushColor = vec4(1,1,1,1 - radius);
			outColor.xyz += (1-outColor.w) * brushColor.w * brushColor.xyz;
			outColor.w += (1-outColor.w) * brushColor.w;
			if(outColor.w>alphaStop) return;
		}
	}

	vec3 step = d * stepsize;
	//insert random displacement to startpositon (prevents bad visual effects)
	startPoint += step * rand(startPoint);

	float prevDensity[30] = float[30](0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
	int safety = 0;

	while(startPoint.x >= 0 && startPoint.x <= 1 && startPoint.y >= 0 && startPoint.y <= 1 && startPoint.z >= 0 && startPoint.z <= 1){
		bool density = false;
		for(int brush = 0; brush<info.amtOfAxis;++brush){
			float curDensity = texture(texSampler[brush],startPoint).x;
			if(bool(curDensity)){
				density = true;
				if(curStepsize > stepsize)
					break;
			}
			if((prevDensity[brush]<isoVal && curDensity>=isoVal) || (prevDensity[brush]>=isoVal&&curDensity<isoVal)){
				vec4 brushColor = vec4(info.colors[brush*4],info.colors[brush*4+1],info.colors[brush*4+2],info.colors[brush*4+3]);
				if(bool(info.shade)){
					//find exact surface position for lighting
					vec3 curPos = startPoint;
					vec3 prevPos = startPoint - curStepsize * d;
					float precDensity = curDensity;
					for(int i = 0;i<refinmentSteps;++i){
						vec3 tmpPoint = .5f * curPos + .5f * prevPos;
						precDensity = texture(texSampler[brush], tmpPoint).x;
						if(precDensity<isoVal){		//intersection is in interval[tmpPoint , curPos]
							prevPos = tmpPoint;
						}
						else{						//intersection is in interval[prevPoint, tmpPoint]
							curPos = tmpPoint;
						}
					}
					curPos = .5f * prevPos + .5f * curPos;

					float xDir = texture(texSampler[brush],curPos+vec3(stepsize * 4,0,0)).x, 
						yDir = texture(texSampler[brush],curPos+vec3(0,stepsize * 4,0)).x,
						zDir = texture(texSampler[brush],curPos+vec3(0,0,stepsize * 4)).x;
					vec3 normal = -normalize(vec3(xDir - precDensity, yDir - precDensity, zDir - precDensity));
					brushColor.xyz = .5f * brushColor.xyz + max(.5 * dot(normal,normalize(-ubo.lightDir)) * brushColor.xyz , vec3(0)) + max(.4 * pow(dot(normal,normalize(.5*normalize(ubo.camPos.xyz) + .5*normalize(-ubo.lightDir))),50) * vec3(1) , vec3(0));
				}
				outColor.xyz += (1-outColor.w) * brushColor.w * brushColor.xyz;
				outColor.w += (1-outColor.w) * brushColor.w;
				if(outColor.w>alphaStop) return;
			}
			prevDensity[brush] = curDensity;
		}

		if(density){
			if(curStepsize > stepsize)
				startPoint -= curStepsize * d;
			curStepsize = stepsize;
		}
		else{
			curStepsize = clamp(curStepsize  * growth,stepsize,maxStepsize);
		}

		startPoint += curStepsize * d;
	}

	//if we stepped out of the cube and a iso surface was active add surface color
	for(int i = 0;i<info.amtOfAxis;++i){
		if(prevDensity[i] > isoVal){
			vec4 brushColor = vec4(info.colors[i*4],info.colors[i*4+1],info.colors[i*4+2],info.colors[i*4+3]);
			if(bool(info.shade)){
				//find exact surface position for lighting
				vec3 curPos = startPoint;
				vec3 prevPos = startPoint - curStepsize * d;
				float precDensity = prevDensity[i];
				curPos = .5f * prevPos + .5f * curPos;

				float xDir = texture(texSampler[i],curPos+vec3(stepsize * 2,0,0)).x, 
					yDir = texture(texSampler[i],curPos+vec3(0,stepsize * 2,0)).x,
					zDir = texture(texSampler[i],curPos+vec3(0,0,stepsize * 2)).x;
				vec3 normal = -normalize(vec3(xDir - precDensity, yDir - precDensity, zDir - precDensity));
				brushColor.xyz = .5f * brushColor.xyz + max(.5 * dot(normal,normalize(-ubo.lightDir)) * brushColor.xyz , vec3(0)) + max(.4 * pow(dot(normal,normalize(.5*normalize(ubo.camPos.xyz) + .5*normalize(-ubo.lightDir))),50) * vec3(1) , vec3(0));
			}
			outColor.xyz += (1-outColor.w) * brushColor.w * brushColor.xyz;
			outColor.w += (1-outColor.w) * brushColor.w;
		}
	}

	//check for grid lines
	startPoint = endPos +.5f;
	if(bool(ubo.camPos.w)){
		float distanceX = min(startPoint.x,1-startPoint.x);
		float distanceY = min(startPoint.y,1-startPoint.y);
		float distanceZ = min(startPoint.z,1-startPoint.z);
		float sorted[3] = float[3](-1,-1,-1);
		int xCount = 0;
		if(distanceX>distanceY) xCount++;
		if(distanceX>distanceZ) xCount++;
		int yCount = 0;
		if(distanceY>=distanceX) yCount++;
		if(distanceY>distanceZ) yCount++;
		int zCount = 0;
		if(distanceZ>=distanceX) zCount++;
		if(distanceZ>=distanceY) zCount++;
		sorted[xCount] = distanceX;
		sorted[yCount] = distanceY;
		sorted[zCount] = distanceZ;

		float radius = (sorted[0] * sorted[0] + sorted[1] * sorted[1]) / ( ubo.camPos.w * ubo.camPos.w);
		if(radius <= 1)		//blend in grid line
		{
			vec4 brushColor = vec4(1,1,1,1 - radius);
			outColor.xyz += (1-outColor.w) * brushColor.w * brushColor.xyz;
			outColor.w += (1-outColor.w) * brushColor.w;
			if(outColor.w>alphaStop) return;
		}
	}

	//dividing the outColor by its w component to account for multiplication with w in the output merger
	outColor.xyz /= outColor.w;

	////for every axis/attribute here the last density is stored
	//float prevDensity[30] = float[30](0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
	//uint brushBits[30] = uint[30](0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff);
	//bool brushBorder[30] = bool[30](false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false);
	//vec4 brushColor[30] = vec4[30](vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0));
	//
	//bool br = false;		//bool to break early
	//for(int i = 0; i < iterations && !br; i++){
	//	//for every axis/attribute
	//	for(int axis = 0;axis<bInfo.amtOfAxis && !br;++axis){
	//		int axisOffset = int(bInfo.brushes[axis]);
	//		//check if there exists a brush on this axis
	//		if(bool(bInfo.brushes[axisOffset])){		//amtOfBrushes > 0
	//			//as there exist brushes we get the density for this attribute
	//			float density = texture(texSampler[axis],startPoint).x;
	//			//for every brush
	//			for(int brush = 0;brush<bInfo.brushes[axisOffset] && !br;++brush){
	//				int brushOffset = int(bInfo.brushes[axisOffset + 1 + brush]);
	//				//for every MinMax
	//				for(int minMax = 0;minMax<bInfo.brushes[brushOffset + 1] && !br;++minMax){
	//					int minMaxOffset = brushOffset + 6 + 2 * minMax;			//+6 as after 1 the brush index lies, then the amtount of Minmax lies and then the color comes in a vec4
	//					int brushIndex = int(bInfo.brushes[brushOffset]);
	//					float mi = bInfo.brushes[minMaxOffset];
	//					float ma = bInfo.brushes[minMaxOffset + 1];
	//					bool stepInOut = prevDensity[axis] < mi && density >= mi ||
	//						prevDensity[axis] > mi && density <= mi ||
	//						prevDensity[axis] > ma && density <= ma ||
	//						prevDensity[axis] < ma && density >= ma;
	//
	//					//this are all the things i have to set to test if a surface has to be drawn
	//					brushBorder[brushIndex] = brushBorder[brushIndex] || stepInOut;
	//					brushBits[brushIndex] &= (uint((density<mi||density>ma)&&!brushBorder[brushIndex]) << axis) ^ 0xffffffff;
	//					brushColor[brushIndex] = vec4(bInfo.brushes[brushOffset + 2],bInfo.brushes[brushOffset + 3],bInfo.brushes[brushOffset + 4],bInfo.brushes[brushOffset + 5]);
	//
	//					//the surface calculation is moved to the end of the for loop, as we have to check for every attribute of the brush if it is inside it
	//					//if(stepInBot^^stepOutBot || stepInTop^^stepOutTop){			//if we stepped in or out of the min max range blend surface color to total color
	//					//	vec4 surfColor = vec4(bInfo.brushes[brushOffset + 1,brushOffset + 2,brushOffset + 3,brushOffset + 4]);
	//					//	outColor.xyz += (1-outColor.w) * surfColor.w * surfColor.xyz;
	//					//	outColor.w += (1-outColor.w) * surfColor.w;
	//					//	//check for alphaStop
	//					//	if(outColor.w > alphaStop) br = true;
	//					//}
	//				}
	//			}
	//			prevDensity[axis] = density;
	//		}
	//	}
	//
	//	//surface rendering
	//	for(int i = 0;i<30;++i){
	//		if(brushBorder[i] && brushBits[i] == 0xffffffff){		//surface has to be drawn TODO: shading
	//			outColor.xyz += (1-outColor.w) * brushColor[i].w * brushColor[i].xyz;
	//			outColor.w += (1-outColor.w) * brushColor[i].w;
	//			if(outColor.w>alphaStop) br = true;
	//		}
	//		//resetting all brush things
	//		brushBorder[i] = false;
	//		brushBits[i] = 0xffffffff;
	//	}
	//
	//	startPoint += step;
	//}
}