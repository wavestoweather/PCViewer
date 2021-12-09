#version 450
#extension GL_GOOGLE_include_directive : enable

layout(std430, binding = 0) buffer Info{
	float alphaMultiplier;
	uint clipNormalize;
	uint amtOfAttributes;
	float offset;
	float Fov;
	uint relative;
	uint padding;
	uint pad;
	vec4 cameraPos;			//contains in w the max size of points when normalized
	uvec4 posIndices;		//indices for the position coordinates
	vec4 grey;
	vec4 boundingRectMin;
	vec4 boundingRectMax;
	vec4 clippingRectMin;
	vec4 clippingRectMax;
	mat4 mvp;
	float attributeInfos[];	//scale to be used: 0->Normal, 1->logarithmic, 2->squareroot
} infos;

layout(std430, binding = 1) buffer Dat{
	float d[];
} data;

layout(binding = 2, r8) uniform imageBuffer act; 

layout (points) in;
layout (location = 0) in uint dataIndex[];
layout (points, max_vertices = 20) out;
layout(location = 0) out vec4 colorV;

#include "dataAccess.glsl"

void main() {
	for(int attributeIndex = 0; attributeIndex<infos.amtOfAttributes; ++attributeIndex){
		float offsetP = infos.attributeInfos[attributeIndex*8 + 4];
		if(offsetP<0){	//discard deactivated items
			gl_Position = vec4(-2,-2,-2,1);
			continue;
		}
		bool activ = bool(imageLoad( act, int(dataIndex[0])));
		colorV = vec4(infos.attributeInfos[attributeIndex*8],infos.attributeInfos[attributeIndex*8 + 1], infos.attributeInfos[attributeIndex*8 + 2], infos.attributeInfos[attributeIndex*8+ 3]);
		vec4 pos = vec4(getPackedData(dataIndex[0], infos.posIndices.x), getPackedData(dataIndex[0], infos.posIndices.y), getPackedData(dataIndex[0], infos.posIndices.z), 1);
		pos.xyz -= infos.boundingRectMin.xyz;
		pos.xyz /= infos.boundingRectMax.xyz - infos.boundingRectMin.xyz;
		pos.y += infos.offset * offsetP;		//offset is applied after normalization to get a higher effect
		gl_Position = infos.mvp * pos;
		vec3 vert = pos.xyz;
		if (true)
		{
			bool clipped = true;
			if(pos.x<=infos.clippingRectMax.x && pos.y<=infos.clippingRectMax.y && pos.z<=infos.clippingRectMax.z
				&& pos.x>=infos.clippingRectMin.x && pos.y>=infos.clippingRectMin.y && pos.z >= infos.clippingRectMin.z){
				clipped = false;
			}
			if (bool(infos.clipNormalize&2) && (clipped || !bool(activ))){
				colorV = vec4(0,0,0,0);
			}
			else if(!bool(infos.clipNormalize&2) && (clipped || !bool(activ))){
				colorV = infos.grey;
			}

		    float pdistance = distance(vert, infos.cameraPos.xyz);
		    if (infos.alphaMultiplier != 0)
		    {
		        colorV.a *= infos.alphaMultiplier/pdistance;
		    }

		    float radius = getPackedData(dataIndex[0], attributeIndex);

			if(bool(infos.clipNormalize&1)){
				float min = infos.attributeInfos[attributeIndex * 8 + 5];
				float max = infos.attributeInfos[attributeIndex * 8 + 6];
				radius = (radius - min)/(max - min);
			}
			else{
				radius = abs(radius);		//safety normalization to avoid negative numbers(important for log and sqrt)
			}

			radius *= infos.cameraPos.w;	//apply scaling to radius before other transformations were applied

			switch(int(infos.attributeInfos[attributeIndex * 8 + 7])){
				case 0: break; //standard case, do nothing
				case 1: radius = log(radius);
						break;
				case 2: radius = sqrt(radius);
						break;
			}

		    if(bool(infos.relative)) pdistance = 1;
		    radius *= (45.0/infos.Fov) / pdistance;

		    gl_PointSize = radius;
			EmitVertex();
			EndPrimitive();
		}
	}
}