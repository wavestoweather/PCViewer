#version 450
layout(std430, binding = 0) buffer infos{
	float alphaMultiplier;
	uint clipNormalize;
	uint amtOfAttributes;
	float offset;
	uint scale;				//scale to be used: 0->Normal, 1->logarithmic, 2->squareroot
	float Fov;
	uint relative;
	vec4 cameraPos;			//contains in w the max size of points when normalized
	vec4 grey;
	vec4 boundingRectMin;
	vec4 boundingRectMax;
	vec4 clippingRectMin;
	vec4 clippingRectMax;
	mat4 mvp;
	float attributeInfos[];
}

layout(std430, binding = 1) buffer data{
	float d[];
} data;

layout(location = 0) in uint attributeIndex;
layout(location = 1) in uint dataIndex;
layout(location = 2) in uvec3 posIndices; //indices for the position coordinates
layout(location = 3) in bool active;
layout(location = 0) out vec4 colorV;

void main() {
    colorV = vec4(infos.attributeInfos[attributeIndex*7],infos.attributeInfos[attributeIndex*7 + 1], infos.attributeInfos[attributeIndex*7 + 2], infos.attributeInfos[attributeIndex*7] + 3);
	vec4 pos = vec4(data.d[dataIndex*infos.amtOfAttributes + posIndices.x], data.d[dataIndex*infos.amtOfAttributes + posIndices.y], data.d[dataIndex*infos.amtOfAttributes + posIndices.z], 1);
    pos.y += infos.offset * infos.attributeInfos[attributeIndex*7 + 4];
    gl_Position = projMatrix * mvPosition;
    vert = pos.xyz;
    if (true)
    {
		bool clipped = true;
		if(pos.x<=infos.clippingRectMax.x && pos.y<=infos.clippingRectMax.y && pos.z<=infos.clippingRectMax.z
			&& pos.x>=info.clippingRectMin.x && pos.y>=infos.clippingRectMin.y && pos.z >= inofs.clippingRectMin.z){
			clipped = false;
		}
		if (infos.clipNormalize&2 && (clipped || !active)){
			colorV = vec4(0,0,0,0);
		}
		else if(!(infos.clipNormalize&2) && (clipped || !active)){
			colorV = infose.grey;
		}

        float pdistance = distance(vert, infos.cameraPos.xyz);
        if (infos.alphaMultiplier != 0)
        {
            colorV.a *= infos.alphaMultiplier/pdistance;
        }

        float radius = data.d[dataIndex*infos.amtOfAttributes + attributeIndex);

		if(infos.clipNormalize&1){
			float min = infos.attributeInfos[attributeIndex * 7 + 5];
			float max = infos.attributeInfos[attributeIndex * 7 + 6];
			radius = (radius - min)/(max - min);
		}
		else{
			radius = abs(radius);		//safety normalization to avoid negative numbers(important for log and sqrt)
		}

		switch(infos.scale){
			case 0: break; //standard case, do nothing
			case 1: radius = log(radius);
					break;
			case 2: radius = sqrt(radius);
					break;
		}

        if(infos.relative) pdistance = 1;
        radius *= (45.0/infos.FoV) / pdistance;

        gl_PointSize = interValue;
    }
};