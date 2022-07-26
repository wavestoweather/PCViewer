#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

layout(push_constant) uniform constants{
	uint aAxis;
	uint bAxis;
	uint aSize;
	uint bSize;
};

layout(binding = 0) buffer UniformBufferObject{
	float alpha;
	uint amtOfVerts;
	uint amtOfAttributes;
	float padding;
	uint dataFlags;						//contains additional data flags
	uint plotWidth, plotHeight, alphaMappingType;
	vec4 color;
	vec4 vertexTransformations[];		//x holds the x position, y and z hold the lower and the upper bound respectivley for the first amtOfAttributes positions
} ubo;

layout(location = 0) in uint count;
layout(location = 0) out vec4 color;	//color contains the count value in its alpha channel
layout(location = 1) out vec4 aPosBPos;	//containts 2 vec2: [aPos, bPos], with xPos containing the x and y coordinates of one end of a line

const uint MappingMultiplicative = 0;	// standard
const uint MappingBound01 = 1;			// all resulting alpha values have at least  alpha = .01
const uint MappingConstAlpha = 2;		// when count > 0 alpha value from color is taken

void main() {
	float gap = 2.0f/(ubo.amtOfVerts - 1.0f); //gap is tested, and is correct
	uint subWidth = ubo.plotWidth / (ubo.amtOfVerts - 1);

	uint aIndex = gl_VertexIndex / bSize;
	uint bIndex = gl_VertexIndex % bSize;
	
	float x1 = -1.0f + ubo.vertexTransformations[aAxis].x * gap;
	float x2 = -1.0f + ubo.vertexTransformations[bAxis].x * gap;
	//addding the padding to x
	x1 *= 1-ubo.padding;
	x2 *= 1-ubo.padding;

	float y1 = float(aIndex) / float(aSize);
	float y2 = float(bIndex) / float(bSize);
	// transforming according to axis scaling
	y1 = y1 * ubo.vertexTransformations[aAxis].y + ubo.vertexTransformations[aAxis].z;
	y2 = y2 * ubo.vertexTransformations[bAxis].y + ubo.vertexTransformations[bAxis].z;
	float yDiff = (y1 - y2) * ubo.plotHeight * .5f;
	//y /= (ubo.vertexTransformations[i].z - ubo.vertexTransformations[i].y);
	y1 -= .5f;
	y1 *= -2;
	y2 -= .5f;
	y2 *= -2;

	aPosBPos.x = x1;
	aPosBPos.y = y1;
	aPosBPos.z = x2;
	aPosBPos.w = y2;

	color = ubo.color;
	//analytical calculation of opacity for clusterAmt wiith opacity a and N lines: a_final = 1-(1-a)^N
	switch(ubo.alphaMappingType){
	case MappingMultiplicative:
		color.a = 1.-pow(1.-color.a, count);
	break;
	case MappingBound01:
		color.a = 1.-pow(1.-color.a, count);
		if(count > 1)
			color.a = max(color.a, .01);
	break;
	case MappingConstAlpha:
		if(count == 0)
			color.a = 0;
		break;
	}
	
	//color.a *= 1 / (1 + yDiff * yDiff / (subWidth * subWidth));//;ubWidth / sqrt(subWidth * subWidth + yDiff * yDiff)
}