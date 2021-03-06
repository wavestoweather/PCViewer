#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject{
	float x;
	float width;
	float maxVal;
	float minVal;
	uint attributeInd;
	uint amtOfAttributes;
	uint amtOfIndices;
	uint padding;
	vec4 color;
} ubo;

layout( set = 0, binding = 1, r8 ) uniform imageBuffer act;

layout(std430,binding = 2) buffer DataSet{
	float d[];
}data;

void main() {
	uint index = gl_VertexIndex;
	if(index >= ubo.amtOfIndices)
		index -= ubo.amtOfIndices;
	bool a = bool(imageLoad( act, int(index)));
	if(!a){
		gl_Position = vec4(2,2,0,1);
		return;
	}
	float val = data.d[int(index * ubo.amtOfAttributes + ubo.attributeInd)];
	float y = (((val - ubo.minVal) / (ubo.maxVal - ubo.minVal)) - .5f) * 2;
	//index = gl_VertexIndex;
    vec4 pos = vec4( ubo.x, y * -1.0f, 0.0, 1.0);
	if(gl_VertexIndex != index){
		pos += vec4( ubo.width, 0.0, 0.0, 0.0);
	}
	gl_Position = pos;
}