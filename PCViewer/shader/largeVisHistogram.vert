#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2: require
#extension GL_EXT_scalar_block_layout: enable

layout(buffer_reference, scalar) buffer UVec {uint i[];};

layout(push_constant) uniform constants{
	uint64_t        histValues;
    float           yLow;
    float           yHigh;
    float           xStart;
    float           xEnd;
    uint            histValuesCount;
    float           alpha;
};

layout(location = 0) out vec4 color;	//color contains the count value in its alpha channel

void main() {
    uint count = UVec(histValues).i[gl_VertexIndex >> 1];
    gl_Position.x = (gl_VertexIndex & 1) == 0 ? xStart: xEnd;
    gl_Position.y = (gl_VertexIndex >> 1) / (histValuesCount - 1) * (yHigh - yLow) + yLow;
    gl_Position.y = gl_Position.y * 2 - 1;
    gl_Position.z = 0;
    color = vec4(vec3(float(count)), alpha);
}