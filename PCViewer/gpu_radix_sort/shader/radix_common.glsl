// defines which are needed for this code to compile properly:
// BITS_PER_PASS        : default 4
// ELEMENTS_PER_THREAD  : default 4
// SRC_TYPE             : default float_vec
// HAS_PAYLOAD          : default undefined
// PAYLOAD_TYPE         : default uint_vec

#ifndef radix_common_glsl
#define radix_common_glsl

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_buffer_reference2: require
#extension GL_EXT_buffer_reference_uvec2: require
#extension GL_EXT_shader_explicit_arithmetic_types: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_shader_subgroup_ballot: enable
#extension GL_KHR_shader_subgroup_arithmetic: enable

#ifndef BITS_PER_PASS
    #pragma diagnostic warning "Missing define BITS_PER_PASS, setting to default value 4"
    #define BITS_PER_PASS 4
#endif
#ifndef ELEMENTS_PER_THREAD
    #pragma diagnostic warning "Missing define ELEMENTS_PER_THREAD, setting to default value 4"
    #define ELEMENTS_PER_THREAD 4
#endif
#ifndef SRC_TYPE
    #pragma diagnostic warning "Missing define SRC_TYPE, setting to default value float_vec"
    #define SRC_TYPE float_vec
    #define SRC_T float
#endif
#ifdef HAS_PAYLOAD
#ifndef PAYLOAD_TYPE
    #pragma diagnostic warning "Missing define PAYLOAD_TYPE, setting to default value none_type"
    #define PAYLOAD_TYPE uint_vec
    #define PAYLOAD_T uint
#endif
#endif
//#if SRC_TYPE != PAYLOAD_TYPE
//    #define PAYLOAD_IS_DIFFERENT_TYPE
//    #pragma message "Payload type is different than value type"
//#else
//    #pragma message "Payload type is the same as value type"
//#endif
#define PAYLOAD_IS_DIFFERENT_TYPE

layout(push_constant) uniform push_constants{
    uint64_t src_values;
    uint64_t dst_values;
    uint64_t src_payload;
    uint64_t dst_payload;
    uint64_t scan_scratch;
    uint64_t scratch_reduced;
    uint     bit_shift;
    uint     num_keys;					
    int      num_blocks_per_threadgroup;
    uint     num_thread_groups;
    uint     num_thread_groups_with_additional_blocks;
    uint     num_reduce_threadgroup_per_bin;
    uint     num_scan_values;
};

layout(buffer_reference, scalar) buffer ubyte_vec   { uint8_t d[]; };
layout(buffer_reference, scalar) buffer byte_vec    { int8_t d[]; };
layout(buffer_reference, scalar) buffer ushort_vec  { uint16_t d[]; };
layout(buffer_reference, scalar) buffer short_vec   { int16_t d[]; };
layout(buffer_reference, scalar) buffer uint_vec    { uint d[]; };
layout(buffer_reference, scalar) buffer int_vec     { int d[]; };
layout(buffer_reference, scalar) buffer ulong_vec   { uint64_t d[]; };
layout(buffer_reference, scalar) buffer long_vec    { int64_t d[]; };
layout(buffer_reference, scalar) buffer half_vec    { float16_t d[]; };
layout(buffer_reference, scalar) buffer float_vec   { float d[]; };
layout(buffer_reference, scalar) buffer double_vec  { double d[]; };

layout(constant_id = 0) const uint local_size       = 128;

layout(local_size_x_id = 0) in;

// remapping to ffx variables
const uint bin_count = 1 << BITS_PER_PASS;
const uint bit_mask = bin_count - 1;
const uint FFX_PARALLELSORT_SORT_BIN_COUNT = bin_count;
const uint elements_per_thread = ELEMENTS_PER_THREAD;
const uint FFX_PARALLELSORT_ELEMENTS_PER_THREAD = elements_per_thread;
const uint FFX_PARALLELSORT_THREADGROUP_SIZE = local_size;
uint localID = gl_LocalInvocationID.x;
uint groupID = gl_WorkGroupID.x;
uint NumThreadGroups = num_thread_groups;
uint NumThreadGroupsWithAdditionalBlocks = num_thread_groups_with_additional_blocks;
uint NumReduceThreadgroupPerBin = num_reduce_threadgroup_per_bin;
int NumBlocksPerThreadGroup = num_blocks_per_threadgroup;
uint NumScanValues = num_scan_values;
uint NumKeys = num_keys;

// data variables
SRC_TYPE        SrcBuffer   = SRC_TYPE(src_values);
SRC_TYPE        DstBuffer   = SRC_TYPE(dst_values);
#ifdef HAS_PAYLOAD
PAYLOAD_TYPE    SrcPayload  = PAYLOAD_TYPE(src_payload);
PAYLOAD_TYPE    DstPayload  = PAYLOAD_TYPE(dst_payload);
#endif
uint_vec        SumTable    = uint_vec(scan_scratch);
uint_vec        ReduceTable = uint_vec(scratch_reduced);
uint_vec        ScanSrc     = uint_vec(scan_scratch);
uint_vec        ScanDst     = uint_vec(scan_scratch);
uint_vec        ScanScratch = uint_vec(scratch_reduced);

// mapping values to uint for radix calculation
uint get_local_key(uint8_t v){ return (uint(v) >> bit_shift) & bit_mask; }
uint get_local_key(int8_t v){ return (uint(v ^ 0x80) >> bit_shift) & bit_mask; }
uint get_local_key(uint16_t v){ return (uint(v) >> bit_shift) & bit_mask; }
uint get_local_key(int16_t v){ return (uint(v ^ 0x8000) >> bit_shift) & bit_mask; }
uint get_local_key(uint v){ return (v >> bit_shift) & bit_mask; }
uint get_local_key(int v){ return (uint(v ^ 0x80000000) >> bit_shift) & bit_mask; }
uint get_local_key(uint64_t v){ return uint(v >> uint64_t(bit_shift)) & bit_mask; }
//uint get_local_key(int64_t v){ return uint((v ^ int64_t(0x8000000000000000)) >> bit_shift) & bit_mask; }

uint get_local_key(float v){ 
    uint val = floatBitsToUint(v);
	uint mask = uint(-int(val >> 31) | 0x80000000);
	return uint((val ^ mask) >> bit_shift) & bit_mask;
}

const uint8_t ubyte_max = uint8_t(0xff);
const uint16_t ushort_max = uint16_t(0xffff);
const uint uint_max = uint(0xffffffff);
const uint64_t ulong_max = uint64_t(0xffffffffffffffffl);
//uint get_local_key(double v){ 
//    uint64_t val = doubleBitsToUint64(v);
//	uint64_t mask = uint64_t(-int64_t(val >> 63) | int64_t(0x8000000000000000));
//	return uint((val ^ mask) >> bit_shift) & bit_mask
//}

shared uint gs_FFX_PARALLELSORT_LDSSums[FFX_PARALLELSORT_THREADGROUP_SIZE];
uint FFX_ParallelSort_ThreadgroupReduce(uint localSum, uint localID){
    // Do wave local reduce
	uint waveReduced = subgroupAdd(localSum);
	// First lane in a wave writes out wave reduction to LDS (this accounts for num waves per group greater than HW wave size)
	// Note that some hardware with very small HW wave sizes (i.e. <= 8) may exhibit issues with this algorithm, and have not been tested.
	uint waveID = gl_SubgroupID;//localID / gl_SubgroupSize;
	if (subgroupElect())
		gs_FFX_PARALLELSORT_LDSSums[waveID] = waveReduced;
	// Wait for everyone to catch up
	barrier();
	// First wave worth of threads sum up wave reductions
	if (waveID == 0)
		waveReduced = subgroupAdd( (localID < FFX_PARALLELSORT_THREADGROUP_SIZE / gl_SubgroupSize) ? gs_FFX_PARALLELSORT_LDSSums[localID] : 0);

    // Returned the reduced sum
	return waveReduced;
}

uint FFX_ParallelSort_BlockScanPrefix(uint localSum, uint localID)
{
    // Do wave local scan-prefix
    uint wavePrefixed = subgroupExclusiveAdd(localSum);

    // Since we are dealing with thread group sizes greater than HW wave size, we need to account for what wave we are in.
    uint waveID = gl_SubgroupID;
    uint laneID = gl_SubgroupInvocationID;

    // Last element in a wave writes out partial sum to LDS
    if (laneID == gl_SubgroupSize - 1)
        gs_FFX_PARALLELSORT_LDSSums[waveID] = wavePrefixed + localSum;

    // Wait for everyone to catch up
    barrier();

    // First wave prefixes partial sums
    if (waveID == 0)
        gs_FFX_PARALLELSORT_LDSSums[localID] = subgroupExclusiveAdd(gs_FFX_PARALLELSORT_LDSSums[localID]);

    // Wait for everyone to catch up
    barrier();

    // Add the partial sums back to each wave prefix
    wavePrefixed += gs_FFX_PARALLELSORT_LDSSums[waveID];

    return wavePrefixed;
}

#endif