// defines which are needed for this code to compile properly:
// BITS_PER_PASS        : default 4
// ELEMENTS_PER_THREAD  : default 4
// SRC_TYPE             : default float_vec
// HAS_PAYLOAD          : default undefined
// PAYLOAD_TYPE         : default uint_vec

#ifndef BITS_PER_PASS
    #warning "Missing define BITS_PER_PASS, setting to default value 4";
    #define BITS_PER_PASS 4
#endif
#ifndef ELEMENTS_PER_THREAD
    #warning "Missing define ELEMENTS_PER_THREAD, setting to default value 4";
    #define ELEMENTS_PER_THREAD 4
#endif
#ifndef SRC_TYPE
    #warning "Missing define SRC_TYPE, setting to default value float_vec";
    #define SRC_TYPE float_vec
#endif
#ifdef HAS_PAYLOAD
#ifndef PAYLOAD_TYPE
    #warning "Missing define PAYLOAD_TYPE, setting to default value none_type";
    #define PAYLOAD_TYPE none_type
#endif
#endif

struct push_constants{
    uint     num_keys_index;
    uint     max_number_threadgroups;
    uint     bit_shift;
    uint     num_blocks_per_threadgroup;

    uint     num_keys;					
    int      num_blocks_per_threadgroup;
    uint     num_thread_groups;
    uint     num_thread_groups_with_additional_blocks;
    uint     num_reduce_threadgroup_per_bin;
    uint     num_scan_values;
    uint64_t src_values;
    uint64_t dst_values;
    uint64_t src_payload;
    uint64_t dst_payload;
    uint64_t sum_table;
    uint64_t reduce_table;
    uint64_t scan_src;
    uint64_t scan_dst;
    uint64_t scan_scratch;
};

const uint none_type = 0;
const uint ubyte_type = 1;
const uint byte_type = 2;
const uint ushort_type = 3;
const uint short_type = 4;
const uint uint_type = 5;
const uint int_type = 6;
const uint uint64_type = 7;
const uint int64_type = 8;
const uint half_type = 9;
const uint float_type = 10;
const uint double_type = 11;

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

// all given over via defines
//layout(constant_id = 1) const uint bits_per_pass    = 4;
//                        const uint bin_count        = 1 << bits_per_pass;
//layout(constant_id = 2) const uint elements_per_thread = 4;
//layout(constant_id = 3) const uint data_type        = float_type; // data type has to be set via defines
//                        const uint block_size       = elements_per_thread * local_size;
layout(local_size_x_id = 0) in;

// remapping to ffx variables
const uint bin_count = 1 << BITS_PER_PASS;
const uint bit_mask = bin_count - 1;
const uint FFX_PARALLELSORT_SORT_BIN_COUNT = bin_count;
const uint FFX_PARALLELSORT_ELEMENTS_PER_THREAD = elements_per_thread;
const uint FFX_PARALLELSORT_THREADGROUP_SIZE = local_size;
const uint localID = gl_LocalInvocationID.x;
const uint groupID = gl_WorkgroupID.x;
const uint NumThreadGroups = num_thread_groups;
const uint NumThreadGroupsWithAdditionalBlocks = num_thread_groups_with_additional_blocks;

// mapping values to uint for radix calculation
uint get_local_key(uint8_t v){ return (uint(v) >> bit_shift) & bit_mask; }
uint get_local_key(int8_t v){ return (uint(v ^ 0x80) >> bit_shift) & bit_mask; }
uint get_local_key(uint16_t v){ return (uint(v) >> bit_shift) & bit_mask; }
uint get_local_key(int16_t v){ return (uint(v ^ 0x8000) >> bit_shift) & bit_mask; }
uint get_local_key(uint v){ return (v >> bit_shift) & bit_mask; }
uint get_local_key(int v){ return (uint(v ^ 0x80000000) >> bit_shift) & bit_mask; }
uint get_local_key(uint64_t v){ return uint(v >> bit_shift) & bit_mask; }
uint get_local_key(int64_t v){ return uint((v ^ 0x8000000000000000) >> bit_shift) & bit_mask; }

uint get_local_key(float v){ 
    uint val = floatBitsToUint(v);
	uint mask = uint(-int(val >> 31) | 0x80000000);
	return uint((val ^ mask) >> bit_shift) & bit_mask;
}
uint get_local_key(double v){ 
    uint64_t val = doubleBitsToUint64(v);
	uint64_t mask = uint64_t(-int64_t(val >> 63) | int64_t(0x8000000000000000));
	return uint((val ^ mask) >> bit_shift) & bit_mask
}

// data variables
SRC_TYPE        SrcBuffer   = SRC_TYPE(src_values);
SRC_TYPE        DstBuffer   = SRC_TYPE(dst_values);
#ifdef HAS_PAYLOAD
PAYLOAD_TYPE    SrcPayload  = PAYLOAD_TYPE(src_payload);
PAYLOAD_TYPE    DstPayload  = PAYLOAD_TYPE(dst_payload);
#endif
uint_vec        SumTable    = uint_vec(sum_table);
uint_vec        ReduceTable = uint_vec(reduce_table);
uint_vec        ScanSrc     = uint_vec(scan_src);
uint_vec        ScanDst     = uint_vec(scan_dst);
uint_vec        ScanScratch = uint_vec(scan_scratch);
