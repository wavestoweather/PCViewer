struct push_constants{
    uint     num_keys_index;
    uint     max_number_threadgroups;
    uint     bit_shift;
    uint     num_blocks_per_threadgroup;

    uint num_keys;					// unused, are handed over via push constants
    int  num_blocks_per_threadgroup; 	// unused, are handed over via push constants
    uint num_thread_groups;
    uint num_thread_groups_with_additional_blocks;
    uint num_reduce_threadgroup_per_bin;
    uint num_scan_values;
    uint64_t src_values;
    uint64_t dst_values;
    uint64_t src_payload;
    uint64_t dst_payload;
    uint64_t sum_table;
    uint64_t reduce_table;
    uint64_t scratch_data;
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
layout(constant_id = 1) const uint bits_per_pass    = 4;
                        const uint bin_count        = 1 << bits_per_pass;
layout(constant_id = 2) const uint elements_per_thread = 4;
layout(constant_id = 3) const uint data_type        = float_type;
                        const uint block_size = elements_per_thread * local_size;
layout(local_size_x_id = 0) in;