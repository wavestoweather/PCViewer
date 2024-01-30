# This cmake file does the follwing things:
#   1. Compiles all shaders to check their validity and create spir-v binaries
#   2. Pack these created binary files into cpp header files which then can be included
#       directly into the source code and are packed into the binary
#   3. Sets up the dependencies such that the shaders are recompiled when any shader changes
#   4. Sets GPU_RADIX_SRC to contain all cpp source files which need to be compiled

# Step 1: Compiling all shaders
file(GLOB shaders "shader/*.comp")
file(GLOB headers "sahder/*.glsl")
file(GLOB conversion_shader "shader/radix_convert.comp")
list(REMOVE_ITEM shaders ${conversion_shader})
file(GLOB gpu_radix_headers "../gpu_radix_sort/FFX_ParallelSort.h" "../gpu_radix_sort/shader/*.hlsl")
make_directory(${CMAKE_CURRENT_SOURCE_DIR}/shader/compiled_shaders)
set(preprocessor_defines 
    "SRC_TYPE=ubyte_vec SRC_T=uint8_t SRC_MAX=ubyte_max"
    "SRC_TYPE=ushort_vec SRC_T=uint16_t SRC_MAX=ushort_max"
    "SRC_TYPE=uint_vec SRC_T=uint SRC_MAX=uint_max"
    "SRC_TYPE=ulong_vec SRC_T=uint64_t SRC_MAX=ulong_max"
    "SRC_TYPE=ubyte_vec SRC_T=uint8_t SRC_MAX=ubyte_max HAS_PAYLOAD"
    "SRC_TYPE=ushort_vec SRC_T=uint16_t SRC_MAX=ushort_max HAS_PAYLOAD"
    "SRC_TYPE=uint_vec SRC_T=uint SRC_MAX=uint_max HAS_PAYLOAD"
    "SRC_TYPE=ulong_vec SRC_T=uint64_t SRC_MAX=ulong_max HAS_PAYLOAD"
)

set(data_type_names 
    ubyte    
    ushort      
    uint     
    ulong
    ubyte_pay   
    ushort_pay
    uint_pay
    ulong_pay
)

set(convert_defines 
    "SRC_TYPE=byte_vec DST_TYPE=ubyte_vec"
    "SRC_TYPE=ubyte_vec DST_TYPE=byte_vec"
    "SRC_TYPE=short_vec DST_TYPE=ushort_vec"
    "SRC_TYPE=ushort_vec DST_TYPE=short_vec"
    "SRC_TYPE=int_vec DST_TYPE=uint_vec"
    "SRC_TYPE=uint_vec DST_TYPE=int_vec"
    "SRC_TYPE=long_vec DST_TYPE=ulong_vec"
    "SRC_TYPE=ulong_vec DST_TYPE=long_vec"
    "SRC_TYPE=float_vec DST_TYPE=uint_vec"
    "SRC_TYPE=uint_vec DST_TYPE=float_vec FLOATS"
    "SRC_TYPE=double_vec DST_TYPE=ulong_vec"
    "SRC_TYPE=ulong_vec DST_TYPE=double_vec FLOATS"
)
set(convert_type_names
    byte_ubyte
    ubyte_byte
    short_ushort
    ushort_short
    int_uint
    uint_int
    long_ulong
    ulong_long
    float_uint
    uint_float
    double_ulong
    ulong_double
)

find_program(GLSLC glslc)
# sorting pipelines --------------------------------------------------------------
foreach(shader IN LISTS shaders)
    foreach(type defines IN ZIP_LISTS data_type_names preprocessor_defines)
        set(macros)
        string(REPLACE " " ";" define_list ${defines})
        foreach(define IN LISTS define_list)
            list(APPEND macros -D${define})
        endforeach()

        set(current_shader_path ${shader})
        get_filename_component(shader_path ${current_shader_path} DIRECTORY)
        get_filename_component(shader_name ${current_shader_path} NAME_WE)
        set(current_shader_out_path ${shader_path}/compiled_shaders/${shader_name}_${type}.comp.spv)

        message(${GLSLC} -O --target-env=vulkan1.2 ${macros} -o ${current_shader_out_path} ${current_shader_path})
        #message("Adding compile shader command: ${current_shader_path} -> ${current_shader_out_path} with compile defines ${macros_string}")
        #message("${current_shader_out_path}")

        # custom command for compilation
        add_custom_command(
            OUTPUT ${current_shader_out_path}
            COMMAND ${GLSLC} -O --target-env=vulkan1.2 ${macros} -o ${current_shader_out_path} ${current_shader_path}
            DEPENDS ${current_shader_path} ${headers}
            IMPLICIT_DEPENDS CXX ${current_shader_path}
            VERBATIM)
        list(APPEND spriv_binary_files ${current_shader_out_path})
    endforeach()
endforeach()
# conversion pipeline -----------------------------------------------------------------
foreach(type defines IN ZIP_LISTS convert_type_names convert_defines)
    set(macros)
    string(REPLACE " " ";" define_list ${defines})
    foreach(define IN LISTS define_list)
        list(APPEND macros -D${define})
    endforeach()

    set(current_shader_path ${conversion_shader})
    get_filename_component(shader_path ${current_shader_path} DIRECTORY)
    get_filename_component(shader_name ${current_shader_path} NAME_WE)
    set(current_shader_out_path ${shader_path}/compiled_shaders/${shader_name}_${type}.comp.spv)

    #message("${current_shader_out_path}")

    # custom command for compilation
    add_custom_command(
        OUTPUT ${current_shader_out_path}
        COMMAND ${GLSLC} -O --target-env=vulkan1.2 ${macros} -o ${current_shader_out_path} ${current_shader_path}
        DEPENDS ${current_shader_path}
        IMPLICIT_DEPENDS CXX ${current_shader_path}
        VERBATIM)
    list(APPEND spriv_binary_files ${current_shader_out_path})
endforeach()

list(LENGTH spriv_binary_files spirv_count)
message("Compiling ${spirv_count} shader for radix sort (Should be 40 sorting + 12 conversion = 52)")

add_custom_target(compile_shaders_sort DEPENDS ${spriv_binary_files}) # custom target is needed to add the compile dpendency

# Step 2: Converting shaders to importable binary headers
set(output_header shaders.h)
set(output_variable_name spriv_binary_files)
set(output_namespace_name ShaderBinaries)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pack_binary_files.cmake.in ${CMAKE_CURRENT_SOURCE_DIR}/pack_binary_files.cmake @ONLY)
# add_custom_target(delete_header COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_SOURCE_DIR}/${output_header} DEPENDS ${spriv_binary_files})
add_custom_target(packed_shaders
                   COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/pack_binary_files.cmake
                   DEPENDS compile_shaders_sort)


# Step 3: Setting up the dependencies to triger re-compilation on change
#           and building shaders/converting them before building main application
add_dependencies(PCViewer_new packed_shaders)
