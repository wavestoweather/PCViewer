# This cmake file does the follwing things:
#   1. Compiles all shaders to check their validity and create spir-v binaries
#   2. Pack these created binary files into cpp header files which then can be included
#       directly into the source code and are packed into the binary
#   3. Sets up the dependencies such that the shaders are recompiled when any shader changes
#   4. Sets GPU_RADIX_SRC to contain all cpp source files which need to be compiled

# Step 1: Compiling all shaders
file(GLOB shaders "shader/*.comp")
file(GLOB headers "sahder/*.glsl")
file(GLOB gpu_radix_headers "../gpu_radix_sort/FFX_ParallelSort.h" "../gpu_radix_sort/shader/*.hlsl")
make_directory(${CMAKE_CURRENT_SOURCE_DIR}/shader/compiled_shaders)
set(preprocessor_defines "SRC_TYPE=float_vec SRC_T=float SRC_MAX=float(1.0/.0)"
                         "SRC_TYPE=int_vec SRC_T=int SRC_MAX=int\(0x7FFFFFFF\)"
                         "SRC_TYPE=uint_vec SRC_T=uint SRC_MAX=uint\(0xFFFFFFFF\)"
                         "SRC_TYPE=float_vec SRC_T=float SRC_MAX=float(1.0/.0) HAS_PAYLOAD")

set(data_type_names 
        float    
        int      
        uint     
        float_32)

find_program(GLSLC glslc)
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

add_custom_target(compile_shaders_sort DEPENDS ${spriv_binary_files}) # custom target is needed to add the compile dpendency

# Step 2: Converting shaders to importable binary headers
set(output_header shaders.h)
set(output_variable_name spriv_binary_files)
set(output_namespace_name ShaderBinaries)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pack_binary_files.cmake.in ${CMAKE_CURRENT_SOURCE_DIR}/pack_binary_files.cmake @ONLY)
add_custom_target(delete_header COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_SOURCE_DIR}/${output_header})
add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${output_header}
                   COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/pack_binary_files.cmake
                   DEPENDS ${spriv_binary_files}
                   COMMENT "Packing binary files into header")

add_custom_target(packed_shaders DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${output_header})

# Step 3: Setting up the dependencies to triger re-compilation on change
#           and building shaders/converting them before building main application
add_dependencies(delete_header compile_shaders_sort)
add_dependencies(packed_shaders delete_header)
add_dependencies(PCViewer_new packed_shaders)
