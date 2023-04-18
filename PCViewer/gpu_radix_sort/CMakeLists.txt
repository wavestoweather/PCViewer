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

foreach(shader IN LISTS shaders)
    find_program(GLSLC glslc)

    set(current_shader_path ${shader})
    get_filename_component(shader_path ${current_shader_path} DIRECTORY)
    get_filename_component(shader_name ${current_shader_path} NAME)
    set(current_shader_out_path ${shader_path}/compiled_shaders/${shader_name}.spv)

    message("Adding compile shader command: ${current_shader_path} -> ${current_shader_out_path}")

    # custom command for compilation
    add_custom_command(
        OUTPUT ${current_shader_out_path}
        COMMAND ${GLSLC} --target-spv=spv1.4 -o ${current_shader_out_path} ${current_shader_path}
        DEPENDS ${current_shader_path} ${headers}
        IMPLICIT_DEPENDS CXX ${current_shader_path}
        VERBATIM)
    list(APPEND spriv_binary_files ${current_shader_out_path})
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

