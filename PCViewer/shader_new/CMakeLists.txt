## compile shaders
file(GLOB shaders "*.comp" "*.vert" "*.frag")

foreach(shader IN LISTS shaders)
    find_program(GLSLC glslc)

    set(current_shader_path ${shader})
    set(current_shader_out_path ${current_shader_path}.spv)

    message("Adding compile shader command: ${current_shader_path} -> ${current_shader_out_path}")

    # custom command for compilation
    add_custom_command(
        OUTPUT ${current_shader_out_path}
        COMMAND ${GLSLC} --target-spv=spv1.4 -o ${current_shader_out_path} ${current_shader_path}
        DEPENDS ${current_shader_path}
        IMPLICIT_DEPENDS CXX ${current_shader_path}
        VERBATIM)
    list(APPEND spriv_binary_files ${current_shader_out_path})
endforeach()

add_custom_target(compile_shaders DEPENDS ${spriv_binary_files})
add_dependencies(PCViewer_new compile_shaders)

### copy shaders

file(GLOB compiled_shaders "*.spv")

foreach(shader IN LISTS compiled_shaders)
    set(current_shader_path ${shader})
    get_filename_component(shader_filename ${current_shader_path} NAME)
    set(current_output_path ${CMAKE_BINARY_DIR}/shader/${shader_filename})

    get_filename_component(current_output_directory ${current_output_path} DIRECTORY)
    file(MAKE_DIRECTORY ${current_output_directory})

    message("Copying spv: ${current_shader_path} -> ${current_output_path}")

    add_custom_command(
           OUTPUT ${current_output_path}
           COMMAND ${CMAKE_COMMAND} -E copy ${current_shader_path} ${current_output_path}
           DEPENDS ${current_shader_path}
           VERBATIM)
    list(APPEND compiled_shaders ${current_output_path})
endforeach()

add_custom_target(copy_shaders DEPENDS ${compiled_shaders})
add_dependencies(PCViewer_new copy_shaders)
