create_test_sourcelist(Tests
    CommonCppTests.cpp
    radix_sort_test.cpp
    struct_to_json_tests.cpp
    iterator_test.cpp
    ranges_test.cpp
    kd_tree_test.cpp
    shader_test.cpp
    gpu_tsne_test.cpp
    gpu_radix_test.cpp
)

file(GLOB src
    "../imgui_nodes/crude_json.cpp"
    "../workbenches/parallel_coordinates_workbench.cpp"
    "../workbenches/scatterplot_workbench.cpp"
    "../imgui/*.cpp"
    "../structures/globals.cpp"
    "../pipelines/*.cpp"
    "../util/shader_compiler.cpp"
    "../gpu_radix_sort/*.cpp"
)

add_executable(CommonCppTests ${Tests} ${src})
target_link_libraries(CommonCppTests PRIVATE Vulkan::Vulkan ${SDL2_LIBRARIES})
if(DEFINED Vulkan_glslang_FOUND)
    target_link_libraries(CommonCppTests PRIVATE Vulkan::glslang)
endif()
if(DEFINED Vulkan_shaderc_combined_FOUND)
    target_link_libraries(CommonCppTests PRIVATE Vulkan::shaderc_combined)
endif()
if(DEFINED glslang_FOUND)
    target_link_libraries(CommonCppTests PRIVATE glslang::glslang ${glslang_LIBRARIES})
endif()

set(TestsToRun ${Tests})
remove(TestsToRun CommonCppTests.cpp)

foreach(test ${TestsToRun})
    get_filename_component(TName ${test} NAME_WE)
    add_test(NAME ${TName} COMMAND CommonCppTests ${TName})
endforeach()

add_dependencies(CommonCppTests packed_shaders)