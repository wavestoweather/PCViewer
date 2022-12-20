create_test_sourcelist(Tests
    CommonCppTests.cpp
    radix_sort_test.cpp
    struct_to_json_tests.cpp
)

file(GLOB src
    "../imgui_nodes/crude_json.cpp"
    "../workbenches/parallel_coordinates_workbench.cpp"
    "../imgui/*.cpp"
    "../structures/globals.cpp"
    "../pipelines/*.cpp"
)

add_executable(CommonCppTests ${Tests} ${src})
target_link_libraries(CommonCppTests Vulkan::Vulkan ${SDL2_LIBRARIES})

set(TestsToRun ${Tests})
remove(TestsToRun CommonCppTests.cpp)

foreach(test ${TestsToRun})
    get_filename_component(TName ${test} NAME_WE)
    add_test(NAME ${TName} COMMAND CommonCppTests ${TName})
endforeach()