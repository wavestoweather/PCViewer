create_test_sourcelist(Tests
    CommonCppTests.cpp
    radix_sort_test.cpp
)

add_executable(CommonCppTests ${Tests})

set(TestsToRun ${Tests})
remove(TestsToRun CommonCppTests.cpp)

foreach(test ${TestsToRun})
    get_filename_component(TName ${test} NAME_WE)
    add_test(NAME ${TName} COMMAND CommonCppTests ${TName})
endforeach()