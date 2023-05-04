find_package(benchmark REQUIRED)

aux_source_directory(../bench DIR_BENCH)

add_executable(bench_tiny ${DIR_BENCH})
# # target_link_directories(bench_tiny PUBLIC ${PROJECT_SOURCE_DIR}/lib)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")
endif()

set(link_lib benchmark::benchmark benchmark::benchmark_main)
target_link_directories(bench_tiny PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(bench_tiny ${link_lib} OpenMP::OpenMP_CXX)
target_link_libraries(bench_tiny tinyinfer)

if (MSVC)
    # find tinyinfer dll
    add_custom_command(TARGET bench_tiny POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:tinyinfer>/tinyinfer.dll"
            $<TARGET_FILE_DIR:bench_tiny>)
endif()

target_include_directories(bench_tiny PUBLIC ${benchmark_INCLUDE_DIRS})
target_include_directories(bench_tiny PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(bench_tiny PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(bench_tiny PUBLIC ${Armadillo_INCLUDE_DIR})