find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

aux_source_directory(../test/test_data DIR_TEST_DATA)
aux_source_directory(../test/test_layer DIR_TEST_LAYER)
aux_source_directory(../test/test_net DIR_TEST_NET)
aux_source_directory(../test/test_runtime DIR_TEST_RUNTIME)

set(link_lib glog::glog GTest::gtest)
if(!WIN32)
    set(link_lib "${link_lib} pthread")
endif()

set(link_math_lib  ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_executable(test_tiny test_main.cpp ${DIR_TEST_DATA} ${DIR_TEST_LAYER} ${DIR_TEST_NET} ${DIR_TEST_RUNTIME})
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O0")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fopenmp -march=native")
endif()

target_link_libraries(test_tiny ${link_lib} ${link_math_lib})
target_link_directories(test_tiny PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(test_tiny tinyinfer)

if (MSVC)
    add_custom_command(TARGET test_tiny POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:tinyinfer>/tinyinfer.dll"
            $<TARGET_FILE_DIR:test_tiny>)
endif()

target_include_directories(test_tiny PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_tiny PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(test_tiny PUBLIC ${Armadillo_INCLUDE_DIR})


