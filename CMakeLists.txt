cmake_minimum_required(VERSION 3.16)

project(TinyInfer)

set(CMAKE_CXX_STANDARD 17)

include_directories(./include)
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

if (MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    # 假如编译x64，那么SSE2永远可用
    # 假如编译x86，那么SSE2可用的话，就开启SSE2
    if (CMAKE_SIZEOF_VOID_P EQUAL 4)
        # check if SSE2 is available
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("/arch:SSE2" HAS_SSE2)
    else ()
        set(HAS_SSE2 ON)
    endif ()
    if (HAS_SSE2)
        # set __SSE2__ and __XOP__ macros
        add_definitions(-D__SSE2__ -D__XOP__)
    endif ()
    # Force LLVM OpenMP on MSVC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:llvm /openmp:experimental")
endif ()

find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

aux_source_directory(./src/data DIR_DATA)
aux_source_directory(./src/kernel/abstract DIR_ABSTRACT_KERNEL)
aux_source_directory(./src/kernel/details DIR_DETAILS_KERNEL)
aux_source_directory(./src/parser DIR_PARSER)
aux_source_directory(./src/runtime DIR_RUNTIME)

set(link_lib glog::glog)
IF (!WIN32)
    set(link_lib ${link_lib} pthread)
ENDIF ()

set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_library(tinyinfer SHARED ${DIR_DATA} ${DIR_ABSTRACT_KERNEL} ${DIR_DETAILS_KERNEL} ${DIR_PARSER} ${DIR_RUNTIME})
target_link_libraries(tinyinfer ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)

target_include_directories(tinyinfer PUBLIC ${benchmark_INCLUDE_DIRS})
target_include_directories(tinyinfer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(tinyinfer PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(tinyinfer PUBLIC ${Armadillo_INCLUDE_DIR})

# mathfun library defines
add_compile_definitions(SSE_MATHFUN_WITH_CODE USE_SSE_AUTO)

# 本项目的开发者请使用set(DEVELOPMENT ON)或者在cmake中添加-DDEVELOPMENT=ON将选项打开
option(DEVELOPMENT ON)
set(DEVELOPMENT ON)
if (${DEVELOPMENT})
    message(STATUS "DEVELOPMENT MODE ON")
    enable_testing()
    add_subdirectory(bench)
    add_subdirectory(test)
endif ()

option(BUILD_DEMO "BUILD THE DEMO PROJECT")

set(BUILD_DEMO ON)

if (BUILD_DEMO)
    MESSAGE(STATUS "BUILD DEMO PROJECT")
    add_subdirectory(demo)
endif ()
