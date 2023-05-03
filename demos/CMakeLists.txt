cmake_minimum_required(VERSION 3.16)

project(resnet_demo)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")
endif()

add_executable(resnet_demo resnet_demo.cpp)

target_include_directories(resnet_demo PUBLIC ../include)
target_link_directories(resnet_demo PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(resnet_demo ${OpenCV_LIBS} tinyinfer)

if (MSVC)
    add_custom_command(TARGET resnet_demo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:tinyinfer>/tinyinfer.dll"
            $<TARGET_FILE_DIR:resnet_demo>)
endif()
