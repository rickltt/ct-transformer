cmake_minimum_required(VERSION 3.18)
project(CTTransformer)

set(APP_NAME CTTransformerTest)
set(LIB_NAME disfluapi)


set(ONNXRUNTIME_ROOTDIR ${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.19.0)
include_directories(${ONNXRUNTIME_ROOTDIR}/include)
link_directories(${ONNXRUNTIME_ROOTDIR}/lib)

find_package(glog REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

FILE(GLOB_RECURSE SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)

SET(ALL_SRCS ${SOURCE_FILES})

ADD_LIBRARY(${LIB_NAME} SHARED ${ALL_SRCS})

target_link_libraries( ${LIB_NAME} PUBLIC onnxruntime glog::glog)

add_executable(debug ${CMAKE_CURRENT_SOURCE_DIR}/src/debug.cpp)
target_link_libraries(debug PRIVATE ${LIB_NAME})