cmake_minimum_required(VERSION 3.5)
project(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(INCLUDE_DIR ./include)

include_directories(${INCLUDE_DIR})

set(HEADER_FILES
        ${INCLUDE_DIR}/GaussSeidel.hpp
        ${INCLUDE_DIR}/Matrix.hpp)

set(SOURCE_FILES
        main.cpp include/cudaUtils.h)

add_executable(CUDA ${SOURCE_FILES} ${HEADER_FILES})