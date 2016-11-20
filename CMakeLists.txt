cmake_minimum_required(VERSION 3.5)
project(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(SOURCE_FILES
    GaussSeidel.hpp
    main.cpp
    Matrix.hpp)

add_executable(CUDA ${SOURCE_FILES})